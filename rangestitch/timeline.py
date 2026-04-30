from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
import logging
from numbers import Real
from time import perf_counter
from typing import Literal

import polars as pl

# Internal switch used after dtype validation to keep the gap-threshold rules
# and logging paths explicit.
# Example: ``pl.Date`` columns resolve to ``"date"`` and matching
# ``pl.Datetime("us")`` columns resolve to ``"datetime"``.
_TemporalKind = Literal["date", "datetime"]

# Module-level logger for standard library-style package logging.
# Example: ``logging.getLogger("rangestitch").setLevel(logging.INFO)``.
_LOGGER = logging.getLogger(__name__)


def _dtype_matches(dtype: pl.DataType, base_type: pl.DataType) -> bool:
    """Check Polars dtypes against concrete or parameterized base types.

    This helper exists because ``pl.Datetime`` carries parameters such as the
    time unit, so direct equality is not always the right family-level check.
    It is used by ``_resolve_temporal_kind()`` while ``interval_stitch()``
    validates that interval bounds are already typed correctly.

    Example:
        ``_dtype_matches(pl.Datetime("us"), pl.Datetime)`` returns ``True``.
    """

    return dtype == base_type or getattr(dtype, "base_type", lambda: None)() == base_type


def _normalize_column_argument(
    value: str | Sequence[str] | None,
    *,
    argument_name: str,
) -> list[str]:
    """Turn optional column-name arguments into validated string lists.

    This helper exists because the public API accepts either one column name, a
    sequence of names, or ``None`` for characteristic-column arguments. It is
    used by ``interval_stitch()`` before validation and aggregation setup.

    Example:
        ``"StatusBeg"`` becomes ``["StatusBeg"]`` and ``None`` becomes ``[]``.
    """

    if value is None:
        return []
    if isinstance(value, str):
        columns = [value]
    else:
        columns = list(value)

    if not all(isinstance(column, str) and column for column in columns):
        raise TypeError(f"{argument_name} must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{argument_name} must not contain duplicate column names")
    return columns


def _ordered_unique(columns: Sequence[str]) -> list[str]:
    """Deduplicate column names without changing the caller's ordering.

    This helper exists so required-column selection stays stable while avoiding
    repeated work when a column is referenced from multiple logical roles. It
    is used by ``_prepare_frame()`` before the working frame is sliced.

    Example:
        ``["ID", "From", "ID", "To"]`` becomes ``["ID", "From", "To"]``.
    """

    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            ordered.append(column)
    return ordered


def _validate_role_columns(
    id_column: str,
    from_column: str,
    to_column: str,
    characteristic_beg_columns: Sequence[str],
    characteristic_end_columns: Sequence[str],
) -> None:
    """Reject overlapping semantic roles across configured source columns.

    This helper exists to prevent ambiguous stitching rules, such as a column
    serving as both an ID and a characteristic source. It is called directly by
    ``interval_stitch()`` before frame preparation begins.

    Example:
        using ``id_column="ID"`` and
        ``characteristic_beg_columns=["ID"]`` raises ``ValueError``.
    """

    role_columns = [id_column, from_column, to_column, *characteristic_beg_columns, *characteristic_end_columns]
    if len(set(role_columns)) != len(role_columns):
        raise ValueError("column roles must not overlap")


def _validate_required_columns(frame: pl.DataFrame, required_columns: Sequence[str]) -> None:
    """Ensure every source column needed by stitching is present in the input.

    This helper exists to fail early with a focused error before any sorting or
    grouping work starts. It is used by ``_prepare_frame()`` after determining
    the columns ``interval_stitch()`` needs.

    Example:
        if the frame has ``["ID", "From"]`` but ``"To"`` is required, the
        helper raises ``ValueError``.
    """

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_columns = ", ".join(missing)
        raise ValueError(f"missing required columns: {missing_columns}")


def _resolve_temporal_kind(
    frame: pl.DataFrame,
    *,
    from_column: str,
    to_column: str,
) -> _TemporalKind:
    """Validate interval dtypes and resolve whether stitching is date or datetime based.

    This helper exists to keep the public API lean and deterministic: callers
    must provide ``From`` and ``To`` columns that are already typed as Polars
    ``Date`` or matching ``Datetime`` dtypes. It is used by
    ``_prepare_frame()`` before gap handling and sorting.

    Example:
        ``pl.Date`` plus ``pl.Date`` resolves to ``"date"``, while
        ``pl.Datetime("us")`` plus ``pl.Datetime("us")`` resolves to
        ``"datetime"``.
    """

    from_dtype = frame.schema[from_column]
    to_dtype = frame.schema[to_column]

    if _dtype_matches(from_dtype, pl.Date) and _dtype_matches(to_dtype, pl.Date):
        return "date"

    if _dtype_matches(from_dtype, pl.Datetime) and _dtype_matches(to_dtype, pl.Datetime):
        if from_dtype != to_dtype:
            raise TypeError(f"{from_column} and {to_column} must use the same Polars Datetime dtype")
        return "datetime"

    raise TypeError(f"{from_column} and {to_column} must both be pl.Date or matching pl.Datetime columns")


def _prepare_frame(
    data_frame: pl.DataFrame,
    *,
    id_column: str,
    from_column: str,
    to_column: str,
    characteristic_beg_columns: Sequence[str],
    characteristic_end_columns: Sequence[str],
) -> tuple[pl.DataFrame, _TemporalKind]:
    """Build the validated working frame consumed by the stitching algorithm.

    This helper exists to centralize type checks, required-column validation,
    row-order preservation, temporal dtype validation, and null checks. It is
    the main setup step called by ``interval_stitch()`` before any block
    detection or aggregation happens.

    Example:
        a typed frame with ``From`` and ``To`` already stored as ``pl.Date`` is
        sliced to the required columns and augmented with
        ``__rangestitch_input_order``.
    """

    if not isinstance(data_frame, pl.DataFrame):
        raise TypeError("data_frame must be a polars.DataFrame")

    required_columns = _ordered_unique(
        [id_column, from_column, to_column, *characteristic_beg_columns, *characteristic_end_columns]
    )
    _validate_required_columns(data_frame, required_columns)

    frame = data_frame.select(required_columns).with_row_index("__rangestitch_input_order")
    temporal_kind = _resolve_temporal_kind(frame, from_column=from_column, to_column=to_column)

    if frame.get_column(id_column).null_count() > 0:
        raise ValueError(f"{id_column} cannot be null")
    if frame.get_column(from_column).null_count() > 0 or frame.get_column(to_column).null_count() > 0:
        raise ValueError(f"{from_column} and {to_column} cannot be null")

    return frame, temporal_kind


def _build_aggregation_expressions(
    *,
    from_column: str,
    to_column: str,
    characteristic_beg_columns: Sequence[str],
    characteristic_end_columns: Sequence[str],
) -> list[pl.Expr]:
    """Assemble the group-by expressions that collapse one stitched block.

    This helper exists to keep the main group-by readable while encoding the
    library's begin/end characteristic rules in one place. It is used by
    ``interval_stitch()`` inside the final block-level aggregation.

    Example:
        the returned expressions make ``From`` use ``first()``, ``To`` use
        ``max()``, begin columns use ``first()``, and end columns come from the
        row matching ``__rangestitch_block_max_to``.
    """

    expressions = [
        pl.col(from_column).first().alias(from_column),
        pl.col(to_column).max().alias(to_column),
    ]
    expressions.extend(pl.col(column).first().alias(column) for column in characteristic_beg_columns)
    expressions.extend(
        pl.col(column)
        .filter(pl.col(to_column) == pl.col("__rangestitch_block_max_to"))
        .first()
        .alias(column)
        for column in characteristic_end_columns
    )
    return expressions


def _normalize_gap_threshold(
    gap_threshold: Real | timedelta,
    *,
    temporal_kind: _TemporalKind,
) -> timedelta:
    """Convert the configured gap threshold into one comparable ``timedelta``.

    This helper exists because the simplified API still supports day-based
    numeric thresholds and explicit ``timedelta`` values for datetime
    workflows. It is called by ``interval_stitch()`` before adjacent periods
    are compared.

    Example:
        ``gap_threshold=1`` becomes ``timedelta(days=1)``, while
        ``gap_threshold=timedelta(minutes=30)`` stays unchanged.
    """

    if isinstance(gap_threshold, bool) or not isinstance(gap_threshold, (Real, timedelta)):
        raise TypeError("gap_threshold must be a non-negative number or datetime.timedelta")

    threshold = gap_threshold if isinstance(gap_threshold, timedelta) else timedelta(days=float(gap_threshold))

    if threshold < timedelta(0):
        raise ValueError("gap_threshold must be greater than or equal to 0")

    if temporal_kind == "date":
        _, remainder = divmod(threshold, timedelta(days=1))
        if remainder != timedelta(0):
            raise ValueError("date-based stitching requires whole-day gap thresholds")

    return threshold


def _empty_result_schema(
    frame: pl.DataFrame,
    *,
    id_column: str,
    from_column: str,
    to_column: str,
    characteristic_beg_columns: Sequence[str],
    characteristic_end_columns: Sequence[str],
) -> dict[str, pl.DataType]:
    """Build the output schema for empty results.

    This helper exists so empty outputs keep the same typed contract as
    non-empty results instead of degrading to ``Null`` columns. It is used by
    ``interval_stitch()`` when the validated input frame has zero rows.

    Example:
        an empty date-based input still returns ``ID`` as ``pl.String`` and
        ``From``/``To`` as ``pl.Date``.
    """

    schema = {
        id_column: pl.String,
        from_column: frame.schema[from_column],
        to_column: frame.schema[to_column],
    }
    for column in [*characteristic_beg_columns, *characteristic_end_columns]:
        schema[column] = frame.schema[column]
    return schema


def interval_stitch(
    data_frame: pl.DataFrame,
    gap_threshold: Real | timedelta = 1,
    id_column: str = "ID",
    from_column: str = "From",
    to_column: str = "To",
    characteristic_beg_columns: str | Sequence[str] | None = None,
    characteristic_end_columns: str | Sequence[str] | None = None,
) -> pl.DataFrame:
    """Merge and stitch typed temporal intervals for each entity in a DataFrame.

    The function groups rows by ``id_column``, sorts them by start bound and
    original input order, then merges overlapping or near-adjacent intervals.
    "Begin" characteristic columns come from the first row in each stitched
    block, while "end" characteristic columns come from the row that
    contributes the block's final end bound.

    This public function exists as the package's single high-level API. The
    contract is intentionally narrow: callers must provide a ``polars.DataFrame``
    and the interval columns must already be typed as Polars ``Date`` or as
    matching Polars ``Datetime`` dtypes.

    Example:
        two rows for the same ``ID`` with ``To=2020-01-10`` and
        ``From=2020-01-11`` are stitched into one block when
        ``gap_threshold=1``.

    Args:
        data_frame: Input ``polars.DataFrame`` containing interval records.
        gap_threshold: Maximum gap allowed before a new block starts. Numeric
            values are interpreted as days. For sub-day datetime gaps, pass a
            ``datetime.timedelta``.
        id_column: Name of the entity identifier column.
        from_column: Name of the interval start column. Must already be typed
            as ``pl.Date`` or ``pl.Datetime``.
        to_column: Name of the interval end column. Must already be typed as
            ``pl.Date`` or ``pl.Datetime`` and match ``from_column``.
        characteristic_beg_columns: Columns whose values should come from the
            first row in a stitched block.
        characteristic_end_columns: Columns whose values should come from the
            row with the stitched block's maximum end bound.

    Returns:
        A ``polars.DataFrame`` containing stitched intervals. The ID column is
        returned as strings. Temporal columns preserve the validated input
        dtypes.

    Raises:
        TypeError: If ``data_frame`` is not a ``polars.DataFrame``, temporal
            columns are not already typed as Polars dates/datetimes, or
            ``gap_threshold`` has an unsupported type.
        ValueError: If required columns are missing, column roles overlap,
            interval bounds are null, or the gap threshold is invalid for
            date-based stitching.
    """

    log_info_enabled = _LOGGER.isEnabledFor(logging.INFO)
    started_at = perf_counter() if log_info_enabled else None

    if log_info_enabled:
        _LOGGER.info(
            "Starting range stitching with id_column=%r, from_column=%r, to_column=%r, gap_threshold=%r.",
            id_column,
            from_column,
            to_column,
            gap_threshold,
        )

    beg_columns = _normalize_column_argument(
        characteristic_beg_columns,
        argument_name="characteristic_beg_columns",
    )
    end_columns = _normalize_column_argument(
        characteristic_end_columns,
        argument_name="characteristic_end_columns",
    )
    _validate_role_columns(id_column, from_column, to_column, beg_columns, end_columns)

    if log_info_enabled:
        _LOGGER.info(
            "Validated column configuration with %s begin characteristic columns and %s end characteristic columns.",
            len(beg_columns),
            len(end_columns),
        )

    frame, temporal_kind = _prepare_frame(
        data_frame,
        id_column=id_column,
        from_column=from_column,
        to_column=to_column,
        characteristic_beg_columns=beg_columns,
        characteristic_end_columns=end_columns,
    )
    normalized_gap_threshold = _normalize_gap_threshold(
        gap_threshold,
        temporal_kind=temporal_kind,
    )

    if log_info_enabled:
        _LOGGER.info(
            "Prepared %s rows using %s precision; normalized gap threshold to %s.",
            frame.height,
            temporal_kind,
            normalized_gap_threshold,
        )

    selected_columns = [id_column, from_column, to_column, *beg_columns, *end_columns]
    if frame.is_empty():
        empty_result = pl.DataFrame(
            schema=_empty_result_schema(
                frame,
                id_column=id_column,
                from_column=from_column,
                to_column=to_column,
                characteristic_beg_columns=beg_columns,
                characteristic_end_columns=end_columns,
            )
        )

        if log_info_enabled:
            _LOGGER.info("Input data is empty; returning an empty DataFrame with %s columns.", empty_result.width)

        return empty_result.select(selected_columns)

    block_frame = (
        frame.sort([id_column, from_column, "__rangestitch_input_order"])
        .with_columns(pl.col(to_column).cum_max().over(id_column).alias("__rangestitch_running_max_to"))
        .with_columns(
            pl.col("__rangestitch_running_max_to")
            .shift(1)
            .over(id_column)
            .alias("__rangestitch_prev_running_max_to")
        )
        .with_columns(
            (
                pl.col("__rangestitch_prev_running_max_to").is_null()
                | (
                    (pl.col(from_column) - pl.col("__rangestitch_prev_running_max_to"))
                    > pl.lit(normalized_gap_threshold)
                )
            ).alias("__rangestitch_new_block")
        )
        .with_columns(
            pl.col("__rangestitch_new_block")
            .cast(pl.Int64)
            .cum_sum()
            .over(id_column)
            .alias("__rangestitch_block_id")
        )
        .with_columns(
            pl.col(to_column)
            .max()
            .over([id_column, "__rangestitch_block_id"])
            .alias("__rangestitch_block_max_to")
        )
    )

    result = (
        block_frame.group_by([id_column, "__rangestitch_block_id"], maintain_order=True)
        .agg(
            *_build_aggregation_expressions(
                from_column=from_column,
                to_column=to_column,
                characteristic_beg_columns=beg_columns,
                characteristic_end_columns=end_columns,
            )
        )
        .with_columns(pl.col(id_column).cast(pl.String).alias(id_column))
        .select(selected_columns)
    )

    if log_info_enabled and started_at is not None:
        elapsed = perf_counter() - started_at
        _LOGGER.info(
            "Range stitching completed in %.3f secs. Reduced %s input rows to %s output rows.",
            elapsed,
            frame.height,
            result.height,
        )

    return result
