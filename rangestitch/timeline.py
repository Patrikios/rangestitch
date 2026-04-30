from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from datetime import date, datetime, timedelta, timezone
import logging
from numbers import Real
from time import perf_counter
from typing import Any, Literal, Mapping

import polars as pl

# Used only for date-based gap diagnostics in ``interval_stitch()`` to mark
# the first stitched period for each ID, which has no previous block to compare
# against.
# Example: the first stitched row for ``ID="A"`` gets ``Difference = inf``.
_FIRST_PERIOD_SENTINEL = float("inf")

# Keeps all normalized Polars datetime columns on one explicit precision so
# casts and mapped values produce the same dtype throughout ``_datetime_expr()``
# and ``_prepare_frame()``.
# Example: date columns promoted to datetimes become ``pl.Datetime("us")``.
_DATETIME_TIME_UNIT = "us"

# Canonical internal unit names used after ``_normalize_gap_units()`` resolves
# the spellings accepted by the public ``gap_units`` argument.
# Example: ``"minutes"`` is normalized to the canonical value ``"mins"``.
_GapUnit = Literal["auto", "days", "hours", "mins", "secs"]

# Internal switch used while preparing the input frame to decide whether the
# whole stitching run should use date or datetime comparisons.
# Example: if either interval column contains ``2024-01-01T09:30:00``, the
# shared kind becomes ``"datetime"``.
_TemporalKind = Literal["date", "datetime"]

# Human-friendly ``gap_units`` spellings accepted by ``interval_stitch()`` and
# normalized by ``_normalize_gap_units()`` into the canonical ``_GapUnit``
# values consumed by ``_normalize_gap_threshold()``.
# Example: ``"hr"`` and ``"hours"`` both map to ``"hours"``.
_GAP_UNIT_ALIASES: dict[str, _GapUnit] = {
    "auto": "auto",
    "day": "days",
    "days": "days",
    "hour": "hours",
    "hours": "hours",
    "hr": "hours",
    "hrs": "hours",
    "min": "mins",
    "mins": "mins",
    "minute": "mins",
    "minutes": "mins",
    "sec": "secs",
    "secs": "secs",
    "second": "secs",
    "seconds": "secs",
}

# Conversion factors used by ``_normalize_gap_threshold()`` to translate
# numeric user input into a comparable ``timedelta``.
# Example: ``2`` with normalized units ``"hours"`` becomes ``7200`` seconds.
_SECONDS_PER_UNIT: dict[str, int] = {
    "days": 86_400,
    "hours": 3_600,
    "mins": 60,
    "secs": 1,
}

# Module-level logger used only by ``interval_stitch()`` when ``verbose=True``.
# Example: ``logging.getLogger("rangestitch.timeline").setLevel(logging.INFO)``.
_LOGGER = logging.getLogger(__name__)


def _normalize_datetime(value: datetime) -> datetime:
    """Normalize timezone-aware datetimes to a common naive UTC form.

    This helper exists so interval comparisons use one clock even when input
    rows mix offsets or ``Z`` timestamps. It is called by
    ``_parse_temporal_string()``, ``_coerce_date()``, and
    ``_coerce_datetime()`` before values reach ``_prepare_frame()``.

    Example:
        ``datetime(2024, 1, 1, 10, 0, tzinfo=timezone(timedelta(hours=1)))``
        becomes ``datetime(2024, 1, 1, 9, 0)``.
    """

    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_temporal_string(value: str, *, column_name: str) -> date | datetime:
    """Parse ISO-like temporal strings into ``date`` or ``datetime`` objects.

    This helper exists to let the public API accept string-based temporal
    columns in addition to Python and Polars date types. It is used by
    ``_coerce_date()``, ``_coerce_datetime()``, and
    ``_infer_temporal_kind()`` when ``interval_stitch()`` receives string
    input.

    Example:
        ``"2024-01-01"`` becomes ``date(2024, 1, 1)``, while
        ``"2024-01-01T09:00:00Z"`` becomes ``datetime(2024, 1, 1, 9, 0)``.
    """

    normalized = value.strip().replace("Z", "+00:00")
    try:
        return date.fromisoformat(normalized)
    except ValueError:
        try:
            return _normalize_datetime(datetime.fromisoformat(normalized))
        except ValueError as exc:
            raise TypeError(
                f"{column_name} must contain ISO date or datetime strings, got {value!r}"
            ) from exc


def _coerce_date(value: Any, *, column_name: str) -> date:
    """Convert supported temporal inputs into a plain ``datetime.date``.

    This helper exists so date-only workflows can accept Python dates,
    datetimes, and ISO strings through one path. It is used inside
    ``_date_expr()`` when ``_prepare_frame()`` normalizes interval bounds for
    ``interval_stitch()``.

    Example:
        ``"2024-01-01T09:30:00Z"`` becomes ``date(2024, 1, 1)`` and
        ``datetime(2024, 1, 1, 9, 30)`` also becomes ``date(2024, 1, 1)``.
    """

    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return _normalize_datetime(value).date()
    if isinstance(value, str):
        parsed = _parse_temporal_string(value, column_name=column_name)
        return parsed if isinstance(parsed, date) and not isinstance(parsed, datetime) else parsed.date()
    raise TypeError(f"{column_name} must contain date-like values, got {type(value).__name__}")


def _coerce_datetime(value: Any, *, column_name: str) -> datetime:
    """Convert supported temporal inputs into a naive normalized ``datetime``.

    This helper exists so datetime workflows can accept Python dates,
    datetimes, and ISO strings while preserving a single comparison format. It
    is used inside ``_datetime_expr()`` when ``_prepare_frame()`` normalizes
    interval bounds for ``interval_stitch()``.

    Example:
        ``date(2024, 1, 1)`` becomes ``datetime(2024, 1, 1, 0, 0)`` and
        ``"2024-01-01T09:30:00+01:00"`` becomes ``datetime(2024, 1, 1, 8, 30)``.
    """

    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        parsed = _parse_temporal_string(value, column_name=column_name)
        if isinstance(parsed, datetime):
            return parsed
        return datetime.combine(parsed, datetime.min.time())
    raise TypeError(f"{column_name} must contain date-like values, got {type(value).__name__}")


def _dtype_matches(dtype: pl.DataType, base_type: pl.DataType) -> bool:
    """Check Polars dtypes against concrete or parameterized base types.

    This helper exists because Polars temporal dtypes may carry parameters such
    as time units, and direct equality is not always enough. It is used by
    ``_date_expr()``, ``_datetime_expr()``, and ``_infer_temporal_kind()`` to
    decide how much coercion ``interval_stitch()`` needs.

    Example:
        ``_dtype_matches(pl.Datetime("us"), pl.Datetime)`` returns ``True``
        even though the dtype carries a time-unit parameter.
    """

    return dtype == base_type or getattr(dtype, "base_type", lambda: None)() == base_type


def _date_expr(column_name: str, dtype: pl.DataType) -> pl.Expr:
    """Build the Polars expression for date-only interval normalization.

    This helper exists to keep the date coercion logic in expression form so
    Polars can handle already-typed columns efficiently and only fall back to
    Python mapping when needed. It is selected by ``_temporal_expr()`` and used
    by ``_prepare_frame()`` before ``interval_stitch()`` performs stitching.

    Example:
        with ``column_name="From"`` and ``dtype=pl.Datetime("us")``, the
        returned expression behaves like ``pl.col("From").dt.date()``.
    """

    if _dtype_matches(dtype, pl.Date):
        return pl.col(column_name)
    if _dtype_matches(dtype, pl.Datetime):
        return pl.col(column_name).dt.date()
    return pl.col(column_name).map_elements(
        lambda value: _coerce_date(value, column_name=column_name),
        return_dtype=pl.Date,
    )


def _datetime_expr(column_name: str, dtype: pl.DataType) -> pl.Expr:
    """Build the Polars expression for datetime interval normalization.

    This helper exists to preserve datetime precision while still accepting
    date-only and string input. It is selected by ``_temporal_expr()`` and used
    by ``_prepare_frame()`` before ``interval_stitch()`` performs stitching.

    Example:
        with ``column_name="From"`` and ``dtype=pl.Date``, the returned
        expression casts that column to ``pl.Datetime("us")``.
    """

    if _dtype_matches(dtype, pl.Datetime):
        return pl.col(column_name).cast(pl.Datetime(_DATETIME_TIME_UNIT))
    if _dtype_matches(dtype, pl.Date):
        return pl.col(column_name).cast(pl.Datetime(_DATETIME_TIME_UNIT))
    return pl.col(column_name).map_elements(
        lambda value: _coerce_datetime(value, column_name=column_name),
        return_dtype=pl.Datetime(_DATETIME_TIME_UNIT),
    )


def _frame_from_input(data_frame: pl.DataFrame | Iterable[Mapping[str, Any]]) -> pl.DataFrame:
    """Convert supported input shapes into a ``polars.DataFrame``.

    This helper exists so the public function can accept either an existing
    frame or an iterable of row mappings without branching throughout the main
    algorithm. It is only used by ``_prepare_frame()`` near the start of
    ``interval_stitch()``.

    Example:
        ``[{"ID": 1, "From": "2024-01-01", "To": "2024-01-02"}]`` becomes a
        one-row ``pl.DataFrame``.
    """

    return data_frame if isinstance(data_frame, pl.DataFrame) else pl.DataFrame(data_frame)


def _infer_temporal_kind(series: pl.Series, *, column_name: str) -> _TemporalKind:
    """Infer whether one interval column behaves like dates or datetimes.

    This helper exists because string and object columns do not always carry
    enough schema information for Polars alone to decide the comparison
    precision. It is called by ``_resolve_temporal_kind()`` while
    ``_prepare_frame()`` chooses the normalization path for ``interval_stitch()``.

    Example:
        a series containing ``"2024-01-01T09:00:00"`` is inferred as
        ``"datetime"``, while ``"2024-01-01"`` stays ``"date"``.
    """

    if _dtype_matches(series.dtype, pl.Datetime):
        return "datetime"
    if _dtype_matches(series.dtype, pl.Date):
        return "date"

    for value in series.drop_nulls().to_list():
        if isinstance(value, datetime):
            return "datetime"
        if isinstance(value, date):
            continue
        if isinstance(value, str):
            parsed = _parse_temporal_string(value, column_name=column_name)
            if isinstance(parsed, datetime):
                return "datetime"
            continue
        raise TypeError(f"{column_name} must contain date-like values, got {type(value).__name__}")

    return "date"


def _resolve_temporal_kind(
    frame: pl.DataFrame,
    *,
    from_column: str,
    to_column: str,
) -> _TemporalKind:
    """Choose one shared temporal precision for both interval bound columns.

    This helper exists so ``From`` and ``To`` are normalized consistently: if
    either side is datetime-like, both columns are treated as datetimes. It is
    used only by ``_prepare_frame()`` before the main stitching logic runs.

    Example:
        if ``From`` contains dates and ``To`` contains datetimes, the resolved
        kind is ``"datetime"`` for both columns.
    """

    kinds = {
        _infer_temporal_kind(frame.get_column(from_column), column_name=from_column),
        _infer_temporal_kind(frame.get_column(to_column), column_name=to_column),
    }
    return "datetime" if "datetime" in kinds else "date"


def _temporal_expr(column_name: str, dtype: pl.DataType, *, temporal_kind: _TemporalKind) -> pl.Expr:
    """Dispatch to the correct temporal normalization expression builder.

    This helper exists to keep ``_prepare_frame()`` simple while still routing
    each interval column through either date or datetime coercion. It chooses
    between ``_date_expr()`` and ``_datetime_expr()`` for ``interval_stitch()``.

    Example:
        ``_temporal_expr("From", pl.Date, temporal_kind="datetime")`` chooses
        ``_datetime_expr()`` rather than ``_date_expr()``.
    """

    if temporal_kind == "datetime":
        return _datetime_expr(column_name, dtype)
    return _date_expr(column_name, dtype)


def _normalize_column_argument(
    value: str | Sequence[str] | None,
    *,
    argument_name: str,
) -> list[str]:
    """Turn optional column-name arguments into validated string lists.

    This helper exists because the public API accepts either one column name, a
    sequence of names, or ``None`` for several parameters. It is used by
    ``interval_stitch()`` for ``characteristic_*_columns`` and
    ``output_columns`` before validation and projection.

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
    repeated work when the same column is referenced across internal lists. It
    is used by ``_prepare_frame()`` when building the frame slice consumed by
    ``interval_stitch()``.

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

    This helper exists to prevent ambiguous output rules, such as a column
    serving as both an ID and a characteristic source. It is called directly by
    ``interval_stitch()`` before any frame preparation begins.

    Example:
        using ``id_column="ID"`` and
        ``characteristic_beg_columns=["ID"]`` raises ``ValueError``.
    """

    role_columns = [id_column, from_column, to_column, *characteristic_beg_columns, *characteristic_end_columns]
    if len(set(role_columns)) != len(role_columns):
        raise ValueError("column roles must not overlap")


def _prepare_frame(
    data_frame: pl.DataFrame | Iterable[Mapping[str, Any]],
    *,
    id_column: str,
    from_column: str,
    to_column: str,
    characteristic_beg_columns: Sequence[str],
    characteristic_end_columns: Sequence[str],
) -> tuple[pl.DataFrame, _TemporalKind]:
    """Build the normalized working frame consumed by the stitching algorithm.

    This helper exists to centralize input conversion, required-column checks,
    row-order preservation, temporal-type resolution, and null validation. It
    is the main setup step called by ``interval_stitch()`` before any sorting,
    grouping, or gap calculations happen.

    Example:
        a frame with string ``From``/``To`` columns is converted into a working
        frame with normalized temporal dtypes plus ``__rangestitch_input_order``.
    """

    frame = _frame_from_input(data_frame)
    required_columns = _ordered_unique(
        [id_column, from_column, to_column, *characteristic_beg_columns, *characteristic_end_columns]
    )
    _validate_required_columns(frame, required_columns)

    frame = frame.select(required_columns).with_row_index("__rangestitch_input_order")
    temporal_kind = _resolve_temporal_kind(frame, from_column=from_column, to_column=to_column)
    schema = frame.schema
    frame = frame.with_columns(
        _temporal_expr(from_column, schema[from_column], temporal_kind=temporal_kind).alias(from_column),
        _temporal_expr(to_column, schema[to_column], temporal_kind=temporal_kind).alias(to_column),
    )

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


def _validate_output_columns(output_columns: Sequence[str], available_columns: Sequence[str]) -> None:
    """Ensure a requested output projection matches the available result schema.

    This helper exists to fail early with a clear error instead of letting a
    later ``select`` raise a less targeted exception. It is used by
    ``interval_stitch()`` for both empty-result handling and final projection.

    Example:
        requesting ``["ID", "MissingColumn"]`` raises ``ValueError`` if
        ``"MissingColumn"`` is not part of the computed result.
    """

    invalid = [column for column in output_columns if column not in available_columns]
    if invalid:
        invalid_columns = ", ".join(invalid)
        raise ValueError(f"unknown output columns requested: {invalid_columns}")


def _normalize_difference_column(difference_column: str) -> str:
    """Validate the configured name for the optional gap-diagnostics column.

    This helper exists so diagnostics-related validation stays separate from
    the core stitching logic. It is called by ``interval_stitch()`` before
    deciding whether to add the gap column.

    Example:
        ``"GapDays"`` is accepted and returned unchanged, while ``""`` raises
        ``TypeError``.
    """

    if not isinstance(difference_column, str) or not difference_column:
        raise TypeError("difference_column must be a non-empty string")
    return difference_column


def _validate_required_columns(frame: pl.DataFrame, required_columns: Sequence[str]) -> None:
    """Ensure every source column needed by stitching is present in the input.

    This helper exists to produce a focused error before any coercion or sort
    work starts. It is used by ``_prepare_frame()`` after determining which
    columns ``interval_stitch()`` needs.

    Example:
        if the frame has ``["ID", "From"]`` but ``"To"`` is required, the
        helper raises ``ValueError``.
    """

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_columns = ", ".join(missing)
        raise ValueError(f"missing required columns: {missing_columns}")


def _validate_difference_column_name(
    difference_column: str,
    *,
    existing_columns: Sequence[str],
) -> None:
    """Reject gap-diagnostics names that would collide with selected columns.

    This helper exists to avoid silently overwriting real output fields when
    diagnostics are enabled. It is used by ``interval_stitch()`` when building
    either empty or populated results with the optional gap column.

    Example:
        if ``difference_column="From"``, enabling diagnostics raises
        ``ValueError`` because that output name already exists.
    """

    if difference_column in existing_columns:
        raise ValueError("difference_column must not overlap other selected columns")


def _normalize_gap_units(gap_units: str) -> _GapUnit:
    """Normalize user-facing gap-unit spellings into the internal enum values.

    This helper exists so the public API can accept a few human-friendly
    aliases while the rest of the code works with one canonical set of units.
    It is called by ``interval_stitch()`` before gap-threshold normalization.

    Example:
        ``"minutes"`` normalizes to ``"mins"`` and ``"hr"`` normalizes to
        ``"hours"``.
    """

    if not isinstance(gap_units, str):
        raise TypeError("gap_units must be a string")
    normalized = _GAP_UNIT_ALIASES.get(gap_units.lower())
    if normalized is None:
        raise ValueError("gap_units must be one of: auto, days, hours, mins, secs")
    return normalized


def _normalize_gap_threshold(
    gap_threshold: Real | timedelta,
    *,
    gap_units: _GapUnit,
    temporal_kind: _TemporalKind,
) -> timedelta:
    """Convert the configured gap threshold into one comparable ``timedelta``.

    This helper exists because the public API allows both numeric thresholds
    and ``timedelta`` objects, with extra rules for date-only workflows. It is
    called by ``interval_stitch()`` before the block-building expressions
    compare gaps between adjacent periods.

    Example:
        ``gap_threshold=30`` with ``gap_units="mins"`` becomes
        ``timedelta(minutes=30)``, while ``gap_threshold=1`` with
        ``gap_units="auto"`` becomes ``timedelta(days=1)``.
    """

    if isinstance(gap_threshold, bool) or not isinstance(gap_threshold, (Real, timedelta)):
        raise TypeError("gap_threshold must be a non-negative number or datetime.timedelta")

    if isinstance(gap_threshold, timedelta):
        if gap_units != "auto":
            raise ValueError("gap_units must be 'auto' when gap_threshold is a timedelta")
        threshold = gap_threshold
    else:
        if gap_threshold < 0:
            raise ValueError("gap_threshold must be greater than or equal to 0")
        resolved_units = "days" if gap_units == "auto" else gap_units
        threshold = timedelta(seconds=float(gap_threshold) * _SECONDS_PER_UNIT[resolved_units])

    if threshold < timedelta(0):
        raise ValueError("gap_threshold must be greater than or equal to 0")

    if temporal_kind == "date":
        _, remainder = divmod(threshold, timedelta(days=1))
        if remainder != timedelta(0):
            raise ValueError("sub-day gap thresholds require datetime-like from/to columns")

    return threshold


def interval_stitch(
    data_frame: pl.DataFrame | Iterable[Mapping[str, Any]],
    gap_threshold: Real | timedelta = 1,
    gap_units: str = "auto",
    id_column: str = "ID",
    from_column: str = "From",
    to_column: str = "To",
    characteristic_beg_columns: str | Sequence[str] | None = None,
    characteristic_end_columns: str | Sequence[str] | None = None,
    keep_all_periods: bool = False,
    verbose: bool = True,
    output_columns: str | Sequence[str] | None = None,
    include_gap_column: bool = True,
    difference_column: str = "Difference",
) -> pl.DataFrame:
    """Merge and stitch temporal intervals for each entity in a dataset.

    The function groups rows by ``id_column``, sorts them by start date and
    original input order, then merges overlapping or near-adjacent intervals.
    "Begin" characteristic columns come from the first row in each stitched
    block, while "end" characteristic columns come from the row that
    contributes the block's final end date.

    ``from_column`` and ``to_column`` may contain Python ``date`` values,
    Python ``datetime`` values, ISO date strings, or Polars date/datetime
    columns. If either interval column is datetime-like, both columns are
    normalized to datetimes and time-of-day precision is preserved. Otherwise,
    interval bounds are normalized to calendar dates.

    This public function exists as the package's single high-level API. It
    orchestrates the private helpers in three phases: argument normalization
    and validation, frame preparation and temporal coercion, then block
    detection plus aggregation into stitched output rows.

    Example:
        two rows for the same ``ID`` with ``To=2020-01-10`` and
        ``From=2020-01-11`` are stitched into one block when the effective gap
        threshold is one day.

    Args:
        data_frame: A ``polars.DataFrame`` or iterable of row mappings
            containing interval records.
        gap_threshold: Maximum gap allowed between two periods before a new
            stitched block starts. Numeric values are interpreted as days by
            default. A new block starts only when
            ``From - previous To > gap_threshold`` after interval bounds have
            been normalized.
        gap_units: Units used for numeric ``gap_threshold`` values. Supported
            values are ``"auto"``, ``"days"``, ``"hours"``, ``"mins"``, and
            ``"secs"``. ``"auto"`` preserves the legacy day-based behavior.
        id_column: Name of the entity identifier column.
        from_column: Name of the interval start column. Values may be Python
            ``date`` objects, Python ``datetime`` objects, ISO date strings,
            or Polars date/datetime columns.
        to_column: Name of the interval end column. Values may be Python
            ``date`` objects, Python ``datetime`` objects, ISO date strings,
            or Polars date/datetime columns.
        characteristic_beg_columns: Columns whose values should come from the
            first row in a stitched block. If omitted, no "begin"
            characteristic columns are included.
        characteristic_end_columns: Columns whose values should come from the
            row with the stitched block's maximum end date. If omitted, no
            "end" characteristic columns are included.
        keep_all_periods: Compatibility flag used together with
            ``include_gap_column`` to retain gap diagnostics in the output.
        verbose: Whether to emit INFO-level logging for the stitching process.
        output_columns: Optional subset and ordering of result columns.
        include_gap_column: Whether to include the configured gap-diagnostics
            column when gap diagnostics are requested.
        difference_column: Name of the gap-diagnostics output column when
            ``include_gap_column`` is enabled.

    Returns:
        A ``polars.DataFrame`` containing the stitched intervals. The ID column
        is returned as strings. Interval bounds are returned as ``pl.Date``
        values for date workflows and ``pl.Datetime("us")`` values for
        datetime workflows.

    Raises:
        ValueError: If required columns are missing, column roles overlap,
            output columns are invalid, interval bounds are null, or
            ``gap_threshold`` is invalid for the detected interval precision.
        TypeError: If temporal columns contain unsupported value types.
    """
    logger = _LOGGER if verbose else None
    try:
        if logger is not None:
            logger.info(
                "Starting interval_stitch with id_column=%r, from_column=%r, to_column=%r, "
                "gap_threshold=%r, gap_units=%r, keep_all_periods=%s, include_gap_column=%s.",
                id_column,
                from_column,
                to_column,
                gap_threshold,
                gap_units,
                keep_all_periods,
                include_gap_column,
            )

        beg_columns = _normalize_column_argument(
            characteristic_beg_columns,
            argument_name="characteristic_beg_columns",
        )
        end_columns = _normalize_column_argument(
            characteristic_end_columns,
            argument_name="characteristic_end_columns",
        )
        normalized_difference_column = _normalize_difference_column(difference_column)
        requested_output_columns = _normalize_column_argument(output_columns, argument_name="output_columns")
        _validate_role_columns(id_column, from_column, to_column, beg_columns, end_columns)

        if logger is not None:
            logger.info(
                "Validated column configuration with %s begin characteristic columns and %s end characteristic columns.",
                len(beg_columns),
                len(end_columns),
            )

        started_at = perf_counter()
        frame, temporal_kind = _prepare_frame(
            data_frame,
            id_column=id_column,
            from_column=from_column,
            to_column=to_column,
            characteristic_beg_columns=beg_columns,
            characteristic_end_columns=end_columns,
        )
        normalized_gap_units = _normalize_gap_units(gap_units)
        normalized_gap_threshold = _normalize_gap_threshold(
            gap_threshold,
            gap_units=normalized_gap_units,
            temporal_kind=temporal_kind,
        )

        if logger is not None:
            logger.info(
                "Prepared %s rows using %s precision; normalized gap threshold to %s.",
                frame.height,
                temporal_kind,
                normalized_gap_threshold,
            )

        if frame.is_empty():
            available_columns = [id_column, from_column, to_column, *beg_columns, *end_columns]
            if keep_all_periods and include_gap_column:
                _validate_difference_column_name(normalized_difference_column, existing_columns=available_columns)
                available_columns.append(normalized_difference_column)
            if requested_output_columns:
                _validate_output_columns(requested_output_columns, available_columns)
                empty_result = pl.DataFrame(schema={column: pl.Null for column in requested_output_columns})
            else:
                empty_result = pl.DataFrame(schema={column: pl.Null for column in available_columns})

            if logger is not None:
                logger.info("Input data is empty; returning an empty DataFrame with %s columns.", empty_result.width)

            return empty_result

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
        )

        selected_columns = [id_column, from_column, to_column, *beg_columns, *end_columns]
        if keep_all_periods and include_gap_column:
            _validate_difference_column_name(normalized_difference_column, existing_columns=selected_columns)
            gap_difference_expr = pl.col(from_column) - pl.col("__rangestitch_prev_block_to")
            result = (
                result.with_columns(
                    pl.col(to_column).shift(1).over(id_column).alias("__rangestitch_prev_block_to")
                ).with_columns(
                    (
                        pl.when(pl.col("__rangestitch_prev_block_to").is_null())
                        .then(pl.lit(_FIRST_PERIOD_SENTINEL, dtype=pl.Float64))
                        .otherwise(gap_difference_expr.dt.total_days().cast(pl.Float64))
                        .alias(normalized_difference_column)
                        if temporal_kind == "date"
                        else gap_difference_expr.alias(normalized_difference_column)
                    )
                )
            )
            selected_columns.append(normalized_difference_column)

            if logger is not None:
                logger.info("Added gap diagnostics column %r.", normalized_difference_column)

        result = result.select(selected_columns)

        if requested_output_columns:
            _validate_output_columns(requested_output_columns, selected_columns)
            result = result.select(requested_output_columns)

            if logger is not None:
                logger.info("Projected result to requested columns: %s.", requested_output_columns)

        if logger is not None:
            elapsed = perf_counter() - started_at
            logger.info(
                "Interval stitching completed in %.3f secs. Reduced %s input rows to %s output rows.",
                elapsed,
                frame.height,
                result.height,
            )

        return result
    except Exception:
        if logger is not None:
            logger.exception("Interval stitching failed.")
        raise
