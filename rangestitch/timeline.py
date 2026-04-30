from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from datetime import date, datetime, timedelta, timezone
import logging
from numbers import Real
from time import perf_counter
from typing import Any, Literal, Mapping

import polars as pl

_FIRST_PERIOD_SENTINEL = float("inf")
_DATETIME_TIME_UNIT = "us"
_GapUnit = Literal["auto", "days", "hours", "mins", "secs"]
_TemporalKind = Literal["date", "datetime"]
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
_SECONDS_PER_UNIT: dict[str, int] = {
    "days": 86_400,
    "hours": 3_600,
    "mins": 60,
    "secs": 1,
}
_LOGGER = logging.getLogger(__name__)


def _normalize_datetime(value: datetime) -> datetime:
    """Return a timezone-normalized naive datetime value."""

    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_temporal_string(value: str, *, column_name: str) -> date | datetime:
    """Parse an ISO-like date or datetime string into a temporal value."""

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
    """Normalize a supported date-like value into a ``datetime.date``."""

    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return _normalize_datetime(value).date()
    if isinstance(value, str):
        parsed = _parse_temporal_string(value, column_name=column_name)
        return parsed if isinstance(parsed, date) and not isinstance(parsed, datetime) else parsed.date()
    raise TypeError(f"{column_name} must contain date-like values, got {type(value).__name__}")


def _coerce_datetime(value: Any, *, column_name: str) -> datetime:
    """Normalize a supported temporal value into a naive ``datetime``."""

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
    """Return whether a Polars dtype matches a concrete or parametric base type."""

    return dtype == base_type or getattr(dtype, "base_type", lambda: None)() == base_type


def _date_expr(column_name: str, dtype: pl.DataType) -> pl.Expr:
    """Build a Polars expression that coerces a column to ``pl.Date``."""

    if _dtype_matches(dtype, pl.Date):
        return pl.col(column_name)
    if _dtype_matches(dtype, pl.Datetime):
        return pl.col(column_name).dt.date()
    return pl.col(column_name).map_elements(
        lambda value: _coerce_date(value, column_name=column_name),
        return_dtype=pl.Date,
    )


def _datetime_expr(column_name: str, dtype: pl.DataType) -> pl.Expr:
    """Build a Polars expression that coerces a column to ``pl.Datetime``."""

    if _dtype_matches(dtype, pl.Datetime):
        return pl.col(column_name).cast(pl.Datetime(_DATETIME_TIME_UNIT))
    if _dtype_matches(dtype, pl.Date):
        return pl.col(column_name).cast(pl.Datetime(_DATETIME_TIME_UNIT))
    return pl.col(column_name).map_elements(
        lambda value: _coerce_datetime(value, column_name=column_name),
        return_dtype=pl.Datetime(_DATETIME_TIME_UNIT),
    )


def _frame_from_input(data_frame: pl.DataFrame | Iterable[Mapping[str, Any]]) -> pl.DataFrame:
    """Return a DataFrame for either frame input or row-mapping input."""

    return data_frame if isinstance(data_frame, pl.DataFrame) else pl.DataFrame(data_frame)


def _infer_temporal_kind(series: pl.Series, *, column_name: str) -> _TemporalKind:
    """Infer whether a column should be treated as date-based or datetime-based."""

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
    """Resolve the temporal precision that should be used for interval bounds."""

    kinds = {
        _infer_temporal_kind(frame.get_column(from_column), column_name=from_column),
        _infer_temporal_kind(frame.get_column(to_column), column_name=to_column),
    }
    return "datetime" if "datetime" in kinds else "date"


def _temporal_expr(column_name: str, dtype: pl.DataType, *, temporal_kind: _TemporalKind) -> pl.Expr:
    """Build the normalization expression for the requested temporal precision."""

    if temporal_kind == "datetime":
        return _datetime_expr(column_name, dtype)
    return _date_expr(column_name, dtype)


def _normalize_column_argument(
    value: str | Sequence[str] | None,
    *,
    argument_name: str,
) -> list[str]:
    """Normalize an optional column argument into a validated list of names."""

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
    """Return unique column names while preserving their first-seen order."""

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
    """Ensure configured column roles do not reuse the same source column."""

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
    """Validate required columns and coerce interval bounds to a shared precision."""

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
    """Build the group-by aggregations used to collapse stitched interval blocks."""

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
    """Validate that requested output columns exist in the computed result."""

    invalid = [column for column in output_columns if column not in available_columns]
    if invalid:
        invalid_columns = ", ".join(invalid)
        raise ValueError(f"unknown output columns requested: {invalid_columns}")


def _normalize_difference_column(difference_column: str) -> str:
    """Validate the configured gap-diagnostics output column name."""

    if not isinstance(difference_column, str) or not difference_column:
        raise TypeError("difference_column must be a non-empty string")
    return difference_column


def _validate_required_columns(frame: pl.DataFrame, required_columns: Sequence[str]) -> None:
    """Validate that all requested columns exist in the input DataFrame."""

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_columns = ", ".join(missing)
        raise ValueError(f"missing required columns: {missing_columns}")


def _validate_difference_column_name(
    difference_column: str,
    *,
    existing_columns: Sequence[str],
) -> None:
    """Ensure the diagnostics column does not collide with existing output columns."""

    if difference_column in existing_columns:
        raise ValueError("difference_column must not overlap other selected columns")


def _normalize_gap_units(gap_units: str) -> _GapUnit:
    """Validate and normalize the configured gap units."""

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
    """Convert the configured gap threshold into a comparable ``timedelta``."""

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
