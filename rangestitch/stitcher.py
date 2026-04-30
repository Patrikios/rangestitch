from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from numbers import Real

import polars as pl

from .timeline import interval_stitch


@dataclass(slots=True, frozen=True)
class RangeStitch:
    """Small configured wrapper around the internal interval stitching function.

    The class exists to keep the package's public API compact while letting
    callers configure column names and gap behavior once, then reuse the same
    stitcher across multiple DataFrames.

    Example:
        ``RangeStitch(gap_threshold=timedelta(minutes=30)).stitch(data_frame)``
        stitches datetime intervals with a 30-minute gap tolerance.
    """

    gap_threshold: Real | timedelta = 1
    id_column: str = "ID"
    from_column: str = "From"
    to_column: str = "To"
    characteristic_beg_columns: str | Sequence[str] | None = None
    characteristic_end_columns: str | Sequence[str] | None = None

    def stitch(self, data_frame: pl.DataFrame) -> pl.DataFrame:
        """Stitch one typed Polars DataFrame using the stored configuration."""

        return interval_stitch(
            data_frame,
            gap_threshold=self.gap_threshold,
            id_column=self.id_column,
            from_column=self.from_column,
            to_column=self.to_column,
            characteristic_beg_columns=self.characteristic_beg_columns,
            characteristic_end_columns=self.characteristic_end_columns,
        )
