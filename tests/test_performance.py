from __future__ import annotations

from datetime import date
import os
from time import perf_counter
import unittest

import polars as pl

from rangestitch import RangeStitch

_ROW_COUNT = 1_000_000
_ID_COUNT = 1_000
_ROWS_PER_ID = _ROW_COUNT // _ID_COUNT
_BLOCK_SIZE = 5
_BLOCK_SPAN_DAYS = 7
_EXPECTED_OUTPUT_ROWS = _ID_COUNT * (_ROWS_PER_ID // _BLOCK_SIZE)
_RUN_PERF_TESTS = os.getenv("RANGESTITCH_RUN_PERF_TESTS") == "1"
_MAX_SECONDS_ENV = "RANGESTITCH_MAX_SECONDS_1M"


def _build_one_million_row_frame() -> pl.DataFrame:
    """Create a deterministic 1,000,000-row date-based benchmark dataset.

    Rows are arranged as 1,000 IDs with 1,000 rows each. Every five rows per
    ID form one stitchable block, and blocks are separated by larger gaps so
    the expected output size remains stable and easy to verify.
    """

    row_numbers = pl.int_range(0, _ROW_COUNT, eager=True)
    return (
        pl.DataFrame({"__row_nr": row_numbers})
        .with_columns(
            (pl.col("__row_nr") // _ROWS_PER_ID).cast(pl.Int64).alias("ID"),
            (pl.col("__row_nr") % _ROWS_PER_ID).cast(pl.Int64).alias("__row_in_id"),
        )
        .with_columns(
            (pl.col("__row_in_id") // _BLOCK_SIZE).alias("__block_index"),
            (pl.col("__row_in_id") % _BLOCK_SIZE).alias("__position_in_block"),
        )
        .with_columns(
            (pl.col("__block_index") * _BLOCK_SPAN_DAYS + pl.col("__position_in_block")).alias(
                "__start_day_offset"
            ),
            pl.col("__row_in_id").alias("CharacteristicBeg"),
            (pl.col("__row_in_id") * 10).alias("CharacteristicEnd1"),
        )
        .with_columns(
            (pl.lit(date(2020, 1, 1)) + pl.duration(days=pl.col("__start_day_offset"))).cast(pl.Date).alias("From"),
            (pl.lit(date(2020, 1, 1)) + pl.duration(days=pl.col("__start_day_offset"))).cast(pl.Date).alias("To"),
        )
        .select(["ID", "From", "To", "CharacteristicBeg", "CharacteristicEnd1"])
    )


@unittest.skipUnless(_RUN_PERF_TESTS, "set RANGESTITCH_RUN_PERF_TESTS=1 to run the 1,000,000-row timing test")
class RangeStitchPerformanceTests(unittest.TestCase):
    def test_range_stitch_times_one_million_rows(self) -> None:
        build_started_at = perf_counter()
        data = _build_one_million_row_frame()
        build_elapsed = perf_counter() - build_started_at

        stitcher = RangeStitch(
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns="CharacteristicEnd1",
        )

        started_at = perf_counter()
        result = stitcher.stitch(data)
        stitch_elapsed = perf_counter() - started_at

        self.assertEqual(data.height, _ROW_COUNT)
        self.assertEqual(result.height, _EXPECTED_OUTPUT_ROWS)
        self.assertEqual(
            result.columns,
            ["ID", "From", "To", "CharacteristicBeg", "CharacteristicEnd1"],
        )

        print(
            (
                f"Built {data.height:,} rows in {build_elapsed:.3f}s; "
                f"stitched to {result.height:,} rows in {stitch_elapsed:.3f}s."
            )
        )

        max_seconds = os.getenv(_MAX_SECONDS_ENV)
        if max_seconds is not None:
            self.assertLessEqual(
                stitch_elapsed,
                float(max_seconds),
                f"1,000,000-row stitch took {stitch_elapsed:.3f}s, exceeding {_MAX_SECONDS_ENV}={max_seconds}",
            )
