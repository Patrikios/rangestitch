from __future__ import annotations

from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from io import StringIO
import logging
import unittest

import polars as pl
from polars.testing import assert_frame_equal

import rangestitch
from rangestitch import RangeStitch
from reference_data import reference_expected_frame, reference_input_frame


class IntervalStitchTests(unittest.TestCase):
    def test_package_exports_only_range_stitch(self) -> None:
        self.assertIn("RangeStitch", rangestitch.__all__)
        self.assertNotIn("interval_stitch", rangestitch.__all__)
        self.assertTrue(hasattr(rangestitch, "RangeStitch"))
        self.assertFalse(hasattr(rangestitch, "interval_stitch"))
        self.assertIs(rangestitch.RangeStitch, RangeStitch)

    def test_reference_dataset_matches_r_output(self) -> None:
        actual = RangeStitch(
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
        ).stitch(reference_input_frame())
        expected = reference_expected_frame()
        assert_frame_equal(actual, expected)

    def test_adjacent_ranges_are_stitched(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 1,
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 10),
                    "CharacteristicBeg": "a",
                    "CharacteristicEnd1": None,
                    "CharacteristicEnd2": None,
                },
                {
                    "ID": 1,
                    "From": date(2020, 1, 11),
                    "To": date(2020, 1, 15),
                    "CharacteristicBeg": "b",
                    "CharacteristicEnd1": 3,
                    "CharacteristicEnd2": 4,
                },
            ]
        )

        actual = RangeStitch(
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
        ).stitch(data)
        expected = pl.DataFrame(
            [
                {
                    "ID": "1",
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 15),
                    "CharacteristicBeg": "a",
                    "CharacteristicEnd1": 3,
                    "CharacteristicEnd2": 4,
                }
            ]
        ).cast(actual.schema)

        assert_frame_equal(actual, expected)

    def test_equal_end_date_keeps_existing_end_characteristics(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 7,
                    "From": date(2021, 6, 1),
                    "To": date(2021, 6, 10),
                    "CharacteristicBeg": "x",
                    "CharacteristicEnd1": 1,
                    "CharacteristicEnd2": 2,
                },
                {
                    "ID": 7,
                    "From": date(2021, 6, 8),
                    "To": date(2021, 6, 10),
                    "CharacteristicBeg": "y",
                    "CharacteristicEnd1": 9,
                    "CharacteristicEnd2": 9,
                },
            ]
        )

        actual = RangeStitch(
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
        ).stitch(data)
        expected = pl.DataFrame(
            [
                {
                    "ID": "7",
                    "From": date(2021, 6, 1),
                    "To": date(2021, 6, 10),
                    "CharacteristicBeg": "x",
                    "CharacteristicEnd1": 1,
                    "CharacteristicEnd2": 2,
                }
            ]
        ).cast(actual.schema)

        assert_frame_equal(actual, expected)

    def test_same_id_and_from_preserves_input_order_for_characteristic_beg(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 4,
                    "From": date(2021, 1, 1),
                    "To": date(2021, 1, 10),
                    "CharacteristicBeg": "first",
                    "CharacteristicEnd1": None,
                    "CharacteristicEnd2": None,
                },
                {
                    "ID": 4,
                    "From": date(2021, 1, 1),
                    "To": date(2021, 2, 1),
                    "CharacteristicBeg": "second",
                    "CharacteristicEnd1": 7,
                    "CharacteristicEnd2": None,
                },
            ]
        )

        actual = RangeStitch(
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
        ).stitch(data)
        expected = pl.DataFrame(
            [
                {
                    "ID": "4",
                    "From": date(2021, 1, 1),
                    "To": date(2021, 2, 1),
                    "CharacteristicBeg": "first",
                    "CharacteristicEnd1": 7,
                    "CharacteristicEnd2": None,
                }
            ]
        ).cast(actual.schema)

        assert_frame_equal(actual, expected)

    def test_range_stitch_supports_custom_columns(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "CustomerID": "A",
                    "StartDate": date(2020, 1, 1),
                    "EndDate": date(2020, 1, 1),
                    "StatusBeg": "New",
                    "StatusEnd": "Active",
                    "TypeBeg": "Basic",
                    "TypeEnd": "Basic",
                },
                {
                    "CustomerID": "A",
                    "StartDate": date(2020, 1, 2),
                    "EndDate": date(2020, 1, 3),
                    "StatusBeg": "New",
                    "StatusEnd": "Active",
                    "TypeBeg": "Basic",
                    "TypeEnd": "Premium",
                },
                {
                    "CustomerID": "A",
                    "StartDate": date(2020, 2, 1),
                    "EndDate": date(2020, 2, 5),
                    "StatusBeg": "Returning",
                    "StatusEnd": "Dormant",
                    "TypeBeg": "Premium",
                    "TypeEnd": "Gold",
                },
                {
                    "CustomerID": "B",
                    "StartDate": date(2020, 2, 10),
                    "EndDate": date(2020, 2, 12),
                    "StatusBeg": "First",
                    "StatusEnd": "Active",
                    "TypeBeg": "Standard",
                    "TypeEnd": "Standard",
                },
            ]
        )

        actual = RangeStitch(
            id_column="CustomerID",
            from_column="StartDate",
            to_column="EndDate",
            characteristic_beg_columns=["StatusBeg", "TypeBeg"],
            characteristic_end_columns=["StatusEnd", "TypeEnd"],
        ).stitch(data)
        expected = pl.DataFrame(
            [
                {
                    "CustomerID": "A",
                    "StartDate": date(2020, 1, 1),
                    "EndDate": date(2020, 1, 3),
                    "StatusBeg": "New",
                    "TypeBeg": "Basic",
                    "StatusEnd": "Active",
                    "TypeEnd": "Premium",
                },
                {
                    "CustomerID": "A",
                    "StartDate": date(2020, 2, 1),
                    "EndDate": date(2020, 2, 5),
                    "StatusBeg": "Returning",
                    "TypeBeg": "Premium",
                    "StatusEnd": "Dormant",
                    "TypeEnd": "Gold",
                },
                {
                    "CustomerID": "B",
                    "StartDate": date(2020, 2, 10),
                    "EndDate": date(2020, 2, 12),
                    "StatusBeg": "First",
                    "TypeBeg": "Standard",
                    "StatusEnd": "Active",
                    "TypeEnd": "Standard",
                },
            ]
        )

        assert_frame_equal(actual, expected)

    def test_range_stitch_can_be_reused_across_multiple_dataframes(self) -> None:
        stitcher = RangeStitch(
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
        )

        first_data = pl.DataFrame(
            [
                {
                    "ID": 1,
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 10),
                    "CharacteristicBeg": "a",
                    "CharacteristicEnd1": None,
                    "CharacteristicEnd2": None,
                },
                {
                    "ID": 1,
                    "From": date(2020, 1, 11),
                    "To": date(2020, 1, 12),
                    "CharacteristicBeg": "b",
                    "CharacteristicEnd1": 4,
                    "CharacteristicEnd2": 5,
                },
            ]
        )
        second_data = pl.DataFrame(
            [
                {
                    "ID": 2,
                    "From": date(2020, 2, 1),
                    "To": date(2020, 2, 2),
                    "CharacteristicBeg": "c",
                    "CharacteristicEnd1": 7,
                    "CharacteristicEnd2": 8,
                },
                {
                    "ID": 2,
                    "From": date(2020, 2, 4),
                    "To": date(2020, 2, 5),
                    "CharacteristicBeg": "d",
                    "CharacteristicEnd1": 9,
                    "CharacteristicEnd2": 10,
                },
            ]
        )

        first_actual = stitcher.stitch(first_data)
        second_actual = stitcher.stitch(second_data)
        first_expected = pl.DataFrame(
            [
                {
                    "ID": "1",
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 12),
                    "CharacteristicBeg": "a",
                    "CharacteristicEnd1": 4,
                    "CharacteristicEnd2": 5,
                }
            ]
        ).cast(first_actual.schema)
        second_expected = pl.DataFrame(
            [
                {
                    "ID": "2",
                    "From": date(2020, 2, 1),
                    "To": date(2020, 2, 2),
                    "CharacteristicBeg": "c",
                    "CharacteristicEnd1": 7,
                    "CharacteristicEnd2": 8,
                },
                {
                    "ID": "2",
                    "From": date(2020, 2, 4),
                    "To": date(2020, 2, 5),
                    "CharacteristicBeg": "d",
                    "CharacteristicEnd1": 9,
                    "CharacteristicEnd2": 10,
                },
            ]
        ).cast(second_actual.schema)

        assert_frame_equal(first_actual, first_expected)
        assert_frame_equal(second_actual, second_expected)

    def test_missing_requested_characteristic_column_raises_validation_error(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 1,
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 1),
                    "CharacteristicBeg": "a",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "missing required columns: CharacteristicEnd1"):
            RangeStitch(
                characteristic_beg_columns="CharacteristicBeg",
                characteristic_end_columns="CharacteristicEnd1",
            ).stitch(data)

    def test_datetime_ranges_support_timedelta_gap_threshold(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 9,
                    "From": datetime(2024, 1, 1, 9, 0, 0),
                    "To": datetime(2024, 1, 1, 10, 0, 0),
                    "CharacteristicBeg": "open",
                    "CharacteristicEnd1": None,
                    "CharacteristicEnd2": None,
                },
                {
                    "ID": 9,
                    "From": datetime(2024, 1, 1, 10, 30, 0),
                    "To": datetime(2024, 1, 1, 11, 0, 0),
                    "CharacteristicBeg": "continued",
                    "CharacteristicEnd1": 5,
                    "CharacteristicEnd2": 6,
                },
            ]
        )

        actual = RangeStitch(
            gap_threshold=timedelta(minutes=30),
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
        ).stitch(data)
        expected = pl.DataFrame(
            [
                {
                    "ID": "9",
                    "From": datetime(2024, 1, 1, 9, 0, 0),
                    "To": datetime(2024, 1, 1, 11, 0, 0),
                    "CharacteristicBeg": "open",
                    "CharacteristicEnd1": 5,
                    "CharacteristicEnd2": 6,
                }
            ]
        ).cast(actual.schema)

        assert_frame_equal(actual, expected)

    def test_date_based_stitching_requires_whole_day_gap_thresholds(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 1,
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 1),
                },
                {
                    "ID": 1,
                    "From": date(2020, 1, 2),
                    "To": date(2020, 1, 2),
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "date-based stitching requires whole-day gap thresholds"):
            RangeStitch(
                gap_threshold=timedelta(minutes=30),
            ).stitch(data)

    def test_range_stitch_requires_polars_dataframe_input(self) -> None:
        with self.assertRaisesRegex(TypeError, "data_frame must be a polars.DataFrame"):
            RangeStitch().stitch([{"ID": 1, "From": date(2020, 1, 1), "To": date(2020, 1, 1)}])

    def test_range_stitch_requires_polars_temporal_columns(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 1,
                    "From": "2024-01-01",
                    "To": "2024-01-02",
                }
            ]
        )

        with self.assertRaisesRegex(TypeError, "must both be pl.Date or matching pl.Datetime columns"):
            RangeStitch().stitch(data)

    def test_range_stitch_rejects_mixed_date_and_datetime_columns(self) -> None:
        data = pl.DataFrame(
            {
                "ID": [1],
                "From": [date(2024, 1, 1)],
                "To": [datetime(2024, 1, 1, 0, 0, 0)],
            },
            schema={
                "ID": pl.Int64,
                "From": pl.Date,
                "To": pl.Datetime("us"),
            },
        )

        with self.assertRaisesRegex(TypeError, "must both be pl.Date or matching pl.Datetime columns"):
            RangeStitch().stitch(data)

    def test_empty_input_preserves_output_schema(self) -> None:
        data = pl.DataFrame(
            schema={
                "ID": pl.Int64,
                "From": pl.Date,
                "To": pl.Date,
                "CharacteristicBeg": pl.String,
                "CharacteristicEnd1": pl.Int64,
            }
        )

        actual = RangeStitch(
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns="CharacteristicEnd1",
        ).stitch(data)

        self.assertEqual(
            actual.schema,
            {
                "ID": pl.String,
                "From": pl.Date,
                "To": pl.Date,
                "CharacteristicBeg": pl.String,
                "CharacteristicEnd1": pl.Int64,
            },
        )
        self.assertEqual(actual.height, 0)

    def test_range_stitch_uses_logging_instead_of_print(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 1,
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 1),
                    "CharacteristicBeg": "a",
                    "CharacteristicEnd1": 1,
                    "CharacteristicEnd2": None,
                }
            ]
        )

        stdout_buffer = StringIO()
        with self.assertLogs("rangestitch.timeline", level="INFO") as captured, redirect_stdout(stdout_buffer):
            RangeStitch().stitch(data)

        self.assertEqual(stdout_buffer.getvalue(), "")
        self.assertIn("Starting range stitching", captured.output[0])
        self.assertIn("Range stitching completed", captured.output[-1])
        self.assertNotIn("interval_stitch", " ".join(captured.output))

    def test_range_stitch_respects_existing_logger_level(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 1,
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 1),
                    "CharacteristicBeg": "a",
                    "CharacteristicEnd1": 1,
                    "CharacteristicEnd2": None,
                }
            ]
        )

        logger = logging.getLogger("rangestitch.timeline")
        original_level = logger.level
        original_handlers = list(logger.handlers)
        original_propagate = logger.propagate
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.handlers = [handler]
        logger.setLevel(logging.WARNING)
        logger.propagate = False

        try:
            RangeStitch().stitch(data)
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        self.assertEqual(stream.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
