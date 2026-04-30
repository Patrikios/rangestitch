from __future__ import annotations

from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from io import StringIO
import logging
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from rangestitch import interval_stitch
from reference_data import reference_expected_frame, reference_input_frame


class IntervalStitchTests(unittest.TestCase):
    def test_reference_dataset_matches_r_output(self) -> None:
        actual = interval_stitch(
            reference_input_frame(),
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
            verbose=False,
        )
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

        actual = interval_stitch(
            data,
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
            verbose=False,
        )
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

        actual = interval_stitch(
            data,
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
            verbose=False,
        )
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

        actual = interval_stitch(
            data,
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
            verbose=False,
        )
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

    def test_interval_stitch_supports_custom_columns(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "CustomerID": "A",
                    "StartDate": "2020-01-01",
                    "EndDate": "2020-01-01",
                    "StatusBeg": "New",
                    "StatusEnd": "Active",
                    "TypeBeg": "Basic",
                    "TypeEnd": "Basic",
                },
                {
                    "CustomerID": "A",
                    "StartDate": "2020-01-02",
                    "EndDate": "2020-01-03",
                    "StatusBeg": "New",
                    "StatusEnd": "Active",
                    "TypeBeg": "Basic",
                    "TypeEnd": "Premium",
                },
                {
                    "CustomerID": "A",
                    "StartDate": "2020-02-01",
                    "EndDate": "2020-02-05",
                    "StatusBeg": "Returning",
                    "StatusEnd": "Dormant",
                    "TypeBeg": "Premium",
                    "TypeEnd": "Gold",
                },
                {
                    "CustomerID": "B",
                    "StartDate": "2020-02-10",
                    "EndDate": "2020-02-12",
                    "StatusBeg": "First",
                    "StatusEnd": "Active",
                    "TypeBeg": "Standard",
                    "TypeEnd": "Standard",
                },
            ]
        )

        actual = interval_stitch(
            data,
            id_column="CustomerID",
            from_column="StartDate",
            to_column="EndDate",
            characteristic_beg_columns=["StatusBeg", "TypeBeg"],
            characteristic_end_columns=["StatusEnd", "TypeEnd"],
            verbose=False,
        )
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

    def test_interval_stitch_supports_gap_diagnostics_and_output_columns(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 1,
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 1),
                    "CharacteristicBeg": "a",
                    "CharacteristicEnd1": 1,
                    "CharacteristicEnd2": None,
                },
                {
                    "ID": 1,
                    "From": date(2020, 1, 2),
                    "To": date(2020, 1, 2),
                    "CharacteristicBeg": "b",
                    "CharacteristicEnd1": 2,
                    "CharacteristicEnd2": None,
                },
            ]
        )

        actual = interval_stitch(
            data,
            gap_threshold=0,
            keep_all_periods=True,
            include_gap_column=True,
            output_columns=["ID", "From", "To", "Difference"],
            verbose=False,
        )
        expected = pl.DataFrame(
            [
                {
                    "ID": "1",
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 1),
                    "Difference": float("inf"),
                },
                {
                    "ID": "1",
                    "From": date(2020, 1, 2),
                    "To": date(2020, 1, 2),
                    "Difference": 1.0,
                },
            ]
        ).cast(actual.schema)

        assert_frame_equal(actual, expected)

    def test_gap_diagnostics_support_custom_difference_column(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 3,
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 1),
                    "CharacteristicBeg": "a",
                    "CharacteristicEnd1": 1,
                    "CharacteristicEnd2": None,
                },
                {
                    "ID": 3,
                    "From": date(2020, 1, 3),
                    "To": date(2020, 1, 3),
                    "CharacteristicBeg": "b",
                    "CharacteristicEnd1": 2,
                    "CharacteristicEnd2": None,
                },
            ]
        )

        actual = interval_stitch(
            data,
            gap_threshold=0,
            keep_all_periods=True,
            include_gap_column=True,
            difference_column="GapDays",
            output_columns=["ID", "From", "To", "GapDays"],
            verbose=False,
        )
        expected = pl.DataFrame(
            [
                {
                    "ID": "3",
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 1),
                    "GapDays": float("inf"),
                },
                {
                    "ID": "3",
                    "From": date(2020, 1, 3),
                    "To": date(2020, 1, 3),
                    "GapDays": 2.0,
                },
            ]
        ).cast(actual.schema)

        assert_frame_equal(actual, expected)

    def test_difference_column_must_not_overlap_existing_columns(self) -> None:
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

        with self.assertRaisesRegex(ValueError, "difference_column must not overlap other selected columns"):
            interval_stitch(
                data,
                keep_all_periods=True,
                include_gap_column=True,
                difference_column="To",
                verbose=False,
            )

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
            interval_stitch(
                data,
                characteristic_beg_columns="CharacteristicBeg",
                characteristic_end_columns="CharacteristicEnd1",
                verbose=False,
            )

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

        actual = interval_stitch(
            data,
            gap_threshold=timedelta(minutes=30),
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
            verbose=False,
        )
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

    def test_datetime_ranges_support_numeric_gap_units(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 12,
                    "From": "2024-02-01T12:00:00",
                    "To": "2024-02-01T12:00:05",
                    "CharacteristicBeg": "start",
                    "CharacteristicEnd1": None,
                    "CharacteristicEnd2": None,
                },
                {
                    "ID": 12,
                    "From": "2024-02-01T12:00:07",
                    "To": "2024-02-01T12:00:10",
                    "CharacteristicBeg": "later",
                    "CharacteristicEnd1": 8,
                    "CharacteristicEnd2": 9,
                },
            ]
        )

        actual = interval_stitch(
            data,
            gap_threshold=2,
            gap_units="secs",
            characteristic_beg_columns="CharacteristicBeg",
            characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
            verbose=False,
        )
        expected = pl.DataFrame(
            [
                {
                    "ID": "12",
                    "From": datetime(2024, 2, 1, 12, 0, 0),
                    "To": datetime(2024, 2, 1, 12, 0, 10),
                    "CharacteristicBeg": "start",
                    "CharacteristicEnd1": 8,
                    "CharacteristicEnd2": 9,
                }
            ]
        ).cast(actual.schema)

        assert_frame_equal(actual, expected)

    def test_sub_day_gap_threshold_requires_datetime_columns(self) -> None:
        data = pl.DataFrame(
            [
                {
                    "ID": 1,
                    "From": date(2020, 1, 1),
                    "To": date(2020, 1, 1),
                    "CharacteristicBeg": "a",
                    "CharacteristicEnd1": 1,
                    "CharacteristicEnd2": None,
                },
                {
                    "ID": 1,
                    "From": date(2020, 1, 2),
                    "To": date(2020, 1, 2),
                    "CharacteristicBeg": "b",
                    "CharacteristicEnd1": 2,
                    "CharacteristicEnd2": None,
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "sub-day gap thresholds require datetime-like from/to columns"):
            interval_stitch(
                data,
                gap_threshold=30,
                gap_units="mins",
                verbose=False,
            )

    def test_verbose_uses_logging_instead_of_print(self) -> None:
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
            interval_stitch(data, verbose=True)

        self.assertEqual(stdout_buffer.getvalue(), "")
        self.assertIn("Starting interval_stitch", captured.output[0])
        self.assertIn("Interval stitching completed", captured.output[-1])

    def test_verbose_respects_existing_logger_level(self) -> None:
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
            interval_stitch(data, verbose=True)
        finally:
            logger.handlers = original_handlers
            logger.setLevel(original_level)
            logger.propagate = original_propagate

        self.assertEqual(stream.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
