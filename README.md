# rangestitch

`rangestitch` is a small Python library for merging, stitching, and analyzing date or datetime intervals with [Polars](https://pola.rs/).

It is designed for datasets that contain repeated periods per entity, such as customer relationship spans, subscription windows, coverage periods, or employment ranges. The library groups intervals by ID, merges overlapping or adjacent ranges, and carries forward "begin" and "end" characteristics in a predictable way.

## Features

- Merge overlapping intervals per entity.
- Stitch adjacent intervals when the gap is within a configurable threshold.
- Preserve the first "begin" characteristics in a stitched block.
- Take the "end" characteristics from the row that contributes the final end date in that block.
- Accept `polars.DataFrame` input or any iterable of row dictionaries.
- Support `date`, `datetime`, and ISO date/datetime string inputs.
- Preserve datetime precision when either interval column is datetime-like.
- Support day-based legacy gaps, explicit `timedelta` thresholds, and numeric thresholds with configurable `gap_units`.
- Expose a single `interval_stitch()` API for stitching temporal intervals.

## Requirements

- Python `>=3.14`
- `polars>=1.40.1`

## Installation

Install from the repository:

```bash
pip install .
```

For local development:

```bash
pip install -e .
```

If you use `uv`:

```bash
uv sync
```

## Quick Start

Date-based stitching:

```python
from datetime import date

import polars as pl

from rangestitch import interval_stitch

data = pl.DataFrame(
    [
        {
            "ID": 1,
            "From": date(2020, 1, 1),
            "To": date(2020, 1, 10),
            "CharacteristicBeg": "new",
            "CharacteristicEnd1": None,
            "CharacteristicEnd2": None,
        },
        {
            "ID": 1,
            "From": date(2020, 1, 11),
            "To": date(2020, 1, 15),
            "CharacteristicBeg": "active",
            "CharacteristicEnd1": 3,
            "CharacteristicEnd2": 4,
        },
        {
            "ID": 2,
            "From": date(2020, 2, 1),
            "To": date(2020, 2, 5),
            "CharacteristicBeg": "single",
            "CharacteristicEnd1": 9,
            "CharacteristicEnd2": 1,
        },
    ]
)

result = interval_stitch(
    data,
    characteristic_beg_columns="CharacteristicBeg",
    characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
    verbose=False,
)
print(result)
```

Output:

```text
shape: (2, 6)
+-----+------------+------------+-------------------+--------------------+--------------------+
| ID  | From       | To         | CharacteristicBeg | CharacteristicEnd1 | CharacteristicEnd2 |
| --- | ---        | ---        | ---               | ---                | ---                |
| str | date       | date       | str               | i64                | i64                |
+=====+============+============+===================+====================+====================+
| 1   | 2020-01-01 | 2020-01-15 | new               | 3                  | 4                  |
| 2   | 2020-02-01 | 2020-02-05 | single            | 9                  | 1                  |
+-----+------------+------------+-------------------+--------------------+--------------------+
```

The two ranges for `ID=1` are stitched because they are adjacent. The stitched record keeps:

- the first `CharacteristicBeg` value in the block
- the final `To` date across the block
- the end characteristics from the row that contributed that final `To` date

Datetime-based stitching:

```python
from datetime import datetime, timedelta

import polars as pl

from rangestitch import interval_stitch

data = pl.DataFrame(
    [
        {
            "ID": 1,
            "From": datetime(2024, 1, 1, 9, 0, 0),
            "To": datetime(2024, 1, 1, 10, 0, 0),
            "CharacteristicBeg": "open",
            "CharacteristicEnd1": None,
            "CharacteristicEnd2": None,
        },
        {
            "ID": 1,
            "From": datetime(2024, 1, 1, 10, 30, 0),
            "To": datetime(2024, 1, 1, 11, 0, 0),
            "CharacteristicBeg": "continued",
            "CharacteristicEnd1": 5,
            "CharacteristicEnd2": 6,
        },
    ]
)

result = interval_stitch(
    data,
    gap_threshold=timedelta(minutes=30),
    characteristic_beg_columns="CharacteristicBeg",
    characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
    verbose=False,
)
```

Numeric thresholds remain day-based by default for backward compatibility. For sub-day numeric gaps, pass `gap_units`:

```python
result = interval_stitch(
    data,
    gap_threshold=30,
    gap_units="mins",
    verbose=False,
)
```

## How Stitching Works

For each ID, rows are sorted by:

1. the ID column
2. the start-date column
3. original input order

Ranges are then grouped into blocks:

- If a range overlaps the current block, it is merged into that block.
- If the gap between the next `From` and the running maximum `To` is less than or equal to `gap_threshold`, it is stitched into the same block.
- Otherwise, a new block starts.

Gap interpretation:

- Numeric `gap_threshold` values use days by default.
- For datetime workflows, you can pass a Python `timedelta` for exact sub-day gaps.
- For datetime workflows, you can also combine numeric `gap_threshold` values with `gap_units="days"`, `"hours"`, `"mins"`, or `"secs"`.
- A new block starts only when `From - previous To > gap_threshold`.

Characteristic handling inside a stitched block:

- "Begin" characteristic columns use the first row in the block.
- "End" characteristic columns use the first row whose end date matches the block's maximum end date.

## API

The package exports a single public function:

```python
from rangestitch import interval_stitch
```

### Key Parameters

- `data_frame`: a `polars.DataFrame` or an iterable of row mappings.
- `gap_threshold`: maximum gap allowed between periods before a new block starts. Numeric values are interpreted as days by default, preserving the legacy API. For datetime workflows you can also pass `datetime.timedelta` values or combine numeric thresholds with `gap_units`. A new block starts only when `From - previous To > gap_threshold`. Default: `1`.
- `gap_units`: units for numeric `gap_threshold` values. One of `"auto"`, `"days"`, `"hours"`, `"mins"`, or `"secs"`. `"auto"` preserves the legacy day-based behavior. Default: `"auto"`.
- `id_column`: entity identifier column. Default: `"ID"`.
- `from_column`: interval start column. Values may be Python `date` objects, Python `datetime` objects, ISO date strings, ISO datetime strings, or Polars date/datetime columns. Default: `"From"`.
- `to_column`: interval end column. Values may be Python `date` objects, Python `datetime` objects, ISO date strings, ISO datetime strings, or Polars date/datetime columns. Default: `"To"`.
- `characteristic_beg_columns`: optional columns whose values should come from the first row of a stitched block.
- `characteristic_end_columns`: optional columns whose values should come from the row with the block's final end date.
- `keep_all_periods`: compatibility flag currently used together with `include_gap_column` to enable gap diagnostics in the returned output.
- `output_columns`: optional subset and order of columns to return.
- `include_gap_column`: when used with `keep_all_periods=True`, adds the configured gap-diagnostics column.
- `difference_column`: name of the gap-diagnostics output column. Default: `"Difference"`.
- `verbose`: when `True`, emits INFO-level log messages for interval normalization, gap handling, and output size through the standard `rangestitch.timeline` logger.

### Important Notes

- Output IDs are cast to strings.
- Characteristic columns are only included if you explicitly pass them.
- Verbose mode uses Python's `logging` module under the `rangestitch.timeline` logger.
- Configure handlers and the logger level yourself, for example with `logging.basicConfig(level=logging.INFO)` or by setting the `rangestitch.timeline` logger level directly before calling `interval_stitch`.
- If either interval column is datetime-like, both interval columns are normalized to datetimes and time-of-day precision is preserved.
- If both interval columns are date-like, they are normalized to dates and gap calculations are day-based.
- Sub-day thresholds require datetime-like interval columns.
- Requested columns are validated against the input DataFrame before stitching begins.
- Missing required columns raise `ValueError`.
- Null IDs or null start/end dates raise `ValueError`.
- Temporal columns can be Python `date` objects, Python `datetime` objects, ISO date/datetime strings, or Polars date/datetime columns.
- Column roles must not overlap. For example, the same column cannot be both the ID column and a characteristic column.

## Custom Column Example

```python
import polars as pl

from rangestitch import interval_stitch

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
    ]
)

result = interval_stitch(
    data,
    id_column="CustomerID",
    from_column="StartDate",
    to_column="EndDate",
    characteristic_beg_columns=["StatusBeg", "TypeBeg"],
    characteristic_end_columns=["StatusEnd", "TypeEnd"],
    verbose=False,
)
```

## Gap Diagnostics

When you want every stitched block per ID plus the distance from the previous block, enable `keep_all_periods=True` and `include_gap_column=True`.

```python
from datetime import date

import polars as pl

from rangestitch import interval_stitch

data = pl.DataFrame(
    [
        {"ID": 1, "From": date(2020, 1, 1), "To": date(2020, 1, 1), "CharacteristicBeg": "a", "CharacteristicEnd1": 1, "CharacteristicEnd2": None},
        {"ID": 1, "From": date(2020, 1, 2), "To": date(2020, 1, 2), "CharacteristicBeg": "b", "CharacteristicEnd1": 2, "CharacteristicEnd2": None},
    ]
)

result = interval_stitch(
    data,
    gap_threshold=0,
    keep_all_periods=True,
    include_gap_column=True,
    difference_column="GapDays",
    output_columns=["ID", "From", "To", "GapDays"],
    verbose=False,
)
```

In the gap-diagnostics column:

- for date workflows, the first period for each ID is marked with positive infinity (`inf`)
- later date-based periods contain the day difference from the previous stitched block's end date

For datetime workflows, the configured gap-diagnostics column is returned as a Polars duration and the first period has a null difference.

## Development

Run the test suite with:

```bash
uv run python -m unittest discover -s tests
```

If your environment requires system certificate discovery for `uv`, use:

```bash
uv run --system-certs python -m unittest discover -s tests
```

The bundled tests cover:

- stitching adjacent ranges
- stitching datetime ranges with `timedelta` thresholds
- stitching datetime ranges with numeric thresholds plus `gap_units`
- preserving input order for equal start dates
- tie behavior when multiple rows share the same final end date
- custom column mappings
- gap-diagnostic output
- validation for sub-day thresholds on date-only inputs
- parity with the repository's bundled reference dataset
