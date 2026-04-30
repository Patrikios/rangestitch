# rangestitch

`rangestitch` is a small Python library for stitching typed date or datetime intervals with [Polars](https://pola.rs/).

It is designed for datasets that contain repeated periods per entity, such as customer relationship spans, subscription windows, coverage periods, or employment ranges. The library groups intervals by ID, merges overlapping or near-adjacent ranges, and carries forward "begin" and "end" characteristics in a predictable way.

## Features

- Merge overlapping intervals per entity.
- Stitch adjacent intervals when the gap is within a configurable threshold.
- Preserve the first "begin" characteristics in a stitched block.
- Take the "end" characteristics from the row that contributes the final end bound in that block.
- Expose a single `RangeStitch` API for typed Polars DataFrames.

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

## Contract

`RangeStitch(...).stitch()` intentionally supports a narrow API:

- `data_frame` must be a `polars.DataFrame`
- `From` and `To` must already be typed as `pl.Date` or matching `pl.Datetime` dtypes
- numeric `gap_threshold` values are interpreted as days
- for sub-day datetime gaps, pass `datetime.timedelta`

The API does not parse ISO strings, normalize timezones, project custom output subsets, or add gap-diagnostic columns.

## Quick Start

Date-based stitching:

```python
from datetime import date

import polars as pl

from rangestitch import RangeStitch

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

stitcher = RangeStitch(
    characteristic_beg_columns="CharacteristicBeg",
    characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
)
result = stitcher.stitch(data)
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

Datetime-based stitching:

```python
from datetime import datetime, timedelta

import polars as pl

from rangestitch import RangeStitch

data = pl.DataFrame(
    [
        {
            "ID": 1,
            "From": datetime(2024, 1, 1, 9, 0, 0),
            "To": datetime(2024, 1, 1, 10, 0, 0),
        },
        {
            "ID": 1,
            "From": datetime(2024, 1, 1, 10, 30, 0),
            "To": datetime(2024, 1, 1, 11, 0, 0),
        },
    ]
)

stitcher = RangeStitch(
    gap_threshold=timedelta(minutes=30),
)
result = stitcher.stitch(data)
```

## How Stitching Works

For each ID, rows are sorted by:

1. the ID column
2. the start-bound column
3. original input order

Ranges are then grouped into blocks:

- If a range overlaps the current block, it is merged into that block.
- If the gap between the next `From` and the running maximum `To` is less than or equal to `gap_threshold`, it is stitched into the same block.
- Otherwise, a new block starts.

Characteristic handling inside a stitched block:

- "Begin" characteristic columns use the first row in the block.
- "End" characteristic columns use the first row whose end bound matches the block's maximum end bound.

## API

The package exports a single public class:

```python
from rangestitch import RangeStitch
```

Create a configured stitcher once, then call `.stitch(data_frame)` for each compatible frame. This is useful when several DataFrames share the same column mapping and gap settings.

### Constructor Parameters

- `gap_threshold`: maximum gap allowed between periods before a new block starts. Numeric values are interpreted as days. For sub-day datetime workflows, pass `datetime.timedelta`. Default: `1`.
- `id_column`: entity identifier column. Default: `"ID"`.
- `from_column`: interval start column. Must already be typed as `pl.Date` or `pl.Datetime`. Default: `"From"`.
- `to_column`: interval end column. Must already be typed as `pl.Date` or `pl.Datetime` and match `from_column`. Default: `"To"`.
- `characteristic_beg_columns`: optional columns whose values should come from the first row of a stitched block.
- `characteristic_end_columns`: optional columns whose values should come from the row with the block's final end bound.

### Method

- `stitch(data_frame)`: stitches one `polars.DataFrame` using the stored configuration.

### Important Notes

- Output IDs are cast to strings.
- Characteristic columns are only included if you explicitly pass them.
- `From` and `To` must already be typed as Polars temporal columns.
- Mixed `pl.Date` / `pl.Datetime` interval bounds are rejected.
- Date-based stitching requires whole-day gap thresholds.
- Missing required columns raise `ValueError`.
- Null IDs or null start/end bounds raise `ValueError`.
- Column roles must not overlap. For example, the same column cannot be both the ID column and a characteristic column.

## Custom Column Example

```python
from datetime import date

import polars as pl

from rangestitch import RangeStitch

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
    ]
)

stitcher = RangeStitch(
    id_column="CustomerID",
    from_column="StartDate",
    to_column="EndDate",
    characteristic_beg_columns=["StatusBeg", "TypeBeg"],
    characteristic_end_columns=["StatusEnd", "TypeEnd"],
)
result = stitcher.stitch(data)
```

## Logging

`rangestitch` uses standard library logging and does not expose a per-call verbosity flag.

By default, the package is silent and installs a `NullHandler`. To see stitching progress logs, configure the `rangestitch` logger hierarchy in your application:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("rangestitch").setLevel(logging.INFO)
```

Progress messages are emitted from `rangestitch.timeline` at `INFO` level.

## Development

Run the test suite with:

```bash
uv run python -m unittest discover -s tests
```

Run the verbose suite with per-test names:

```bash
uv run python -m unittest discover -s tests -v
```

Run the dedicated 1,000,000-row timing test:

```bash
RANGESTITCH_RUN_PERF_TESTS=1 uv run python -m unittest tests.test_performance -v
```

In PowerShell:

```powershell
$env:RANGESTITCH_RUN_PERF_TESTS='1'
uv run python -m unittest tests.test_performance -v
```

To turn that benchmark into a pass/fail regression check, also set a maximum
allowed stitching time in seconds:

```bash
RANGESTITCH_RUN_PERF_TESTS=1 RANGESTITCH_MAX_SECONDS_1M=5 uv run python -m unittest tests.test_performance -v
```

In PowerShell:

```powershell
$env:RANGESTITCH_RUN_PERF_TESTS='1'
$env:RANGESTITCH_MAX_SECONDS_1M='5'
uv run python -m unittest tests.test_performance -v
```

If your environment requires system certificate discovery for `uv`, use:

```bash
uv run --system-certs python -m unittest discover -s tests
```

The bundled tests cover:

- stitching adjacent ranges
- stitching datetime ranges with `timedelta` thresholds
- preserving input order for equal start dates
- tie behavior when multiple rows share the same final end bound
- custom column mappings
- empty-result schema preservation
- validation for non-DataFrame inputs
- validation for non-temporal or mixed temporal interval columns
- parity with the repository's bundled reference dataset
