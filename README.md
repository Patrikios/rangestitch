# rangestitch

`rangestitch` is a Python library for stitching typed date or datetime intervals with [Polars](https://pola.rs/).

It is designed for datasets that contain repeated periods per entity, such as customer relationship spans, subscription windows, coverage periods, or employment ranges. The library groups intervals by ID, merges overlapping or near-adjacent ranges, and carries forward "begin" and "end" characteristics in a predictable way.

## Features

- Merge overlapping intervals per entity.
- Stitch adjacent intervals when the gap is within a configurable threshold.
- Support day-based thresholds for `pl.Date` columns and `datetime.timedelta` thresholds for `pl.Datetime` columns.
- Preserve the first "begin" characteristics in a stitched block.
- Take the "end" characteristics from the row that contributes the final end bound in that block.
- Preserve typed Polars temporal columns in the output.
- Reuse one immutable `RangeStitch` configuration across multiple DataFrames.
- Expose a single `RangeStitch` API for typed Polars DataFrames.

## Requirements

- Python `>=3.14`
- `polars>=1.40.1`

## Installation

Install directly from a public GitHub repository:

```bash
pip install git+https://github.com/patrikios/rangeStitch.git
```

Install a specific branch, tag, or commit:

```bash
pip install git+https://github.com/patrikios/rangeStitch.git@main
pip install git+https://github.com/patrikios/rangeStitch.git@<tag>
pip install git+https://github.com/patrikios/rangeStitch.git@<commit-hash>
```

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
- `ID`, `From`, and `To` values must not be null
- numeric `gap_threshold` values are interpreted as days
- for sub-day datetime gaps, pass `datetime.timedelta`

The API does not try to clean or reshape raw interval data for you. In particular, it does not:

- parse ISO date or datetime strings into Polars temporal columns
- normalize or reconcile timezones for you
- choose an arbitrary custom output projection beyond the configured ID, interval, and characteristic columns
- add extra diagnostic columns such as gap sizes, overlap flags, or block IDs to the final result

That preparation should happen before you call `.stitch()`. A typical workflow is:

1. load the raw data
2. parse or cast the interval columns to `pl.Date` or matching `pl.Datetime` dtypes
3. normalize timezone handling and datetime precision if your source data needs it
4. rename or select the columns you want `RangeStitch` to use
5. call `RangeStitch(...).stitch(prepared_data_frame)`

If you need extra analysis columns, compute them before or after stitching as a separate Polars step. `rangestitch` focuses only on the stitching operation itself.

## Quick Start

The example below uses every constructor argument. It shows:

- custom column mappings
- a date-based `gap_threshold`
- overlapping and near-adjacent periods that stitch into one block
- a larger gap that forces a new block
- multiple "begin" and "end" characteristic columns

```python
from datetime import date

import polars as pl

from rangestitch import RangeStitch

data = pl.DataFrame(
    [
        {
            "CustomerID": "A100",
            "CoverageStart": date(2024, 1, 1),
            "CoverageEnd": date(2024, 1, 31),
            "PlanAtStart": "Basic",
            "ChannelAtStart": "Broker",
            "StatusAtEnd": "Active",
            "PlanAtEnd": "Basic",
            "PremiumAtEnd": 120,
        },
        {
            "CustomerID": "A100",
            "CoverageStart": date(2024, 2, 1),
            "CoverageEnd": date(2024, 2, 15),
            "PlanAtStart": "Basic",
            "ChannelAtStart": "SelfService",
            "StatusAtEnd": "Suspended",
            "PlanAtEnd": "Plus",
            "PremiumAtEnd": 130,
        },
        {
            "CustomerID": "A100",
            "CoverageStart": date(2024, 2, 17),
            "CoverageEnd": date(2024, 3, 31),
            "PlanAtStart": "Plus",
            "ChannelAtStart": "RenewalDesk",
            "StatusAtEnd": "Active",
            "PlanAtEnd": "Premium",
            "PremiumAtEnd": 160,
        },
        {
            "CustomerID": "A100",
            "CoverageStart": date(2024, 4, 5),
            "CoverageEnd": date(2024, 4, 30),
            "PlanAtStart": "Premium",
            "ChannelAtStart": "RenewalDesk",
            "StatusAtEnd": "Cancelled",
            "PlanAtEnd": "Premium",
            "PremiumAtEnd": 0,
        },
        {
            "CustomerID": "B200",
            "CoverageStart": date(2024, 2, 10),
            "CoverageEnd": date(2024, 2, 20),
            "PlanAtStart": "Starter",
            "ChannelAtStart": "Partner",
            "StatusAtEnd": "Active",
            "PlanAtEnd": "Starter",
            "PremiumAtEnd": 90,
        },
        {
            "CustomerID": "B200",
            "CoverageStart": date(2024, 2, 21),
            "CoverageEnd": date(2024, 2, 25),
            "PlanAtStart": "Starter",
            "ChannelAtStart": "Partner",
            "StatusAtEnd": "Active",
            "PlanAtEnd": "Pro",
            "PremiumAtEnd": 95,
        },
    ]
)

stitcher = RangeStitch(
    gap_threshold=2,
    id_column="CustomerID",
    from_column="CoverageStart",
    to_column="CoverageEnd",
    characteristic_beg_columns=["PlanAtStart", "ChannelAtStart"],
    characteristic_end_columns=["StatusAtEnd", "PlanAtEnd", "PremiumAtEnd"],
)
result = stitcher.stitch(data)
print(result.to_dicts())
```

Output:

```python
[
    {
        "CustomerID": "A100",
        "CoverageStart": date(2024, 1, 1),
        "CoverageEnd": date(2024, 3, 31),
        "PlanAtStart": "Basic",
        "ChannelAtStart": "Broker",
        "StatusAtEnd": "Active",
        "PlanAtEnd": "Premium",
        "PremiumAtEnd": 160,
    },
    {
        "CustomerID": "A100",
        "CoverageStart": date(2024, 4, 5),
        "CoverageEnd": date(2024, 4, 30),
        "PlanAtStart": "Premium",
        "ChannelAtStart": "RenewalDesk",
        "StatusAtEnd": "Cancelled",
        "PlanAtEnd": "Premium",
        "PremiumAtEnd": 0,
    },
    {
        "CustomerID": "B200",
        "CoverageStart": date(2024, 2, 10),
        "CoverageEnd": date(2024, 2, 25),
        "PlanAtStart": "Starter",
        "ChannelAtStart": "Partner",
        "StatusAtEnd": "Active",
        "PlanAtEnd": "Pro",
        "PremiumAtEnd": 95,
    },
]
```

Why the first `A100` block looks that way:

- the first three rows stitch together because the last gap is exactly two days, which matches `gap_threshold=2`
- `PlanAtStart` and `ChannelAtStart` come from the first row in the stitched block
- `StatusAtEnd`, `PlanAtEnd`, and `PremiumAtEnd` come from the row with the final `CoverageEnd` of `2024-03-31`

## Datetime Example

For sub-day stitching, pass a `datetime.timedelta` gap threshold:

```python
from datetime import datetime, timedelta

import polars as pl

from rangestitch import RangeStitch

data = pl.DataFrame(
    [
        {
            "ServiceID": "SRV-01",
            "WindowStart": datetime(2024, 5, 1, 9, 0),
            "WindowEnd": datetime(2024, 5, 1, 9, 25),
            "StageAtStart": "opened",
            "RegionAtStart": "eu-central",
            "StageAtEnd": "triaged",
            "OwnerAtEnd": "alice",
        },
        {
            "ServiceID": "SRV-01",
            "WindowStart": datetime(2024, 5, 1, 9, 40),
            "WindowEnd": datetime(2024, 5, 1, 10, 10),
            "StageAtStart": "triaged",
            "RegionAtStart": "eu-central",
            "StageAtEnd": "investigating",
            "OwnerAtEnd": "bob",
        },
        {
            "ServiceID": "SRV-01",
            "WindowStart": datetime(2024, 5, 1, 11, 0),
            "WindowEnd": datetime(2024, 5, 1, 11, 20),
            "StageAtStart": "investigating",
            "RegionAtStart": "eu-central",
            "StageAtEnd": "resolved",
            "OwnerAtEnd": "dana",
        },
        {
            "ServiceID": "SRV-02",
            "WindowStart": datetime(2024, 5, 1, 8, 0),
            "WindowEnd": datetime(2024, 5, 1, 8, 45),
            "StageAtStart": "opened",
            "RegionAtStart": "us-east",
            "StageAtEnd": "waiting",
            "OwnerAtEnd": "erin",
        },
        {
            "ServiceID": "SRV-02",
            "WindowStart": datetime(2024, 5, 1, 8, 30),
            "WindowEnd": datetime(2024, 5, 1, 9, 15),
            "StageAtStart": "waiting",
            "RegionAtStart": "us-east",
            "StageAtEnd": "recovered",
            "OwnerAtEnd": "frank",
        },
    ]
)

stitcher = RangeStitch(
    gap_threshold=timedelta(minutes=45),
    id_column="ServiceID",
    from_column="WindowStart",
    to_column="WindowEnd",
    characteristic_beg_columns=["StageAtStart", "RegionAtStart"],
    characteristic_end_columns=["StageAtEnd", "OwnerAtEnd"],
)
result = stitcher.stitch(data)
print(result.to_dicts())
```

Output:

```python
[
    {
        "ServiceID": "SRV-01",
        "WindowStart": datetime(2024, 5, 1, 9, 0),
        "WindowEnd": datetime(2024, 5, 1, 10, 10),
        "StageAtStart": "opened",
        "RegionAtStart": "eu-central",
        "StageAtEnd": "investigating",
        "OwnerAtEnd": "bob",
    },
    {
        "ServiceID": "SRV-01",
        "WindowStart": datetime(2024, 5, 1, 11, 0),
        "WindowEnd": datetime(2024, 5, 1, 11, 20),
        "StageAtStart": "investigating",
        "RegionAtStart": "eu-central",
        "StageAtEnd": "resolved",
        "OwnerAtEnd": "dana",
    },
    {
        "ServiceID": "SRV-02",
        "WindowStart": datetime(2024, 5, 1, 8, 0),
        "WindowEnd": datetime(2024, 5, 1, 9, 15),
        "StageAtStart": "opened",
        "RegionAtStart": "us-east",
        "StageAtEnd": "recovered",
        "OwnerAtEnd": "frank",
    },
]
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
- If several rows share the same start bound, original input order decides which row is "first".

## API

The package exports a single public class:

```python
from rangestitch import RangeStitch
```

Create a configured stitcher once, then call `.stitch(data_frame)` for each compatible frame. This is useful when several DataFrames share the same column mapping and gap settings.

`RangeStitch` is a frozen dataclass. That means you set the configuration when you create the instance, and if you want different settings later, you create a new instance rather than mutating the old one.

Full constructor:

```python
RangeStitch(
    gap_threshold=1,
    id_column="ID",
    from_column="From",
    to_column="To",
    characteristic_beg_columns=None,
    characteristic_end_columns=None,
)
```

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
- Output columns are returned in this order: ID column, start column, end column, begin characteristic columns, end characteristic columns.
- Characteristic columns are only included if you explicitly pass them.
- `From` and `To` must already be typed as Polars temporal columns.
- Mixed `pl.Date` / `pl.Datetime` interval bounds are rejected.
- Matching datetime columns must use the same Polars datetime dtype.
- `gap_threshold` must be a non-negative number or `datetime.timedelta`.
- Date-based stitching requires whole-day gap thresholds.
- Boolean `gap_threshold` values are rejected.
- Missing required columns raise `ValueError`.
- Null IDs or null start/end bounds raise `ValueError`.
- Column roles must not overlap. For example, the same column cannot be both the ID column and a characteristic column.
- Characteristic column arguments must contain non-empty, non-duplicate column names.
- Empty inputs return an empty DataFrame with the expected typed output schema.

## Equivalent Full Default-Column Configuration

If your source columns already use the defaults `ID`, `From`, and `To`, the fully expanded configuration looks like this:

```python
stitcher = RangeStitch(
    gap_threshold=1,
    id_column="ID",
    from_column="From",
    to_column="To",
    characteristic_beg_columns="CharacteristicBeg",
    characteristic_end_columns=["CharacteristicEnd1", "CharacteristicEnd2"],
)
```

You can shorten that to `RangeStitch(characteristic_beg_columns=..., characteristic_end_columns=...)` when the other arguments should use their defaults.

## Logging

`rangestitch` uses standard library logging.

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

- package export behavior
- parity with the repository's bundled reference dataset
- stitching adjacent ranges
- stitching overlapping ranges
- stitching datetime ranges with `timedelta` thresholds
- preserving input order for equal start dates
- tie behavior when multiple rows share the same final end bound
- custom column mappings and stitcher reuse across multiple DataFrames
- empty-result schema preservation
- validation for missing required characteristic columns
- validation for non-DataFrame inputs
- validation for non-temporal or mixed temporal interval columns
- validation for invalid sub-day thresholds on date columns
- logging behavior
- an optional deterministic 1,000,000-row timing test
- an optional 1,000,000-row performance regression gate via `RANGESTITCH_MAX_SECONDS_1M`
