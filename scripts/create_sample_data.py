import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.globals import PATH_DAM_TESTING_DATA, PATH_DAM_TRAINING_DATA, PATH_RTM_DATA

DEFAULT_START = pd.Timestamp("2025-07-01 00:00:00")
DEFAULT_END = pd.Timestamp("2025-10-31 23:59:59")


@dataclass(frozen=True)
class DatasetSpec:
    source_path: Path
    timestamp_column: str
    parser: Callable[[pd.Series], pd.Series]


def parse_dam_key_timestamps(series: pd.Series) -> pd.Series:
    """Parse DAM key timestamps in format MM/DD/YYYY H where H is 1-24."""
    date_part = pd.to_datetime(
        series.str.slice(0, 10), format="%m/%d/%Y", errors="coerce"
    )
    hour_part = pd.to_numeric(series.str.slice(11), errors="coerce")
    return date_part + pd.to_timedelta(hour_part - 1, unit="h")


def parse_datetime_timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def sample_output_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}_sample_3m{source_path.suffix}")


def filter_dataset(
    spec: DatasetSpec,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    df = pd.read_csv(spec.source_path)
    source_rows = len(df)
    output_path = sample_output_path(spec.source_path)
    if spec.timestamp_column not in df.columns:
        raise KeyError(
            f"Column '{spec.timestamp_column}' not found in {spec.source_path}."
        )

    timestamps = spec.parser(df[spec.timestamp_column])
    valid_ts = timestamps.notna()
    bad_timestamp_rows = int((~valid_ts).sum())

    in_window = valid_ts & (timestamps >= start) & (timestamps <= end)
    sampled_df = df.loc[in_window]
    sampled_rows = len(sampled_df)
    sampled_min_ts = timestamps.loc[in_window].min() if sampled_rows else None
    sampled_max_ts = timestamps.loc[in_window].max() if sampled_rows else None

    if sampled_df.empty:
        df.iloc[0:0].to_csv(output_path, index=False)
    else:
        sampled_df.to_csv(output_path, index=False)

    return {
        "source": spec.source_path,
        "output": output_path,
        "source_rows": source_rows,
        "sampled_rows": sampled_rows,
        "bad_timestamp_rows": bad_timestamp_rows,
        "min_ts": sampled_min_ts,
        "max_ts": sampled_max_ts,
    }


def build_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            source_path=Path(PATH_DAM_TRAINING_DATA),
            timestamp_column="key",
            parser=parse_dam_key_timestamps,
        ),
        DatasetSpec(
            source_path=Path(PATH_DAM_TESTING_DATA),
            timestamp_column="key",
            parser=parse_dam_key_timestamps,
        ),
        DatasetSpec(
            source_path=Path(PATH_RTM_DATA),
            timestamp_column="hour_timestamp",
            parser=parse_datetime_timestamps,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create sample CSV copies for GitHub publication using a fixed datetime window."
        )
    )
    parser.add_argument(
        "--start",
        default=str(DEFAULT_START),
        help=f"Inclusive start timestamp. Default: {DEFAULT_START}",
    )
    parser.add_argument(
        "--end",
        default=str(DEFAULT_END),
        help=f"Inclusive end timestamp. Default: {DEFAULT_END}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start > end:
        raise ValueError(f"Start must be <= end. Got {start} > {end}.")

    print(f"Creating sample CSV files for window: [{start}, {end}]")
    print()

    summaries: list[dict[str, object]] = []
    for spec in build_specs():
        summary = filter_dataset(spec=spec, start=start, end=end)
        summaries.append(summary)

    print("Summary")
    print("=" * 80)
    for summary in summaries:
        print(f"Source: {summary['source']}")
        print(f"Output: {summary['output']}")
        print(
            "Rows: "
            f"source={summary['source_rows']:,}, "
            f"sample={summary['sampled_rows']:,}, "
            f"bad_timestamps={summary['bad_timestamp_rows']:,}"
        )
        print(f"Sample bounds: {summary['min_ts']} -> {summary['max_ts']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
