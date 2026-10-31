"""Verify ZIP checksums and replay saved QuantAlloc analyses offline."""

import argparse
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import pandas as pd

from src.dashboard import (
    run_sample_analysis,
    run_sample_sensitivity,
    run_strategy_comparison,
)
from src.dashboard_settings import DashboardSettings
from src.export import create_export_archive
from src.sample_config import BENCHMARK_TICKER

MAX_ARCHIVE_BYTES = 50_000_000
MAX_UNCOMPRESSED_BYTES = 150_000_000
MAX_FILES = 256


def _read_archive(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    """Read a bounded ZIP without extracting files."""
    if not isinstance(payload, bytes):
        raise TypeError("archive must be bytes")

    if not payload or len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("archive size is invalid")

    try:
        with ZipFile(BytesIO(payload)) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]

            if not names or len(names) > MAX_FILES or len(set(names)) != len(names):
                raise ValueError("invalid or duplicated ZIP entries")

            total_size = sum(item.file_size for item in infos)

            if total_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("uncompressed ZIP exceeds size limit")

            for info in infos:
                path = PurePosixPath(info.filename)

                if (
                    info.is_dir()
                    or path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or "\\" in info.filename
                    or info.flag_bits & 1
                ):
                    raise ValueError("unsafe ZIP entry")

            if "manifest.json" not in names:
                raise ValueError("manifest.json is missing")

            if archive.getinfo("manifest.json").file_size > 1_000_000:
                raise ValueError("manifest is too large")

            files = {name: archive.read(name) for name in names}

    except BadZipFile as exc:
        raise ValueError("invalid or damaged ZIP") from exc

    try:
        manifest = json.loads(files.pop("manifest.json").decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid manifest JSON") from exc

    if not isinstance(manifest, dict):
        raise TypeError("manifest must be an object")

    return manifest, files


def _load_prices(data: bytes) -> pd.DataFrame:
    """Load prices while preserving serialized float precision."""
    try:
        table = pd.read_csv(
            BytesIO(data),
            index_col=0,
            float_precision="round_trip",
        )

        table.index = pd.to_datetime(
            table.index,
            errors="raise",
        )

    except (
        ValueError,
        TypeError,
        pd.errors.ParserError,
    ) as exc:
        raise ValueError("invalid price CSV") from exc

    if (
        table.empty
        or table.index.has_duplicates
        or table.index.hasnans
        or not table.index.is_monotonic_increasing
    ):
        raise ValueError("invalid price index")

    return table


def _compare_csv(
    name: str,
    actual: bytes,
    expected: bytes,
) -> None:
    """Compare saved and recomputed tables with numeric tolerance."""
    try:
        saved = pd.read_csv(
            BytesIO(actual),
            index_col=0,
        )
        replayed = pd.read_csv(
            BytesIO(expected),
            index_col=0,
        )

        pd.testing.assert_frame_equal(
            saved,
            replayed,
            check_dtype=False,
            rtol=1e-9,
            atol=1e-11,
        )

    except (
        AssertionError,
        ValueError,
        pd.errors.ParserError,
    ) as exc:
        raise ValueError(f"replay mismatch: {name}") from exc


def verify_export_archive(
    payload: bytes,
    *,
    replay: bool = True,
) -> dict:
    """Verify file integrity and optionally rerun the saved analysis."""
    manifest, files = _read_archive(payload)

    if manifest.get("schema_version") != 1 or manifest.get("project") != "QuantAlloc":
        raise ValueError("unsupported manifest schema")

    if manifest.get("release") != "v0.22":
        raise ValueError("unsupported export release")

    hashes = manifest.get("file_sha256")

    if not isinstance(hashes, dict) or set(hashes) != set(files):
        raise ValueError("manifest file listing does not match ZIP")

    for name, data in files.items():
        if hashes[name] != sha256(data).hexdigest():
            raise ValueError(f"SHA-256 mismatch: {name}")

    analysis_type = manifest.get("analysis_type")

    if analysis_type not in (
        "single_strategy",
        "strategy_comparison",
        "parameter_sensitivity",
    ):
        raise ValueError("unknown analysis type")

    required = {
        "input/asset_prices.csv",
        "input/benchmark_prices.csv",
    }

    if not required <= set(files):
        raise ValueError("missing saved prices")

    report = {
        "analysis_type": analysis_type,
        "verified_files": len(files),
        "sha256_verified": True,
        "replay_verified": False,
    }

    if not replay:
        return report

    metadata = manifest.get("input_data")

    if not isinstance(metadata, dict):
        raise TypeError("input metadata must be an object")

    if metadata.get("benchmark_ticker") != BENCHMARK_TICKER:
        raise ValueError("unsupported benchmark ticker")

    if not all(
        key in metadata
        for key in (
            "requested_start_date",
            "requested_end_date_exclusive",
        )
    ):
        raise ValueError("missing input date metadata")

    try:
        settings = DashboardSettings(**manifest["settings"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid saved settings") from exc

    assets = _load_prices(files["input/asset_prices.csv"])
    benchmark = _load_prices(files["input/benchmark_prices.csv"])

    if (
        assets.columns.tolist() != metadata.get("asset_tickers")
        or benchmark.columns.tolist() != [BENCHMARK_TICKER]
        or len(assets) != metadata.get("asset_observations")
        or len(benchmark) != metadata.get("benchmark_observations")
    ):
        raise ValueError("saved price data do not match metadata")

    if analysis_type == "single_strategy":
        result = run_sample_analysis(
            assets,
            benchmark,
            manifest.get("strategy"),
            settings=settings,
        )

    elif analysis_type == "strategy_comparison":
        result = run_strategy_comparison(
            assets,
            benchmark,
            settings=settings,
        )

    else:
        result = run_sample_sensitivity(
            assets,
            manifest.get("strategy"),
            settings=settings,
        )

    expected_payload = create_export_archive(
        result,
        settings,
        assets,
        benchmark,
        start_date=metadata["requested_start_date"],
        end_date=metadata["requested_end_date_exclusive"],
        benchmark_ticker=BENCHMARK_TICKER,
    )

    expected_manifest, expected_files = _read_archive(expected_payload)

    def without_hashes(data: dict) -> dict:
        return {key: value for key, value in data.items() if key != "file_sha256"}

    if without_hashes(manifest) != without_hashes(expected_manifest):
        raise ValueError("replay metadata mismatch")

    if set(files) != set(expected_files):
        raise ValueError("replay file listing mismatch")

    for name in sorted(files):
        if name.startswith(("results/", "backtests/")):
            _compare_csv(
                name,
                files[name],
                expected_files[name],
            )

    report["replay_verified"] = True

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a QuantAlloc v0.22 export ZIP offline."
    )

    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="skip backtest replay",
    )

    args = parser.parse_args()

    try:
        report = verify_export_archive(
            args.archive.read_bytes(),
            replay=not args.integrity_only,
        )
    except (ValueError, TypeError, OSError) as exc:
        print(f"VERIFICATION FAILED: {exc}")
        return 1

    print(json.dumps(report, sort_keys=True))
    print("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
