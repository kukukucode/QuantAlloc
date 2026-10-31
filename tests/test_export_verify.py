"""Verify archived results can be checked and replayed offline."""

import json
from hashlib import sha256
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from src.dashboard import (
    run_sample_analysis,
    run_sample_sensitivity,
    run_strategy_comparison,
)
from src.dashboard_settings import DashboardSettings
from src.export import create_export_archive
from src.export_verify import verify_export_archive


def _make_bundle(prices, mode):
    assets, benchmark = prices
    settings = DashboardSettings()

    if mode == "single":
        settings = DashboardSettings(252, 126, 0.003)
        result = run_sample_analysis(
            assets,
            benchmark,
            "equal_weight",
            settings=settings,
        )

    elif mode == "comparison":
        result = run_strategy_comparison(
            assets,
            benchmark,
            settings=settings,
        )

    else:
        result = run_sample_sensitivity(
            assets,
            "equal_weight",
            settings=settings,
        )

    return create_export_archive(
        result,
        settings,
        assets,
        benchmark,
    )


def _replace_entries(payload, updates):
    with ZipFile(BytesIO(payload)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}

    files.update(updates)

    output = BytesIO()

    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)

    return output.getvalue()


@pytest.mark.parametrize(
    "mode",
    ["single", "comparison", "sensitivity"],
)
def test_replay_saved_analysis(sample_market_prices, mode):
    archive = _make_bundle(sample_market_prices, mode)

    report = verify_export_archive(archive)

    assert report["sha256_verified"] is True
    assert report["replay_verified"] is True
    assert report["verified_files"] > 2


def test_integrity_only(sample_market_prices):
    archive = _make_bundle(sample_market_prices, "single")

    report = verify_export_archive(
        archive,
        replay=False,
    )

    assert report["sha256_verified"] is True
    assert report["replay_verified"] is False


def test_rejects_corrupt_file(sample_market_prices):
    archive = _make_bundle(sample_market_prices, "single")

    tampered = _replace_entries(
        archive,
        {"results/performance.csv": b"invalid,csv\n"},
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_export_archive(tampered)


def test_rehashed_tampering_is_caught_by_replay(
    sample_market_prices,
):
    archive = _make_bundle(sample_market_prices, "single")

    with ZipFile(BytesIO(archive)) as original:
        manifest = json.loads(original.read("manifest.json"))
        csv = original.read("results/performance.csv")

    changed = csv.replace(b"cagr", b"fraud")
    assert changed != csv

    manifest["file_sha256"]["results/performance.csv"] = sha256(changed).hexdigest()

    tampered = _replace_entries(
        archive,
        {
            "results/performance.csv": changed,
            "manifest.json": json.dumps(manifest).encode("utf-8"),
        },
    )

    assert verify_export_archive(
        tampered,
        replay=False,
    )["sha256_verified"]

    with pytest.raises(ValueError, match="replay mismatch"):
        verify_export_archive(tampered)


def test_rejects_unsafe_member(sample_market_prices):
    archive = _make_bundle(sample_market_prices, "single")

    tampered = _replace_entries(
        archive,
        {"../escape.txt": b"unsafe"},
    )

    with pytest.raises(ValueError, match="unsafe ZIP entry"):
        verify_export_archive(tampered)


def test_rejects_bad_zip():
    with pytest.raises(
        ValueError,
        match="invalid or damaged ZIP",
    ):
        verify_export_archive(b"not a zip")
