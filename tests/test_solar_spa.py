from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from forge3d.geo import SolarTime, solar_position
from scripts import helios_evidence_report as helios_reporter


REFERENCE = Path(__file__).parent / "data" / "spa_reference.csv"
ANGLE_TOLERANCE_DEG = 0.0003


def _angle_error(actual: float, expected: float) -> float:
    delta = abs(actual - expected) % 360.0
    return min(delta, 360.0 - delta)


def _reference_rows() -> list[dict[str, float]]:
    with REFERENCE.open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def test_spa_worked_example_matches_nrel() -> None:
    result = solar_position(
        (2003, 10, 17, 12, 30, 30),
        39.742476,
        -105.1786,
        1830.14,
        tz_offset_hours=-7,
        delta_t_seconds=67,
        pressure_mbar=820,
        temperature_c=11,
    )

    zenith_error = abs(result["zenith_deg"] - 50.11162)
    azimuth_error = _angle_error(result["azimuth_deg"], 194.34024)
    print(
        "HELIOS SPA worked example: "
        f"delta_zenith_deg={zenith_error:.12f}, "
        f"delta_azimuth_deg={azimuth_error:.12f}"
    )
    assert zenith_error <= ANGLE_TOLERANCE_DEG
    assert azimuth_error <= ANGLE_TOLERANCE_DEG


def test_spa_matches_official_reference_rows() -> None:
    # Every row is independently traceable to an official NREL publication or
    # the public NLR SPA calculator; see spa_reference.PROVENANCE.md.
    rows = _reference_rows()
    assert len(rows) >= 20
    assert min(row["lat"] for row in rows) <= -80
    assert max(row["lat"] for row in rows) >= 80
    assert min(row["year"] for row in rows) <= 1900
    assert max(row["year"] for row in rows) >= 2100

    maxima = {
        "zenith_deg": 0.0,
        "azimuth_deg": 0.0,
        "true_elevation_deg": 0.0,
    }
    for row in rows:
        result = solar_position(
            tuple(int(row[key]) for key in ("year", "month", "day", "hour", "minute", "second")),
            row["lat"],
            row["lon"],
            row["elev_m"],
            tz_offset_hours=row["tz_offset_hours"],
            delta_t_seconds=row["delta_t_seconds"],
            pressure_mbar=row["pressure_mbar"],
            temperature_c=row["temperature_c"],
        )
        label = f"{int(row['year'])}-{int(row['month']):02}-{int(row['day']):02}@{row['lat']},{row['lon']}"
        zenith_error = abs(result["zenith_deg"] - row["zenith_deg"])
        azimuth_error = _angle_error(result["azimuth_deg"], row["azimuth_deg"])
        elevation_error = abs(
            result["true_elevation_deg"] - row["true_elevation_deg"]
        )
        maxima["zenith_deg"] = max(maxima["zenith_deg"], zenith_error)
        maxima["azimuth_deg"] = max(maxima["azimuth_deg"], azimuth_error)
        maxima["true_elevation_deg"] = max(
            maxima["true_elevation_deg"], elevation_error
        )
        assert zenith_error <= ANGLE_TOLERANCE_DEG, label
        assert azimuth_error <= ANGLE_TOLERANCE_DEG, label
        assert elevation_error <= ANGLE_TOLERANCE_DEG, label
        # The public NLR calculator exports these two fields to six decimals.
        assert abs(result["distance_au"] - row["distance_au"]) <= 5.1e-7, label
        assert abs(result["equation_of_time_min"] - row["equation_of_time_min"]) <= 5.1e-7, label
    print(
        "HELIOS SPA reference rows: "
        f"rows={len(rows)}, "
        f"max_zenith_error_deg={maxima['zenith_deg']:.12f}, "
        f"max_azimuth_error_deg={maxima['azimuth_deg']:.12f}, "
        f"max_true_elevation_error_deg={maxima['true_elevation_deg']:.12f}"
    )


def test_solar_time_is_an_explicit_timezone_free_contract() -> None:
    solar_time = SolarTime(
        utc=(2025, 6, 21, 12, 0, 0),
        observer_lat=48.2082,
        observer_lon=16.3738,
        observer_elev_m=171,
        tz_offset_hours=2,
        delta_t_seconds=74.5,
        pressure_mbar=1000,
        temperature_c=25,
    )

    assert solar_time.position()["azimuth_deg"] == pytest.approx(150.7305003541, abs=ANGLE_TOLERANCE_DEG)
    assert solar_time.to_native()["utc"] == (2025, 6, 21, 12, 0, 0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latitude", 90.0001),
        ("longitude", 180.0001),
        ("pressure_mbar", 0.0),
        ("temperature_c", -273.15),
        ("tz_offset_hours", 19.0),
    ],
)
def test_spa_rejects_invalid_physical_inputs(field: str, value: float) -> None:
    kwargs = {
        "tz_offset_hours": 0.0,
        "delta_t_seconds": 74.0,
        "pressure_mbar": 1013.25,
        "temperature_c": 15.0,
    }
    latitude = 45.0
    longitude = 5.0
    if field == "latitude":
        latitude = value
    elif field == "longitude":
        longitude = value
    else:
        kwargs[field] = value

    with pytest.raises(ValueError):
        solar_position((2025, 1, 1, 12, 0, 0), latitude, longitude, 0.0, **kwargs)


HELIOS_TEST_HEAD = "1" * 40
HELIOS_ADAPTER = {
    "status": "ok",
    "name": "NVIDIA GeForce RTX 4090",
    "vendor": 0x10DE,
    "device": 9860,
    "backend": "Vulkan",
    "device_type": "DiscreteGpu",
    "driver": "NVIDIA",
    "driver_info": "555.85",
    "software_fallback": False,
}
HELIOS_PATH_MEMORY = {
    "shadow_mask": {
        "current_host_visible_bytes": 0,
        "expected_readback_bytes": 512,
        "peak_host_visible_bytes": 512,
    },
    "viewshed": {
        "current_host_visible_bytes": 0,
        "expected_readback_bytes": 65_536,
        "peak_host_visible_bytes": 65_536,
    },
}


def _helios_evidence(directory: Path) -> None:
    for name in ("checked-out-head.txt", "rust-checked-out-head.txt"):
        (directory / name).write_text(HELIOS_TEST_HEAD + "\n", encoding="utf-8")
    (directory / "adapter-probe.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "requested_backend": "vulkan",
                "probe": HELIOS_ADAPTER,
            }
        ),
        encoding="utf-8",
    )
    cases = "".join(
        f'<testcase name="{name}" />'
        for name in sorted(helios_reporter.REQUIRED_TESTS)
    )
    (directory / "pytest-junit.xml").write_text(
        f'<testsuite tests="{len(helios_reporter.REQUIRED_TESTS)}">{cases}</testsuite>',
        encoding="utf-8",
    )
    (directory / "rust-conservative-descent.log").write_text(
        "HELIOS 2D 256x256 conservative descent: rays=10000, false_misses=0, "
        "false_hit_rate=0.000000, shadow_mask_agreement=1.000000, "
        "mask_false_misses=0, mask_false_hits=0\n"
        "test path_tracing::hybrid_compute::terrain_heightfield::tests::"
        "curvature_descent_is_conservative ... ok\n",
        encoding="utf-8",
    )
    production_gpu = {
        "schema": "forge3d.helios_production_terrain_trace_gpu/1",
        "status": "PASS",
        "candidate_sha": HELIOS_TEST_HEAD,
        "strict_physical_required": True,
        "adapter": {
            **HELIOS_ADAPTER,
            "software_classification": "hardware",
        },
        "metrics": {
            "rays": 10_000,
            "false_misses": 0,
            "false_hit_rate": 0.0,
            "shadow_mask_pixels": 65_025,
            "shadow_mask_agreement": 1.0,
            "mask_false_misses": 0,
            "mask_false_hits": 0,
        },
    }
    (directory / "production-gpu-conservative-descent.log").write_text(
        "HELIOS_PRODUCTION_TERRAIN_TRACE_GPU_JSON "
        + json.dumps(production_gpu, sort_keys=True)
        + "\n"
        "test path_tracing::hybrid_compute::terrain_heightfield::tests::"
        "curvature_descent_production_gpu_is_conservative ... ok\n",
        encoding="utf-8",
    )
    before_metrics = {
        "gpu_resource_bytes": 100,
        "minmax_pyramid_bytes": 40,
        "peak_host_visible_bytes": 100,
    }
    after_metrics = {
        "gpu_resource_bytes": 105,
        "minmax_pyramid_bytes": 40,
        "peak_host_visible_bytes": 105,
    }
    common = {
        "returncode": 0,
        "package_path": "/worktree/python/forge3d/__init__.py",
        "native_path": "/worktree/python/forge3d/_forge3d.abi3.so",
        "native_sha256": "a" * 64,
        "adapter": HELIOS_ADAPTER,
        "workload": {"width": 64, "height": 64, "seed": 7},
    }
    memory = {
        "schema": "forge3d.helios_memory_comparison/1",
        "before": {**common, "mode": "flat_baseline", "metrics": before_metrics},
        "after": {**common, "mode": "helios", "metrics": after_metrics},
        "delta": {
            key: after_metrics[key] - before_metrics[key] for key in before_metrics
        },
    }
    (directory / "pytest.log").write_text(
        "HELIOS SPA worked example: delta_zenith_deg=0.000002024000, "
        "delta_azimuth_deg=0.000000510200\n"
        "HELIOS SPA reference rows: rows=24, max_zenith_error_deg=0.000000474721, "
        "max_azimuth_error_deg=0.000000411981, "
        "max_true_elevation_error_deg=0.000000456648\n"
        "HELIOS load-bearing: IoU=0.950000000, flipped=205, "
        "extent=76400.0x88900.0 m\n"
        "HELIOS curved Whitebox reference: IoU=0.990000000, flipped=12, "
        "flat_negative_iou=0.900000000, flat_negative_flipped=205\n"
        "HELIOS shadow-tip: bearing_deg=75.435809247000, "
        "expected_bearing_deg=75.435809247000, bearing_error_deg=0.000000000000, "
        "length_m=98138.152083000000, curved_prediction_m=98138.152083000000, "
        "relative_error=0.000000000000, flat_length_m=85145.998358000004, "
        "flat_delta_fraction=0.132386370100\n"
        "HELIOS memory: "
        + json.dumps(memory, sort_keys=True)
        + "\n"
        + "HELIOS path memory: "
        + json.dumps(HELIOS_PATH_MEMORY, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def test_helios_verifier_writes_exact_metrics_and_pass_markers(
    tmp_path: Path,
) -> None:
    _helios_evidence(tmp_path)
    report = helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)
    assert report["status"] == "PASS"
    assert report["metrics"]["conservative_descent"]["false_misses"] == 0
    production_gpu = report["metrics"]["production_gpu_conservative_descent"]
    assert production_gpu["status"] == "PASS"
    assert production_gpu["candidate_sha"] == HELIOS_TEST_HEAD
    assert production_gpu["adapter"] == {
        **HELIOS_ADAPTER,
        "software_classification": "hardware",
    }
    assert production_gpu["metrics"]["shadow_mask_pixels"] == 65_025
    assert report["metrics"]["load_bearing_viewshed"]["flipped_pixels"] == 205
    assert report["metrics"]["memory"]["after"]["metrics"]["peak_host_visible_bytes"] == 105
    assert report["metrics"]["path_memory"]["viewshed"] == {
        "current_host_visible_bytes": 0,
        "expected_readback_bytes": 65_536,
        "peak_host_visible_bytes": 65_536,
    }
    assert json.loads((tmp_path / "lane-ran.json").read_text())["status"] == "RAN"
    assert json.loads((tmp_path / "verification.json").read_text())["status"] == "PASS"


def test_helios_verifier_rejects_candidate_mismatch(tmp_path: Path) -> None:
    _helios_evidence(tmp_path)
    with pytest.raises(helios_reporter.EvidenceError, match="does not match candidate"):
        helios_reporter.verify(tmp_path, "2" * 40)


def test_helios_verifier_rejects_nonphysical_adapter(tmp_path: Path) -> None:
    _helios_evidence(tmp_path)
    probe_path = tmp_path / "adapter-probe.json"
    probe = json.loads(probe_path.read_text())
    probe["probe"]["software_fallback"] = True
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    with pytest.raises(
        helios_reporter.EvidenceError, match="physical NVIDIA Vulkan"
    ):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


def test_helios_verifier_rejects_nonproducer_adapter_envelope(tmp_path: Path) -> None:
    _helios_evidence(tmp_path)
    probe_path = tmp_path / "adapter-probe.json"
    probe = json.loads(probe_path.read_text())
    probe["status"] = "ok"
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    with pytest.raises(
        helios_reporter.EvidenceError, match="envelope status must equal 'passed'"
    ):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


def test_helios_verifier_rejects_relaxed_load_bearing_metric(
    tmp_path: Path,
) -> None:
    _helios_evidence(tmp_path)
    log_path = tmp_path / "pytest.log"
    log_path.write_text(
        log_path.read_text().replace("IoU=0.950000000", "IoU=0.970000000"),
        encoding="utf-8",
    )
    with pytest.raises(helios_reporter.EvidenceError, match="load-bearing viewshed"):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


def _mutate_production_gpu(directory: Path, section: str, field: str, value: object) -> None:
    log_path = directory / "production-gpu-conservative-descent.log"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    prefix = "HELIOS_PRODUCTION_TERRAIN_TRACE_GPU_JSON "
    index = next(index for index, line in enumerate(lines) if line.startswith(prefix))
    record = json.loads(lines[index].removeprefix(prefix))
    target = record if section == "root" else record[section]
    target[field] = value
    lines[index] = prefix + json.dumps(record, sort_keys=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("root", "status", "NOT_PROVEN", "candidate and strict lane"),
        ("root", "candidate_sha", "2" * 40, "candidate and strict lane"),
        ("root", "strict_physical_required", False, "candidate and strict lane"),
        ("metrics", "rays", 9_999, "GPU ray gate"),
        ("metrics", "false_misses", 1, "GPU ray gate"),
        ("metrics", "shadow_mask_agreement", 0.998, "GPU shadow-mask gate"),
    ],
)
def test_helios_verifier_rejects_invalid_production_gpu_proof(
    tmp_path: Path, section: str, field: str, value: object, message: str
) -> None:
    _helios_evidence(tmp_path)
    _mutate_production_gpu(tmp_path, section, field, value)
    with pytest.raises(helios_reporter.EvidenceError, match=message):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "missing"),
        ("name", "NVIDIA stale adapter"),
        ("vendor", 0x1234),
        ("device", 1),
        ("device_type", "IntegratedGpu"),
        ("backend", "Dx12"),
        ("driver", "stale-driver"),
        ("driver_info", "stale-driver-info"),
        ("software_fallback", True),
        ("software_classification", "software"),
    ],
)
def test_helios_verifier_rejects_each_mutated_producer_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    _helios_evidence(tmp_path)
    _mutate_production_gpu(tmp_path, "adapter", field, value)
    with pytest.raises(helios_reporter.EvidenceError, match="acceptance adapter"):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


def test_helios_verifier_rejects_missing_production_gpu_proof(tmp_path: Path) -> None:
    _helios_evidence(tmp_path)
    (tmp_path / "production-gpu-conservative-descent.log").unlink()
    with pytest.raises(helios_reporter.EvidenceError, match="missing or unreadable"):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


def _replace_path_memory(directory: Path, metrics: dict) -> None:
    log_path = directory / "pytest.log"
    prefix = "HELIOS path memory: "
    lines = [
        prefix + json.dumps(metrics, sort_keys=True) if line.startswith(prefix) else line
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _replace_memory(directory: Path, metrics: dict) -> None:
    log_path = directory / "pytest.log"
    prefix = "HELIOS memory: "
    lines = [
        prefix + json.dumps(metrics, sort_keys=True) if line.startswith(prefix) else line
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _memory(directory: Path) -> dict:
    line = next(
        line
        for line in (directory / "pytest.log").read_text(encoding="utf-8").splitlines()
        if line.startswith("HELIOS memory: ")
    )
    return json.loads(line.removeprefix("HELIOS memory: "))


def test_helios_verifier_rejects_historical_global_peak_schema(tmp_path: Path) -> None:
    _helios_evidence(tmp_path)
    _replace_memory(
        tmp_path,
        {
            "flat_baseline": {"peak_host_visible_bytes": 100},
            "helios": {"peak_host_visible_bytes": 100},
        },
    )
    with pytest.raises(helios_reporter.EvidenceError, match="unsupported schema"):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("native_sha256", "b" * 64, "native SHA-256 mismatch"),
        ("adapter", {**HELIOS_ADAPTER, "name": "different"}, "adapter mismatch"),
        ("workload", {"width": 65, "height": 64, "seed": 7}, "workload mismatch"),
    ],
)
def test_helios_verifier_rejects_unbound_memory_comparison(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    _helios_evidence(tmp_path)
    memory = _memory(tmp_path)
    memory["after"][field] = value
    _replace_memory(tmp_path, memory)
    with pytest.raises(helios_reporter.EvidenceError, match=message):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


def test_helios_verifier_rejects_failed_memory_subprocess(tmp_path: Path) -> None:
    _helios_evidence(tmp_path)
    memory = _memory(tmp_path)
    memory["after"]["returncode"] = 1
    _replace_memory(tmp_path, memory)
    with pytest.raises(helios_reporter.EvidenceError, match="subprocess did not succeed"):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


@pytest.mark.parametrize("metric", ["peak_host_visible_bytes", "gpu_resource_bytes"])
def test_helios_verifier_rejects_memory_growth_over_five_percent(
    tmp_path: Path, metric: str
) -> None:
    _helios_evidence(tmp_path)
    memory = _memory(tmp_path)
    memory["after"]["metrics"][metric] = 106
    memory["delta"][metric] = 6
    _replace_memory(tmp_path, memory)
    with pytest.raises(helios_reporter.EvidenceError, match=f"memory gate failed for {metric}"):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


def test_helios_verifier_rejects_minmax_pyramid_mismatch(tmp_path: Path) -> None:
    _helios_evidence(tmp_path)
    memory = _memory(tmp_path)
    memory["after"]["metrics"]["minmax_pyramid_bytes"] = 41
    memory["delta"]["minmax_pyramid_bytes"] = 1
    _replace_memory(tmp_path, memory)
    with pytest.raises(helios_reporter.EvidenceError, match="min-max pyramid allocation changed"):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


def test_helios_verifier_rejects_incomplete_path_memory(tmp_path: Path) -> None:
    _helios_evidence(tmp_path)
    metrics = json.loads(json.dumps(HELIOS_PATH_MEMORY))
    metrics.pop("shadow_mask")
    _replace_path_memory(tmp_path, metrics)
    with pytest.raises(helios_reporter.EvidenceError, match="path memory.*incomplete"):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


def test_helios_verifier_rejects_unreleased_path_memory(tmp_path: Path) -> None:
    _helios_evidence(tmp_path)
    metrics = json.loads(json.dumps(HELIOS_PATH_MEMORY))
    metrics["viewshed"]["current_host_visible_bytes"] = 1
    _replace_path_memory(tmp_path, metrics)
    with pytest.raises(helios_reporter.EvidenceError, match="viewshed memory accounting"):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


def test_helios_verifier_rejects_path_memory_over_budget(tmp_path: Path) -> None:
    _helios_evidence(tmp_path)
    metrics = json.loads(json.dumps(HELIOS_PATH_MEMORY))
    budget = 512 * 1024 * 1024
    metrics["shadow_mask"]["expected_readback_bytes"] = budget
    metrics["shadow_mask"]["peak_host_visible_bytes"] = budget
    _replace_path_memory(tmp_path, metrics)
    with pytest.raises(helios_reporter.EvidenceError, match="exceeds the 512 MiB budget"):
        helios_reporter.verify(tmp_path, HELIOS_TEST_HEAD)


def test_helios_ci_path_filter_covers_trust_boundary() -> None:
    workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    helios_paths = workflow.split("            helios:\n", 1)[1].split(
        "            determinism_f3dz:\n", 1
    )[0]
    for path in (
        ".github/workflows/build-wheel.yml",
        "scripts/terrain_ci_probe.py",
        "scripts/assert_junit_zero_skips.py",
        "scripts/install_compatible_wheel.py",
    ):
        assert f"              - '{path}'" in helios_paths

    helios_job = workflow.split("  test-helios-gpu:\n", 1)[1].split(
        "\n  # ============================================================================\n  # Hosted determinism families",
        1,
    )[0]
    assert "python scripts/install_compatible_wheel.py dist" in helios_job
    assert "python -m pip install pytest numpy rasterio pyproj" in helios_job
    assert "FORGE3D_HELIOS_REQUIRE_PHYSICAL_GPU: '1'" in helios_job
    assert (
        "path_tracing::hybrid_compute::terrain_heightfield::tests::"
        "curvature_descent_production_gpu_is_conservative -- --exact --nocapture"
        in helios_job
    )
    assert "production-gpu-conservative-descent.log" in helios_job
