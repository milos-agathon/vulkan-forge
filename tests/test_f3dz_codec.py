from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import forge3d
from forge3d.codec import compress_dem, decompress_dem, verify_dem
from _toml_compat import load_toml


CORPUS = Path(__file__).parent / "data" / "codec_corpus"
EPSILONS = (0.05, 0.1, 0.5, 1.0)
TILES = ("alpine", "rolling", "coastal_flat_nodata", "canyon")


def _source(name: str) -> np.ndarray:
    return np.load(CORPUS / f"{name}.npy", allow_pickle=False)


@lru_cache(maxsize=None)
def _case(name: str, eps: float, progressive: bool) -> tuple[bytes, dict]:
    source = _source(name)
    encoded = compress_dem(source, eps, progressive=progressive)
    return encoded, verify_dem(encoded, source)


def _finite_max_error(source: np.ndarray, decoded: np.ndarray) -> float:
    source_nan = np.isnan(source)
    assert np.array_equal(source_nan, np.isnan(decoded))
    finite = ~source_nan
    return float(np.max(np.abs(decoded[finite] - source[finite]), initial=0.0))


def _require_f3dz_nvidia_vulkan() -> dict:
    probe = forge3d.device_probe("vulkan")
    assert isinstance(probe, dict) and probe.get("status") == "ok", probe
    assert str(probe.get("backend", "")).lower() == "vulkan", probe
    assert str(probe.get("device_type", "")).lower() == "discretegpu", probe
    assert probe.get("vendor") == 0x10DE or "nvidia" in str(probe.get("name", "")).lower(), probe

    artifact_dir = os.getenv("FORGE3D_F3DZ_ARTIFACT_DIR")
    if artifact_dir:
        artifact_path = Path(artifact_dir) / "adapter-probe.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {"requested_backend": "vulkan", "probe": probe},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return probe


def test_committed_real_corpus_manifest_and_hashes() -> None:
    manifest = load_toml(CORPUS / "MANIFEST.toml")
    assert manifest["format"] == "forge3d-f3dz-corpus/1"
    assert manifest["license"].startswith("Public domain")
    assert [tile["name"] for tile in manifest["tiles"]] == list(TILES)
    for tile in manifest["tiles"]:
        payload = (CORPUS / tile["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == tile["output_sha256"]
        source = _source(tile["name"])
        assert source.shape == (256, 256)
        assert source.dtype == np.dtype("<f4")
    coastal = _source("coastal_flat_nodata")
    assert 0 < np.count_nonzero(np.isnan(coastal)) < coastal.size
    assert all(not np.isnan(_source(name)).any() for name in TILES if name != "coastal_flat_nodata")


@pytest.mark.parametrize("eps", EPSILONS)
@pytest.mark.parametrize("name", TILES)
def test_error_bound_stored_page_error_nan_and_determinism(name: str, eps: float) -> None:
    source = _source(name)
    encoded, report = _case(name, eps, True)
    second = compress_dem(source, eps, progressive=True)
    decoded, info = decompress_dem(encoded)

    assert encoded == second
    assert hashlib.sha256(encoded).digest() == hashlib.sha256(second).digest()
    assert info["codec"] == "f3dz/1"
    assert info["eps"] == float(np.float32(eps))
    assert not info["base_quality"]
    assert _finite_max_error(source, decoded) <= eps
    assert report["valid"] and report["crc_ok"] and report["header_consistent"]
    assert report["all_within_epsilon"]
    assert report["stored_errors_match"]
    assert all(page["within_epsilon"] for page in report["pages"])
    assert all(page["stored_error_matches"] for page in report["pages"])


def test_flate2_compression_win_and_predictor_ablation() -> None:
    rows: list[str] = []
    for eps in EPSILONS:
        f3dz_total = 0
        baseline_total = 0
        ablation_total = 0
        for name in TILES:
            encoded, report = _case(name, eps, False)
            f3dz = len(encoded)
            baseline = report["baseline_flate2_stream_size"]
            ablation = report["ablation_stream_size"]
            assert baseline is not None and ablation is not None
            assert f3dz < baseline, f"{name} eps={eps}: {f3dz=} must beat {baseline=}"
            f3dz_total += f3dz
            baseline_total += baseline
            ablation_total += ablation
            rows.append(
                f"{name:24s} eps={eps:>4} f3dz={f3dz:6d} "
                f"flate2={baseline:6d} order0={ablation:6d} "
                f"saving={1.0 - f3dz / baseline:7.2%}"
            )
        ratio = f3dz_total / baseline_total
        ablation_ratio = ablation_total / baseline_total
        assert ratio <= 0.70, f"eps={eps}: corpus ratio {ratio:.6f} exceeds 0.70"
        assert ablation_ratio > 0.70, (
            f"eps={eps}: order-0 ratio {ablation_ratio:.6f} unexpectedly retains the 30% win"
        )
        rows.append(
            f"{'CORPUS MEAN':24s} eps={eps:>4} f3dz={f3dz_total:6d} "
            f"flate2={baseline_total:6d} order0={ablation_total:6d} "
            f"saving={1.0 - ratio:7.2%}"
        )
    print("\n".join(rows))


@pytest.mark.parametrize("eps", EPSILONS)
@pytest.mark.parametrize("name", TILES)
def test_progressive_base_layer_is_bounded_and_exactly_declared(name: str, eps: float) -> None:
    _, report = _case(name, eps, True)
    base = report["base"]
    assert base is not None
    assert base["base_quality"]
    assert base["within_4epsilon"]
    assert base["max_abs_err"] <= 4.0 * eps
    assert base["stored_errors_match"]
    assert all(page["stored_error_matches"] for page in base["pages"])
    assert base["degradation_kind"] == "base_quality"
    assert base["degradation_name"] == "f3dz_unrefined_pages"


def test_base_quality_render_capture_declares_degradation_and_refined_does_not() -> None:
    source = _source("alpine")
    encoded = compress_dem(source, 0.1, progressive=True)

    forge3d.clear_native_degradations()
    forge3d.begin_render_execution_capture("f3dz.base-quality.test")
    report = verify_dem(encoded, source)
    assert report["base"]["base_quality"]
    forge3d.finish_render_execution_capture("f3dz.decode", 1)
    base_capture = json.loads(forge3d.render_execution_report())
    assert base_capture["codec"] == {
        "codec": "f3dz/1",
        "eps": 0.1,
        "pages_base_quality": 16,
    }
    assert any(
        entry["kind"] == "base_quality" and entry["name"] == "f3dz_unrefined_pages"
        for entry in base_capture["degradations"]
    )

    forge3d.clear_native_degradations()
    forge3d.begin_render_execution_capture("f3dz.refined.test")
    decompress_dem(encoded)
    forge3d.finish_render_execution_capture("f3dz.decode", 1)
    refined_capture = json.loads(forge3d.render_execution_report())
    assert refined_capture["codec"] == {
        "codec": "f3dz/1",
        "eps": 0.1,
        "pages_base_quality": 0,
    }
    assert not any(
        entry["kind"] == "base_quality" and entry["name"] == "f3dz_unrefined_pages"
        for entry in refined_capture["degradations"]
    )


def test_gpu_matches_cpu_for_every_corpus_page() -> None:
    if os.getenv("FORGE3D_REQUIRE_F3DZ_GPU") == "1":
        _require_f3dz_nvidia_vulkan()
    unavailable: list[str] = []
    for name in TILES:
        _, report = _case(name, 0.1, True)
        gpu = report["gpu"]
        if not gpu["available"]:
            unavailable.append(f"{name}: {gpu['error']}")
            continue
        assert gpu["bit_identical"]
        assert gpu["gpu_sha256"] == gpu["cpu_sha256"]
        assert gpu["gpu_page_sha256"] == gpu["cpu_page_sha256"]
        assert len(gpu["gpu_page_sha256"]) == 16
    if unavailable:
        if os.getenv("FORGE3D_REQUIRE_F3DZ_GPU") == "1":
            pytest.fail("physical GPU gate required but unavailable:\n" + "\n".join(unavailable))
        pytest.skip("no GPU adapter available for F3DZ identity gate")


def test_cross_platform_determinism_hashes() -> None:
    expected = load_toml(CORPUS / "DETERMINISM.toml")
    for name in TILES:
        source = _source(name)
        for eps in EPSILONS:
            encoded = compress_dem(source, eps, progressive=True)
            key = f"eps_{str(eps).replace('.', '_')}"
            assert hashlib.sha256(encoded).hexdigest() == expected[name][key]


def test_physical_gpu_ci_contract_is_zero_skip_and_records_benchmark() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    job = workflow.split("  test-f3dz-gpu:", 1)[1].split(
        "\n  test-anamnesis-portability-seed:", 1
    )[0]
    assert "needs: [build-wheel-windows, terrain-golden-paths]" in job
    assert "if: >-" in job
    assert "github.event_name == 'schedule'" in job
    assert "inputs.scope == 'full'" in job
    assert "inputs.scope == 'f3dz'" in job
    for forbidden in (
        "github.event_name == 'pull_request'",
        "github.event_name == 'push'",
        "run-physical",
        "needs.terrain-golden-paths.outputs.f3dz",
    ):
        assert forbidden not in job

    f3dz_paths = workflow.split("            f3dz:\n", 1)[1].split(
        "\n            anamnesis:\n", 1
    )[0]
    required_paths = (
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "pyproject.toml",
        "benches/f3dz_bench.rs",
        "python/forge3d/codec.py",
        "python/forge3d/__init__.py",
        "python/forge3d/_native.py",
        "python/forge3d/_gpu.py",
        "src/codec/**",
        "src/lib.rs",
        "src/core/mod.rs",
        "src/core/error.rs",
        "src/core/certificate.rs",
        "src/core/degradation.rs",
        "src/core/gpu.rs",
        "src/core/capabilities.rs",
        "src/core/resource_tracker.rs",
        "src/core/memory_tracker.rs",
        "src/core/shader_registry.rs",
        "src/core/shader_contract_runtime.rs",
        "src/core/provenance.rs",
        "src/core/memory_tracker/**",
        "src/py_functions/codec.rs",
        "src/py_functions/diagnostics.rs",
        "src/py_functions/mod.rs",
        "src/py_module/mod.rs",
        "src/py_module/functions.rs",
        "src/py_module/functions/codec.rs",
        "src/py_module/functions/diagnostics.rs",
        "src/shaders/f3dz_decode.wgsl",
        "scripts/install_compatible_wheel.py",
        "scripts/assert_junit_zero_skips.py",
        "tests/data/codec_corpus/**",
        "tests/_toml_compat.py",
        "tests/test_f3dz_codec.py",
    )
    for path in required_paths:
        assert f"              - '{path}'" in f3dz_paths
    excluded_paths = (
        "docs/formats/f3dz.md",
        "tools/f3dz_determinism_report.py",
        ".cargo/**",
        "scripts/terrain_ci_probe.py",
        "src/terrain/cog/cog_reader.rs",
        "src/terrain/stream/height.rs",
        "python/forge3d/__init__.pyi",
    )
    for path in excluded_paths:
        assert f"              - '{path}'" not in f3dz_paths
    for workflow_path in (
        ".github/workflows/ci.yml",
        ".github/workflows/build-wheel.yml",
    ):
        assert f"              - '{workflow_path}'" in f3dz_paths
    assert "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in job
    assert "FORGE3D_REQUIRE_F3DZ_GPU: '1'" in job
    assert "python scripts/terrain_ci_probe.py" not in job
    assert "python -m pytest tests/test_f3dz_codec.py" in job
    assert "scripts/assert_junit_zero_skips.py" in job
    assert "cargo bench --bench f3dz_bench" in job
    assert "rustup default stable" in job
    assert "rust-toolchain.txt" in job
    assert job.index("rustup default stable") < job.index(
        "cargo bench --bench f3dz_bench"
    )
    benchmark = job.split(
        "      - name: Record separate source benchmark (nightly/full only)", 1
    )[1].split("\n      - name: Upload F3DZ physical-GPU evidence", 1)[0]
    assert "github.event_name == 'schedule'" in benchmark
    assert "inputs.scope == 'full'" in benchmark
    assert "source-benchmark-not-installed-wheel-acceptance" in benchmark
    assert "benchmark-provenance.json" in benchmark
    assert "f3dz-physical-gpu-evidence" in job
    assert "Get-PSDrive -PSProvider FileSystem" in job
    assert "Sort-Object Free -Descending" in job
    assert (
        '$scratchScope = "$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT-$env:GITHUB_JOB"'
        in job
    )
    assert "FORGE3D_F3DZ_SCRATCH_DIR" in job
    assert '"CARGO_TARGET_DIR=$targetDir"' in job
    assert "name: Clean F3DZ build scratch" in job
    assert "(Split-Path -Leaf $parentDir) -ne 'forge3d-ci-scratch'" in job
    assert job.index("name: Upload F3DZ physical-GPU evidence") < job.index(
        "name: Clean F3DZ build scratch"
    )


def test_f3dz_gpu_probe_writes_adapter_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    probe = {
        "status": "ok",
        "backend": "Vulkan",
        "device_type": "DiscreteGpu",
        "vendor": 0x10DE,
        "name": "NVIDIA GeForce RTX 3070",
    }
    monkeypatch.setattr(forge3d, "device_probe", lambda backend: probe)
    monkeypatch.setenv("FORGE3D_F3DZ_ARTIFACT_DIR", str(tmp_path))

    assert _require_f3dz_nvidia_vulkan() == probe
    assert json.loads((tmp_path / "adapter-probe.json").read_text(encoding="utf-8")) == {
        "requested_backend": "vulkan",
        "probe": probe,
    }
