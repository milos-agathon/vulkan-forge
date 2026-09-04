from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import forge3d as f3d
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "check_anamnesis_portability.py"


def test_ci_portability_jobs_use_available_physical_gpu_runner():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    seed_job = ci.split("  test-anamnesis-portability-seed:", 1)[1].split(
        "\n  test-anamnesis-portability:", 1
    )[0]
    consumer_job = ci.split("  test-anamnesis-portability:", 1)[1].split(
        "\n  test-anamnesis-production:", 1
    )[0]
    assert (
        "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in seed_job
    )
    assert (
        "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]"
        in consumer_job
    )
    assert "anamnesis-producer" not in seed_job
    assert "anamnesis-consumer" not in consumer_job
    assert "WGPU_BACKENDS: vulkan" in seed_job
    assert "WGPU_BACKENDS: dx12" in consumer_job
    assert "name: wheels-windows" in seed_job
    assert "--producer-backend vulkan --consumer-backend dx12" in seed_job
    assert "--machine-id-file" in seed_job
    assert "terra_determinata_v1.dx12.sha256" in consumer_job
    assert "--runner-name '${{ runner.name }}'" in seed_job
    assert "metal" not in seed_job.lower()


def test_portability_driver_requires_distinct_backends_not_distinct_machines():
    source = DRIVER.read_text(encoding="utf-8")
    assert 'machine_id == record["producer_machine_id"]' not in source
    assert 'runner_name == record.get("producer_runner_name")' not in source
    assert '"forge3d.anamnesis.native-portability/1"' in source
    assert 'report["hits"] != _PASS_LABELS' in source
    assert 'report["graph_command_submissions"] != 0' in source
    assert "png_to_numpy" not in source
    assert "render_sequence" not in source


def _run(*arguments: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(DRIVER), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.splitlines()[-1])


def test_engine_wgsl_fingerprint_uses_portable_relative_paths():
    digest = hashlib.sha256()
    shader_root = ROOT / "src" / "shaders"
    for path in sorted(shader_root.rglob("*.wgsl")):
        relative = path.relative_to(shader_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    native = json.loads(f3d.anamnesis_engine_fingerprint())
    assert native["wgsl_tree_sha256"] == digest.hexdigest()


def test_portable_store_hits_and_capability_mismatch_misses(tmp_path):
    if os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1":
        pytest.skip(
            "native portability is exercised by the protected producer/consumer jobs"
        )
    if not f3d.has_gpu():
        pytest.skip("portable resource rehydration requires a GPU adapter")
    frame_blob = tmp_path / "terra.png"
    rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    rgba[..., 0] = np.arange(512, dtype=np.uint16)[None, :] % 256
    rgba[..., 1] = np.arange(512, dtype=np.uint16)[:, None] % 256
    rgba[..., 2] = 127
    rgba[..., 3] = 255
    f3d.numpy_to_png(frame_blob, rgba)
    golden = tmp_path / "terra.sha256"
    golden.write_text(
        hashlib.sha256(frame_blob.read_bytes()).hexdigest() + "  terra.png\n",
        encoding="utf-8",
    )
    adapter_record = tmp_path / "adapter.jsonl"
    adapter_record.write_text(
        json.dumps(
            {
                "adapter": {
                    "backend": "vulkan",
                    "name": "physical-test-adapter",
                    "software_fallback": False,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    record = tmp_path / "record.json"
    consumer_blob = tmp_path / "consumer.png"
    consumer_blob.write_bytes(frame_blob.read_bytes())
    consumer_adapter = tmp_path / "consumer-adapter.jsonl"
    consumer_adapter.write_text(
        json.dumps(
            {
                "adapter": {
                    "backend": "dx12",
                    "name": "physical-test-consumer",
                    "software_fallback": False,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    producer_machine = tmp_path / "producer-machine.txt"
    producer_machine.write_text(
        "11111111-1111-1111-1111-111111111111\n", encoding="utf-8"
    )
    consumer_machine = tmp_path / "consumer-machine.txt"
    consumer_machine.write_text(
        "11111111-1111-1111-1111-111111111111\n", encoding="utf-8"
    )

    seeded = _run(
        "seed",
        "--cache",
        str(cache),
        "--record",
        str(record),
        "--frame-blob",
        str(frame_blob),
        "--golden",
        str(golden),
        "--adapter-record",
        str(adapter_record),
        "--machine-id-file",
        str(producer_machine),
        "--runner-name",
        "producer-runner",
    )
    checked = _run(
        "check",
        "--cache",
        str(cache),
        "--record",
        str(record),
        "--consumer-frame-blob",
        str(consumer_blob),
        "--consumer-golden",
        str(golden),
        "--consumer-adapter-record",
        str(consumer_adapter),
        "--machine-id-file",
        str(consumer_machine),
        "--runner-name",
        "producer-runner",
    )
    mismatched = _run("mismatch", "--cache", str(cache), "--record", str(record))

    assert seeded["misses"] > 0
    assert checked["hit_rate"] == 1.0
    assert checked["backend_goldens_match"] is True
    assert checked["distinct_machine"] is False
    assert mismatched["hit_rate"] == 0.0
    assert mismatched["cache_isolated"] is True
    assert mismatched["mismatch_dimension"] == "compatibility_profile"


def test_seed_rejects_blob_that_differs_from_committed_golden(tmp_path):
    frame_blob = tmp_path / "terra.png"
    frame_blob.write_bytes(b"wrong pixels")
    golden = tmp_path / "terra.sha256"
    golden.write_text(hashlib.sha256(b"golden pixels").hexdigest() + "\n", encoding="utf-8")
    adapter_record = tmp_path / "adapter.jsonl"
    adapter_record.write_text(
        json.dumps({"adapter": {"software_fallback": False}}) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "seed",
            "--cache",
            str(tmp_path / "cache"),
            "--record",
            str(tmp_path / "record.json"),
            "--frame-blob",
            str(frame_blob),
            "--golden",
            str(golden),
            "--adapter-record",
            str(adapter_record),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "differs from committed golden" in completed.stderr
