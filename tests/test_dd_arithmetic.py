"""DUPLA double-float source and generation contracts."""

from __future__ import annotations

import re
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WGSL = ROOT / "src" / "shaders" / "includes" / "dd.wgsl"
RUST = ROOT / "src" / "core" / "dd.rs"
RUST_VECTOR = ROOT / "src" / "core" / "dd" / "vector.rs"
RUST_PRODUCT = ROOT / "src" / "core" / "dd" / "product.rs"
GENERATOR = ROOT / "scripts" / "generate_dd.py"
HARNESS = ROOT / "src" / "shaders" / "dd_harness.wgsl"
GPU = ROOT / "src" / "core" / "dd" / "gpu.rs"
PY_FUNCTIONS = ROOT / "src" / "py_functions" / "precision.rs"
PY_REGISTRAR = ROOT / "src" / "py_module" / "functions" / "precision.rs"


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"\bfn\s+{re.escape(name)}\b[^{{]*\{{", source)
    assert match, f"missing function {name}"
    depth = 1
    cursor = match.end()
    while cursor < len(source) and depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    assert depth == 0, f"unterminated function {name}"
    return source[match.end() : cursor - 1]


def _without_comments(source: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)


def test_generated_mirrors_are_current() -> None:
    assert GENERATOR.is_file()
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "lockstep operation order verified" in result.stdout


def test_generator_rejects_cross_language_statement_reordering() -> None:
    spec = importlib.util.spec_from_file_location("generate_dd", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rust = "\n".join(
        path.read_text(encoding="utf-8") for path in (RUST, RUST_PRODUCT, RUST_VECTOR)
    )
    wgsl = WGSL.read_text(encoding="utf-8")
    first = "let e1 = dd_barrier(e0 + dd_barrier(sa.hi * sb.lo));"
    second = "let e2 = dd_barrier(e1 + dd_barrier(sa.lo * sb.hi));"
    mutated = wgsl.replace(first + "\n    " + second, second + "\n    " + first)
    assert mutated != wgsl
    try:
        module.validate_lockstep(rust, mutated)
    except SystemExit as error:
        assert "operation-order mismatch" in str(error)
    else:
        raise AssertionError("generator accepted divergent Rust/WGSL statement order")

    newton = "y = dd_barrier(y * (1.5 - dd_barrier(half_x * dd_barrier(y * y))));"
    reordered = "y = dd_barrier((1.5 - dd_barrier(half_x * dd_barrier(y * y))) * y);"
    mutated = wgsl.replace(newton, reordered, 1)
    assert mutated != wgsl
    try:
        module.validate_lockstep(rust, mutated)
    except SystemExit as error:
        assert "operation-order mismatch" in str(error)
    else:
        raise AssertionError("generator accepted divergent Newton operand order")

    subnormal = "let total = a_signed + b_signed;"
    mutated = wgsl.replace(subnormal, "let total = b_signed + a_signed;", 1)
    assert mutated != wgsl
    try:
        module.validate_lockstep(rust, mutated)
    except SystemExit as error:
        assert "operation-order mismatch" in str(error)
    else:
        raise AssertionError("generator accepted divergent subnormal integer path")


def test_wgsl_exports_complete_dd_surface_and_sources() -> None:
    source = WGSL.read_text(encoding="utf-8")
    assert "struct DD" in source
    assert "struct DDVec3" in source
    for name in (
        "two_sum",
        "quick_two_sum",
        "two_prod_split",
        "two_prod_fma",
        "two_prod",
        "dd_renorm",
        "dd_add",
        "dd_sub",
        "dd_mul",
        "dd_div",
        "dd_sqrt",
        "dd_dot3",
        "dd_length3",
        "dd_sub_vec3",
    ):
        _function_body(source, name)
    for citation in ("Knuth", "Dekker", "Veltkamp", "Joldes", "Muller", "Popescu"):
        assert citation in source
    assert "DD_ADD_BOUND_U2: f32 = 3.0" in source
    assert "DD_MUL_BOUND_U2: f32 = 7.0" in source
    assert "DD_DIV_BOUND_U2: f32 = 15.0" in source
    assert "DD_SQRT_BOUND_U2: f32 = 15.0" in source
    assert "fn dd_barrier" in source
    assert "__DD_BARRIER_BODY__" in source
    assembly = (ROOT / "src" / "core" / "dd" / "gpu_exec.rs").read_text(encoding="utf-8")
    assert "DETERMINISM" in assembly
    assert "0x7fffffffu" in assembly and "0x80000000u" in assembly
    assert "atomicStore" not in assembly and "atomicLoad" not in assembly
    assert "Requires determinism.wgsl" in source


def test_codegen_files_respect_repository_size_guideline() -> None:
    for path in (
        GENERATOR,
        ROOT / "scripts" / "dd.rs.in",
        ROOT / "scripts" / "dd_product.rs.in",
        ROOT / "scripts" / "dd_vector.rs.in",
        ROOT / "scripts" / "dd.wgsl.in",
        RUST,
        RUST_PRODUCT,
        RUST_VECTOR,
        ROOT / "src" / "core" / "dd" / "types.rs",
    ):
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 300, path


def test_error_free_and_refinement_bodies_avoid_forbidden_ops() -> None:
    source = _without_comments(WGSL.read_text(encoding="utf-8"))
    for name in (
        "two_sum",
        "quick_two_sum",
        "two_prod_split",
        "dd_renorm",
        "dd_add",
        "dd_sub",
        "dd_mul",
        "dd_reciprocal_refine",
        "dd_div",
        "dd_inverse_sqrt_refine",
        "dd_sqrt",
    ):
        body = _function_body(source, name)
        assert "/" not in body, f"{name} contains hardware division"
        assert not re.search(r"\bsqrt\s*\(", body), f"{name} contains hardware sqrt"
    seed = _function_body(source, "dd_inverse_sqrt_seed")
    assert seed.count("inverseSqrt(") == 1


def test_product_selection_is_shader_compile_time_not_runtime() -> None:
    source = WGSL.read_text(encoding="utf-8")
    body = _function_body(source, "two_prod")
    assert "__DD_TWO_PROD_CALL__" in body
    assert "select(" not in body
    assert "if" not in body


def test_gpu_proof_source_has_required_scale_phases_and_refusal() -> None:
    shader = HARNESS.read_text(encoding="utf-8")
    rust = GPU.read_text(encoding="utf-8")
    assert "generated_pair" in shader and "adversarial_pair" in shader and "canary_pair" in shader
    assert "return generated_pair" not in _function_body(shader, "adversarial_pair")
    assert "MIN_GENERATED_COUNT: u64 = 100_000_000" in rust
    assert "ADVERSARIAL_COUNT: u64 = 1_000_000" in rust
    assert "FORGE3D_DD_FORCE_SELFTEST_FAIL" in rust
    assert "0x0001_2345" in (ROOT / "src" / "core" / "dd" / "generator.rs").read_text(encoding="utf-8")
    assert "8 => (0.0, -0.0)" in (ROOT / "src" / "core" / "dd" / "generator.rs").read_text(encoding="utf-8")
    assert '"precision_selftest_failed"' in rust and '"double_float"' in rust
    assert "RenderError::degraded_capability" in rust


def test_gpu_proof_uses_tracked_buffers_and_certificate_evidence() -> None:
    source = (ROOT / "src" / "core" / "dd" / "gpu_exec.rs").read_text(encoding="utf-8")
    assert "tracked_create_buffer(" in source
    assert "queue.write_buffer(&params_buffer" in source
    assert ".create_buffer(" not in source
    certificate = (ROOT / "src" / "core" / "certificate.rs").read_text(encoding="utf-8")
    assert "PrecisionEvidence" in certificate
    assert 'skip_serializing_if = "Option::is_none"' in certificate


def test_all_dupla_shaders_are_in_the_proof_coverage_ledger() -> None:
    ledger = (ROOT / "tests" / "shader_proofs_ledger.toml").read_text(encoding="utf-8")
    for path in (
        "src/shaders/includes/dd.wgsl",
        "src/shaders/dd_harness.wgsl",
        "src/shaders/dd_jitter.wgsl",
    ):
        assert f'path = "{path}"' in ledger


def test_precision_native_surface_is_registered_and_stubbed() -> None:
    functions = PY_FUNCTIONS.read_text(encoding="utf-8")
    registrar = PY_REGISTRAR.read_text(encoding="utf-8")
    package = (ROOT / "python" / "forge3d" / "__init__.py").read_text(encoding="utf-8")
    package_stub = (ROOT / "python" / "forge3d" / "__init__.pyi").read_text(encoding="utf-8")
    module_stub = (ROOT / "python" / "forge3d" / "precision.pyi").read_text(encoding="utf-8")
    for name in ("dd_selftest", "dd_harness", "dd_jitter_demo"):
        assert f"fn {name}" in functions
        assert f"py_functions::{name}" in registrar
        assert f'"{name}"' in package
        assert name in package_stub
        assert f"def {name}" in module_stub
    assert functions.count("map_err(PyErr::from)") == 3
    assert functions.count("allow_threads") == 3


def test_pinned_backend_ci_archives_full_dupla_proof() -> None:
    workflow = (ROOT / ".github" / "workflows" / "determinism-matrix.yml").read_text(
        encoding="utf-8"
    )
    required_render = workflow.split("  render:\n", 1)[1].split(
        "\n  # Optional Apple/Metal", 1
    )[0]
    optional_metal = workflow.split("  metal-diagnostic:\n", 1)[1].split(
        "\n  wasm-policy:\n", 1
    )[0]
    proof = (ROOT / "scripts" / "run_dupla_proof.py").read_text(encoding="utf-8")
    assert "Prove DUPLA precision" in workflow
    assert "WGPU_BACKENDS: ${{ matrix.backend }}" in workflow
    assert "run_dupla_proof.py" in workflow
    assert "dupla-proof-${{ matrix.leg }}.json" in workflow
    assert "--validate-artifacts hashes" in workflow
    assert "continue-on-error" not in required_render
    assert "continue-on-error: true" in optional_metal
    assert 'OPERATIONS = ("add", "mul", "div", "sqrt")' in proof
    assert "100_000_000" in proof
    assert "dd_jitter_demo(frames=1_000)" in proof


def test_gpu_unit_tests_skip_only_non_proof_adapters() -> None:
    source = (ROOT / "src" / "core" / "dd_tests.rs").read_text(encoding="utf-8")
    helper = _function_body(source, "require_physical_gpu_or_skip")
    assert "is_physical_proof_adapter" in helper
    assert "Ok(context) =>" in helper
    assert "is not proof hardware" in helper
    assert 'message.starts_with("No suitable GPU adapter found")' in helper
    assert "Err(error) => panic!" in helper
    assert "Err(_)" not in helper
    gpu = (ROOT / "src" / "core" / "gpu.rs").read_text(encoding="utf-8")
    classifier = _function_body(gpu, "is_physical_proof_adapter")
    assert "!software_fallback" in classifier
    assert "!is_virtualized_adapter_name" in classifier
    for name in (
        "forced_canary_failure_records_degradation_refuses_and_releases_resources",
        "forced_jitter_failure_releases_every_tracked_resource",
    ):
        body = _function_body(source, name)
        assert body.count("require_physical_gpu_or_skip") == 1
        assert re.search(
            r"if std::env::var_os\(CHILD\)\.is_none\(\) \{\s*"
            r"if !require_physical_gpu_or_skip\(",
            body,
        )
