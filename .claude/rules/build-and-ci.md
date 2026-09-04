---
paths: ["Cargo.toml", "pyproject.toml", ".cargo/**", ".github/workflows/**", "pytest.ini", "**/conftest.py"]
---

# Build, test, and CI facts

- Build with maturin/PyO3. Wheels use the `release-lto` profile.
- Use `cargo forge3d-clippy` for routine work and
  `cargo forge3d-clippy-acceptance` for an explicit full acceptance candidate;
  never use plain `cargo clippy`.
- The current portable feature inventory is:
  `default,async_readback,copc_laz,cog_streaming,gis-remote,geos-topology,weighted-oit,wsI_bigbuf,wsI_double_buf,enable-pbr,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-renderer-config,enable-staging-rings,shader-contract-asserts`.
  `.github/workflows/ci.yml` and `.cargo/config.toml` are authoritative and the
  honesty gate locks routing and duplicate inventories. `PR Core Success` runs
  the stable hosted pull-request contract. A manual `scope=full` dispatch runs
  the complete matrix, including the dedicated system-PROJ check.
- Every tracked `tests/test_*.py` remains assigned to a current core, compat,
  full, slow, or explicit lane, or to a dated entry in `tests/UNRUN.toml`.
  Routine pull requests run `scripts/ci_pytest_lane.py --profile fast`; the
  complete acceptance lanes run `--profile full`. A downstream feature does
  not inherit the complete suite merely because it consumes a CENSOR contract.
- Golden comparison logic, update safety, and probe outcome classification are
  routine invariants. Candidate-selected physical NVIDIA/Vulkan goldens and
  GPU lanes are acceptance evidence summarized by `Full Acceptance Summary`;
  a probe crash or pixel mismatch remains fatal whenever that lane is selected.
  Metal coverage is an opt-in support diagnostic and is never a prerequisite
  for moonshot, pull-request, or release acceptance.
- Production signing is required only by protected acceptance/release work.
  Routine internal and fork PRs remain explicitly untrusted and verify schema,
  canonicalization, and tamper rejection without the production secret.
- `determinism-matrix` is a reusable acceptance diagnostic until the
  documented fixed-function filtering and downstream compiler differences in
  `src/shaders/includes/determinism.wgsl` are eliminated. Its per-backend
  render jobs and zero-byte diff must stay loud; cross-adapter hash divergence
  is permitted as diagnostic evidence and must never be presented as a green
  determinism guarantee or used to replace the committed hash casually. The
  stable hosted pull-request gate is `PR Core Success`; scheduled or manually
  selected determinism and physical evidence is reported separately by
  `Full Acceptance Summary`.
