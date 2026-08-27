# forge3d behavior-preserving refactor ledger

This file is the durable execution ledger for the refactor begun on 2026-08-12.
It is evidence, not a declaration that a change is safe. Update it at every
reviewed checkpoint; do not rewrite prior checkpoint-log rows.

> **Current-state boundary (2026-08-27):** The original refactor narrative and
> checkpoint rows below are retained as historical evidence at their recorded
> identities. Statements in those sections such as "current candidate," "no PR
> yet," and unrun full/slow lanes describe those historical checkpoints, not the
> PR #170 continuation. The authoritative current local state is the final
> section, **PR #170 terminal-run audit and uncommitted continuation
> (2026-08-27)**.

## Mission

Exhaustively inspect the tracked product-relevant forge3d surfaces and apply the
smallest verified simplifications required by evidence-backed claims while preserving
every supported function, public API, shader contract, recipe, render path,
test, example, golden, certificate, build lane, and documented behavior.

### Behavior-preservation rule

Every audit item is a claim. Accept it only when repository evidence shows that
deleting it would leave this mission unmet or unproven. Lock behavior before an
edit where existing proof is insufficient, make the smallest complete change,
and advance its status only as far as exact evidence permits. Reduced line count
is never proof. Skipped, unavailable, mocked, wrong-SHA, wrong-adapter, or
wrong-backend evidence is `NOT_PROVEN`.

### Scope

- Every path returned by `git ls-files` is in audit coverage scope, classified
  before cleanup. C01-C07 retain their preliminary descriptions for provenance
  but are `REJECTED`: that audit was incorrectly routed and is not implementation
  authority. The correctly routed audit exposes only conditional N01-N03. The
  read-only W1-A/B/C coverage and W1-V review are complete. The historical W1-F
  checkpoint froze N01, N02, A01-A05, W1B-01, W1C-01, and W1C-02 as `PLANNED`;
  separately authorized I1-I9 implementation and focused proof have since
  advanced all ten claims to `LOCALLY_PROVEN` at the uncommitted local
  candidate. Necessary physical adjudication remediation also corrected the
  Metal timing/certificate interaction, the 80-byte host/WGSL `Sphere` stride,
  and a shadow-accumulation lost-update race; its exact installed-wheel Apple
  M4 Metal gate is locally proven below. N03 and C01-C07 remain evidence-backed
  rejections. The applicable current-source local final gates are locally
  proven below; broad full Python and slow acceptance remain `NOT_PROVEN`.
- Product code and seams include Rust, WGSL, Python, PyO3, stubs, build and
  packaging files, tests, examples, docs, scripts, tools, workflows, specs,
  fixtures, assets, goldens, certificates, and tracked root files.
- The implementation branch is `codex/refactor-forge3d-20260812` in isolated
  worktree `/private/tmp/forge3d-refactor-20260812`.

### Non-goals

- No redesign, feature work, migration, dependency upgrade, performance rewrite,
  correctness/security project, formatting churn, golden or certificate refresh,
  generated-source edit, or public API removal.
- No consolidation of AETHER's independent acceptance oracle into production.
- No cross-subsystem generic WGPU abstraction, blind `cargo fix`, Clippy
  quarantine changes, or cleanup claim added after it was already visible.
- Large assets, fixtures, corpora, binaries, ignored output, historical material,
  and generated files are not slop merely because of size or shape.

### Authority

The controlling prompt is the live content of
`codex/refactor-runbook-refresh-20260811:docs/refactor-forge3d-sol-ultra-runbook.md`;
the user's literal `docs/prompts/refactor-forge3d-sol-ultra.md` path is absent.
The live root `AGENTS.md` and its MSW kernel, `.claude/rules/build-and-ci.md`,
`.claude/rules/rust-core.md`, `CONTRIBUTING.md`, `Cargo.toml`, `pyproject.toml`,
`.cargo/config.toml`, and `.github/workflows/*.yml` govern this checkout. The
invocation authorizes repository inspection, in-scope edits, local proof,
review, commits, a push, and a new PR only after applicable local tests are
green. It does not authorize golden/certificate rotation, dependency changes,
production signing, or claiming unavailable physical evidence.

This post-I9 checkpoint is narrower: only this ledger correction and review are
authorized. It does not authorize source or manifest edits, commits, pushes,
network access, or a PR. The earlier W1-F freeze was documentation-only; the
later I1-I9, physical-gate remediations, integration runs, and reviewed
static-contract fixes were separately authorized and cannot be inferred from
that frozen plan.

### Completion conditions

- W1 manifests derived from `git ls-files` have an exact union equal to the full
  tracked set, empty pairwise intersections, and evidence in the coverage matrix
  for every tracked path; classification may reject non-product files from
  cleanup but may not silently omit them from coverage.
- Every finding has an evidence-backed disposition; accepted implementation
  claims start as `PLANNED` and advance only as far as exact evidence permits,
  rejected claims are `REJECTED`, and unavailable execution/remote/physical
  evidence lanes remain `NOT_PROVEN`.
- Every necessary transformation is proven at the exact final head by all
  applicable local and remote gates; unavailable physical evidence remains
  `NOT_PROVEN`.
- Public behavior and affected contract families remain preserved, the final
  ledger matches the exact PR head, all mandated reviews are approved, required
  checks are green, the PR is mergeable, and the primary checkout is unchanged.

## Baseline identity and environment

| Fact | Exact observation | Status |
|---|---|---|
| Intended base | `origin/main` resolved locally to `f5db54f95d202681f95dad649162d18efdae8987` | `VALIDATED` |
| Isolated branch and worktree | `codex/refactor-forge3d-20260812`; `/private/tmp/forge3d-refactor-20260812` | `VALIDATED` |
| Isolated initial head/state | exact base above; no porcelain entries after worktree creation | `VALIDATED` |
| Primary checkout identity | `/Users/mpopovic3/forge3d`; branch `codex/m06-fgv-audit-remediation`; head `b5cc2048dae6eeca972b0b33b11f54ffbcdb60fe` | `VALIDATED` |
| Primary tracked changes | `AGENTS.md`; `docs/fable5-moonshots-remediation-plan.md` | `VALIDATED` |
| Primary untracked paths | `.agents/`; `skills-lock.json` | `VALIDATED` |
| Platform | macOS 26.5.2, Darwin 25.5.0, arm64 | `VALIDATED` |
| Git and Git LFS | Git 2.50.1 (Apple Git-155); git-lfs 3.7.1 | `VALIDATED` |
| Rust | `stable-aarch64-apple-darwin`; rustc 1.90.0; cargo 1.90.0 | `VALIDATED` |
| Python | CPython 3.13.5 at `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`; pytest found at the same framework prefix | `VALIDATED` |
| Other tools | uv 0.10.2; CMake 4.0.3; Node 24.3.0; npm 11.4.2 | `VALIDATED` |
| Required build tools | executable shims are absent from `PATH`; `python3 -m maturin --version` reports `maturin 1.14.1` and `python3 -m ninja --version` reports `1.11.1.git.kitware.jobserver-1` (Ninja 1.11.1) | `VALIDATED` |
| Backend/adapter | installed final-candidate wheel exercised an Apple M4 through Metal | `LOCALLY_PROVEN` |
| Remote/physical evidence | exact installed-wheel Apple M4 Metal adjudication gate passed; no exact-branch remote, hosted matrix, NVIDIA/Vulkan, signing, or cross-platform acceptance run exists yet | `LOCALLY_PROVEN` for the named Metal gate; otherwise `NOT_PROVEN` |

### Stale-snapshot warning

The runbook's 2026-08-11 metrics, CI links, protection snapshot, and code
navigation facts are bound to `de97cedd1da91ebcb234aa2edd729ea4778a8222`.
They are non-authoritative for this base. All figures in this ledger were
remeasured at `f5db54f95d202681f95dad649162d18efdae8987`; all test and acceptance
results must be measured again at each later candidate head.

## Architecture and contract map

| Contract family | Code/spec authority inspected for mapping | Behavior that must remain true | Baseline map status |
|---|---|---|---|
| CENSOR | `docs/censor-validation-policy.md`; `src/core/{capabilities,degradation,resource_tracker,certificate,shader_registry}.rs`; CI certificate, probe, allocation, fast/full-lane routes | capability/provenance truth; explicit degradation; tracked allocation and enforced budget; canonical certificate/tamper semantics; shader-use reporting; honest ABSENT/CRASH | `VALIDATED` |
| SUTURA | `python/forge3d/map_scene.py`, `_map_scene_render.py`, `_map_scene_validation.py`, `recipe_manifest.py`, bundle modules, recipe/bundle/SUTURA tests | no placeholder render; compiled-plan ownership; canonical serialization and bundle round trip; compile-time label culling; structured diagnostic blocks | `VALIDATED` |
| Rust-PyO3-stub parity | `src/lib.rs`, `src/py_module/**`, `src/py_functions/**`, `src/scene/py_api/**`, `python/forge3d/*.py`, `*.pyi`, `pyproject.toml`, API tests | native signature, registration, module/class ownership, exports, typing, feature gates, and installed-extension behavior remain aligned | `VALIDATED` |
| Rust-WGSL/GPU | Rust renderer/pipeline/layout modules, `src/shaders/**/*.wgsl`, `build.rs`, shader contract tests | struct alignment/stride, binding and pipeline layouts, stage limits, entry points, lifetimes, and renderer-owned assembly remain unchanged | `VALIDATED` |
| LIMES | `src/vector/coverage/**`, `src/py_functions/vector/coverage*.rs`, coverage shaders and `tests/test_vector_coverage.py` | opt-in/default-preserving coverage and compiled-scene cache boundaries | `VALIDATED` |
| VT, visibility, terrain, TESSELLA | `src/terrain/renderer/{virtual_texture,visibility_buffer}.rs`, clipmap/culling/feedback modules, TESSELLA scripts/tests, CI scope | one VT store; residency/streaming/picking seams; fail-closed SHA/adapter-bound physical proof | `VALIDATED` |
| SIDERA astronomy | `src/astro/**`, `python/forge3d/astro*`, astro tests/assets, deterministic night workflow | public 2000-2050 numerical window and backend-specific deterministic night evidence | `VALIDATED` |
| AETHER atmosphere | `src/core/atmosphere/**`, atmosphere shaders/wrappers/tests, `src/path_tracing/hybrid_compute/aether_reference.rs` | production LUT/provenance path; acceptance oracle stays independent; no deduplication across that boundary | `VALIDATED` |
| SUBSTRATIA | terrain golden/certificate paths, `_substratia_evidence.py`, report script/tests, NVIDIA visual lane | exact-SHA, adapter-bound, golden-bound, zero-skip physical evidence; portable/mock evidence never substitutes | `VALIDATED` |
| Determinism/certificates | `src/core/certificate.rs`, determinism Python/scripts/workflow, committed hash and certificate fixtures | canonical inputs/outputs and provenance; signing/tamper contracts; exact identity on acceptance evidence | `VALIDATED` |
| Text/Unicode | `src/labels/unicode/**`, Unicode corpora/provenance, text modules/stubs/tests | generated-source ownership, Unicode version/provenance, shaping and behavior | `VALIDATED` |
| GIS/units | `src/gis/**`, `src/geo/units.rs`, Python GIS wrappers/stubs, GIS tests/docs | coordinate, CRS, height, epoch, unit, raster/vector and narrowing semantics | `VALIDATED` |
| Examples/recipes/docs | `examples/**`, recipe manifests/tests/goldens, docs/tutorials, packaging and CI path filters | real runnable paths, parser interfaces, manifest/schema/canonical behavior, documented API alignment | `VALIDATED` |

Mapping status means the named authority and preservation boundary are known;
it is not runtime acceptance of any family.

## Baseline validation matrix

| ID | Exact command or evidence | Identity | Result | Status |
|---|---|---|---|---|
| B01 | `git rev-parse HEAD` | isolated worktree before ledger edit | `f5db54f95d202681f95dad649162d18efdae8987` | `VALIDATED` |
| B02 | `git status --porcelain=v2 --branch` | isolated worktree before ledger edit | branch exact; no entries | `VALIDATED` |
| B03 | `git -C /Users/mpopovic3/forge3d status --porcelain=v2 --branch` | primary checkout before T0 | exact dirty snapshot recorded above | `VALIDATED` |
| B04 | `git ls-files` classifications and extension/directory counts | exact base | metrics below | `VALIDATED` |
| B05 | `cargo fmt --check` | exact base | not run before source work | `NOT_PROVEN` |
| B06 | `cargo forge3d-clippy` | exact base | not run before source work | `NOT_PROVEN` |
| B07 | `python3 -m maturin develop` | exact base | module entry point verified available; build not run before source work | `PLANNED` |
| B08 | `FORGE3D_NO_BOOTSTRAP=1 python scripts/ci_pytest_lane.py --profile fast -v --tb=short` | exact base | not run; installed exact-base extension unavailable | `NOT_PROVEN` |
| B09 | affected focused Rust/Python/shader/example characterization commands | exact base | assigned to claim executors before edits | `PLANNED` |
| B10 | full/slow/platform/physical/signing acceptance | exact base | no run | `NOT_PROVEN` |

## Final validation matrix

This matrix is populated only with final-head evidence. Its commands are copied
from the live CI authority; a claim executor may add narrower proof before these
integration gates.

Current local candidate identity is commit
`f5db54f95d202681f95dad649162d18efdae8987` plus 51 staged paths and zero
unstaged paths immediately before this self-describing ledger edit. The staged
`git diff --cached --binary` SHA256 was
`9ae70342541a6257ea2523f69e501aae847938572bac613b92b585a7a0e8574a`.
The ledger-only edit is
intentionally left unstaged for the orchestrator to review. This is not a final
commit SHA or remote identity.

| ID | Exact command/evidence | Required identity | Result | Status |
|---|---|---|---|---|
| F01 | `cargo fmt --check` | current base-plus-uncommitted candidate | passed in the final integration run | `LOCALLY_PROVEN` |
| F02 | `cargo forge3d-clippy` | current source candidate; differs from the release-LTO wheel only by this ledger's documentation bytes | passed | `LOCALLY_PROVEN` |
| F03 | `python3 -m maturin develop` / `maturin build --interpreter python3 --locked` | current source candidate before this ledger-only edit | fresh release-LTO wheel SHA256 `b0b1ef2c...`; installed native SHA256 `2365c2ff...`; install/import and affected evidence below passed | `LOCALLY_PROVEN` |
| F04 | `FORGE3D_NO_BOOTSTRAP=1 python scripts/ci_pytest_lane.py --profile fast -v --tb=short` | current source candidate before this ledger-only edit | 642 passed, 28 skipped by policy, 0 failed | `LOCALLY_PROVEN` |
| F05 | focused proof for every Wave 1 accepted claim, recorded per finding | current base-plus-uncommitted I1-I9 candidate | all ten accepted findings locally proven and per-change reviewer-approved; exact evidence below | `LOCALLY_PROVEN` |
| F06 | exact `cargo check`, Rust test, doctest, and `cargo forge3d-clippy-acceptance` commands from `.github/workflows/ci.yml` | current source candidate before this ledger-only edit | current `cargo forge3d-clippy-acceptance` passed; affected complete-file suite passed 29 with 19 policy GPU skips and 0 failures | `LOCALLY_PROVEN` |
| F07 | independent per-change and whole-diff review; ponytail review; Standards and Spec review | clean commit `a3260432...` plus the two-path TERMINUS/ledger continuation | the exact current two-path candidate received combined `gpt-5.6-sol:xhigh` `APPROVED` review with zero contract defects, Ponytail `Lean already. Ship.` with net `-0` necessary lines, and Standards and Spec `APPROVED` review with zero findings; the source remediation also received focused independent `APPROVED` review | `APPROVED / CURRENT_REVIEW_CLOSED`; execution and publication remain separate and `NOT_PROVEN` where required |
| F08 | required remote `PR Core Success` plus PR-head and mergeability readback | exact PR head | no PR yet | `NOT_PROVEN` |
| F09 | full/slow/wheel/platform/NVIDIA/Vulkan/Metal/signing acceptance | exact SHA and authoritative environment | fresh release-LTO wheel's exact Apple M4 Metal adjudication gate passed 1/1, 0 skipped in 352.45 s, with zero-skip JUnit; broad full Python ran but failed as recorded below; the slow and other listed acceptance lanes are not green or remain unavailable | `LOCALLY_PROVEN` for this exact Metal gate; otherwise `NOT_PROVEN` |

## Metrics

Methodology: counts use `git ls-files` at exact base
`f5db54f95d202681f95dad649162d18efdae8987`; line counts feed those tracked path
lists to `wc -l`. Python combines `.py` and `.pyi`. Directory counts use tracked
paths below each named directory. These are navigation and coverage evidence,
not quality gates, targets, priorities, or permission to delete. Extension and
line counts do not measure semantic duplication; binary/generated content and
language classification can create false impressions.

| Metric | Exact-base measurement | Status |
|---|---:|---|
| All tracked paths | 2,793 | `VALIDATED` |
| Rust | 1,189 files / 301,831 lines | `VALIDATED` |
| Python plus stubs | 559 files / 205,416 lines | `VALIDATED` |
| WGSL | 143 files / 32,486 lines | `VALIDATED` |
| `src/` | 1,338 paths | `VALIDATED` |
| `tests/` | 988 paths | `VALIDATED` |
| `python/` | 140 paths | `VALIDATED` |
| `docs/` | 139 paths | `VALIDATED` |
| `examples/` | 46 paths | `VALIDATED` |
| `scripts/`, `tools/`, `bench*/` | 37 / 4 / 2 paths | `VALIDATED` |
| `assets/` | 50 paths | `VALIDATED` |
| superseded audit estimate | up to 450 lines and zero dependencies removable; incorrectly routed, rejected, and not a gate | `REJECTED` |
| accepted implementation set | ten evidence-backed claims with focused local proof; count is not a size gate, target, or remote/physical execution claim | `LOCALLY_PROVEN` |

## W1-V exact manifest and path-classification proof

Identity is `f5db54f95d202681f95dad649162d18efdae8987`. The
manifest serialization is Git-index order, UTF-8 path bytes, and one `awk
print` newline per path, piped directly to `shasum -a 256`. Predicate A is
`Cargo.toml|Cargo.lock|build.rs|CMakeLists.txt|cmake/**|.cargo/**|src/**`
excluding `src/lib.rs`, `src/(py_functions|py_module|py_types)/**`, and
`src/scene/py_api/**`. Predicate B is exactly those excluded Rust/PyO3 roots
plus `src/lib.rs|conftest.py|pytest.ini|python/**|tests/**|examples/**|bench/**|
benches/**|data/**`. C is `!A && !B`.

| Manifest | Count | SHA256 |
|---|---:|---|
| A | 1,252 | `3abf0c315b4bb7416935a54253ad9920ed588b08307be8439e215ab1f4697a33` |
| B | 1,272 | `51efb1a88ad236ce8a0214d189408ccb041d83af061d83f2e9fcb86b6245bcdf` |
| C | 269 | `e685bff1ab3790153131bc7dc3c57baa9da3fe955378de2ed67552595b457781` |

The union has 2,793 paths and is byte-identical to `git ls-files`; A∩B,
A∩C, and B∩C each have zero paths. Ignored outputs are not tracked and therefore
are recorded as outside A/B/C, not silently classified.

The lossless TSV below is the path-bound W1-V inspection ledger:
`path<TAB>wave<TAB>classification<TAB>evidence`. Its exact UTF-8 bytes,
including the final newline, have SHA256
`7b731e1b83f79dc1f147ba1cb0611195a6df2e575e326f4c14108fd4d6af200f`.
It has exactly 2,793 rows, 2,793 unique first fields, and its first field is
byte-for-byte `git ls-files`. Counts are: ordinary-source 2,502; binary 75;
golden 66; asset 37; corpus 25; certificate 24; historical 20; policy-or-CI 18;
build-or-package-config 14; fixture 11; generated 1. Class assignment used
provenance/manifests/consumers: certificate and golden consumer trees first;
`docs/superpowers/**` historical; the exact generated Unicode source;
test corpus and fixture consumer trees; asset/package/runtime/documentation
consumer trees with binary format only inside those provenance-bound trees;
live policy/spec/workflow authority; exact build/package configuration; then
ordinary code/test/example/doc/script/tool source. Thus extensions never
classify a path without a provenance/consumer root. The TSV is authoritative
for every exact-path decision and resolves all rule precedence.

```tsv
.cargo/config.toml	A	build-or-package-config	build/package/tool configuration consumer
.claude/rules/build-and-ci.md	C	policy-or-CI	live policy/spec/workflow authority
.claude/rules/rust-core.md	C	policy-or-CI	live policy/spec/workflow authority
.gitattributes	C	build-or-package-config	build/package/tool configuration consumer
.github/scripts/cartographer_prime_evidence.py	C	policy-or-CI	live policy/spec/workflow authority
.github/scripts/verify_cartographer_prime_evidence.py	C	policy-or-CI	live policy/spec/workflow authority
.github/workflows/build-wheel.yml	C	policy-or-CI	live policy/spec/workflow authority
.github/workflows/cartographer-prime.yml	C	policy-or-CI	live policy/spec/workflow authority
.github/workflows/certificate-refresh.yml	C	policy-or-CI	live policy/spec/workflow authority
.github/workflows/ci.yml	C	policy-or-CI	live policy/spec/workflow authority
.github/workflows/determinism-matrix.yml	C	policy-or-CI	live policy/spec/workflow authority
.github/workflows/docs.yml	C	policy-or-CI	live policy/spec/workflow authority
.github/workflows/public-funnel-monitor.yml	C	policy-or-CI	live policy/spec/workflow authority
.github/workflows/publish.yml	C	policy-or-CI	live policy/spec/workflow authority
.github/workflows/test-python-wheel.yml	C	policy-or-CI	live policy/spec/workflow authority
.gitignore	C	build-or-package-config	build/package/tool configuration consumer
.pre-commit-config.yaml	C	build-or-package-config	build/package/tool configuration consumer
AGENTS.md	C	policy-or-CI	live policy/spec/workflow authority
CHANGELOG.md	C	ordinary-source	code/test/example/doc/script or tool consumer
CMakeLists.txt	A	build-or-package-config	build/package/tool configuration consumer
CONTRIBUTING.md	C	policy-or-CI	live policy/spec/workflow authority
Cargo.lock	A	build-or-package-config	build/package/tool configuration consumer
Cargo.toml	A	build-or-package-config	build/package/tool configuration consumer
LICENSE	C	ordinary-source	code/test/example/doc/script or tool consumer
LICENSE-APACHE	C	ordinary-source	code/test/example/doc/script or tool consumer
MANIFEST.in	C	build-or-package-config	build/package/tool configuration consumer
README.md	C	ordinary-source	code/test/example/doc/script or tool consumer
SECURITY.md	C	policy-or-CI	live policy/spec/workflow authority
assets/astro/MANIFEST.toml	C	asset	asset/package/documentation consumer or manifest
assets/astro/THIRD_PARTY_NOTICES.md	C	asset	asset/package/documentation consumer or manifest
assets/astro/bright_stars.bin	C	binary	asset/package/runtime consumer plus binary format
assets/astro/delta_t_fit.dat	C	binary	asset/package/runtime consumer plus binary format
assets/astro/leap_seconds.dat	C	binary	asset/package/runtime consumer plus binary format
assets/astro/moon_albedo.bin	C	binary	asset/package/runtime consumer plus binary format
assets/astro/moon_terms.bin	C	binary	asset/package/runtime consumer plus binary format
assets/astro/vsop87d.bin	C	binary	asset/package/runtime consumer plus binary format
assets/colormaps/magma_256x1.png	C	binary	asset/package/runtime consumer plus binary format
assets/colormaps/terrain_256x1.png	C	binary	asset/package/runtime consumer plus binary format
assets/colormaps/viridis_256x1.png	C	binary	asset/package/runtime consumer plus binary format
assets/fonts/NotoSans-OFL.txt	C	asset	asset/package/documentation consumer or manifest
assets/fonts/NotoSans-subset.ttf	C	binary	asset/package/runtime consumer plus binary format
assets/fonts/NotoSansArabic-OFL.txt	C	asset	asset/package/documentation consumer or manifest
assets/fonts/NotoSansArabic-subset.ttf	C	binary	asset/package/runtime consumer plus binary format
assets/fonts/NotoSansDevanagari-OFL.txt	C	asset	asset/package/documentation consumer or manifest
assets/fonts/NotoSansDevanagari-subset.ttf	C	binary	asset/package/runtime consumer plus binary format
assets/fonts/NotoSansHebrew-OFL.txt	C	asset	asset/package/documentation consumer or manifest
assets/fonts/NotoSansHebrew-subset.ttf	C	binary	asset/package/runtime consumer plus binary format
assets/fonts/NotoSansLatin-subset.ttf	C	binary	asset/package/runtime consumer plus binary format
assets/fonts/NotoSansSC-OFL.txt	C	asset	asset/package/documentation consumer or manifest
assets/fonts/NotoSansSC-subset.ttf	C	binary	asset/package/runtime consumer plus binary format
assets/fonts/PROVENANCE.md	C	asset	asset/package/documentation consumer or manifest
assets/fonts/default_atlas.json	C	asset	asset/package/documentation consumer or manifest
assets/frames.mp4	C	binary	asset/package/runtime consumer plus binary format
assets/fuji_labels.png	C	binary	asset/package/runtime consumer plus binary format
assets/geoid/README.md	C	asset	asset/package/documentation consumer or manifest
assets/geoid/egm96_n120.bin	C	binary	asset/package/runtime consumer plus binary format
assets/geoid/mars_areoid_n179.bin	C	binary	asset/package/runtime consumer plus binary format
assets/geoid/mars_areoid_n179.manifest.json	C	asset	asset/package/documentation consumer or manifest
assets/geojson/10-270-592.city.json	C	asset	asset/package/documentation consumer or manifest
assets/geojson/mount_fuji_buildings.geojson	C	asset	asset/package/documentation consumer or manifest
assets/geojson/sample_buildings.city.json	C	asset	asset/package/documentation consumer or manifest
assets/gpkg/Mount_Fuji_places.gpkg	C	binary	asset/package/runtime consumer plus binary format
assets/gpkg/luxembourg_rail.gpkg	C	binary	asset/package/runtime consumer plus binary format
assets/highres.png	C	binary	asset/package/runtime consumer plus binary format
assets/lidar/MtStHelens.laz	C	binary	asset/package/runtime consumer plus binary format
assets/objects/bunny.obj	C	asset	asset/package/documentation consumer or manifest
assets/objects/cornell_box.obj	C	asset	asset/package/documentation consumer or manifest
assets/objects/cornell_sphere.obj	C	asset	asset/package/documentation consumer or manifest
assets/swiss-legend.png	C	binary	asset/package/runtime consumer plus binary format
assets/tif/Bryce_Canyon.tif	C	binary	asset/package/runtime consumer plus binary format
assets/tif/Gore_Range_Albers_1m.tif	C	binary	asset/package/runtime consumer plus binary format
assets/tif/Mount_Fuji_30m.tif	C	binary	asset/package/runtime consumer plus binary format
assets/tif/dem_rainier.tif	C	binary	asset/package/runtime consumer plus binary format
assets/tif/luxembourg_dem.tif	C	binary	asset/package/runtime consumer plus binary format
assets/tif/moon_south_pole_lola.manifest.json	C	asset	asset/package/documentation consumer or manifest
assets/tif/moon_south_pole_lola.tif	C	binary	asset/package/runtime consumer plus binary format
assets/tif/switzerland_dem.tif	C	binary	asset/package/runtime consumer plus binary format
assets/tif/switzerland_land_cover.tif	C	binary	asset/package/runtime consumer plus binary format
bench/upload_policies/policies.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
benches/f3dz_bench.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
build.rs	A	build-or-package-config	build/package/tool configuration consumer
cmake/ForgeConfig.cmake	A	build-or-package-config	build/package/tool configuration consumer
cmake/README.md	A	build-or-package-config	build/package/tool configuration consumer
conftest.py	B	build-or-package-config	build/package/tool configuration consumer
data/pol_pd_2020_1km_UNadj.tif	B	binary	asset/package/runtime consumer plus binary format
docs/3d-map-rendering-quality-blender-outmatch-plan.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/Makefile	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/anamnesis.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/api/api_reference.rst	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/api/precision.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/assets/highres.png	C	binary	asset/package/runtime consumer plus binary format
docs/assets/logo/forge3d_dark.svg	C	asset	asset/package/documentation consumer or manifest
docs/assets/logo/forge3d_light.svg	C	asset	asset/package/documentation consumer or manifest
docs/assets/readme/california-smoke.webp	C	binary	asset/package/runtime consumer plus binary format
docs/assets/readme/egypt.webp	C	binary	asset/package/runtime consumer plus binary format
docs/assets/readme/france.webp	C	binary	asset/package/runtime consumer plus binary format
docs/assets/readme/germany.webp	C	binary	asset/package/runtime consumer plus binary format
docs/assets/readme/iberia.webp	C	binary	asset/package/runtime consumer plus binary format
docs/assets/readme/lyon.webp	C	binary	asset/package/runtime consumer plus binary format
docs/assets/readme/shasta-hero.webp	C	binary	asset/package/runtime consumer plus binary format
docs/assets/readme/turkiye.webp	C	binary	asset/package/runtime consumer plus binary format
docs/assets/thumbnails/f16_instancing.svg	C	asset	asset/package/documentation consumer or manifest
docs/assets/thumbnails/f18_gltf.svg	C	asset	asset/package/documentation consumer or manifest
docs/assets/thumbnails/f2_city_demo.svg	C	asset	asset/package/documentation consumer or manifest
docs/assets/thumbnails/f3_thick_polyline.svg	C	asset	asset/package/documentation consumer or manifest
docs/audits/fable5-moonshots/14-censor-implementation-audit.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/g-001-p1-1-recipe-manifest-implementation-plan.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/g-002-later-domain-remote-helpers-implementation-plan.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/g-002b-support-matrix.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/g-002c-c2-reproject-vector-implementation-plan.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/g-002c-c3-implementation-plan.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/g-002c-c4-implementation-plan.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/g-002c-c5-implementation-plan.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/g-002c-c6-implementation-plan.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/g-002c-implementation-plan.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/gis-contract-evidence.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/gis-operation-api-crosswalk.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/mapscene-enrichment-capability-ranking.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/mensura-m06-full-geospatial-viewer-spec.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/mensura-m06-world-coord-anchoring.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/recipe-family-manifest-schema.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/carto-engine/rust-gis-implementation-plan.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/censor-validation-policy.md	C	policy-or-CI	live policy/spec/workflow authority
docs/ci-validation.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/conf.py	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/examples/3d-map-project-ideas.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/examples/index.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/examples/rotterdam-solar-potential-shadow-study.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/fable5-moonshots-remediation-plan.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/formats/f3dz.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/gallery/01-mount-rainier.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/gallery/02-mount-fuji-labels.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/gallery/03-swiss-landcover.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/gallery/04-luxembourg-rail-network.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/gallery/05-3d-buildings.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/gallery/06-point-cloud.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/gallery/07-camera-flyover.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/gallery/08-vector-export.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/gallery/09-shadow-comparison.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/gallery/10-map-plate.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/gallery/images/01-mount-rainier.png	C	binary	asset/package/runtime consumer plus binary format
docs/gallery/images/02-mount-fuji-labels.png	C	binary	asset/package/runtime consumer plus binary format
docs/gallery/images/03-swiss-landcover.png	C	binary	asset/package/runtime consumer plus binary format
docs/gallery/images/04-luxembourg-rail-network.png	C	binary	asset/package/runtime consumer plus binary format
docs/gallery/images/05-3d-buildings.png	C	binary	asset/package/runtime consumer plus binary format
docs/gallery/images/06-point-cloud.png	C	binary	asset/package/runtime consumer plus binary format
docs/gallery/images/07-camera-flyover.png	C	binary	asset/package/runtime consumer plus binary format
docs/gallery/images/08-vector-export.png	C	binary	asset/package/runtime consumer plus binary format
docs/gallery/images/09-shadow-comparison.png	C	binary	asset/package/runtime consumer plus binary format
docs/gallery/images/10-map-plate.png	C	binary	asset/package/runtime consumer plus binary format
docs/gallery/index.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/building_support_matrix.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/color-management.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/competitive_positioning.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/data_and_scene_workflows.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/diagnostics_reference.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/feature_map.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/label_plan_guide.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/label_support_matrix.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/large_scene_support.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/offline_3d_map_rendering.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/output_and_integration.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/rendering_and_analysis.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/style_support_matrix.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/tiles3d_support_matrix.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/guides/virtual_texturing_support_matrix.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/index.rst	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/prompts/fable5-moonshots/14-censor-audit.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/prompts/fable5-moonshots/14-censor-remediation.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/prompts/fable5-moonshots/14-censor.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/prompts/fable5-moonshots/17-anamnesis.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/prompts/fable5-moonshots/README-round-2.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/prompts/fable5-moonshots/README-round-3.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/start/architecture.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/start/quickstart.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/superpowers/plans/2026-04-25-khumbu-sentinel-timelapse-implementation.md	C	historical	superseded design/plan archive
docs/superpowers/plans/2026-05-05-khumbu-smooth-orbit-light-render-implementation.md	C	historical	superseded design/plan archive
docs/superpowers/plans/2026-06-08-humanity-globe-video-implementation.md	C	historical	superseded design/plan archive
docs/superpowers/plans/2026-06-11-cigar-smoke-aesthetic-tuning.md	C	historical	superseded design/plan archive
docs/superpowers/plans/2026-06-11-cigar-smoke-refinement-pass2.md	C	historical	superseded design/plan archive
docs/superpowers/plans/2026-06-12-cigar-smoke-fire-only-improvement.md	C	historical	superseded design/plan archive
docs/superpowers/plans/2026-06-15-reference-film-first30-smoke-gap-plan.md	C	historical	superseded design/plan archive
docs/superpowers/plans/2026-07-12-littera-implementation.md	C	historical	superseded design/plan archive
docs/superpowers/plans/2026-07-16-open-pr-closure-audit.md	C	historical	superseded design/plan archive
docs/superpowers/plans/3d-map-rendering-gaps-assessment.md	C	historical	superseded design/plan archive
docs/superpowers/specs/2026-04-25-khumbu-sentinel-timelapse-design.md	C	historical	superseded design/plan archive
docs/superpowers/specs/2026-05-05-khumbu-smooth-orbit-light-render-design.md	C	historical	superseded design/plan archive
docs/superpowers/specs/2026-06-08-humanity-globe-video-design.md	C	historical	superseded design/plan archive
docs/superpowers/specs/2026-06-11-cigar-smoke-aesthetic-tuning-design.md	C	historical	superseded design/plan archive
docs/superpowers/specs/2026-06-11-cigar-smoke-refinement-pass2-design.md	C	historical	superseded design/plan archive
docs/superpowers/specs/2026-06-12-cigar-smoke-source-wisp-requirements.md	C	historical	superseded design/plan archive
docs/superpowers/specs/2026-07-10-censor-closure-design.md	C	historical	superseded design/plan archive
docs/superpowers/specs/2026-07-12-littera-design.md	C	historical	superseded design/plan archive
docs/superpowers/specs/2026-07-13-general-hdr-terrain-mood-design.md	C	historical	superseded design/plan archive
docs/superpowers/specs/2026-07-18-dupla-design.md	C	historical	superseded design/plan archive
docs/terrain/offline-render-quality.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/tutorials/gis-track/01-visualize-your-first-dem.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/tutorials/gis-track/02-drape-overlays-on-terrain.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/tutorials/gis-track/03-build-a-map-plate.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/tutorials/gis-track/04-3d-buildings.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/tutorials/gis-track/index.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/tutorials/images/gis-01-first-dem.png	C	binary	asset/package/runtime consumer plus binary format
docs/tutorials/images/gis-01-first-dem.svg	C	asset	asset/package/documentation consumer or manifest
docs/tutorials/images/gis-02-overlays.png	C	binary	asset/package/runtime consumer plus binary format
docs/tutorials/images/gis-02-overlays.svg	C	asset	asset/package/documentation consumer or manifest
docs/tutorials/images/gis-03-map-plate.png	C	binary	asset/package/runtime consumer plus binary format
docs/tutorials/images/gis-03-map-plate.svg	C	asset	asset/package/documentation consumer or manifest
docs/tutorials/images/gis-04-buildings.png	C	binary	asset/package/runtime consumer plus binary format
docs/tutorials/images/gis-04-buildings.svg	C	asset	asset/package/documentation consumer or manifest
docs/tutorials/images/python-01-first-terrain.png	C	binary	asset/package/runtime consumer plus binary format
docs/tutorials/images/python-01-first-terrain.svg	C	asset	asset/package/documentation consumer or manifest
docs/tutorials/images/python-02-camera-lighting.png	C	binary	asset/package/runtime consumer plus binary format
docs/tutorials/images/python-02-camera-lighting.svg	C	asset	asset/package/documentation consumer or manifest
docs/tutorials/images/python-03-point-clouds.png	C	binary	asset/package/runtime consumer plus binary format
docs/tutorials/images/python-03-point-clouds.svg	C	asset	asset/package/documentation consumer or manifest
docs/tutorials/images/python-04-scene-bundles.png	C	binary	asset/package/runtime consumer plus binary format
docs/tutorials/images/python-04-scene-bundles.svg	C	asset	asset/package/documentation consumer or manifest
docs/tutorials/index.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/tutorials/python-track/01-your-first-3d-terrain.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/tutorials/python-track/02-camera-lighting-and-animation.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/tutorials/python-track/03-point-clouds.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/tutorials/python-track/04-scene-bundles.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/tutorials/python-track/index.md	C	ordinary-source	code/test/example/doc/script or tool consumer
docs/viewer/index.md	C	ordinary-source	code/test/example/doc/script or tool consumer
examples/_import_shim.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/bosnia_terrain_landcover_viewer.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/bryce_canyon_storm_timelapse.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/california_cigar_smoke_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/california_fire_smoke_effect.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/california_wildfire_smoke_video.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/camera_animation_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/colorado_rem_forge3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/forest_cover_copernicus/italy_forest_cover_3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/fuji_labels_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/helsinki_transit_daycycle.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/humanity_globe_video.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/khumbu_icefall_sentinel_timelapse.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/label_api_truth_basic.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/luxembourg_rail_overlay.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/mapscene_buildings_labels.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/mapscene_bundled_datasets_showcase.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/mapscene_offline_quality.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/mapscene_p1_assets_bundle_showcase.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/mapscene_terrain_raster.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/mapscene_vector_labels.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/moon_south_pole.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/notebooks/map_plate.ipynb	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/notebooks/quickstart.ipynb	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/notebooks/terrain_explorer.ipynb	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/osm_city_daycycle.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/osm_city_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/platte_rem_forge3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/pointcloud_viewer_interactive.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/population_ghsl/iberia_builtup_cover_3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/population_ghsl/romania_builtup_cover_3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/population_spike_worldpop/france_population_spikes_height_shade.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/population_spike_worldpop/germany_population_spikes_height_shade.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/population_spike_worldpop/poland_population_contour_3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/population_spike_worldpop/poland_population_spikes.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/population_spike_worldpop/poland_population_spikes_height_shade.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/presets/baseline_no_vector_overlays.json	B	fixture	test/example fixture consumer
examples/presets/rainier_showcase.json	B	fixture	test/example fixture consumer
examples/rotterdam_solar_potential_shadow_study.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/sample_style.json	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/swiss_terrain_landcover_viewer.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/terrain_camera_rigs_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/terrain_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/terrain_viewer_interactive.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/turkiye_river_basins_3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
examples/uk_ireland_lighthouse_map.py	B	ordinary-source	code/test/example/doc/script or tool consumer
pyproject.toml	C	build-or-package-config	build/package/tool configuration consumer
pytest.ini	B	build-or-package-config	build/package/tool configuration consumer
python/dask/__init__.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/dask/array/__init__.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/dask/array/random.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/dask/base.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/__init__.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/__init__.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_canonical_json.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_degradation.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_ed25519.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_gpu.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_license.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_map_scene_common.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_map_scene_labels.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_map_scene_render.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_map_scene_validation.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_memory.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_native.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_png.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_validate.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_viewer_binary.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/_viewer_entry.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/alignment.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/anamnesis.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/anamnesis.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/animation.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/animation.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/astro.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/astro.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/atmosphere.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/atmosphere.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/bench.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/buildings.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/bundle.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/camera_rigs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/camera_rigs.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/certificate.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/codec.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/codec.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/cog.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/colormaps/__init__.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/colormaps/core.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/colormaps/core_palettes.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/colormaps/io.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/colormaps/providers.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/colormaps/registry.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/colors.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/config.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/crs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/data/fonts/FONT-INVENTORY.json	B	asset	asset/package/documentation consumer or manifest
python/forge3d/data/fonts/FONT-NOTICES.json	B	asset	asset/package/documentation consumer or manifest
python/forge3d/data/fonts/NotoSansArabic-subset.ttf	B	binary	asset/package/runtime consumer plus binary format
python/forge3d/data/fonts/NotoSansDevanagari-subset.ttf	B	binary	asset/package/runtime consumer plus binary format
python/forge3d/data/fonts/NotoSansHebrew-subset.ttf	B	binary	asset/package/runtime consumer plus binary format
python/forge3d/data/fonts/NotoSansLatin-subset.ttf	B	binary	asset/package/runtime consumer plus binary format
python/forge3d/data/fonts/NotoSansSC-subset.ttf	B	binary	asset/package/runtime consumer plus binary format
python/forge3d/data/fonts/OFL-1.1.txt	B	asset	asset/package/documentation consumer or manifest
python/forge3d/data/fonts/atlas_latin_default.json	B	asset	asset/package/documentation consumer or manifest
python/forge3d/data/fonts/atlas_latin_default.png	B	binary	asset/package/runtime consumer plus binary format
python/forge3d/data/mini_dem.npy	B	binary	asset/package/runtime consumer plus binary format
python/forge3d/data/sample_boundaries.geojson	B	asset	asset/package/documentation consumer or manifest
python/forge3d/datasets.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/denoise.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/denoise_oidn.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/determinism.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/diagnostics.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/export.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/forge3d.pdb	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/geometry.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/gis.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/gis.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/graticule.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/graticule.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/guiding.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/helpers/__init__.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/helpers/aov_io.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/helpers/frame_dump.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/helpers/ipython_display.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/helpers/mpl_display.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/helpers/offscreen.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/interactive.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/io.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/label_plan.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/legend.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/lighting.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/map_plate.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/map_scene.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/map_scene.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/materials.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/mem.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/mesh.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/north_arrow.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/offline.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/path_tracing.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/path_tracing.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/pointcloud.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/precision.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/precision.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/presets.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/provenance.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/py.typed	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/recipe_manifest.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/recipe_manifest.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/scale_bar.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/sdf.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/sky.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/sky.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/smoke.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/smoke.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/style.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/style_expressions.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/terrain.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/terrain_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/terrain_params.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/terrain_pbr_pom.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/terrain_scatter.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/text.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/text.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/text_atlas.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/text_atlas.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/textures.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/thematic.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/tiles3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/vector.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/verify.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/viewer.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/viewer.pyi	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/viewer_contract.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/viewer_ipc.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/forge3d/widgets.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/pyproj_stub/__init__.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/rasterio/__init__.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/rasterio/enums.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/rasterio/transform.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/rasterio/windows.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/tools/backends_runner.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/tools/device_diagnostics.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/tools/perf_sanity.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/tools/terrain_spike.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/vshade/__init__.py	B	ordinary-source	code/test/example/doc/script or tool consumer
python/xarray/__init__.py	B	ordinary-source	code/test/example/doc/script or tool consumer
scripts/aether_acceptance_evidence.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/assert_junit_zero_skips.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/build_selene_areoid.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/check_anamnesis_portability.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/check_determinism_hashes.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/check_public_funnel.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/ci_pytest_lane.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/compare_images.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/dd.rs.in	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/dd.wgsl.in	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/dd_product.rs.in	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/dd_vector.rs.in	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/detail_normals.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/gen_gallery_images.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/generate_audit_snapshot.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/generate_dd.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/generate_license_keypair.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/histogram_match.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/install_compatible_wheel.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/regenerate_gallery.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/run_dupla_proof.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/run_nvidia_determinism_acceptance.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/run_nvidia_visual_acceptance.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/sign_license_key.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/style_match_eval.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/substratia_evidence_report.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/summarize_m06_evidence.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/terrain_ci_probe.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/terrain_validation.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/tessella_evidence_contract.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/tessella_evidence_provenance.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/tessella_evidence_report.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/tessella_evidence_thresholds.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/transcribe_feedback.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/validate_gore_strict.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/validate_terrain.py	C	ordinary-source	code/test/example/doc/script or tool consumer
scripts/verify_mensura_fixtures.py	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/brdf_tile.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/determinism.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/gi_composite.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/hybrid_terrain_traversal.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/line_aa.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/overlays.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/polygon_fill.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/pt_shade.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/pt_shade_guard.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/terrain_pbr_pom.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/tonemap_common.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/unguarded_zero_div.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
shaders/contracts/water_surface.toml	C	ordinary-source	code/test/example/doc/script or tool consumer
specs/001-diagnostics-support-matrices/tasks.md	C	policy-or-CI	live policy/spec/workflow authority
src/accel/cpu_bvh/build.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/cpu_bvh/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/cpu_bvh/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/cpu_bvh/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/instancing.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/lbvh_gpu.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/lbvh_gpu/buffers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/lbvh_gpu/build.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/lbvh_gpu/morton.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/lbvh_gpu/refit.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/lbvh_gpu/setup.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/lbvh_gpu/sort.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/lbvh_gpu/sort_bitonic.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/lbvh_gpu/topology.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/sah_cpu.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/accel/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/animation/interpolation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/animation/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/animation/render_queue.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/astro/catalog.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/astro/frames.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/astro/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/astro/moon.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/astro/night.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/astro/night_gpu.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/astro/observation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/astro/time.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/astro/vsop.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/bin/forge3d-vtpack.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/bin/interactive_viewer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/bundle/manifest.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/bundle/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/camera/anchor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/camera/dof.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/camera/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/camera/validation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/cli/args.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/cli/gi_config_output.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/cli/gi_config_parse.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/cli/gi_formatting.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/cli/gi_params.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/cli/gi_parsing.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/cli/gi_types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/cli/interactive_viewer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/cli/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/codec/f3dz/decode.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/codec/f3dz/encode.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/codec/f3dz/format.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/codec/f3dz/gpu.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/codec/f3dz/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/codec/f3dz/predict.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/codec/f3dz/rans.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/codec/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/colormap/assets/magma_256x1.png	A	binary	asset/package/runtime consumer plus binary format
src/colormap/assets/terrain_256x1.png	A	binary	asset/package/runtime consumer plus binary format
src/colormap/assets/viridis_256x1.png	A	binary	asset/package/runtime consumer plus binary format
src/colormap/colormap1d.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/colormap/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/converters/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/converters/multipolygonz_to_obj.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/anamnesis/key.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/anamnesis/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/anamnesis/report.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/anamnesis/scheduler.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/anamnesis/store.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/async_compute/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/async_compute/scheduler.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/async_compute/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/async_readback.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/atmosphere/bake.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/atmosphere/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/atmosphere/precomputed.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/atmosphere/precomputed/turbidity-1.bin	A	binary	asset/package/runtime consumer plus binary format
src/core/atmosphere/precomputed/turbidity-10.bin	A	binary	asset/package/runtime consumer plus binary format
src/core/atmosphere/precomputed/turbidity-2.bin	A	binary	asset/package/runtime consumer plus binary format
src/core/atmosphere/precomputed/turbidity-4.bin	A	binary	asset/package/runtime consumer plus binary format
src/core/atmosphere/precomputed/turbidity-8.bin	A	binary	asset/package/runtime consumer plus binary format
src/core/atmosphere/runtime.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/atmosphere/spectral.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/big_buffer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/bloom.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/bloom/config.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/capabilities.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/cascade_split.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/certificate.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/cloud_shadows/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/cloud_shadows/renderer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/cloud_shadows/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/cloud_shadows/utils.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/clouds/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/clouds/renderer/controls.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/clouds/renderer/data.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/clouds/renderer/init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/clouds/renderer/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/clouds/renderer/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/clouds/renderer/resources.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/clouds/renderer/textures.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/clouds/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/compressed_textures.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/compressed_textures/bc4.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/compressed_textures/bc5.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/compressed_textures/bc7.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/compressed_textures/compression.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/compressed_textures/load.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/compressed_textures/parsing.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/compressed_textures/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/compressed_textures/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/compressed_textures/upload.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/context.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd/generator.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd/gpu.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd/gpu_exec.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd/gpu_report.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd/jitter.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd/jitter_model.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd/jitter_pipeline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd/product.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd/proof.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd/vector.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dd_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/degradation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/device_caps.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dof/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dof/pipeline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dof/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/double_buffer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dual_source_oit/constructor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dual_source_oit/controls.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dual_source_oit/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dual_source_oit/pass.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/dual_source_oit/pipeline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/envmap.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/error.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/feedback_buffer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/fence_tracker.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/framegraph_impl/barriers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/framegraph_impl/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/framegraph_impl/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/gbuffer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/gpu.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/gpu_timing.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/gpu_types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ground_plane/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ground_plane/presets.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ground_plane/renderer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ground_plane/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/hdr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/hdr_readback.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/hdr_tonemapping.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/hdr_types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ibl.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ibl/brdf_lut.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ibl/cache.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ibl/constructor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ibl/environment.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ibl/image_io.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ibl/irradiance.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ibl/prefilter.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ibl/runtime.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/jitter.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ltc_area_lights.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ltc_lut.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/ltc_types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/material.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/matrix_stack.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/memory_tracker.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/memory_tracker/helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/memory_tracker/pool.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/memory_tracker/registry.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/memory_tracker/reporting.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/memory_tracker/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/memory_tracker/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/mipmap.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/multi_thread/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/multi_thread/pool.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/multi_thread/tasks.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/overlay_layer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/overlays.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/pbr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/point_spot_lights/creation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/point_spot_lights/draw.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/point_spot_lights/management.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/point_spot_lights/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/point_spot_lights/presets.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/point_spot_lights/structs.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/point_spot_lights/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/provenance.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/reflections.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/reflections_math.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/reflections_types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/resource_tracker.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/sampler_modes.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/scene_graph/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/scene_graph/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/scene_graph/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/scene_graph/traversal.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/scene_graph/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/hzb.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/hzb_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/manager.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/manager/accessors.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/manager/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/manager/execute.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/settings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssao/accessors.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssao/constructor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssao/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssao/passes.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssao/pipelines.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssao/resources.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssao/runtime.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssao/temporal.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssgi.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssgi/accessors.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssgi/constructor/layouts.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssgi/constructor/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssgi/constructor/pipelines.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssgi/constructor/resources.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssgi/controls.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssgi/runtime.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssr/accessors.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssr/constructor/layouts.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssr/constructor/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssr/constructor/pipelines.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssr/constructor/resources.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssr/runtime.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/screen_space_effects/ssr/stats.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/session.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/shader_contract_runtime.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/shader_registry.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/shadow_mapping/bind_group.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/shadow_mapping/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/shadow_mapping/system.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/shadow_mapping/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/shadows/frustum.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/shadows/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/shadows/resources.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/shadows/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/soft_light_radius.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/staging_rings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/taa.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/temporal_history.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/text_mesh/builder.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/text_mesh/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/text_mesh/renderer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/text_mesh/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/text_overlay.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/texture_format.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/texture_format_defs.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/texture_upload/hdr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/texture_upload/height.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/texture_upload/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/texture_upload/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/texture_upload/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/tile_cache/allocator.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/tile_cache/cache.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/tile_cache/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/tile_cache/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/tile_cache/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/tonemap.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/water_surface/constructor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/water_surface/controls.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/water_surface/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/water_surface/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/core/water_surface/uniforms.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/export/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/export/projection.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/export/svg.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/export/svg_labels.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/external_image/decode.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/external_image/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/external_image/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/external_image/upload.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/formats/exr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/formats/hdr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/formats/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/body.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/geodesic.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/geoid.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/projections/aea.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/projections/eqc.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/projections/geocentric.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/projections/lcc.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/projections/merc.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/projections/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/projections/stere.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/projections/tmerc.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/reproject.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geo/units.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/array_convert.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/curves.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/displacement.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/exact/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/exact/oracle.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/exact/predicates.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/exact/py.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/extrude.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/grid.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/mesh_python.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/overlay/faces.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/overlay/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/overlay/rectangles.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/overlay/rings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/overlay/snap.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/overlay/sweep.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/overlay/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/overlay/validity.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/overlay/verification.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/overlay/verification_oracle.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/primitives.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/py_advanced.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/py_bindings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/simplify.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/subdivision.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/tangents.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/thick_polyline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/transform.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/transforms.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/validate.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/geometry/weld.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/affine.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/cog_range.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/crs.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/domain.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/error.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/antimeridian.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/centroid.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/crs_resolve.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/line_ops.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/math.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/measure.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/model.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/parse.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/py.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/tests/dateline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/tests/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/topology.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/topology_buffer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/topology_simplify.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/geometry/validate.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/osm.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/py_json.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/raster_info.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/raster_read.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/raster_tags.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/raster_values.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/raster_window.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/raster_write.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/rasterize.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/remote.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/terrarium.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/thematic.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/tiles.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/vector.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/gis/warp.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/import/building_materials.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/import/cityjson.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/import/cityjson/bindings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/import/cityjson/geometry.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/import/cityjson/parser.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/import/cityjson/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/import/cityjson/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/import/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/import/osm_buildings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/io/gltf_read.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/io/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/io/obj_read.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/io/obj_write.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/io/stl_write.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/atlas.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/callout.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/collision.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/curved.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/declutter.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/font/face.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/font/fvar.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/font/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/font/outline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/font/variation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/layer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/leader.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/line_label.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/msdf/atlas.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/msdf/distance.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/msdf/edge.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/msdf/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/optimal.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/positioned.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/projection.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/py_bindings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/py_text.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/raster.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/rtree.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/arabic.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/bidi.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/bidi_brackets.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/bidi_conformance_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/bidi_explicit.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/bidi_resolve.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/bidi_rule_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/devanagari.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/generate_unicode_tables.py	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gpos.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gpos_attach.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gpos_buffer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gpos_edge_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gpos_kern.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gpos_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gpos_validate.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gpos_value.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gsub.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gsub_apply.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gsub_context.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gsub_filter.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gsub_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gsub_tests/context.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/gsub_tests/feature.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/layout.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/layout_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/linebreak.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/linebreak_conformance_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/linebreak_emoji.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/linebreak_rules.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/ot.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/shape/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/typography.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/unicode/LICENSE-UNICODE	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/unicode/PROVENANCE.md	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/unicode/SOURCES.sha256	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/unicode/UCD_VERSION	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/unicode/generate.py	A	ordinary-source	code/test/example/doc/script or tool consumer
src/labels/unicode/generated.rs	A	generated	Unicode generator provenance and consumers
src/labels/unicode/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lib.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/license/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/area_lights.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/atmospherics.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/ephemeris.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/ibl_cache.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/ibl_wrapper.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/light.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/light_buffer/creation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/light_buffer/frame.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/light_buffer/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/light_buffer/r2.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/light_buffer/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/light_buffer/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/light_buffer/update.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/material.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/py_bindings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/py_bindings/atmosphere.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/py_bindings/gi.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/py_bindings/light.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/py_bindings/material.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/py_bindings/screen_space.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/py_bindings/shadow.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/py_bindings/sky.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/py_bindings/sun_position.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/py_bindings/volumetrics.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/screen_space.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/shadow.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/shadow_map.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/lighting/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/loaders/ktx2/loader.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/loaders/ktx2/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/loaders/ktx2/parser.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/loaders/ktx2/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/loaders/ktx2/validation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/loaders/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/mesh/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/mesh/tbn.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/mesh/vertex.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/adjudication_raster.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/brdf_tile.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/brdf_tile/api.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/brdf_tile/debug.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/brdf_tile/math.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/brdf_tile/params.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/brdf_tile/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/brdf_tile/request.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/brdf_tile/resources.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/brdf_tile/resources/render_pass.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/brdf_tile/resources/timestamps.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/brdf_tile/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/forward.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/pipeline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/offscreen/sphere.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/p5/meta/constants.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/p5/meta/defaults.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/p5/meta/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/p5/meta/ssr_status.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/p5/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/p5/ssr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/p5/ssr_analysis/luminance.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/p5/ssr_analysis/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/p5/ssr_analysis/roi.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/passes/gi/bind_groups.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/passes/gi/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/passes/gi/params.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/passes/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/passes/ssgi.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/passes/ssr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/accel.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/adjudication.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/alias_table.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/aov.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/compute/dispatch.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/compute/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/compute/readback.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/compute/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/compute/setup.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/compute_types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/hybrid_compute/aether_post.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/hybrid_compute/aether_reference.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/hybrid_compute/layouts.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/hybrid_compute/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/hybrid_compute/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/hybrid_compute/render_terrain.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/hybrid_compute/setup.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/hybrid_compute/terrain_heightfield.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/importance.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/io.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/lighting.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/mesh/bind_groups.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/mesh/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/mesh/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/mesh/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/mesh/upload.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/mesh/validation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/reference_scene.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/restir/buffers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/restir/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/restir/system.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/restir/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/aov.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/bindings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/control.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/dispatch.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/instances.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/pipeline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/pipeline/layouts.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/pipeline/restir.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/pipeline/scene_layout.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/pipeline/stages_primary.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/pipeline/stages_secondary.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/queues.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/queues/intersect_shade.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/queues/raygen_shadow.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/queues/scatter_compact.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/queues/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/path_tracing/wavefront/restir.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/picking/bounds.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/picking/heightfield_ray.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/picking/highlight.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/picking/id_buffer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/picking/lasso.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/picking/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/picking/ray.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/picking/selection.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/picking/terrain_query.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/picking/tile_id.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/picking/unified.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/hdr_offscreen/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/hdr_offscreen/pipeline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/hdr_offscreen/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/normal_mapping.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/pbr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/pbr/bindings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/pbr/constructor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/pbr/ibl.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/pbr/material.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/pbr/rendering.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/pbr/scene_uniforms.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/pbr/shadow.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/pbr/state.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/pbr/textures.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pipeline/pbr/tone_mapping.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pointcloud/copc.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pointcloud/copc_decode.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pointcloud/ept.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pointcloud/error.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pointcloud/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pointcloud/octree.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pointcloud/renderer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/pointcloud/traversal.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/adjudication.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/astro.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/atmosphere.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/brdf.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/brdf/render.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/brdf/wrappers.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/codec.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/csm.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/diagnostics.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/frame.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/geodesy.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/labels.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/mod.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/path_tracing.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/path_tracing/aether_reference.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/path_tracing/gpu.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/path_tracing/gpu_mesh.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/path_tracing/hybrid.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/path_tracing/terrain_reference.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/pointcloud.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/precision.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/provenance.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/tiles3d.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/basic.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/coverage.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/coverage_ablation.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/coverage_cache.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/demo.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/inputs.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/oit.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/pick.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/polygon_fill.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/readback.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/render.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/vector/timing.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_functions/viewer.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/classes.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/anamnesis.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/astro.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/atmosphere.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/camera.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/codec.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/diagnostics.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/geodesy.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/geometry.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/gis.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/interactive.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/io_import.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/labels.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/license.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/precision.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/provenance.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/rendering.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/functions/tiles3d.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_module/mod.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_types/aov.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_types/atmosphere.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_types/frame.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_types/hdr_frame.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_types/mod.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_types/offline.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_types/picking.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_types/pointcloud.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_types/screen_space_gi.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/py_types/styles.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/render/colormap.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/instancing.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/material_set.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/material_set/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/material_set/gpu.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/material_set/gpu_helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/material_set/py_api.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/mesh_instanced.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/atmosphere.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/common.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/config/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/config/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/config/tests/defaults.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/config/tests/enums.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/config/tests/validation/atmosphere.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/config/tests/validation/gi.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/config/tests/validation/lights.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/config/tests/validation/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/config/tests/validation/shadows.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/gi.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/lights.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/shading.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/params/shadows.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/render/pbr_pass.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/renderer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/renderer/readback.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/core/constructor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/core/height.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/core/helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/postfx_cpu.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/private_impl.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/private_impl/clouds.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/private_impl/effects.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/private_impl/msaa.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/atmosphere.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/base.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/bloom.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/cloud_shadows.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/clouds.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/dof.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/ground_plane.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/ibl.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/instanced_mesh.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/native_overlays.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/native_text.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/oit.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/point_spot_lights_core.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/point_spot_lights_query.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/point_spot_lights_update.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/raster_overlay.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/rect_area_lights.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/reflections.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/shoreline.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/soft_light.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/ssgi.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/ssr.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/stats.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/text_mesh.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/py_api/water_surface.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/render_paths.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/render_paths/helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/render_paths/png.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/render_paths/rgba.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/render_paths/shared.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/render_paths/timing.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/ssao.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/ssao/constructor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/ssao/helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/ssao/runtime.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/ssao/setup.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/stats.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/texture_helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/scene/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/sdf/hybrid.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/sdf/hybrid_types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/sdf/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/sdf/operations.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/sdf/primitives.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/sdf/py.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shader_sources.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/adjudication_raster.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ao_from_aovs.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/atmosphere/evaluation_core.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/atmosphere/prometheus_aerial.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/atmosphere/prometheus_spectral_reference.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/atmosphere/scattering.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/bloom_blur_h.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/bloom_blur_v.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/bloom_brightpass.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/bloom_composite.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf/ashikhmin_shirley.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf/common.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf/cook_torrance.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf/disney_principled.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf/dispatch.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf/lambert.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf/minnaert.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf/oren_nayar.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf/phong.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf/toon.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf/ward.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/brdf_tile.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/bvh_refit.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/clipmap_lod_select.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/cloud_shadows.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/clouds.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/culling_compute.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/dd_harness.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/dd_jitter.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/denoise_atrous.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/dof.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/extrusion.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/f3dz_decode.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/filters/bilateral_separable.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/filters/edge_aware_upsample.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/fog_upsample.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/gbuffer/common.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/gi/composite.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/gi/debug.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ground_plane.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/heightfield_ao.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/heightfield_sun_vis.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/hybrid_kernel.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/hybrid_terrain_traversal.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/hybrid_traversal.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/hzb_build.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/hzb_cull.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ibl.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ibl_brdf.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ibl_equirect.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ibl_prefilter.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/includes/dd.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/includes/determinism.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/includes/shadow_moments.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/includes/tonemap_common.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/lbvh_link.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/lbvh_morton.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/lighting.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/lighting_ibl.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/lights.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/line_aa.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/mesh_basic.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/mesh_instanced.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/moment_generation.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/normal_mapping_vertex.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/offline_accumulate.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/offline_depth_expand.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/offline_depth_extract.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/offline_luminance.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/offline_resolve.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/oit_compose.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/oit_dual_source.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/oit_dual_source_compose.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/overlays.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/pbr.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/planar_reflections.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/point_edl.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/point_instanced.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/point_spot_lights.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/polygon_fill.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/postprocess_tonemap.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/pt_intersect.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/pt_kernel.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/pt_raygen.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/pt_restir_init.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/pt_restir_spatial.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/pt_restir_temporal.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/pt_scatter.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/pt_shade.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/pt_shadow.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/radix_sort_pairs.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/restir_spatial.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/restir_temporal.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/sdf_operations.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/sdf_primitives.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/shadow_blur.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/shadows.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/sky.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/soft_light_radius.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssao.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssao/common.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssao/composite.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssao/gtao.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssao/ssao.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssgi/composite.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssgi/resolve_temporal.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssgi/shade.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssgi/trace.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssr/composite.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssr/fallback_env.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssr/shade.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssr/temporal.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/ssr/trace.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/stars.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/taa.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/temporal/resolve_ao.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain_aether_blit.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain_blit.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain_minimal.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain_noise.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain_normal_blit.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain_pbr_pom.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain_probes.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain_shadow_depth.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain_visbuffer_resolve.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain_visbuffer_write.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/terrain_visibility_fullscreen.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/text_overlay.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/tone_map.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/tonemap_terrain_offline.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/vector_coverage_bin.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/vector_coverage_raster.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/vector_coverage_resolve.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/velocity.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/viewer_volumetrics.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/volumetric.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shaders/water_surface.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/blur_pass.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/cascade_math.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/csm_depth_control.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/csm_renderer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/csm_types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/manager/budget.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/manager/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/manager/system.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/manager/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/moment_pass.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/msm_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/shadows/state.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/smoke/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/smoke/py.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/smoke/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/smoke/sampling.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/smoke/sim.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/smoke/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/converters.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/expressions/comparison.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/expressions/control.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/expressions/dispatch.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/expressions/logic.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/expressions/math.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/expressions/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/expressions/property.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/expressions/strings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/expressions/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/parser.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/sprite.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/style/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/accumulation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/analysis.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/bloom_processor/config.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/bloom_processor/constructor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/bloom_processor/execute.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/bloom_processor/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/bloom_processor/uniforms.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/camera.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/clipmap/geomorph.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/clipmap/gpu_lod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/clipmap/level.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/clipmap/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/clipmap/py_bindings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/clipmap/ring.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/clipmap/streaming.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/clipmap/vertex.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/cog/cache.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/cog/cog_reader.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/cog/content_range.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/cog/error.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/cog/ifd_parser.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/cog/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/cog/py_bindings.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/cog/range_cache.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/cog/range_reader.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/cog/range_stats.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/colormap_lut.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/culling/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/culling/two_phase.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/globals.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/hosek_rgb_data.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/hosek_sky.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/lights.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/lod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/mesh.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/page_table/common.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/page_table/gpu.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/page_table/height_loader.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/page_table/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/page_table/overlay_loader.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/page_table/queue.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/page_table/readers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/pipeline/bind_groups.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/pipeline/creation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/pipeline/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/probes/baker.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/probes/gpu.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/probes/heightfield_baker.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/probes/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/probes/reflection_baker.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/probes/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/decode_atmosphere.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/decode_core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/decode_effects.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/decode_lighting.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/decode_materials.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/decode_postfx.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/decode_probes.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/decode_vt.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/native_effects.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/native_lighting.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/native_material.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/native_overlays.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/native_postfx.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/native_postfx/atmosphere.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/native_postfx/camera.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/native_postfx/quality.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/native_postfx/tonemap.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/native_probes.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/native_vt.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/parse.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/private_impl.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/render_params/py_api.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/anamnesis.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/aov.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/atmosphere.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/atmosphere/luts.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/bind_groups.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/bind_groups/base_layouts.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/bind_groups/layouts.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/bind_groups/terrain_pass.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/constructor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/draw/execute.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/draw/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/draw/setup/context.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/draw/setup/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/draw/setup/pipeline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/geometry.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/height_ao/init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/height_ao/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/height_ao/passes.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/height_ao/pipelines.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/msaa.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/offline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/pipeline_cache.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/probes.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/py_api.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/render_graph.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/resources/ao.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/resources/init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/resources/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/resources/resize.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/runtime_contract.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/scatter.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/shadows.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/shadows/main_bind_group.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/shadows/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/shadows/resources.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/shadows/setup.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/streaming.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/uniforms.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/upload.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/viewer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/virtual_texture.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/visibility_buffer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/water_reflection/bind_group.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/water_reflection/init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/water_reflection/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/renderer/water_reflection/uniforms.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/scatter.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/scene.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/spike.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/spike/analysis.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/spike/async_loader.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/spike/constructor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/spike/height_mosaic.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/spike/overlay_stream.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/spike/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/spike/terrain_ops.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/spike/tiling.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/stats.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/stream/color.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/stream/config.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/stream/height.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/stream/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/stream/util.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/tiling.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/uniforms.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/vt/footprint.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/vt/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/vt/procedural.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/vt/requests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/vt/store.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/terrain/vt_family_residency.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/tiles3d/b3dm.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/tiles3d/bounds.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/tiles3d/error.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/tiles3d/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/tiles3d/pnts.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/tiles3d/renderer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/tiles3d/sse.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/tiles3d/tile.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/tiles3d/tileset.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/tiles3d/traversal.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/util/debug_pattern.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/util/exr_write.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/util/image_write.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/util/memory_budget.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/util/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/uv/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/uv/unwrap.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/api.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/api/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/api/extrusion.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/api/py.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/api/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/batch/aabb.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/batch/frustum.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/batch/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/batch/stats.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/coverage/binning.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/coverage/ingest.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/coverage/math.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/coverage/math_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/coverage/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/coverage/raster.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/coverage/raster_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/coverage/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/coverage/resolve.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/coverage/resolve_tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/coverage/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/data.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/extrusion.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/gpu_extrusion/buffers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/gpu_extrusion/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/gpu_extrusion/pipeline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/gpu_extrusion/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/gpu_extrusion/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/graph.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/indirect/culling.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/indirect/draw.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/indirect/init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/indirect/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/indirect/renderer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/indirect/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/indirect/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/layer.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/line.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/line_helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/line_pipeline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/line_types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/oit/blend.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/oit/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/oit/pipeline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/atlas.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/instance.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/pipeline.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/renderer/config.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/renderer/draw.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/renderer/init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/renderer/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/renderer/pipelines.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/renderer/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/renderer/upload.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/point/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/vector/polygon.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/verify/contract.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/verify/domain.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/verify/ir/engine.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/verify/ir/eval.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/verify/ir/expr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/verify/ir/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/verify/ir/ops.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/verify/ir/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/verify/ir/value.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/verify/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/camera_controller.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/effects_command.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/gi_command.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/handler.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/ipc_command.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/labels_command.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/legacy_handler.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/pointcloud_command.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/scene_command.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/scene_review_command.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/terrain_command.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/cmd/vector_overlay_command.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/cmd_parse_init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/command_batch.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/command_preflight.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/command_preflight_helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/ipc_state.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/runner.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/environment/atmosphere.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/environment/ibl.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/environment/lighting.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/environment/misc.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/environment/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/environment/parse.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/gi/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/gi/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/gi/ssao.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/gi/ssgi.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/gi/ssr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/parser/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/spawn.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/event_loop/stdin_reader/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/hud.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/image_analysis.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/init/composite_init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/init/device_init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/init/fallback_init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/init/fog_init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/init/gbuffer_init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/init/gi_baseline_init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/init/lit_init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/init/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/init/sky_init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/init/viewer_new.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/input/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/input/viewer_input.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/defaults.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/parse.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/payloads.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/request.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/response.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/translate/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/translate/labels.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/translate/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/translate/overlays.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/translate/scene_review.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/protocol/translate/terrain.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/ipc/server.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/ao.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/cornell.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/gbuffer_dump.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/gi_ablation.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/gi_verification.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/ssgi_cornell.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/ssgi_temporal.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/ssr_glossy.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/ssr_helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/ssr_scene_impl.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/p5/ssr_thickness.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/pointcloud.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/pointcloud/load.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/pointcloud/pointcloud.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/pointcloud/shader.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/pointcloud/state.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/pointcloud/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/finalize.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/frame_anchor.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/frame_anchor_stats.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/frame_setup.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/geometry/fog.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/geometry/fog_dispatch.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/geometry/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/geometry/pass.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/postfx.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/postfx_cache.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/secondary.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/main_loop/snapshot_sky.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/render/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/scene_review.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/labels.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/mesh_upload.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/resize.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/viewer_helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/viewer_helpers/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/viewer_helpers/gi.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/viewer_helpers/gi/geometry.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/viewer_helpers/gi/reexecute.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/viewer_helpers/ibl.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/viewer_helpers/readback.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/state/viewer_helpers/snapshot_sky.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/denoise.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/dof.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/dof/dof.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/dof/pass.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/dof/pass/execute.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/dof/pass/setup.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/dof/shader.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/dof/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/motion_blur.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/motion_blur_depth.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/motion_blur_depth/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/overlay.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/overlay/config.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/overlay/sampling.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/overlay/stack.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/overlay/stack/composite.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/overlay/stack/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/overlay/tests.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/pbr_renderer/defaults.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/pbr_renderer/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/pbr_renderer/types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/pbr_renderer/updates.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/post_process.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/motion_blur.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/offscreen/effects.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/offscreen/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/offscreen/scene.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/offscreen/setup.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/screen/effects.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/screen/mod.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/screen/resources.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/screen/scene.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/screen/setup.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/render/shadow.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/scene.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/scene/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/scene/overlays.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/scene/pbr_compute.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/scene/pipeline_init.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/scene/scatter.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/scene/terrain_load.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/shader.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/shader_pbr.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/shader_pbr/terrain_pbr.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/shader_pbr/terrain_shadow_depth.wgsl	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/vector_overlay.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/vector_overlay/core.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/vector_overlay/pipelines.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/volume_density.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/terrain/volumetrics.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_analysis.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_config.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_constants.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_enums.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_enums/commands.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_enums/config.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_enums/modes.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_image_utils.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_render_helpers.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_ssr_scene.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_struct.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
src/viewer/viewer_types.rs	A	ordinary-source	code/test/example/doc/script or tool consumer
tests/UNRUN.toml	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/__init__.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_aether_physical_probe.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_aether_pt_oracle.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_aether_quadrature.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_bc_fixtures.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_cog_http_fixtures.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_coverage_ref.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_deltae.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_fuzz.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_geomfuzz.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_golden_variants.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_license_test_keys.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_loopback.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_ssim.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_substratia_evidence.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_sutura_recipes.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_terrain_flythrough.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_terrain_runtime.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_tessella_evidence.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_toml_compat.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_torture.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/_torture_materiality.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/allocation_allowlist.toml	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/anamnesis_gpu_acceptance.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/anamnesis_irrelevant_inputs.toml	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/astro_catalog.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/astro_ephemeris.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/astro_oracle.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/astro_time.rs	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/conftest.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/data/codec_corpus/DETERMINISM.toml	B	corpus	test corpus provenance/consumer
tests/data/codec_corpus/MANIFEST.toml	B	corpus	test corpus provenance/consumer
tests/data/codec_corpus/alpine.npy	B	corpus	test corpus provenance/consumer
tests/data/codec_corpus/canyon.npy	B	corpus	test corpus provenance/consumer
tests/data/codec_corpus/coastal_flat_nodata.npy	B	corpus	test corpus provenance/consumer
tests/data/codec_corpus/rolling.npy	B	corpus	test corpus provenance/consumer
tests/data/egm96_test_values.txt	B	corpus	test corpus provenance/consumer
tests/data/geodtest_mars.dat	B	corpus	test corpus provenance/consumer
tests/data/geodtest_subset.dat	B	corpus	test corpus provenance/consumer
tests/data/horizons_vectors.MANIFEST.toml	B	corpus	test corpus provenance/consumer
tests/data/horizons_vectors.dat	B	corpus	test corpus provenance/consumer
tests/data/mars_areoid_reference.txt	B	corpus	test corpus provenance/consumer
tests/data/shader_proofs/unguarded_zero_div.wgsl	B	corpus	test corpus provenance/consumer
tests/data/shaping/PROVENANCE.md	B	corpus	test corpus provenance/consumer
tests/data/shaping/arabic.json	B	corpus	test corpus provenance/consumer
tests/data/shaping/cjk.json	B	corpus	test corpus provenance/consumer
tests/data/shaping/devanagari.json	B	corpus	test corpus provenance/consumer
tests/data/shaping/hebrew.json	B	corpus	test corpus provenance/consumer
tests/data/shaping/latin.json	B	corpus	test corpus provenance/consumer
tests/data/shaping/mixed.json	B	corpus	test corpus provenance/consumer
tests/data/unicode/BidiCharacterTest.txt	B	corpus	test corpus provenance/consumer
tests/data/unicode/BidiTest.txt	B	corpus	test corpus provenance/consumer
tests/data/unicode/LineBreakTest.txt	B	corpus	test corpus provenance/consumer
tests/data/unicode/PROVENANCE.md	B	corpus	test corpus provenance/consumer
tests/data/vector_torture/cases.json	B	corpus	test corpus provenance/consumer
tests/degradation_allowlist.toml	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/fixtures/mapbox_streets_v8.json	B	fixture	test/example fixture consumer
tests/fixtures/provenance/provenance.json	B	fixture	test/example fixture consumer
tests/fixtures/provenance/source_map.npy	B	fixture	test/example fixture consumer
tests/fixtures/recipe_manifests/climate_bivariate.json	B	fixture	test/example fixture consumer
tests/fixtures/recipe_manifests/hydrology_river.json	B	fixture	test/example fixture consumer
tests/fixtures/recipe_manifests/landcover_esri_terrain_viewer.json	B	fixture	test/example fixture consumer
tests/fixtures/recipe_manifests/mapscene_showcases.json	B	fixture	test/example fixture consumer
tests/fixtures/recipe_manifests/terrain_demo.json	B	fixture	test/example fixture consumer
tests/fixtures/recipe_manifests/terrain_label.json	B	fixture	test/example fixture consumer
tests/golden/adjudication/pt_reference.png	B	golden	golden test/provenance consumer
tests/golden/adjudication/raster_reference.png	B	golden	golden test/provenance consumer
tests/golden/atmosphere/aether_gpu_sunset_sweep.png	B	golden	golden test/provenance consumer
tests/golden/atmosphere/aether_sunset_sweep.png	B	golden	golden test/provenance consumer
tests/golden/certificates/README.md	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_alignment_utm.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_auto_water.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_buildings.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_clipmap_large_region.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_cloud_shadows.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_copc_points.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_furniture_graticule.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_label_arabic_joining.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_label_halo_depth.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_label_occlusion_ridge.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_material_maps.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_offline_aovs.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_png16_color.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_screen_space_contact.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_screen_space_reflection.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_terrain_raster.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_textured_gltf_landmark.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_thematic_choropleth.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_tiles3d_points.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_vector_labels.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_vector_stroke_quality.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/mapscene_vector_stroke_quality_4x.json	B	certificate	certificate fixture/provenance consumer
tests/golden/certificates/signing.pub	B	certificate	certificate fixture/provenance consumer
tests/golden/hybrid_terrain/mini_dem_reference.png	B	golden	golden test/provenance consumer
tests/golden/labels/optimal_plan_hash.json	B	golden	golden test/provenance consumer
tests/golden/presets/rainier_showcase_mapscene.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_alignment_utm.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_auto_water.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_buildings.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_clipmap_large_region.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_cloud_shadows.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_copc_points.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_furniture_graticule.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_label_arabic_joining.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_label_halo_depth.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_label_occlusion_ridge.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_material_maps.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_offline_aovs.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_png16_color.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_screen_space_contact.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_screen_space_reflection.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_terrain_raster.metal.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_terrain_raster.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_terrain_raster.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_textured_gltf_landmark.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_thematic_choropleth.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_tiles3d_points.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_vector_labels.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_vector_stroke_quality.png	B	golden	golden test/provenance consumer
tests/golden/recipes/mapscene_vector_stroke_quality_4x.png	B	golden	golden test/provenance consumer
tests/golden/sidera_night.png	B	golden	golden test/provenance consumer
tests/golden/terrain/substratia_grazing_baseline.metal.png	B	golden	golden test/provenance consumer
tests/golden/terrain/substratia_grazing_baseline.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/terrain/substratia_grazing_normal.metal.png	B	golden	golden test/provenance consumer
tests/golden/terrain/substratia_grazing_normal.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_atmosphere.metal.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_atmosphere.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_atmosphere.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_low_sun_sky.metal.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_low_sun_sky.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_low_sun_sky.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_pbr.metal.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_pbr.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_pbr.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_pom.metal.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_pom.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_pom.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_tv10_scene_a_sss.metal.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_tv10_scene_a_sss.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_tv10_scene_a_sss.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_tv10_scene_b_sss.metal.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_tv10_scene_b_sss.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_tv10_scene_b_sss.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_tv10_zero_sss.metal.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_tv10_zero_sss.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_tv10_zero_sss.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_water.metal.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_water.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_water.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_water_reflection.metal.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_water_reflection.nvidia-vulkan.png	B	golden	golden test/provenance consumer
tests/golden/terrain/terrain_water_reflection.png	B	golden	golden test/provenance consumer
tests/goldens/determinism/sidera_night.sha256	B	golden	golden test/provenance consumer
tests/goldens/determinism/terra_determinata_v1.dx12.sha256	B	golden	golden test/provenance consumer
tests/goldens/determinism/terra_determinata_v1.sha256	B	golden	golden test/provenance consumer
tests/helpers_namespace.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/requirements.txt	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/robustness_ratchet.toml	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/shader_proofs_ledger.toml	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/smoke_test.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_3dtiles_parse.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_3dtiles_sse.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_accumulation_aa.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_adjudication_gate.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_aether_acceptance_evidence.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_allocation_gate.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_anamnesis_adversarial_keys.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_anamnesis_hermeticity.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_anamnesis_incremental.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_anamnesis_inertness.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_anamnesis_p1.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_anamnesis_portability.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_anamnesis_uniform_layout.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_animation_mvp.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_aov.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_api_contracts.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_api_truth_pass.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_assert_junit_zero_skips.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_astro_ephemeris.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_astro_night_golden.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_atmosphere_golden.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_atmosphere_lut_handoff.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_atmosphere_pt_reference.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_atmosphere_reference.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_atmosphere_spectral.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_bc_encoders.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_bench_diagnostics.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_bidi_conformance.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_blender_quality_integration.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_bloom_effect.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_bloom_execute_behavior.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_boolean_overlay.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_bosnia_terrain_landcover_viewer.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_bryce_canyon_storm_timelapse.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_budget_enforce.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_buildings_cityjson.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_buildings_extrude.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_buildings_materials.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_buildings_roof.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_bundle_cli.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_bundle_render.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_bundle_roundtrip.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_california_cigar_smoke_hybrid.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_california_fire_smoke_effect.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_california_wildfire_smoke_wind.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_cam_phi_wiring.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_camera_rigs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_capability_negotiation.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_cartographer_prime_evidence.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_certificate_verifier.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_ci_cost_controls.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_ci_lfs_fanout.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_clipmap_structure.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_cog_streaming.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_color_management.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_colorado_rem_example.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_colors_mood.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_copc_laz_fixture.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_crs_auto.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_crs_reproject.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_dask_stub.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_datasets.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_dd_arithmetic.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_dd_jitter.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_dead_render_structure_gate.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_degradation_behavior.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_dem_loading.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_denoise_settings.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_determinism_hash.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_determinism_matrix.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_device_init_failure.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_diagnostics_bundle_serialization.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_diagnostics_contract.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_diagnostics_no_op_policy.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_diagnostics_quickstart.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_diagnostics_style_support.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_diagnostics_support_paths.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_dof.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_epsg_g7_2.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_exact_predicates.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_export_projection.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_export_svg.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_exr_output.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_f3dz_codec.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_flythrough_popping.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_fog_offline.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_france_population_spikes_height_shade.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_geodesic_karney.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_geodetic_conservation.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_geoid_egm96.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_geomorph_seams.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_alignment_windowing.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_cog_range.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_cog_range_tiled.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_cog_tiles.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_crs_affine.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_domain.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_osm.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_raster.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_rasterize_mask.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_read_raster.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_remote.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_resample_reproject.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_thematic.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_vector_crs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_vector_geom.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_vector_io.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gis_vector_overlay.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gltf_import.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_gpu_lod_selection.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_height_system_safety.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_heightfield_ao.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_heightfield_compute_contracts.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_helsinki_transit_daycycle.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_humanity_globe_video.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_hybrid_terrain_pt.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_hzb_culling.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_iberia_builtup_cover_3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_ibl_from_hdr_formats.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_install_compatible_wheel.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_install_smoke.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_italy_forest_cover_example.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_jitter_sequence.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_khumbu_icefall_sentinel_timelapse.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_api_configuration_truth.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_api_diagnostics.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_api_docs_support.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_api_line_curved_paths.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_api_line_edge_cases.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_api_public_workflow.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_api_quickstart.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_api_stable_ids.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_api_state_noops.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_optimal_solver.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_candidate_bridge.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_contract.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_determinism.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_docs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_keepouts.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_payloads.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_point_candidates.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_polygon_candidates.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_priority.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_quickstart.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_rejection_reasons.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_label_plan_terrain.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_labels_pybindings.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_lens_effects.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_license.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_light_feature_enablement.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_lighting_alignment.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_lighting_preset.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_luxembourg_rail_overlay.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_m06_anchoring_boundary.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_m06_command_transaction.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_m06_full_geospatial_viewer.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_m06_python_viewer_contracts.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_m06_scene_review_transaction.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_m06_single_rebase_contract.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_m06_temporal_resource_contract.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_m06_viewer_matrix_contract.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_map_plate_layout.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapbox_streets_fixture.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_alignment.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_buildings.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_docs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_examples.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_furniture.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_label_occlusion.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_label_plan_integration.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_presets.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_quickstart.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_recipe_contract.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_render_png.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_render_policy.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_save_bundle.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_screen_space_settings.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_support_status.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_sutura_integrity.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_typing.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_validation.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mapscene_vector_strokes.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_memory_budget_policy.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_mesh_tbn.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_motion_blur.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_motion_blur_runner.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_motion_vectors.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_msdf_fidelity.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_no_silent_degradation.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_oit_transparency.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_osm_city_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_building_workflow_support.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_bundle_guardrails.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_bundle_roundtrip.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_diagnostics_contract.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_docs_support_matrix.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_fixture_inventory.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_label_expressions_plan_diagnostics.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_label_layer_crs_terrain.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_label_layer_geometry.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_prerequisite_contracts.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_tiles3d_support.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p1_typography_font_support.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_advanced_label_rules.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_advanced_labels_repeated_curved.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_building_texture_diagnostics.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_building_texture_docs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_complex_shaping_decision.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_determinism_noop.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_diagnostics_contract.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_large_scene_bottlenecks.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_large_scene_cache_lod_instancing.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_large_scene_docs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_large_scene_memory.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_quickstart.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_support_docs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_textured_building_mapscene.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_vt_docs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p2_vt_family_validation.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_p7_preset_cli_merge.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_perspective_probe.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_perspective_projection.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_picking_ipc.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_picking_premium.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_pipeline_validation_gate.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_planetary_datums.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_platte_rem_example.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_png_io_fallback.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_pointcloud_gpu_integration.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_pointcloud_lod.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_poland_population_contour_3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_preset_visual_parity.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_pro_gating.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_projection_oracle.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_provenance_offline_verify.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_provenance_veritas.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_recipe_goldens.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_recipe_manifest.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_render_certificate.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_render_certificate_contract.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_reproject_error_policy.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_reproject_no_silent_suppression.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_robustness_ratchet.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_romania_builtup_cover_3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_rotterdam_solar_potential_shadow_study.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_shader_proofs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_shader_reachability.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_shadow_techniques.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_shaping_conformance.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_smoke.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_ssgi_ssr_wiring.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_style_parser.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_style_pixel_diff.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_style_render.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_style_visual.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_substratia_evidence_report.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_sun_ephemeris.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_sun_visibility.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_support_matrices_docs.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_swiss_terrain_landcover_viewer.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_taa_convergence.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_taa_toggle.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_analysis_api.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_camera_rigs_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_clipmap_streaming.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_data_revision.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_demo_cli_smoke.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_demo_preset_integration.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_hue_variation_strength.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_material_maps.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_materials.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_overlay_stack.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_probes.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_reflection_probe_exports.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_render_color_space.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_renderer.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_runtime.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_scatter.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_sky_parity.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_tv10_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_tv10_goldens.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_tv10_subsurface.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_tv10_subsurface_materials.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_tv13_lod_pipeline.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_tv21_blending.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_tv21_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_tv24_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_tv4_demo.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_tv4_material_variation.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_tv6_heterogeneous_volumetrics.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_viewer_pbr.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_visual_goldens.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_terrain_vt_pbr_families.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_tessella_certificate_evidence.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_tessella_evidence_report.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_text_atlas.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_text_three_surfaces.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_thematic.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_tonemap_lut.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_torture_atlas.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_trust_boundary_diagnostics.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_turkiye_river_basins_3d.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_tv12_offline_architecture.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_tv12_offline_quality.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_tv12_oidn.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_tv20_virtual_texturing.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_tv22_scatter_wind.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_uk_ireland_lighthouse_map.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_vector_coverage.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_vector_drape.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_vector_overlay_drape.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_vector_overlay_rendering.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_viewer_ipc.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_visibility_buffer.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_volumetrics_sky.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_vt_out_of_core.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_vt_request_retention.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_water_surface_shader.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_widgets.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/test_world_coord_f32_gate.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/COVERAGE.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/MANIFEST.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/README.md	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-001.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-002.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-003.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-004.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-005.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-006.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-007.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-008.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-009.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-010.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-011.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-012.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-013.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-014.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-015.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-016.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-017.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-018.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-019.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-020.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-021.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-022.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-023.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-024.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-025.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-026.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-027.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-028.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-029.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-030.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-031.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-032.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-033.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-034.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-035.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-036.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-037.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-038.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-039.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-040.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-041.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-042.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-043.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-044.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-045.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-046.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-047.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-048.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-049.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-050.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-051.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-052.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-053.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-054.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-055.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-056.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-057.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-058.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-059.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-060.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-061.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-062.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-063.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-064.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-065.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-066.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-067.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-068.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-069.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-070.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-071.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-072.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-073.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-074.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-075.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-076.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-077.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-078.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-079.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-080.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-081.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-082.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-083.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-084.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-085.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-086.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-087.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-088.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-089.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/crs/crs-090.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-001.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-002.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-003.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-004.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-005.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-006.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-007.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-008.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-009.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-010.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-011.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-012.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-013.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-014.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-015.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-016.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-017.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-018.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-019.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-020.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-021.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-022.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-023.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-024.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-025.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-026.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-027.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-028.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-029.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-030.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-031.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-032.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-033.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-034.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-035.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-036.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-037.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-038.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-039.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-040.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-041.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-042.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-043.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-044.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-045.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-046.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-047.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-048.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-049.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-050.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-051.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-052.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-053.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-054.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-055.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-056.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-057.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-058.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-059.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/dems/dems-060.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-001.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-002.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-003.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-004.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-005.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-006.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-007.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-008.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-009.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-010.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-011.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-012.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-013.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-014.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-015.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-016.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-017.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-018.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-019.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-020.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-021.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-022.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-023.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-024.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-025.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-026.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-027.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-028.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-029.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-030.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-031.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-032.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-033.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-034.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-035.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-036.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-037.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-038.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-039.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-040.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-041.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-042.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-043.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-044.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-045.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-046.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-047.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-048.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-049.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-050.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-051.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-052.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-053.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-054.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-055.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-056.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-057.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-058.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-059.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-060.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-061.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-062.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-063.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-064.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-065.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-066.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-067.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-068.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-069.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-070.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-071.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-072.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-073.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-074.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-075.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-076.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-077.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-078.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-079.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-080.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-081.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-082.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-083.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-084.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-085.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-086.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-087.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-088.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-089.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-090.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-091.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-092.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-093.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-094.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-095.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-096.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-097.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-098.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-099.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-100.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-101.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-102.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-103.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-104.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-105.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-106.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-107.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-108.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-109.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-110.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-111.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-112.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-113.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-114.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-115.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-116.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-117.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-118.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-119.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-120.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-121.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-122.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-123.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-124.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-125.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-126.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-127.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-128.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-129.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-130.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-131.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-132.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-133.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-134.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-135.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-136.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-137.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-138.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-139.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/geometry/geometry-140.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-001.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-002.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-003.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-004.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-005.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-006.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-007.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-008.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-009.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-010.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-011.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-012.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-013.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-014.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-015.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-016.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-017.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-018.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-019.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-020.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-021.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-022.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-023.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-024.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-025.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-026.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-027.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-028.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-029.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-030.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-031.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-032.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-033.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-034.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-035.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-036.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-037.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-038.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-039.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-040.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-041.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-042.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-043.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-044.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-045.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-046.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-047.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-048.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-049.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-050.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-051.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-052.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-053.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-054.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-055.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-056.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-057.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-058.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-059.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/labels/labels-060.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-001.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-002.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-003.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-004.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-005.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-006.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-007.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-008.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-009.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-010.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-011.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-012.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-013.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-014.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-015.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-016.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-017.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-018.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-019.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-020.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-021.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-022.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-023.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-024.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-025.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-026.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-027.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-028.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-029.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-030.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-031.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-032.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-033.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-034.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-035.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-036.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-037.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-038.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-039.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-040.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-041.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-042.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-043.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-044.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-045.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-046.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-047.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-048.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-049.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-050.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-051.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-052.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-053.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-054.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-055.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-056.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-057.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-058.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-059.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-060.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-061.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-062.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-063.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-064.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-065.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-066.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-067.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-068.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-069.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-070.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-071.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-072.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-073.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-074.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-075.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-076.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-077.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-078.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-079.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/rasters/rasters-080.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-001.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-002.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-003.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-004.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-005.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-006.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-007.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-008.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-009.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-010.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-011.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-012.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-013.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-014.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-015.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-016.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-017.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-018.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-019.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-020.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-021.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-022.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-023.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-024.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-025.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-026.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-027.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-028.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-029.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-030.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-031.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-032.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-033.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-034.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-035.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-036.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-037.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-038.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-039.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-040.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-041.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-042.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-043.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-044.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-045.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-046.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-047.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-048.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-049.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/semantic/semantic-050.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-001.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-002.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-003.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-004.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-005.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-006.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-007.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-008.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-009.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-010.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-011.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-012.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-013.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-014.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-015.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-016.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-017.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-018.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-019.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/torture/viewer_scene/viewer_scene-020.json	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/validate_terrain_output.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/validate_terrain_p4.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tests/verify_terrain_pbr_pom_shader.py	B	ordinary-source	code/test/example/doc/script or tool consumer
tools/f3dz_determinism_report.py	C	ordinary-source	code/test/example/doc/script or tool consumer
tools/generate_f3dz_corpus.py	C	ordinary-source	code/test/example/doc/script or tool consumer
tools/generate_sidera_assets.py	C	ordinary-source	code/test/example/doc/script or tool consumer
tools/verify_provenance.py	C	ordinary-source	code/test/example/doc/script or tool consumer
```

## Persisted W1 read-only report appendices

The complete W1 owner and verifier records are persisted as separate ledger
appendices at exact base
`f5db54f95d202681f95dad649162d18efdae8987`. Each digest is over the complete
file byte sequence from byte 0 through the final LF byte (UTF-8, including that
final LF), with no newline, Unicode, path, or whitespace normalization.
Independent reproduction is `shasum -a 256 <path>` and `wc -c -l <path>`.

| Anchor | Exact appendix | Lines | Bytes | Whole-file SHA256 | Required content verified |
|---|---|---:|---:|---|---|
| `W1-A-REPORT` | `docs/refactor-forge3d-w1-a.md` | 110 | 15,620 | `5acdb5d4bb785cba09d51be8a46aeb4ac87cbf0e330480fc33792125fe8a589a` | exact A predicate/count/digest; commands/searches; Rust/API/features/WGSL/resources and named contract maps; N01/N02/A01-A05 full finding rows; N03/shader and other rejection records; `NOT_PROVEN` lanes |
| `W1-B-REPORT` | `docs/refactor-forge3d-w1-b.md` | 227 | 18,874 | `a2cff1e4acc39fb327bc0854be1133222b160c85dfc5b6069043824136c90e35` | exact B predicate/count/digest; commands/searches; PyO3/stub/tests/examples/recipes and named contract maps; W1B-01 full finding row; N03/C02/C03/C05 and other rejection records; `NOT_PROVEN` lanes |
| `W1-C-REPORT` | `docs/refactor-forge3d-w1-c.md` | 200 | 18,615 | `683399467be20f3ce5d25aae8ee70d8a5fae163511bfd2b2821e08ecc5f4e129` | exact C complement/count/digest; commands/searches; docs/scripts/tools/CI/packaging/nonordinary and named contract maps; W1C-01/W1C-02 and cross-surface N02 full rows; rejection records; `NOT_PROVEN` lanes |
| `W1-V-REPORT` | `docs/refactor-forge3d-w1-v.md` | 330 | 14,910 | `52aee00821771c24d2c72cd92180725c5d63242c6b27218ac6034c34bc34a312` | exact A/B/C predicates/counts/digests/union/intersections; approved conditions; evidence-based rejections; classification requirement; disjoint I1-I9 ownership; exact proof and `NOT_PROVEN` boundaries |

These whole-file appendices are the complete report records. The integrated
tables below freeze their decisions; they do not replace the inspection proof.

## Tracked-surface coverage matrix

W1-A/B/C inspected the exact path-bound manifests proven below. `VALIDATED`
means every path was inspected and mapped to its owning contract/nonordinary
classification; it does not claim that unrun implementation or physical evidence
has passed.

| Area | Paths | Evidence inspected | Result | Accepted/rejected claim IDs | Remaining uncertainty |
|---|---|---|---|---|---|
| Rust/API/features/WGSL/resources | W1-A manifest plus W1-B PyO3 roots | cfg/features, public traits/APIs, layouts, shader assembly/entry points, resource ownership, tests/specs/CI and consumer seams | `VALIDATED` | N01, N02, A01-A05 locally proven; N03, C01, C04, C05-C07 rejected | final integration and exact-head validation remain pending; remote/platform proof is `NOT_PROVEN` |
| CENSOR | capability, degradation, resource tracker, certificate, shader registry and routed tests/workflows | execution-truth policy, allocation/budget, certificate/tamper, shader-use and probe-result consumers | `VALIDATED` | protected by R01-R04, R07, R13; adjudication pass ordering locally proven | production signing and remote release evidence remain `NOT_PROVEN` |
| SUTURA | MapScene, validation/render, recipe/bundle and tests | placeholder prohibition, compiled-plan/culling ownership, canonical round trip and diagnostics | `VALIDATED` | protected; W1B-01 does not change it | runtime proof after implementation is `NOT_PROVEN` |
| PyO3/stubs | `src/lib.rs`, PyO3 roots, `python/**`, API/install tests | registrations, root/runtime exports, signatures/defaults/properties, typing, reflection and installed-package seams | `VALIDATED` | W1B-01 locally proven; C03, C05, N03 rejected | local installed-wheel parity is proven; remote/platform parity is `NOT_PROVEN` |
| LIMES | vector coverage Rust/Python/WGSL/tests | opt-in/default behavior, cache and exact protected bindings | `VALIDATED` | C06 rejected | physical behavior remains `NOT_PROVEN` |
| VT/TESSELLA/SUBSTRATIA | terrain VT/visibility/clipmap paths, tests/scripts/goldens/certificates/workflows | store/residency/streaming/picking and fail-closed SHA/adapter/golden provenance | `VALIDATED` | protected by R01, R07, R13 | physical acceptance remains `NOT_PROVEN` |
| SIDERA/AETHER | astronomy/atmosphere production, oracle, assets and tests | numerical window, deterministic provenance, production/oracle independence | `VALIDATED` | AETHER consolidation rejected by R02 | backend-specific evidence remains `NOT_PROVEN` |
| Determinism/certificates | certificate/determinism code, fixtures, scripts and workflows | canonical inputs/outputs, signing/tamper and exact-identity boundaries | `VALIDATED` | protected by R01/R13; adjudication certificate order is locally proven as path trace then raster | production signing remains `NOT_PROVEN` |
| Unicode | generated Unicode source, corpora/provenance, label/text code and tests | generator ownership, version/provenance, shaping/behavior consumers | `VALIDATED` | generated source classified, no edit | post-change runtime proof is `NOT_PROVEN` |
| GIS/units | GIS/geo Rust, Python/stubs/tests/docs | CRS/coordinate/height/epoch/unit/raster/vector/narrowing semantics | `VALIDATED` | A04, A05 locally proven | focused candidate proof is complete; nonlocal/platform integration remains pending |
| Examples/recipes | examples, manifests, fixtures/goldens/certificates and tests | tracked runnable paths, parser/API/schema/canonical consumers | `VALIDATED` | W1C-01 locally proven; C02 rejected | four backend-dependent SUTURA cases and remote/physical behavior remain `NOT_PROVEN` |
| Docs/scripts/tools/CI/packaging | W1-C manifest | live policy/CI, package inclusion, referents, scripts/tools consumers and historical boundaries | `VALIDATED` | W1C-01 and W1C-02 locally proven | local Sphinx, sdist inventory and installed-wheel smoke are proven; remote packaging is `NOT_PROVEN` |
| Nonordinary paths | exact classification ledger below | provenance/manifests/consumers, explicit generated/historical/golden/certificate overrides | `VALIDATED` | R01, R13 | no content edit authorized or required |

## Finding register

Priority is dependency order plus contract necessity and regression risk; no
score or threshold is used.

| ID | priority basis | subsystem | path:symbol | claim | evidence | behavior/contract | smallest transformation | risk/dependencies | required proof | status | commit/PR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N01 | build provenance before package proof | build provenance | `build.rs:git_revision,main` | one helper can own both local Git revision queries | lines 51-76 repeat Git command/output parsing | valid 40-hex `GITHUB_SHA` precedence; 12-character short SHA; trim; failure to `unknown`; rerun variables | call unchanged `git_revision` with `["--short=12", "HEAD"]` and `["HEAD"]`; delete only duplicate full-SHA local ladder | provenance/certificate identity; I1 precedes integration | valid/invalid/absent env; Git success/failure/no-Git; provenance, certificate and ANAMNESIS tests | `LOCALLY_PROVEN` | `NOT_PROVEN` |
| N02 | manifest truth before locked builds | Cargo manifest | `Cargo.toml:[dev-dependencies]` | duplicate dev declarations add no target availability | normal and dev both declare `env_logger = "0.10"` and `sha2 = "0.10"`; build separately requires `sha2` | normal/dev/bench/bin/build availability, features and locked resolution unchanged | remove only the two dev-dependency lines; retain normal dependencies and build `sha2` | Cargo target semantics; I2 inspects but does not own `Cargo.lock` | metadata/tree before-after; unchanged `Cargo.lock`; locked build/test/bench | `LOCALLY_PROVEN` | `NOT_PROVEN` |
| A01 | independent build-system cleanup | CMake | `CMakeLists.txt` platform/build-type branches | five assignments have no reader | W1-A def-use inspection found unused three `RUST_LIB_EXT` and two `CARGO_BUILD_TYPE` assignments | all other variables, configure output/cache and target graph unchanged | delete exactly those five assignments | cross-platform CMake behavior; independent I3 | configure output/cache/target graph comparison | `LOCALLY_PROVEN` | `NOT_PROVEN` |
| A02 | exact local duplication with shared geometry ownership | geometry normals | `src/geometry/{mod.rs,displacement.rs,subdivision.rs}:recompute_normals` | byte-identical normal recomputation should have one geometry-private owner | identical 38-line bodies and five callsites | float accumulation/order, normalization and degenerate fallback unchanged | move the exact body to a geometry-private helper in `mod.rs`; replace only the five callsites | numeric/output drift; I4 owns all three files | pre/post triangle, degenerate and missing-normal characterization; focused Rust/Python geometry tests | `LOCALLY_PROVEN` | `NOT_PROVEN` |
| A03 | exact local duplication with overlay ownership | geometry overlay | `src/geometry/overlay/{mod.rs,sweep.rs,faces.rs}:point_on_segment` | byte-identical predicate should have one overlay-private owner | identical bodies, shared origin and three callsites | orientation/sign evaluation order and inclusive bounds unchanged | move exact body to overlay-private `mod.rs`; replace only three callsites | topology/order drift; I5 owns the three files | full overlay Rust/Python corpus, fuzz and order/source proof | `LOCALLY_PROVEN` | `NOT_PROVEN` |
| A04 | exact GIS constructor truth | GIS raster metadata | `src/gis/{mod.rs,domain.rs,rasterize.rs,thematic.rs}:synthetic_info` | three exact synthetic `RasterInfo` constructors should share GIS-private ownership | W1-A found exact constructors in the three consumers | memory driver, dimensions, dtype, band/nodata and every default field unchanged | add one `cfg(extension-module)` GIS-private helper in `mod.rs`; replace exact constructors | Python dict and cfg surface; shares disjoint I6 with A05 | array-only inputs through all three public surfaces and exact dict equality | `LOCALLY_PROVEN` | `NOT_PROVEN` |
| A05 | consolidate only proven WKT syntax truth | GIS CRS parsing | `src/gis/{crs.rs,raster_tags.rs,raster_write.rs}:looks_like_wkt,validate_wkt_structure` | common WKT token recognition and bracket structure should have one GIS-private owner | raster tags/write implementations share exact token/structure logic; writer has additional WGS84/IAU policy | token/case/whitespace/brackets and exact errors; writer's stricter WGS84/IAU validation remains local | move only common token/structure logic to `crs.rs`; keep writer policy/error branches | read/write and planetary semantics; shares I6 with A04 | token/case/whitespace/bracket matrix, planetary cases, read-versus-write behavior and exact errors | `LOCALLY_PROVEN` | `NOT_PROVEN` |
| W1B-01 | public runtime/stub parity | package root stub | `python/forge3d/__init__.pyi`; `tests/test_api_contracts.py` | root stub lacks four declarations already exported at runtime | runtime `__all__` contains `MaterialSet`, `SunPosition`, `sun_position`, `sun_position_utc`; `MaterialSet` is already an unresolved forward annotation | runtime identity, signatures, defaults, properties and exports unchanged | add accurate class/function declarations only and parity coverage; do not change runtime `__all__` | typing/API drift; I7 | `python3 -m pytest tests/test_api_contracts.py -v` before/after an exact-head installed build; inspect the four signatures/defaults/properties | `LOCALLY_PROVEN` | `NOT_PROVEN` |
| W1C-01 | documented runnable surface must match tracked truth | examples documentation | `docs/examples/index.md`; `docs/guides/data_and_scene_workflows.md`; `docs/guides/feature_map.md`; new `tests/test_example_catalog_docs.py` | index says complete while naming absent `examples/support/_png.py`, omitting tracked runnable examples, and routing Fuji to `viewer_ipc` instead of MapScene/LabelLayer | tracked example manifest plus current imports/tests/API consumers | no new capability claim; every named path exists; Fuji ownership remains MapScene/LabelLayer | reconcile only the three docs and add exact reference tests to tracked/current API truth | docs drift; I8 | `python3 -m pytest tests/test_example_catalog_docs.py tests/test_mapscene_sutura_integrity.py -v`; `python3 -m sphinx -b html docs docs/_build/html` | `LOCALLY_PROVEN` | `NOT_PROVEN` |
| W1C-02 | package manifest truth | source distribution | `MANIFEST.in`; new `tests/test_sdist_manifest.py` | manifest includes an absent/untracked `rust-toolchain.toml` | `git ls-files` and filesystem show no `rust-toolchain.toml`; no live policy requires it | all other sdist/wheel members unchanged | remove only `include rust-toolchain.toml` and add exact archive-member proof | archive drift; I9 | `python3 -m pytest tests/test_sdist_manifest.py -v`; build sdist/wheel with the repository's exact-head packaging route and smoke-install the wheel | `LOCALLY_PROVEN` | `NOT_PROVEN` |
| N03 | public/fallible semantics defeat whole-dependency substitution | synchronization | `Cargo.toml` plus ten direct-use Rust files recorded in the prior row | replace `once_cell` with std and remove dependency | rustc 1.90 reports E0658 for `OnceLock::get_or_try_init`; four fallible call paths and public concrete `once_cell` statics require current semantics | initialization/retry/public concrete types and project convention remain intact | none | partial conversion fragments convention and retains dependency | rejection is proven by compiler/API and all-use manifest evidence | `REJECTED` | `NOT_PROVEN` |
| C01 | no necessity for layout-risk abstraction | terrain offline renderer | `src/terrain/renderer/offline.rs` | extract descriptor helpers | W1-A found no contract gap; bindings/layout ownership raise regression risk | exact WGSL/pipeline layouts unchanged | none | layout risk without necessity | evidence-backed rejection; no execution proof required | `REJECTED` | `NOT_PROVEN` |
| C02 | no necessity for wider example ownership | example CLI | two terrain example CLIs | share eight CLI groups | W1-B found ownership expansion without a contract gap | parser/help/runtime behavior unchanged | none | coupling and parser drift | evidence-backed rejection | `REJECTED` | `NOT_PROVEN` |
| C03 | reachability is not proven | Python private helpers | exact five-helper set above | delete definition-only helpers | W1-B could not exclude dynamic/downstream reachability | public/dynamic/import behavior unchanged | none | destructive deletion under uncertainty | evidence-backed rejection | `REJECTED` | `NOT_PROVEN` |
| C04 | physical/evaluation equivalence unproven | shadow shader | `src/shaders/shadows.wgsl` | extract shader helpers/constants | W1-A did not prove evaluation-order or physical equivalence | bindings, operation/sample order and pixels unchanged | none | physical shader regression | evidence-backed rejection | `REJECTED` | `NOT_PROVEN` |
| C05 | borrow/error equivalence unproven | scene light PyO3 | point/spot update mutators | extract lookup/update helper | W1-B did not prove borrow and exact error preservation | public signatures, mutations and exceptions unchanged | none | PyO3 borrow/error drift | evidence-backed rejection | `REJECTED` | `NOT_PROVEN` |
| C06 | protected LIMES binding needs no abstraction | vector coverage | binning/raster/resolve layout helpers | consolidate protected helpers | W1-A found no necessary gap across protected binding ownership | LIMES opt-in/default and exact bindings unchanged | none | protected binding regression | evidence-backed rejection | `REJECTED` | `NOT_PROVEN` |
| C07 | broad diagnostics are style churn, not contract gaps | Rust tree | preliminary Clippy candidate families | apply native-idiom rewrites broadly | W1-A found no expression-by-expression necessity; counts are navigation only | evaluation order, overflow, errors and APIs unchanged | none | broad semantic/style churn | evidence-backed rejection | `REJECTED` | `NOT_PROVEN` |

## Dependency-ordered implementation waves

W1-A/B/C and independent W1-V are complete read-only evidence lanes. The
following implementation ownership is exact and pairwise disjoint; no executor
may edit another row's files. `Cargo.lock` is inspection-only for I2. I1-I9 are
now `LOCALLY_PROVEN` at the base-plus-uncommitted-diff candidate. Final
integration still requires applicable gates, whole-diff review, and an
exact-head ledger update.

| Wave | Claims | Exact owned files | Smallest transformation and preserved behavior | Exact proof requirement |
|---|---|---|---|---|
| I1 | N01 | `build.rs` | reuse `git_revision` for short and full local queries only; preserve env precedence, trim, rerun and `unknown` semantics | valid/invalid/absent env; Git success/failure/no-Git; provenance/certificate/ANAMNESIS tests |
| I2 | N02 | `Cargo.toml`; inspect-only `Cargo.lock` | remove only duplicate dev `env_logger` and `sha2`; retain normal/build declarations | metadata/tree before-after, unchanged lockfile, locked build/test/bench |
| I3 | A01 | `CMakeLists.txt` | delete only three unread `RUST_LIB_EXT` and two unread `CARGO_BUILD_TYPE` assignments | configure output/cache/target graph comparison |
| I4 | A02 | `src/geometry/mod.rs`, `src/geometry/displacement.rs`, `src/geometry/subdivision.rs` | move exact normal body to geometry-private owner and replace five calls; preserve arithmetic/order/fallback | triangle, degenerate, missing normals; focused Rust/Python geometry |
| I5 | A03 | `src/geometry/overlay/mod.rs`, `src/geometry/overlay/sweep.rs`, `src/geometry/overlay/faces.rs` | move exact predicate to overlay-private owner and replace three calls; preserve sign/order/inclusive bounds | full overlay Rust/Python corpus, fuzz and source/order proof |
| I6 | A04, A05 | `src/gis/mod.rs`, `src/gis/crs.rs`, `src/gis/domain.rs`, `src/gis/rasterize.rs`, `src/gis/thematic.rs`, `src/gis/raster_tags.rs`, `src/gis/raster_write.rs` | share exact synthetic-info constructor and only common WKT token/structure; preserve defaults, stricter writer policy and exact errors | three array-only public surfaces/exact dict; WKT token/case/space/bracket/planetary/read-write matrix |
| I7 | W1B-01 | `python/forge3d/__init__.pyi`; `tests/test_api_contracts.py` | add four accurate declarations and parity assertions only; runtime exports already complete | `python3 -m pytest tests/test_api_contracts.py -v` before/after exact-head installed build; inspect four signatures/defaults/properties |
| I8 | W1C-01 | `docs/examples/index.md`; `docs/guides/data_and_scene_workflows.md`; `docs/guides/feature_map.md`; new `tests/test_example_catalog_docs.py` | reconcile exact tracked examples/support paths and Fuji MapScene/LabelLayer ownership | `python3 -m pytest tests/test_example_catalog_docs.py tests/test_mapscene_sutura_integrity.py -v`; `python3 -m sphinx -b html docs docs/_build/html` |
| I9 | W1C-02 | `MANIFEST.in`; new `tests/test_sdist_manifest.py` | remove only absent `rust-toolchain.toml` include | `python3 -m pytest tests/test_sdist_manifest.py -v`; exact-head sdist/wheel member diff and wheel smoke-install |

Rollback boundaries are the exact owned-file sets above. No golden, certificate,
generated, historical, asset, corpus, fixture, dependency-version, public-API,
workflow, or unrelated formatting change is authorized. Applicable integration
proof remains F01-F09; unavailable remote/physical/signing evidence remains
`NOT_PROVEN`, never a local pass.

### W1 freeze decision

The exact manifests, per-path inspection mapping, contract-family coverage,
finding dispositions, proofs, and I1-I9 ownership were independently reconciled
at the unchanged base. The freeze is bound to `W1-A-REPORT` SHA256 `5acdb5d4...a589a`,
`W1-B-REPORT` `a2cff1e4...0e35`, `W1-C-REPORT` `68339946...e129`, and
`W1-V-REPORT` `52aee008...a312`; the appendix table records every full digest
and exact byte boundary. All ten accepted findings are `LOCALLY_PROVEN` at the
uncommitted local candidate. None is remote-proven. N03 and C01-C07 are
terminal `REJECTED`. No vague
`DISCOVERED`, `VALIDATED`, or `IN_PROGRESS`
finding remains; `VALIDATED` elsewhere describes identity, mapping, or
read-only evidence only.

## Rejected claim records

| ID | One-line decision | Status |
|---|---|---|
| R01 | Generated sources, fixtures, assets, corpora, binaries, ignored output, goldens, and certificates are classified evidence, not ordinary cleanup; no contract requires changing them. | `REJECTED` |
| R02 | AETHER's independent `aether_reference.rs` oracle must not be consolidated with production atmosphere code. | `REJECTED` |
| R03 | A cross-subsystem generic WGPU layout abstraction would widen ownership beyond the proven local duplicates. | `REJECTED` |
| R04 | Dependency upgrades, redesign, performance/correctness/security work, style churn, blind Clippy rewrites, and Clippy-quarantine edits are outside the refactor contract. | `REJECTED` |
| R05 | Public `ProbeBaker` removal would violate public-API preservation because it is re-exported. | `REJECTED` |
| R06 | Any C03 helper with uncertain dynamic, export, registration, history, or downstream reachability will remain untouched. | `REJECTED` |
| R07 | Any shader extraction that changes evaluation order, resource contracts or assembly, or demands a golden/certificate refresh, will remain untouched. | `REJECTED` |
| R08 | A later-round cleanup claim already visible in the audit is inadmissible under the live fuse. | `REJECTED` |
| C01 | Superseding the incorrectly routed preliminary rationale, W1-A found no necessary contract gap and unacceptable layout/binding risk. | `REJECTED` |
| C02 | Superseding the preliminary rationale, W1-B found ownership expansion and parser coupling without a necessary contract gap. | `REJECTED` |
| C03 | Superseding the preliminary rationale, W1-B could not prove dynamic and downstream reachability absent for destructive deletion. | `REJECTED` |
| C04 | Superseding the preliminary rationale, W1-A could not prove operation-order and physical-pixel equivalence. | `REJECTED` |
| C05 | Superseding the preliminary rationale, W1-B could not prove PyO3 borrow and exact error preservation. | `REJECTED` |
| C06 | Superseding the preliminary rationale, W1-A found no necessity for abstraction across protected LIMES binding ownership. | `REJECTED` |
| C07 | Superseding the preliminary rationale, W1-A found broad diagnostic rewrites to be style churn without expression-level contract necessity. | `REJECTED` |
| N03 | rustc 1.90 rejects fallible `OnceLock::get_or_try_init` with E0658; four fallible paths and public concrete `once_cell` statics mean partial conversion fragments convention and cannot remove the dependency. | `REJECTED` |
| R09 | CMake integration is preserved; no correctly routed evidence proves it is redundant or authorizes removal. | `REJECTED` |
| R10 | Public `QueueFenceExt` removal is rejected because public-API preservation forbids deleting it without an explicit contract change. | `REJECTED` |
| R11 | Public `ProbeBaker` removal remains rejected because it is re-exported and public-API preservation governs this refactor. | `REJECTED` |
| R12 | Shader consolidation remains rejected unless W1 produces equivalence and ownership evidence; textual similarity is not implementation authority. | `REJECTED` |
| R13 | Assets, generated files, fixtures, corpora, goldens, certificates, historical material, and locked-feature inventory are coverage/provenance evidence, not ordinary cleanup. | `REJECTED` |

## Append-only checkpoint log

Do not edit or delete existing rows. Append corrections as later rows.

| Timestamp | IDs | Files | Before/after facts | Tests/live runs and identities | Review findings | Commit SHA | Residual risk | Next necessary claim |
|---|---|---|---|---|---|---|---|---|
| 2026-08-12T13:40:41+02:00 | T0 | `docs/refactor-forge3d.md` only | before: exact base, clean isolated worktree, primary dirty snapshot captured; after: bootstrap ledger created | identity, policy, manifest, CI, toolchain and tracked-surface inspection at `f5db54f9`; source tests not run | reviewer not yet run | `NOT_PROVEN` | build tools unavailable; source, runtime, remote and physical behavior not yet proven | independent T0 ledger review, then Wave A |
| 2026-08-12T13:46:05+02:00 | T0 self-check | `docs/refactor-forge3d.md` only | isolated HEAD/branch unchanged; status contains only this untracked ledger; required headings/schema and C01-C07 present; primary status byte-for-byte matches snapshot | `git rev-parse HEAD`; branch/status/name checks; `git diff --no-index --check /dev/null docs/refactor-forge3d.md` emitted no whitespace errors (exit 1 denotes an untracked-file difference); heading/status/schema `rg`; no source or runtime tests | reviewer found incomplete Phase 1 coverage, premature C03 validation/path error, and premature build-tool blocker | `NOT_PROVEN` | untracked-file diff is outside ordinary `git diff` until staged; content inspected directly | correct findings, then re-review T0 |
| 2026-08-12T13:52:40+02:00 | T0 correction | `docs/refactor-forge3d.md` only | added mandatory read-only Wave 1 before claim freeze/implementation; corrected C03 symbol ownership and status; replaced PATH-only tool conclusion with verified Python module versions | `python3 -m maturin --version` = `maturin 1.14.1`; `python3 -m ninja --version` = `1.11.1.git.kitware.jobserver-1` (Ninja 1.11.1); exact authorized build environment not yet executed | corrections address the first T0 review; re-review required | `NOT_PROVEN` | exhaustive coverage, source/runtime behavior and remote/physical proof remain outstanding | independent T0 re-review, then Wave 1 |
| 2026-08-12T13:54:46+02:00 | T0 correction 2 | `docs/refactor-forge3d.md` only | replaced final/integration claim-set assumptions with all Wave 1 accepted claims; made preliminary C04/C06/C07 execution conditional on Wave 1 acceptance; corrected audit boundary wording | ledger-only review correction; no source/runtime test | second reviewer blocker addressed; re-review required | `NOT_PROVEN` | Wave 1 has not run, so the implementation set remains unfrozen | independent T0 re-review, then Wave 1 |
| 2026-08-12T14:09:38+02:00 | L0 corrected-audit reconciliation | `docs/refactor-forge3d.md` only | correctly routed audit supersedes preliminary research; retained C01-C07 descriptions but rejected every claim and the up-to-450-line estimate; added conditional N01-N03 and conservative non-gate evidence of 12 fewer lines/one fewer direct dependency; replaced implementation waves with read-only W1-A/B/C, reviewer, and ledger freeze | ledger/source inspection at exact unchanged `f5db54f9`; no source/runtime/network command, commit, push, or PR | prior C01-C07 research was incorrectly routed and cannot authorize implementation; W1 evidence and reviewer approval remain required | `NOT_PROVEN` | N01-N03 are `DISCOVERED`, conditional, and unfrozen; exhaustive tracked-surface coverage and runtime proof have not run | orchestrator derives/reconciles W1 manifests, executes W1-A/B/C read-only coverage, obtains review, then freezes ledger |
| 2026-08-12T14:51:08+02:00 | W1-F freeze | `docs/refactor-forge3d.md` only | W1-V reconciled exact A/B/C manifests and all 2,793 classified paths; W1-A/B/C contract evidence freezes N01, N02, A01-A05, W1B-01, W1C-01, W1C-02 as `PLANNED`; N03 and C01-C07 are evidence-backed `REJECTED`; I1-I9 ownership is disjoint | exact base/head/branch/status; manifest count/hash/union/intersection regeneration; classification row/uniqueness/first-column/hash regeneration; ledger schema/status/whitespace and primary-status checks; no source test/build/network command | W1-V approved manifest, coverage, dispositions, proof conditions, and ownership; preliminary rejection rationales superseded by evidence | `NOT_PROVEN` | implementation, local runtime/build, remote, signing and physical proof have not run | separately authorize and execute I1-I9 with per-task review |
| 2026-08-12T15:07:02+02:00 | W1-F-R correction | `docs/refactor-forge3d.md`; read-only `W1-A-REPORT`, `W1-B-REPORT`, `W1-C-REPORT`, `W1-V-REPORT` appendices | persisted and byte-bound all four complete owner/verifier reports; corrected I7 to `python/forge3d/__init__.pyi` plus `tests/test_api_contracts.py`, I8 to the three exact docs plus new `tests/test_example_catalog_docs.py`, and I9 to `MANIFEST.in` plus new `tests/test_sdist_manifest.py`; all I1-I9 ownership intersections are empty; corrected W1-V artifact is 330 lines/14,910 bytes/SHA256 `52aee00821771c24d2c72cd92180725c5d63242c6b27218ac6034c34bc34a312` | appendix base/section/final-LF/size/SHA256 checks; manifest/classification regeneration; exact ownership union/intersection audit; finding schema/status, whitespace, exact head/branch/status, changed-path and primary-snapshot checks; no source test/build/network command | resolves W1-F-R report-persistence and provisional-ownership blockers; freeze cites full report hashes and byte delimiters | `NOT_PROVEN` | implementation, runtime/build, remote, signing and physical proof remain unrun | separately authorize and execute the now-exact I1-I9 plan with per-task review |
| 2026-08-12T15:53:08+02:00 | I1 / N01 | `build.rs` | 5 insertions/14 deletions; helper accepts an argument slice and is called exactly as `rev-parse --short=12 HEAD` and `rev-parse HEAD`; valid environment SHA skips the full query; HEAD commit remains `f5db54f9` plus uncommitted diff | pre/post controlled probes passed valid 40-hex `GITHUB_SHA`, invalid/absent fallback, trimmed output, nonzero status, invalid UTF-8, and missing Git; `cargo fmt --check`, diff check, `cargo check --locked --lib` passed; provenance 10/10, certificate 7/7, ANAMNESIS 12/12, zero skips; direct CPU build-script proof, no candidate wheel required | reviewer `APPROVED` at exact base; no blocker | `NOT_PROVEN` | remote/platform matrix not run; local build-script contract is proven | I7 / W1B-01 after remaining locally proven I1-I6 records are integrated |
| 2026-08-12T15:53:08+02:00 | I2 / N02 | `Cargo.toml`; `Cargo.lock` inspected only | exactly 2 deletions; normal `env_logger`/`sha2` and build `sha2` remain; metadata stays 353 nodes/10 targets with identical package/edge/target digests; `Cargo.lock` SHA256 `152339c2ddae5195920068029940c163119bcd19e5cc9f3bc82107f7f43b2313` and Git blob `dfefdfd59fd23719074064b66b42a3545530ec8d` unchanged; HEAD remains base plus uncommitted diff | pre/post `cargo check --locked --workspace --lib --bins --tests --benches --features async_readback` passed; pre/post `cargo test --locked --workspace --no-run --features async_readback` passed; post bench `f3dz` no-run, format and diff checks passed; no candidate wheel required | reviewer `APPROVED`; no blocker | `NOT_PROVEN` | runtime/full/Clippy remain deferred to integration; structural build proof complete | I7 / W1B-01 after remaining locally proven I1-I6 records are integrated |
| 2026-08-12T15:53:08+02:00 | I3 / A01 | `CMakeLists.txt` | 0 insertions/5 deletions: exactly three `RUST_LIB_EXT` and two `CARGO_BUILD_TYPE` assignments; no remaining references; HEAD remains base plus uncommitted diff | Debug and Release Unix Makefiles configure with optional targets enabled passed; normalized output, full cache, and target inventory were byte-identical; diff check passed; no candidate wheel required; Ninja generator unavailable and not treated as a pass | reviewer `APPROVED`; no blocker | `NOT_PROVEN` | cross-platform remote CMake was not run | I7 / W1B-01 after remaining locally proven I1-I6 records are integrated |
| 2026-08-12T15:53:08+02:00 | I4 / A02 | `src/geometry/mod.rs`; `src/geometry/displacement.rs`; `src/geometry/subdivision.rs` | 41 insertions/80 deletions; exact moved body SHA256 `9ceaaa7e74ca8e158d95cce876743b2650d5269d072cd514e124504fee1a48c5`; five callsites; HEAD remains base plus uncommitted diff | pre/post temporary harness produced byte/f32-bit equality for triangle/missing normals, degenerate, heightmap, procedural and subdivision cases; locked library checks with default and extension features passed; `cargo test --locked --lib geometry::` passed 63/63, zero failed/ignored; preinstalled Python API registration check passed 19 with 427 deselected, but was not candidate-runtime proof; format/diff passed; no candidate wheel required because no export changed and native bit-exact plus extension compile proved the claim | reviewer `APPROVED`; no blocker | `NOT_PROVEN` | nonlocal/platform integration remains; Python candidate runtime was not claimed | I7 / W1B-01 after remaining locally proven I1-I6 records are integrated |
| 2026-08-12T15:53:08+02:00 | I5 / A03 | `src/geometry/overlay/mod.rs`; `src/geometry/overlay/sweep.rs`; `src/geometry/overlay/faces.rs` | 17 insertions/29 deletions; moved predicate SHA256 `427f343458d2f7b278114de2659bb2392d7e42503d1d740d96a21f3174f743d9`; three callsites; HEAD remains base plus uncommitted diff | baseline/post Rust overlay passed 6/6; baseline/post Python passed 8/8; locked candidate wheel under `/private/tmp/forge3d-i5-wheel.3U6BP5` was temporarily installed and candidate Python passed 8/8; zero skips; format/diff passed | reviewer `APPROVED`; no blocker | `NOT_PROVEN` | only nonlocal/platform integration remains | I7 / W1B-01 after I6 record integration |
| 2026-08-12T15:53:08+02:00 | I6 / A04+A05 | `src/gis/mod.rs`; `src/gis/crs.rs`; `src/gis/domain.rs`; `src/gis/rasterize.rs`; `src/gis/thematic.rs`; `src/gis/raster_tags.rs`; `src/gis/raster_write.rs` | 84 insertions/116 deletions; shared exact synthetic metadata and WKT lexical/structural logic only; stricter writer policy retained; HEAD remains `f5db54f9` plus the complete uncommitted I1-I6 diff | `cargo test --locked --lib gis::` passed 68/68; extension check passed; candidate venv `/private/tmp/forge3d-i6-venv-019ff5b5`, `python3 -m maturin develop` passed; focused Python passed 150 with one skip: unrelated loopback bind `EPERM` at `tests/_loopback.py:19`; raw-ndarray `prepare_dem`/`mask_raster`/`normalize_raster`, exact non-WGS Earth rejection, and WKT/WGS/IAU cases passed; format/diff passed | reviewer independently reran the same proof and `APPROVED`; no blocker | `NOT_PROVEN` | quoted-bracket behavior is intentionally retained by the structural validator; physical/nonlocal proof is irrelevant to these CPU contracts | I7 / W1B-01 |
| 2026-08-12T20:01:39+02:00 | I7 / W1B-01 | `python/forge3d/__init__.pyi`; `tests/test_api_contracts.py` | exactly 188 insertions/0 deletions across two files; only four already-runtime-exported declarations and exact parity assertions added; locked wheel retained at `target/wheels/forge3d-1.34.0-cp310-abi3-macosx_11_0_arm64.whl`, SHA256 `30199e3ea1ecd9346dc825ac87562fb6b6124c6a9fa193754536a1bada1037fa`; installed stub/native hashes match that wheel | candidate installed-wheel API contracts passed 441 with seven explicit GPU-only skips; installed-wheel smoke passed 14/14; strict mypy passed; live candidate reports `enumerate_adapters() == []`, `device_probe().status == "no_adapter"`, and `has_gpu() == False`; diff/whitespace checks passed | reviewer `APPROVED`; authoritative ledger advisor independently reproduced the focused, install, typing, wheel-content and backend-visibility evidence; no blocker | `NOT_PROVEN` | remote/platform and physical GPU behavior remain unrun; all seven GPU-dependent cases remain explicitly skipped, not passed | I8 / W1C-01 |
| 2026-08-12T20:01:39+02:00 | I8 / W1C-01 | `docs/examples/index.md`; `docs/guides/data_and_scene_workflows.md`; `docs/guides/feature_map.md`; `tests/test_example_catalog_docs.py` | exactly 226 insertions/112 deletions across four files; catalog exactly covers 46 tracked examples as 39 runnable scripts + three notebooks + four support files; staged test blob `0f13de160d34067053f7b646f48aae98bd1b3b4d`; HEAD remains base plus the complete uncommitted I1-I9 diff | focused catalog/SUTURA proof passed 18 with four explicit native-terrain-backend skips; Sphinx exited 0 with 94 unrelated warnings and no warning in the three edited docs; diff/whitespace checks passed | reviewer `APPROVED` after focused re-review; authoritative ledger advisor independently reproduced the focused and Sphinx proof and exact catalog partition; no blocker | `NOT_PROVEN` | the four backend-dependent cases remain explicitly skipped; remote/physical docs behavior remains unrun; existing unrelated Sphinx warnings were not expanded into this claim | I9 / W1C-02 |
| 2026-08-12T20:01:39+02:00 | I9 / W1C-02 | `MANIFEST.in`; `tests/test_sdist_manifest.py` | exactly one manifest deletion plus one new 55-line test; base and post sdists each contain 2,629 members with byte-identical sorted inventories, SHA256 `3c33fb12389bfc4bf689b9d350eca54c2f368c815ff3c53c8e1803e32464b418`; staged test blob `ac7df5bbe84fd1f07a64e05909a130d34c3124ca`; only absent `rust-toolchain.toml` include removed | focused sdist proof passed 1/1; retained base/post reviewer archives independently re-counted and compared; installed-wheel smoke passed 14/14; complete source/test/docs diff is 24 files, 616 insertions/359 deletions, with `git diff HEAD --binary` SHA256 `a3110d4e44712476890e668bd2d60a029a89f425f83493e4065246ee4c9e1b30`; diff/whitespace checks passed | reviewer `APPROVED`; authoritative ledger advisor independently reproduced the focused proof, inventory equality, installed smoke, staged blob and full-diff identity; no blocker | `NOT_PROVEN` | remote/platform package builds remain unrun; no physical claim applies | final integration gates and whole-diff reviews |
| 2026-08-12T20:09:36+02:00 | I7-I9 ledger checkpoint | `docs/refactor-forge3d.md` only | corrected stale current-state scope, authority, completion, metrics, and coverage summaries from the historical W1 freeze to the locally proven I1-I9 candidate; historical log rows remain unchanged; I7 is 188 insertions/0 deletions, I8 is 226/112, I9 is 55/1, and the 24-file source/test/docs diff remains 616/359 with SHA256 `a3110d4e44712476890e668bd2d60a029a89f425f83493e4065246ee4c9e1b30` | independently reran I7 441 passed/7 GPU-only skips, I8 18 passed/4 native-backend skips, and I9 1 passed; Sphinx exited 0 with the same 94 unrelated warnings and none in the three I8 docs; wheel, staged blobs, two byte-identical 2,629-member sdist inventories, finding schema/status, whitespace, branch/head/status, and primary-checkout snapshot reproduced | blind Standards and Spec reviewers `APPROVED`; requested route was `gpt-5.6-sol:xhigh`, while runtime model identity was not exposed for independent confirmation | `NOT_PROVEN` | final integration, whole-diff/ponytail review, remote/platform, signing, and physical evidence remain pending or `NOT_PROVEN`; skips remain skips | final integration gates and whole-diff reviews |
| 2026-08-12T20:34:37+02:00 | Final-review and executor-routing checkpoint | `docs/refactor-forge3d.md` only; all 29 reviewed paths staged | before this append, the staged and full binary diffs were byte-identical at SHA256 `1c697d289c034fd2de719410993a0ddafebdee0a64c5e05415a54b0bdfaf5b83`, with no unstaged path; excluding this self-describing ledger, the exact 28-path staged and full payloads remain byte-identical at SHA256 `90c1166286469c38767264635a96aea75a73ed90d0bad6ee6de01db30782ef22`; the staging guard is therefore closed by exactly 29 staged paths | no integration/local final gate was executed at this checkpoint; task-execution integration dispatch was attempted twice with exact routing `fork_turns=none`, `model=gpt-5.6-luna`, `reasoning_effort=max`, and the platform rejected both with `Unknown model`, exposing only `gpt-5.6-sol` and `gpt-5.6-terra` as available; the advisor route was requested as `gpt-5.6-sol:xhigh`, but runtime model identity is not exposed for independent confirmation | whole-diff content review found the content sound apart from the now-closed staging guard; the visibility claim was `REJECTED` by xhigh MSW/fuse adjudication because the frozen `pub(super)`/`pub(crate)` items are internal/nonpublic, no contract necessity was proved, and the claim was late; ponytail review returned exactly `Lean already. Ship.`; mandatory code-review Standards reported zero hard violations and one judgement-only Mysterious Name observation at `tests/test_api_contracts.py:2095-2097` for local aliases `f`, `fd`, and `i`, rejected by xhigh adjudication because they are local tuple templates within one expected-signature map, change no behavior or proof, the exact I7 hunk was already approved, and the late-round fuse applies; mandatory code-review Spec alleged one blocker at `docs/guides/feature_map.md:61` against line 15, rejected by xhigh adjudication because line 15 truthfully says `No dedicated tracked runnable example`, line 61 is unchanged broad vocabulary that names, links, and recommends no nonexistent runnable path, exact path truth is enforced by `tests/test_example_catalog_docs.py:51-75`, and the issue was visible during I8 approval so the late-round fuse applies | `NOT_PROVEN` | integration/local final gates and a commit remain `PLANNED`; remote publication, PR identity, CI, exact-head physical/platform evidence, and mergeability remain `NOT_PROVEN`; no commit or push was performed | reroute the necessary integration execution through an available authoritative executor, then run applicable local final gates before any commit or publication |
| 2026-08-13T14:30:16+02:00 | Post-I9 physical remediation and ledger recovery | physical remediation in `src/path_tracing/{adjudication.rs,reference_scene.rs}`, `src/py_functions/adjudication.rs`, four `src/shaders/pt_*.wgsl` consumers, affected atmosphere/certificate tests and this ledger | the isolated worktree was accidentally deleted and recovered with the staged candidate intact; pre-ledger state is exactly 42 staged paths, zero unstaged paths, staged binary-diff SHA256 `01b6f07ee3651770f31e2a7d3164ac6b91537751308bc8ee71eb10ab6089a223`; final wheel SHA256 `b302be2f97e445b3d64f5403f4fe23340dc963994f9778d5ef03b941de0cbd42` | timing baseline wheel/native `26fd8358`/`f403580c` failed at frame 0 after one iteration; fixed timing wheel/native `f631ff83`/`afe7b381` completed multibounce and 451 API/certificate tests passed, with exact certificate order `adjudication.path_trace` (`gpu_ms=0.0`, `draw_calls=spp`) then `adjudication.raster`; Naga regression was red before and green after all four WGSL `Sphere` consumers gained explicit `_pad1: array<f32, 3>` at offset 68, size 12, ending at host stride 80; the physical trace showed a valid material-3 hit with throughput 1 but runtime albedo 0, roughness 0, emissive `[0,.42,.42]` and immediate matching accumulation from the Metal stride-68 mismatch; stride-fixed wheel `9acffe...` changed the fast pixel from `[0,148,148]` to `[149,154,163]` and produced dE pass `0.400993`, shadow SSIM `0.989860`; a 64x64/512-spp serial-shadow differential improved dE pass `0.340159840` to `0.988557214`, proving the remaining lost update; the fixed-slot implementation then produced two byte-identical fast physical runs with dE pass `0.988557214`, mean dE `0.328809`, p95 dE `1.037433`; exact installed-wheel `FORGE3D_TEST_INSTALLED_WHEEL=1 FORGE3D_NO_BOOTSTRAP=1 python -m pytest tests/test_adjudication_gate.py::test_adjudication_gate` on Apple M4 Metal passed with dE pass `0.994660`, shadow SSIM `0.995548`, 1 passed/0 skipped in 358.73 s | timing/certificate, Sphere-stride, and portable fixed-slot reviewers each `APPROVED` with no findings; the ABI-changing provisional path and CAS/deterministic provisional shadow attempts were rejected and are not part of the candidate; this ledger task was explicitly routed `gpt-5.6-sol:medium` with `fork_turns=none`, no delegation, while runtime model identity is not exposed for independent confirmation | `NOT_PROVEN` | the physical Apple M4 Metal adjudication gate is locally proven; final integration remains `PLANNED` while its executor is active; remote/NVIDIA/Vulkan/signing, commit, push, PR, CI and mergeability remain `NOT_PROVEN` | wait for the integration executor; reconcile its exact result before any commit or publication |
| 2026-08-13T15:48:36+02:00 | Final integration, parity adjudication and static-contract correction reconciliation | integration evidence plus reviewed nine-file static-contract fixes; this row changes only `docs/refactor-forge3d.md` | immediately before this ledger edit the candidate was exact base `f5db54f95d202681f95dad649162d18efdae8987` plus exactly 51 staged paths and zero unstaged paths; staged binary-diff SHA256 `f9ee7adf0f7075f618b577a95069e86a13e7192cf9e27f63b7130128102ba6a7`; staged diff 5,104 insertions/462 deletions | integration `cargo fmt --check`, `cargo forge3d-clippy`, the applicable Rust commands and the fast Python lane passed on the pre-static-fix candidate; installed-wheel focused evidence passed adjudication 14/14, AETHER 71/71 and CARTOGRAPHER-PRIME 16/16, all with zero failures/skips; broad full Python executed 4,170 outcomes: 4,100 passed and 70 failed, then serial `--lf` reproduced 54 failures and 16 passes; exact-base comparison and state-reset controls classified the broad failures as baseline parity/state contamination rather than refactor regressions, without converting failures into passes; the nine-file static contract set was red 11 before its fixes and green 22 after | the static-contract fixes were reviewed and approved; this reconciliation was explicitly routed `gpt-5.6-sol:medium` with `fork_turns=none` and no delegation, while runtime model identity is not exposed for independent confirmation | `NOT_PROVEN` | the broad full Python and full/slow acceptance suites are not green; the integration and focused passes predate the later static-contract fixes, so current exact-candidate final-gate reproof remains `PLANNED`; commit, push, PR, hosted CI, mergeability, NVIDIA/Vulkan, signing and other remote/platform evidence remain `NOT_PROVEN` | rerun the applicable final gates against the current exact candidate, then reconcile the result before any commit or publication |
| 2026-08-13T16:01:54+02:00 | Current-source final-gate reproof reconciliation | fresh release-LTO wheel and current source evidence; this row changes only `docs/refactor-forge3d.md` | immediately before this ledger edit the exact candidate was the base plus 51 staged paths, zero unstaged paths, and staged binary-diff SHA256 `9ae70342541a6257ea2523f69e501aae847938572bac613b92b585a7a0e8574a`; fresh release-LTO wheel SHA256 `b0b1ef2c...` and installed native SHA256 `2365c2ff...` predate only this ledger's documentation bytes; the primary-checkout snapshot remained unchanged | current `cargo forge3d-clippy` and `cargo forge3d-clippy-acceptance` passed; Fast passed 642 with 28 policy skips and 0 failures; the affected complete-file suite passed 29 with 19 policy GPU skips and 0 failures; exact physical Apple M4 Metal adjudication passed 1/1 with 0 skipped in 352.45 s and its JUnit zero-skip check passed | source/runtime evidence closes the prior current-candidate reproof gap; this reconciliation was explicitly routed `gpt-5.6-sol:medium` with `fork_turns=none` and no delegation, while runtime model identity is not exposed for independent confirmation | `NOT_PROVEN` | broad full Python remains red as previously recorded and slow acceptance remains `NOT_PROVEN`; commit, push, PR, hosted CI, mergeability, NVIDIA/Vulkan, signing and other remote/platform evidence remain `NOT_PROVEN` | review and stage this ledger-only reconciliation before any commit or publication |
| 2026-08-14T22:16:23+02:00 | Exact-candidate publication handoff | `docs/refactor-forge3d.md` only; source/test payload unchanged | immediately before this ledger edit the 73-path index was staged tree object `00b574f9899875ed7e4b9b5b8b9313f89e10c1e7` (a tree object, not a commit SHA) on base `f5db54f95d202681f95dad649162d18efdae8987`; its locked release-LTO wheel SHA256 is `d4300727425ba645d2bc62a88582957a8de2acd2fa3c81457187c17d675c4022` and the isolated installed native SHA256 is `e6a77c5fc917951c95a49a48c4cb4ff63cc4087f99e99e5e273486bd469440a1` | green: `cargo fmt --check` (1.62 s), `cargo forge3d-clippy` (37.95 s), authoritative portable-feature `cargo check` (10.37 s), serial Rust workspace (151.91 s: main library 1,396 passed, 0 failed, three repository-authorized ignored and four explicitly CI-filtered; auxiliary binaries, benches and integration green), explicit doctests (12 passed, six repository-authorized ignored in 3.65 s), `cargo forge3d-clippy-acceptance` (19.22 s), locked release-LTO wheel build (206.58 s), and isolated installed-wheel smoke/license (29 passed in 1.35 s); from the repository root, exact `WGPU_BACKENDS=metal FORGE3D_NO_BOOTSTRAP=1 FORGE3D_TEST_INSTALLED_WHEEL=1 MPLCONFIGDIR=/private/tmp/forge3d-exact-candidate-r2.5Fumt8/mpl /usr/bin/time -p /private/tmp/forge3d-exact-candidate-r2.5Fumt8/venv/bin/python scripts/ci_pytest_lane.py --profile full -v --tb=short` failed `tests/test_cam_phi_wiring.py::test_cam_phi_changes_output` at 23% after about 818 s; the diagnostic variant ending `--profile full --maxfail=1 -q --tb=short` passed that node but failed `tests/test_determinism_hash.py::test_intra_backend_bit_identity` at 29% after 778.33 s, first `3e1cd11741884b7fcee36be0adac73da141e72ee385952a41d4fbd1edddbce5f`, second `58ddaf202100cdcaaede7d0477a71196f84457a1a109bcfc8b9cfdd92630a58a`; the prior fixed-wheel first-render/adjudication evidence remains recorded in the 2026-08-13T14:30:16 row and is not reclassified as current full-profile proof | user directed draft publication now so another Mac can continue; publication is a handoff with known red and unproven gates, not a PR-ready or merge-ready finding | `NOT_PROVEN` | full non-slow is red; the required slow profile and remaining dedicated physical/platform, NVIDIA/Vulkan, signing, hosted CI, PR-head and mergeability lanes remain `NOT_PROVEN` | publish only as the user-directed draft handoff; on the other Mac reproduce, minimize and clear both ordered Metal failures, then run the required slow and dedicated physical/platform lanes against the resulting exact PR head |

## Historical final-proof checkpoint (superseded 2026-08-25)

| Proof item | Exact final evidence | Status |
|---|---|---|
| Final branch/head | branch named above; final SHA not yet created | `NOT_PROVEN` |
| Claim-to-commit map | no refactor commits yet | `NOT_PROVEN` |
| Before/after structural facts | baseline and current 51-path staged candidate recorded; final committed facts unavailable | `NOT_PROVEN` |
| Local validation | current-source Clippy and acceptance, Fast, affected complete-file, release-LTO wheel and exact installed-wheel Apple M4 Metal adjudication evidence are locally proven; broad full Python remains red and slow acceptance remains `NOT_PROVEN` | `LOCALLY_PROVEN` |
| Review closure | I1-I9 per-change, timing/certificate, Sphere-stride, fixed-slot and static-contract-fix reviews approved; prior whole-diff, ponytail, Standards and Spec reviews completed against their recorded candidates; current exact-candidate final review remains pending | `PLANNED` |
| PR identity/head/checks/mergeability | no PR yet | `NOT_PROVEN` |
| Primary checkout preservation | post-ledger porcelain output matches the recorded pre-work branch, head, tracked modifications, and untracked paths exactly | `VALIDATED` |

## Historical residual-risk checkpoint (superseded 2026-08-25)

- Dynamic Python access and unknown downstream consumers cannot be inferred from
  a definition-only grep; C03 requires per-symbol evidence and preserves any
  uncertain helper.
- The exact installed-wheel adjudication gate is physically proven only on the
  recorded Apple M4 Metal adapter. It cannot prove hosted platform wheels,
  NVIDIA/Vulkan, another adapter/backend, production signing, or full physical
  acceptance; those remain `NOT_PROVEN` until exact-SHA authoritative runs.
- Naga and unit proof alone did not establish pixel behavior; the recorded
  installed-wheel Metal gate supplies physical proof for adjudication only. No
  golden or certificate refresh is authorized for this refactor.
- The broad full Python run is not green: 4,100 tests passed and 70 failed, and
  serial `--lf` reproduced 54 failures with 16 passes. Exact-base parity and
  state-reset controls classify those failures as baseline parity/state
  contamination rather than refactor regressions, but classification is not a
  passing full-suite result. Full/slow acceptance remains `NOT_PROVEN`.
- The repository-wide audit is bounded by exhaustive coverage of tracked
  product-relevant surfaces and their cross-language/runtime/build/documentation
  seams. Preliminary claim families do not bound or freeze that coverage. A
  metric, file length, diagnostic count, or attractive cleanup is not an
  accepted claim by itself.

## PR #170 continuation evidence (2026-08-26)

This section supersedes every earlier provisional current-state summary. It does
not rewrite historical checkpoints, but no earlier use of “current candidate”
or “current evidence” remains authoritative for this continuation.

### Current candidate and stable source identity

| Fact | Exact continuation evidence | Status |
|---|---|---|
| Branch | `codex/refactor-forge3d-20260812` | `VALIDATED` |
| Exact comparison base | `92cf80d20d7d5c6e9a564b853e79d596f3f5088f` | `VALIDATED` |
| Current committed predecessor | HEAD `57dd32830549cc499699256afa368a166ab1436b`; tree `8fcfbdf0be1f6d7b73dd3f1e25263cc2e2a97cff` | `VALIDATED_PREDECESSOR` |
| Live candidate | the committed predecessor plus four modified non-ledger paths and this modified ledger | `VALIDATED` |
| Dirty paths and porcelain status | ` M .github/workflows/ci.yml`<br>` M docs/ci-validation.md`<br>` M docs/gallery/index.md`<br>` M docs/refactor-forge3d.md`<br>` M tests/test_ci_cost_controls.py` | `VALIDATED` |
| Current full base delta | 255 paths and 255 unique paths versus the exact comparison base: 64 `A`, 191 `M`, no deletion or rename | `VALIDATED` |
| Canonical changed-path list | `git diff --name-only 92cf80d20d7d5c6e9a564b853e79d596f3f5088f`, newline after every path including the last; SHA-256 `0e45d47a7a1b783f4dcf17040427966819c10bb795694d9706e7283df9a035b6` | `VALIDATED` |
| Canonical non-ledger source manifest | 254 rows; SHA-256 `616a022b8e5e74fb728b04f5094eca15cbe581c414fe5cbb79c1697c373cb088` | `VALIDATED` |
| Final committed HEAD and tree | no post-change commit exists | `PENDING / NOT_PROVEN` |

The canonical non-ledger manifest makes the implementation/source identity
stable when this tracked ledger changes. Its exact UTF-8 serialization is one
line per path in Git diff order:

`status<TAB>path<TAB>SHA256(current file bytes)<LF>`

It excludes only `docs/refactor-forge3d.md` from the 255-path delta. The
verified command is:

```bash
git diff --name-status 92cf80d20d7d5c6e9a564b853e79d596f3f5088f -- . ':(exclude)docs/refactor-forge3d.md' |
while IFS=$'\t' read -r diff_state file_path; do
  digest="$(shasum -a 256 "$file_path" | cut -d ' ' -f 1)"
  printf '%s\t%s\t%s\n' "$diff_state" "$file_path" "$digest"
done
```

Hashing that stdout, including its final newline, produced the 254-row manifest
SHA-256 above. A second enumeration proved 254 unique paths, no missing path,
and no `D` or rename status. The 255-path mapping below independently binds
every delta path to its claim, owner, and proof category.

### Diagnostic predecessor evidence only

The following values are preserved because they diagnose the clean `57dd3283`
candidate. They are not current-candidate, final-commit, or hosted acceptance
evidence: the two-file workflow-parser fix and this documentation reconciliation
changed tracked files after they were produced.

| Predecessor fact | Recorded value | Status |
|---|---|---|
| Commit and tree | `57dd32830549cc499699256afa368a166ab1436b`; `8fcfbdf0be1f6d7b73dd3f1e25263cc2e2a97cff` | `DIAGNOSTIC_PREDECESSOR` |
| Evidence root and index | `target/pr170-exact-final-20260826-evidence/`; `MATRIX-RESULTS.md`; `ACCEPTANCE-MANIFEST.tsv`; `SHA256SUMS` | `DIAGNOSTIC_PREDECESSOR` |
| Evidence indexes | matrix SHA-256 `2c7f11f60c2ab130ce0ca4ac61da18e61c9524ab7c43133dc908c77c68d70086`; manifest SHA-256 `1e00a77a41d2dc35ceca0279a9d431f782f32525886afdcf1713e25507704d07` | `DIAGNOSTIC_PREDECESSOR` |
| Wheel | `wheelhouse/forge3d-1.35.0-cp310-abi3-macosx_11_0_arm64.whl`; SHA-256 `8bd34610d5791bbca7909e3e9a628c184c05026e07ca0fc0dd9262651c52dea9` | `DIAGNOSTIC_PREDECESSOR` |
| Installed native library | `native/_forge3d.abi3.so`; SHA-256 `67de5718ac3a5956e17d77c2fc735bc62259c971b05a5bd0b15dc6f136551a79` | `DIAGNOSTIC_PREDECESSOR` |
| Adapter and aggregate | Apple M4 / Metal / integrated / `software_fallback=false`; 4,817 non-duplicated required JUnit executions with zero failures, errors, or skips | `DIAGNOSTIC_PREDECESSOR` |

No `57dd3283` wheel, native library, manifest, JUnit, count, review, or physical
result may be relabelled as proof for the live candidate or final committed
head.

### Predeclared exact-head evidence boundary

`target/pr170-final-evidence/` remains stale RED diagnostic evidence for
`8ee3d04aa69cc898a1eba9c60f5cf2d3855ce258`. The newer
`target/pr170-exact-final-20260826-evidence/` is complete local evidence only
for `57dd32830549cc499699256afa368a166ab1436b`. Neither root nor any artifact,
wheel, native library, manifest, checksum, count, or result beneath it may be
reused or relabelled as post-change proof. No post-change evidence root or
index has been created by this documentation task.

| Final value | Authoritative destination | Current status |
|---|---|---|
| Final committed HEAD and tree | new post-change `MATRIX-RESULTS.md`, `ACCEPTANCE-MANIFEST.tsv`, and PR #170 body | `PENDING / NOT_PROVEN` |
| Fresh locked release-LTO wheel path and SHA-256 | new post-change `ACCEPTANCE-MANIFEST.tsv` and `SHA256SUMS` | `PENDING / NOT_PROVEN` |
| Fresh installed native-library path and SHA-256 | new post-change `ACCEPTANCE-MANIFEST.tsv` and `SHA256SUMS` | `PENDING / NOT_PROVEN` |
| Exact local matrix commands, artifact hashes, adapter/backend/fallback identity, and test accounting | new post-change evidence root and PR #170 body | `PENDING / NOT_PROVEN` |
| Exact hosted run IDs, head SHAs, artifact links, and selected-job accounting | PR #170 body and downloaded external evidence | `PENDING / NOT_PROVEN` |

No future wheel, native, artifact hash, or test count is guessed here. Once the
final documentation commit exists and the exact-head matrix begins, tracked
files must not change. The final SHA, tree, wheel/native hashes, matrix counts,
artifact inventory, and hosted evidence must therefore be recorded in the
external manifest/checksum files and the PR body. Any later tracked edit creates
a new head and requires a fresh final matrix.

### Current reviews and remaining proof

Direct user/owner authority approved a five-file ownership-boundary correction.
It supersedes and rejects as PR scope the attempted fresh-render certificate
binding, exact fresh/committed WGSL equality, fresh-clean and failure-guidance
checks in ordinary/backend physical lanes, and Metal timestamp-policy closure.
The candidate runtime WGSL hash
prefix `7e1980...` differs from the protected signed-base hash
`eb5127b8502f542a78926eeba26f9e0211c3ecad712a8abfa86bd57672b52b42`,
so fresh equality was impossible without an owner-authorized certificate
rotation. The PR's certificate JSON catalog and `signing.pub` remain
byte-identical to `origin/main`; the protected-base public verifier is the
pre-merge signing proof, while secret-backed refresh remains protected-main-only.

The ordinary and backend-specific Metal/NVIDIA recipe tests are scoped only to
physical pixel, adapter/backend, no-software-fallback, feature, and provenance
evidence. Protected refresh remains separately gated by update mode, the
production signing requirement and secret, a clean fresh certificate, pinned
public-key equality, and cryptographic verification before write. Both
the protected public-verification and refresh workflows remain unchanged by
this correction. The required independent `gpt-5.6-sol:high` review returned
`APPROVED` for the exact five-file correction after the user's direct approval.

A subsequent review found that `tests/golden/certificates/README.md` still
described the superseded fresh-render certificate boundary. The approved
first correction separated certificate-independent physical pixel lanes from
protected-base public verification and protected-main secret-backed refresh;
13 focused controls passed. A later topology P1 correctly rejected its remaining
implication that the dedicated base-owned verifier ran on ordinary pull-request
or push events. The final correction now states that the verifier runs only on
schedule and required manual `workflow_dispatch` with `scope=full`; ordinary PR
and push events are excluded and use candidate-owned Fast/static contracts.
Pre-merge acceptance therefore requires the separate full-scope dispatch. Four
focused topology controls passed, and the required independent
`gpt-5.6-sol:high` review returned `APPROVED` for the corrected final wording.

A root diagnostic Cargo rerun stopped on `ENOSPC` during build, before any test
executed. This is neither a test failure nor final evidence. Disk recovery
removed only generated Cargo debug, release, release-LTO, and old cycle-4
wheel-build caches, reclaiming 6.9 GiB; every evidence directory was preserved.
Later work regenerated `target/debug`; a second precise development-cache
cleanup removed only that regenerated cache and reclaimed another 1.6 GiB. All
four named cache roots are currently absent, every prior evidence directory
remains preserved, and the future fresh final build cannot reuse those caches.

All local-matrix and exact-tree review results below predate the two-file
workflow-parser fix and this documentation reconciliation. They remain valid
only for the named `57dd3283` predecessor. The parser fix arrived as an
already-approved input to this task; that component approval does not establish
post-change exact-head acceptance.

| Required proof | Exact evidence boundary | Status |
|---|---|---|
| Protected-public signing implementation review | certificate JSON and `signing.pub` remain byte-identical to the protected base; the base-owned public verifier remains the pre-merge signing proof and secret refresh remains protected-main-only | `APPROVED` |
| Owner-boundary correction review | the user directly approved the exact five-file correction; ordinary/Metal/NVIDIA physical recipe proof is certificate-independent, protected refresh retains every signing guard, protected workflows are unchanged, and independent high review approved | `APPROVED` |
| Initial certificate README boundary review | its physical-pixel/public-verifier/refresh separation passed 13 focused controls, but the later topology P1 found its ordinary-event implication inaccurate | `REJECTED / SUPERSEDED` |
| Final certificate README topology review | dedicated base-owned verification is schedule/full-manual-only, ordinary PR/push is excluded and candidate-owned Fast/static remains local scope; pre-merge requires separate `scope=full`, 4 focused controls passed, and independent high review approved | `APPROVED` |
| Fresh-render certificate binding and Metal timestamp-policy closure | superseded and rejected as PR scope under explicit owner authority; their prior focused results do not establish current-candidate or final acceptance | `REJECTED / SUPERSEDED` |
| Protected policy review | all 10 protected-policy checks are approved, `10/10` | `APPROVED` |
| `57dd3283` exact-local matrix and reviews | exact-tree Standards, Spec, combined-diff, and ponytail reviews plus the 4,817-test local matrix apply only to the committed predecessor | `DIAGNOSTIC_PREDECESSOR` |
| Two-file workflow-parser fix | `.github/workflows/ci.yml` and `tests/test_ci_cost_controls.py` arrived already approved; no approval is inferred for later documentation or the combined candidate | `APPROVED_COMPONENT` |
| Post-change focused documentation/source-contract checks | `git diff --check` passed; 13 focused workflow-parser and Apple selection/topology contracts passed; documentation wording was checked against the current workflow | `VALIDATED_FOCUSED_ONLY` |
| Final post-ledger combined-diff audit | independent `gpt-5.6-sol:xhigh` review must run against the exact post-change candidate | `PENDING / NOT_PROVEN` |
| Final ponytail review | must run against the post-ledger candidate before push | `PENDING / NOT_PROVEN` |
| Final two-axis code-review reruns | Standards and Spec review must rerun against the post-ledger candidate before push | `PENDING / NOT_PROVEN` |
| Final documentation commit and push | this task performs neither | `PENDING / NOT_PROVEN` |
| Fresh exact-final local matrix and physical full runs | must run after the final commit using a new locked release-LTO wheel, fresh isolated venv, exact committed head, complete required lanes, physical Metal and NVIDIA/Vulkan runs, and zero-skip accounting | `PENDING / NOT_PROVEN` |
| Protected public certificate verification | required exact-head hosted proof in the full workflow using the protected-base verifier and key | `PENDING / NOT_PROVEN` |
| Secret-backed certificate refresh | not required because this delta changes neither certificate JSON files nor `signing.pub`; required only if certificates genuinely rotate | `NOT_REQUIRED` |
| Other selected hosted acceptance | full dispatch, exact-head artifact inspection, and every dependency of Full Acceptance Summary remain unrun for the final head | `PENDING / NOT_PROVEN` |
| Live PR head, body, draft/readiness, and `MERGEABLE/CLEAN` | require post-push GitHub readback after all exact-head checks | `PENDING / NOT_PROVEN` |

The `APPROVED` signing row is a review verdict, not a claim that the production
secret-backed signer executed. Pre-merge acceptance requires protected public
certificate verification at the exact hosted head. Secret-backed refresh stays
protected-main-only and is `NOT_REQUIRED` unless certificates rotate.

PR #170 therefore preserves the previously approved implementation components,
but the live candidate still awaits post-change reviews, the final commit, a
fresh post-commit wheel/venv matrix, physical full runs, hosted CI, push,
live-head identity, and mergeability/readiness readback. Every one of those
post-change gates remains `PENDING / NOT_PROVEN`; the `57dd3283` evidence is
diagnostic only, and the PR is not final, merge-ready, or ready for review.

### Lossless final changed-path mapping

The canonical mapping is `path<TAB>claim<TAB>task owner<TAB>proof category`.
It has exactly 255 rows and 255 unique paths. Its first column is byte-for-byte
the same ordered path list as `git diff --name-only
92cf80d20d7d5c6e9a564b853e79d596f3f5088f`, with no
missing, extra, or duplicate path. Its exact TSV serialization, including the
final newline, has SHA-256
`4e131c8da9cbac5860e6c06f910fcaf652b64ffe2e3d22a45c9e5cf02e9ef277`. The canonical newline-delimited
changed-path list has SHA-256
`0e45d47a7a1b783f4dcf17040427966819c10bb795694d9706e7283df9a035b6`.

The deterministic ownership rule is: current paths retain their distinct
signing-boundary, owner-boundary, Apple-preflight, Apple-manifest,
example-typing, standards-remediation, workflow-parser, documentation, or
final-ledger claim. Every path retains the latest contract-task owner that
changed its current content; earlier shared touches remain in Git history.
Proof categories name the local proof family; `HOSTED-PENDING` never upgrades
source or policy review into hosted execution.

| Claim | Paths | Task owner | Contract claim |
|---|---:|---|---|
| `PR170-INITIAL` | 48 | initial refactor and Metal-recovery tasks | original behavior-preserving refactor, first-render AOV, text/adjudication, and static-contract payload |
| `PR170-MAIN-RECONCILIATION` | 7 | current-main reconciliation task | preserve final PR behavior while reconciling overlapping current-main content |
| `PR170-METAL-AETHER-STATE` | 5 | Metal state and AETHER task | correct Metal capability state and AETHER binding |
| `PR170-CONTRACT-CLOSURE` | 44 | local contract-closure task | close visibility, VT provenance, fail-closed CI, examples, API/export, and exact local acceptance gaps |
| `PR170-METAL-PROVENANCE` | 1 | Metal provenance-routing task | generate and validate backend-bound Metal fixture provenance |
| `PR170-METAL-FIXTURE-FREEZE` | 52 | Metal fixture-freeze task | freeze reviewed Metal fixtures, provenance, determinism hash, and terrain source contract |
| `PR170-PROMETHEUS-SCOPE` | 1 | PROMETHEUS runtime-proof task | scope runtime proof checks to the authoritative renderer path |
| `PR170-TESSELLA-TIMING` | 1 | TESSELLA timing-acceptance task | separate functional correctness from physical timing acceptance |
| `PR170-ACCEPTANCE-OWNERSHIP` | 73 | acceptance lane-ownership task | make generic, physical, COG, timing, and hosted routes disjoint, complete, and auditable |
| `PR170-APPLE-SELECTION-ORDER` | 1 | Apple Metal selection-audit task | audit skip and xfail decorators after marker deselection |
| `PR170-SIGNING-BOUNDARY` | 3 | signing-boundary task | close protected trust-root, verifier isolation, catalog binding, protected signing intent, and certificate-boundary documentation |
| `PR170-APPLE-PREFLIGHT` | 3 | Apple-preflight task | make Apple acceptance fail closed and preserve hosted fixture prerequisites |
| `PR170-APPLE-MANIFEST` | 1 | Apple-manifest correction task | make the TOML ownership comment match the authoritative 265 executions |
| `PR170-EXAMPLE-TYPING` | 6 | example-typing task | preserve the six independently approved example typing corrections |
| `PR170-STANDARDS-REMEDIATION` | 3 | standards remediation task | make provenance signatures explicit, centralize projection errors, and lock the regression |
| `PR170-OWNER-BOUNDARY` | 1 | owner-boundary/signing correction task | keep physical recipe proof certificate-independent while preserving protected public verification and secret-backed refresh ownership |
| `PR170-WORKFLOW-PARSER` | 2 | workflow-parser fix task | remove job-level use of step-only runner context and lock the parser contract |
| `PR170-DOC-RECONCILIATION` | 2 | acceptance-documentation reconciliation task | distinguish ordinary CI from required Apple physical acceptance and record its selection topology |
| `PR170-FINAL-LEDGER` | 1 | final evidence-ledger task | bind the live candidate, pending exact-head proof, and lossless changed-path mapping |

| Path | Claim | Task owner | Proof category |
|---|---|---|---|
| `.claude/rules/build-and-ci.md` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `WORKFLOW-CONTRACT; HOSTED-PENDING` |
| `.github/workflows/certificate-refresh.yml` | `PR170-SIGNING-BOUNDARY` | signing-boundary task | `FOCUSED-CERTIFICATE-CONTRACT + PROTECTED-POLICY; HOSTED-PENDING` |
| `.github/workflows/ci.yml` | `PR170-WORKFLOW-PARSER` | workflow-parser fix task | `FOCUSED-WORKFLOW-PARSER-CONTRACT; HOSTED-PENDING` |
| `.github/workflows/determinism-matrix.yml` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `WORKFLOW-CONTRACT; HOSTED-PENDING` |
| `.github/workflows/test-python-wheel.yml` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `WORKFLOW-CONTRACT; HOSTED-PENDING` |
| `.gitignore` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `STATIC-CONTRACT + LOCAL-MATRIX` |
| `CMakeLists.txt` | `PR170-MAIN-RECONCILIATION` | current-main reconciliation task | `STATIC-CONTRACT + LOCAL-MATRIX` |
| `Cargo.toml` | `PR170-MAIN-RECONCILIATION` | current-main reconciliation task | `STATIC-CONTRACT + LOCAL-MATRIX` |
| `MANIFEST.in` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `STATIC-CONTRACT + LOCAL-MATRIX` |
| `build.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `STATIC-CONTRACT + LOCAL-MATRIX` |
| `docs/ci-validation.md` | `PR170-DOC-RECONCILIATION` | acceptance-documentation reconciliation task | `DOC-SOURCE-CONTRACT; HOSTED-PENDING` |
| `docs/examples/index.md` | `PR170-TESSELLA-TIMING` | TESSELLA timing-acceptance task | `DOC-REFERENCE + FULL-PROFILE` |
| `docs/gallery/index.md` | `PR170-DOC-RECONCILIATION` | acceptance-documentation reconciliation task | `DOC-SOURCE-CONTRACT; HOSTED-PENDING` |
| `docs/guides/data_and_scene_workflows.md` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `DOC-REFERENCE + FULL-PROFILE` |
| `docs/guides/feature_map.md` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `DOC-REFERENCE + FULL-PROFILE` |
| `docs/refactor-forge3d-w1-a.md` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `LEDGER-SOURCE-LOCK` |
| `docs/refactor-forge3d-w1-b.md` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `LEDGER-SOURCE-LOCK` |
| `docs/refactor-forge3d-w1-c.md` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `LEDGER-SOURCE-LOCK` |
| `docs/refactor-forge3d-w1-v.md` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `LEDGER-SOURCE-LOCK` |
| `docs/refactor-forge3d.md` | `PR170-FINAL-LEDGER` | final evidence-ledger task | `LEDGER-INVARIANT` |
| `examples/_import_shim.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `DOC-REFERENCE + FULL-PROFILE` |
| `examples/_terrain_feature_demo.py` | `PR170-EXAMPLE-TYPING` | example-typing task | `FOCUSED-TYPING-CONTRACT + CANDIDATE-AUDIT` |
| `examples/fuji_labels_demo.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `DOC-REFERENCE + FULL-PROFILE` |
| `examples/mapscene_p1_assets_bundle_showcase.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `DOC-REFERENCE + FULL-PROFILE` |
| `examples/terrain_tv10_subsurface_demo.py` | `PR170-EXAMPLE-TYPING` | example-typing task | `FOCUSED-TYPING-CONTRACT + CANDIDATE-AUDIT` |
| `examples/terrain_tv21_blending_demo.py` | `PR170-EXAMPLE-TYPING` | example-typing task | `FOCUSED-TYPING-CONTRACT + CANDIDATE-AUDIT` |
| `examples/terrain_tv24_reflection_probe_demo.py` | `PR170-EXAMPLE-TYPING` | example-typing task | `FOCUSED-TYPING-CONTRACT + CANDIDATE-AUDIT` |
| `examples/terrain_tv4_material_variation_demo.py` | `PR170-EXAMPLE-TYPING` | example-typing task | `FOCUSED-TYPING-CONTRACT + CANDIDATE-AUDIT` |
| `examples/terrain_tv6_heterogeneous_volumetrics_demo.py` | `PR170-EXAMPLE-TYPING` | example-typing task | `FOCUSED-TYPING-CONTRACT + CANDIDATE-AUDIT` |
| `pyproject.toml` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `STATIC-CONTRACT + LOCAL-MATRIX` |
| `pytest.ini` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `STATIC-CONTRACT + LOCAL-MATRIX` |
| `python/forge3d/__init__.pyi` | `PR170-MAIN-RECONCILIATION` | current-main reconciliation task | `WHEEL-API + FULL-PROFILE` |
| `python/forge3d/determinism.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `WHEEL-API + FULL-PROFILE` |
| `scripts/check_determinism_hashes.py` | `PR170-STANDARDS-REMEDIATION` | standards remediation task | `TYPED-SIGNATURE + DETERMINISM-PROVENANCE-38` |
| `scripts/ci_pytest_lane.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `STATIC-CONTRACT + SELECTION-LEDGER` |
| `scripts/run_apple_metal_acceptance.py` | `PR170-APPLE-SELECTION-ORDER` | Apple Metal selection-audit task | `STATIC-CONTRACT + PHYSICAL-METAL` |
| `scripts/run_nvidia_visual_acceptance.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `STATIC-CONTRACT + LOCAL-MATRIX` |
| `shaders/contracts/terrain_pbr_pom.toml` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/core/capabilities.rs` | `PR170-METAL-AETHER-STATE` | Metal state and AETHER task | `RUST-MATRIX + FULL-PROFILE` |
| `src/core/dd/jitter.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/core/gpu.rs` | `PR170-METAL-AETHER-STATE` | Metal state and AETHER task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/core/gpu_timing.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/core/text_overlay.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/export/mod.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + FULL-PROFILE` |
| `src/export/projection.rs` | `PR170-STANDARDS-REMEDIATION` | standards remediation task | `CENTRALIZED-RENDERERROR + RUST-PROJECTION-7` |
| `src/geometry/displacement.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/geometry/mod.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/geometry/overlay/faces.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/geometry/overlay/mod.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/geometry/overlay/sweep.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/geometry/subdivision.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/gis/crs.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/gis/domain.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/gis/mod.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/gis/raster_tags.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/gis/raster_write.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/gis/rasterize.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/gis/thematic.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/labels/atlas.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/labels/mod.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/lib.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/path_tracing/adjudication.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/path_tracing/hybrid_compute/aether_reference.rs` | `PR170-METAL-AETHER-STATE` | Metal state and AETHER task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/path_tracing/hybrid_compute/render_terrain.rs` | `PR170-PROMETHEUS-SCOPE` | PROMETHEUS runtime-proof task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/path_tracing/reference_scene.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/picking/unified.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + FULL-PROFILE` |
| `src/py_functions/adjudication.rs` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `RUST-MATRIX + ADJUDICATION-CERTIFICATE` |
| `src/py_functions/path_tracing/terrain_reference.rs` | `PR170-MAIN-RECONCILIATION` | current-main reconciliation task | `RUST-MATRIX + FULL-PROFILE` |
| `src/py_module/functions.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + FULL-PROFILE` |
| `src/py_module/functions/export.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + FULL-PROFILE` |
| `src/py_types/aov.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/scene/core/constructor.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/scene/mod.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/scene/py_api/native_text.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + FULL-PROFILE` |
| `src/scene/render_paths/timing.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + FULL-PROFILE` |
| `src/shader_sources.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + FULL-PROFILE` |
| `src/shaders/offline_accumulate.wgsl` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/shaders/pt_intersect.wgsl` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/shaders/pt_raygen.wgsl` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/shaders/pt_shade.wgsl` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/shaders/pt_shadow.wgsl` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/shaders/terrain_pbr_pom.wgsl` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/shaders/terrain_visbuffer_write.wgsl` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/shaders/terrain_visibility_fullscreen.wgsl` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/shaders/text_overlay.wgsl` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/cog/py_bindings.rs` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `RUST-MATRIX + COG-LOOPBACK` |
| `src/terrain/renderer/aov.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/renderer/atmosphere.rs` | `PR170-MAIN-RECONCILIATION` | current-main reconciliation task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/renderer/core.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/renderer/draw/execute.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/renderer/draw/mod.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/renderer/geometry.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/renderer/offline.rs` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/renderer/pipeline_cache.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/renderer/py_api.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/renderer/runtime_contract.rs` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/renderer/virtual_texture.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/renderer/visibility_buffer.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/spike/constructor.rs` | `PR170-METAL-AETHER-STATE` | Metal state and AETHER task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/terrain/vt_family_residency.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + PHYSICAL-METAL` |
| `src/verify/ir/engine.rs` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `RUST-MATRIX + FULL-PROFILE` |
| `src/viewer/init/device_init.rs` | `PR170-METAL-AETHER-STATE` | Metal state and AETHER task | `RUST-MATRIX + PHYSICAL-METAL` |
| `tests/UNRUN.toml` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/_generated_assets.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/_terrain_runtime.py` | `PR170-APPLE-PREFLIGHT` | Apple-preflight task | `FOCUSED-APPLE-CONTRACT + POLICY-AUDIT; HOSTED-PENDING` |
| `tests/apple_metal_acceptance.toml` | `PR170-APPLE-MANIFEST` | Apple-manifest correction task | `STATIC-CONTRACT + CANDIDATE-AUDIT` |
| `tests/golden/certificates/README.md` | `PR170-SIGNING-BOUNDARY` | signing-boundary task | `FOCUSED-CERTIFICATE-CONTRACT + PROTECTED-POLICY; HOSTED-PENDING` |
| `tests/golden/recipes/mapscene_alignment_utm.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_alignment_utm.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_auto_water.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_auto_water.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_buildings.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_buildings.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_clipmap_large_region.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_clipmap_large_region.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_cloud_shadows.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_cloud_shadows.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_copc_points.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_copc_points.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_furniture_graticule.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_furniture_graticule.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_label_arabic_joining.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_label_arabic_joining.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_label_halo_depth.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_label_halo_depth.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_label_occlusion_ridge.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_label_occlusion_ridge.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_material_maps.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_material_maps.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_offline_aovs.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_offline_aovs.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_png16_color.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_png16_color.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_screen_space_contact.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_screen_space_contact.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_screen_space_reflection.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_screen_space_reflection.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_terrain_raster.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_terrain_raster.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_textured_gltf_landmark.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_textured_gltf_landmark.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_thematic_choropleth.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_thematic_choropleth.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_tiles3d_points.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_tiles3d_points.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_vector_labels.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_vector_labels.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_vector_stroke_quality.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_vector_stroke_quality.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_vector_stroke_quality_4x.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/recipes/mapscene_vector_stroke_quality_4x.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/terrain/substratia_grazing_baseline.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/terrain/substratia_grazing_baseline.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/terrain/substratia_grazing_normal.metal.png` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/golden/terrain/substratia_grazing_normal.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/goldens/determinism/terra_determinata_v1.metal.provenance.json` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/goldens/determinism/terra_determinata_v1.metal.sha256` | `PR170-METAL-FIXTURE-FREEZE` | Metal fixture-freeze task | `FIXTURE-PROVENANCE + PHYSICAL-METAL` |
| `tests/requirements.txt` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_adjudication_gate.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + ADJUDICATION-CERTIFICATE` |
| `tests/test_anamnesis_incremental.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_anamnesis_inertness.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_anamnesis_p1.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_anamnesis_portability.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_aov.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_api_contracts.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_api_truth_pass.py` | `PR170-MAIN-RECONCILIATION` | current-main reconciliation task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_astro_night_golden.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_atmosphere_golden.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_atmosphere_lut_handoff.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_atmosphere_pt_reference.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_atmosphere_reference.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_atmosphere_spectral.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_bloom_execute_behavior.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_budget_enforce.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_buildings_extrude.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_california_cigar_smoke_hybrid.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_cam_phi_wiring.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_capability_negotiation.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_certificate_verifier.py` | `PR170-SIGNING-BOUNDARY` | signing-boundary task | `FOCUSED-CERTIFICATE-CONTRACT + PROTECTED-POLICY; HOSTED-PENDING` |
| `tests/test_ci_cost_controls.py` | `PR170-WORKFLOW-PARSER` | workflow-parser fix task | `FOCUSED-WORKFLOW-PARSER-CONTRACT; HOSTED-PENDING` |
| `tests/test_cog_streaming.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + COG-LOOPBACK` |
| `tests/test_color_management.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_dem_loading.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_determinism_hash.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_determinism_matrix.py` | `PR170-METAL-PROVENANCE` | Metal provenance-routing task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_example_catalog_docs.py` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_export_projection.py` | `PR170-STANDARDS-REMEDIATION` | standards remediation task | `PROJECTION-REGRESSION + PYTHON-PROJECTION-15` |
| `tests/test_exr_output.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_f3dz_codec.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_flythrough_popping.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_heightfield_ao.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_hybrid_terrain_pt.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_hzb_culling.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_light_feature_enablement.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_lighting_alignment.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_lighting_preset.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_mapscene_examples.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_mapscene_label_occlusion.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_mapscene_quickstart.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_mapscene_vector_strokes.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_motion_vectors.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_msdf_fidelity.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_no_silent_degradation.py` | `PR170-APPLE-PREFLIGHT` | Apple-preflight task | `FOCUSED-APPLE-CONTRACT + POLICY-AUDIT; HOSTED-PENDING` |
| `tests/test_p2_advanced_label_rules.py` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_p2_advanced_labels_repeated_curved.py` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_p2_determinism_noop.py` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_p2_quickstart.py` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_perspective_probe.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_perspective_projection.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_preset_visual_parity.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_provenance_offline_verify.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_provenance_veritas.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_recipe_goldens.py` | `PR170-OWNER-BOUNDARY` | owner-boundary/signing correction task | `PHYSICAL-RECIPE-SCOPE + PROTECTED-REFRESH-CONTRACT; FINAL-PHYSICAL-HOSTED-PENDING` |
| `tests/test_render_certificate.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_render_certificate_contract.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_sdist_manifest.py` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_shader_proofs.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_shadow_techniques.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_shadow_tip.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_ssgi_ssr_wiring.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_sun_visibility.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_analysis_api.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_clipmap_streaming.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_demo.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_material_maps.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_probes.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_render_color_space.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_renderer.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_runtime.py` | `PR170-APPLE-PREFLIGHT` | Apple-preflight task | `FOCUSED-APPLE-CONTRACT + POLICY-AUDIT; HOSTED-PENDING` |
| `tests/test_terrain_scatter.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_sky_parity.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_tv10_demo.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_tv10_goldens.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_tv10_subsurface.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_tv10_subsurface_materials.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_tv13_lod_pipeline.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_tv21_blending.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_tv21_demo.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_tv24_demo.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_tv4_demo.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_tv4_material_variation.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_tv6_heterogeneous_volumetrics.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_visual_goldens.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_terrain_vt_pbr_families.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_trust_boundary_diagnostics.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_tv12_offline_quality.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_tv20_virtual_texturing.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_tv22_scatter_wind.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_vector_coverage.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_viewshed_curvature.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_visibility_buffer.py` | `PR170-CONTRACT-CLOSURE` | local contract-closure task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_widgets.py` | `PR170-ACCEPTANCE-OWNERSHIP` | acceptance lane-ownership task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/test_world_coord_f32_gate.py` | `PR170-MAIN-RECONCILIATION` | current-main reconciliation task | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/torture/COVERAGE.json` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `FULL-PROFILE + FOCUSED-CONTRACT` |
| `tests/torture/labels/labels-057.json` | `PR170-INITIAL` | initial refactor and Metal-recovery tasks | `FULL-PROFILE + FOCUSED-CONTRACT` |

## PR #170 terminal-run audit and uncommitted continuation (2026-08-27)

This section supersedes the 2026-08-26 continuation section only as a statement
of current state. The earlier section and its 255-path mapping remain useful
historical evidence for committed head `7db2414d816fd02b4fce776fe0ed37d4d4506802`.
They do not describe the later uncommitted implementation or prove its
acceptance.

### Current identity

| Fact | Exact observation | Status |
|---|---|---|
| Local committed HEAD | `a3260432341585eea1e486bccff4fa898c824f4a`; tree `97eb7a74612f2c12b645e124dff37c5c13524997` | `VALIDATED_COMMITTED_PREDECESSOR`; exact phase-A and full lanes ran before the current source/ledger remediation |
| Comparison base | `92cf80d20d7d5c6e9a564b853e79d596f3f5088f` | `VALIDATED` |
| Hosted audit | [CI run 32994300065](https://github.com/milos-agathon/forge3d/actions/runs/32994300065), `workflow_dispatch`, `scope=full`, exact head `7db2414d...`, terminal `completed/failure` | `VALIDATED_RED_PREDECESSOR` |
| Current worktree including this ledger | two tracked modified files and no untracked files: `docs/refactor-forge3d.md` and `src/terrain/renderer/visibility_buffer.rs`; exact newline-terminated `git status --porcelain=v1 -uall` SHA-256 `4e5771c8d2e9be1ab395b4ee022255d990ba65cbc81f562f81d558596284067d` via `git status --porcelain=v1 -uall \| shasum -a 256` | `VALIDATED_UNCOMMITTED` |
| Current tracked diff path set | two paths; exact newline-terminated `git diff --name-only` SHA-256 `dabe6e2d31f5675297abd2cdee7ce025b2f9457a5ab8d99ce64c1d0db75db452` | `VALIDATED_UNCOMMITTED` |
| Final candidate commit/tree | commit `a3260432...` contains the reviewed checked-extent parity predecessor, but no commit contains the later TERMINUS `.expect` removal and ledger reconciliation | `PENDING / NOT_PROVEN` |
| Live PR state | last supplied state was open, draft, and `MERGEABLE/BLOCKED`; no post-change live readback exists | `STALE_SNAPSHOT / NOT_PROVEN` |

Commit `a3260432...` contains the previously reviewed 50-path set. The live
worktree identity is now the two-path uncommitted continuation above, not the
old 50-entry porcelain snapshot; there are no untracked paths.

### Current implementation map

Every row below describes bytes committed no later than `a3260432...`, except that
`src/terrain/renderer/visibility_buffer.rs` and this ledger have later live
modifications. A commit or implementation status is not test, physical, hosted,
publication, or merge acceptance.

| Claim mapped from the terminal audit | Current implementation paths | Current status | Proof boundary |
|---|---|---|---|
| Rust 1.98 lint drift | `Cargo.toml`; `src/geometry/mod.rs`; `src/py_types/aov.rs` | `APPROVED_COMPONENT / COMMITTED_AT_57F4B43E` | component proof and current combined/Standards/Spec review are approved; run 32994300065 predates the correction and hosted Rust proof remains `NOT_PROVEN` |
| HELIOS conservative proof portability | `src/path_tracing/hybrid_compute/terrain_heightfield.rs` | `COMMITTED_AT_57F4B43E` | fixed 75,025-element proof arrays, explicit layouts, checked pipeline creation, sentinels, and count guards are present; Windows and Apple-hosted causality remain `NOT_PROVEN` |
| Linux generic collection boundary | `tests/test_astro_night_golden.py`; `tests/test_no_silent_degradation.py` | `COMMITTED_AT_57F4B43E` | the unused eager `Session` is removed and import/selection locks are added; exact hosted Linux full/slow proof is pending |
| Windows CRLF source-test portability | `src/path_tracing/adjudication.rs`; `src/terrain/analysis/viewshed.rs` | `COMMITTED_AT_57F4B43E` | inspected source is normalized and the FXC mutant must change; Windows proof is pending |
| Windows certificate helper, Windows sdist decoding, and Python 3.10 TOML compatibility | `tests/test_certificate_verifier.py`; `tests/test_sdist_manifest.py`; `tests/test_api_contracts.py` | `COMMITTED_AT_57F4B43E` | Git-for-Windows Bash selection, strict UTF-8 decoding, and `_toml_compat` use are present; the exact cross-platform matrix is pending |
| M-06 fixture/selection and ANAMNESIS deterministic production route | `.github/workflows/ci.yml`; `tests/test_ci_lfs_fanout.py`; `tests/test_no_silent_degradation.py` | `COMMITTED_AT_57F4B43E` | Rainier DEM fanout, interactive-viewer selection, and deterministic ANAMNESIS configuration are present; required NVIDIA reruns remain `NOT_PROVEN` |
| Artifact self-identification | `.github/actions/bind-artifact-head/action.yml`; `.github/workflows/{build-wheel,ci,determinism-matrix,test-python-wheel}.yml`; `tests/test_ci_artifact_binding.py` | `COMMITTED_AT_57F4B43E` | uploads are wired to add a full checked-out-head marker and fail closed on a wrong head; no hosted artifact from this implementation exists |
| NVIDIA Rainier visual-parity candidate | `python/forge3d/{map_scene,presets,terrain_demo,terrain_params}.py`; `src/shader_sources.rs`; `src/shaders/{terrain_pbr_pom,terrain_shadow_depth}.wgsl`; `src/terrain/render_params/{decode_lighting,native_lighting}.rs`; `src/terrain/renderer/{draw/execute,shadows/setup,upload,visibility_buffer}.rs`; `tests/{test_lighting_alignment,test_mapscene_presets,test_terrain_demo_preset_integration,test_visibility_buffer}.py` | `COMMITTED_AT_A3260432 + TERMINUS_REMEDIATION_UNCOMMITTED` | committed parity keeps the `<= 32,768` invariant and exact `u16 -> f32` WGSL arithmetic; the live source removes its two ratchet-breaking `.expect` calls using guarded, mathematically lossless `as u16` conversions with no fallback. Focused remediation proof and independent review are `APPROVED / LOCALLY_PROVEN`; a fresh exact-current wheel/full matrix and NVIDIA golden parity remain `NOT_PROVEN` |
| Current evidence ledger | `docs/refactor-forge3d.md` | `COMMITTED_AT_A3260432 + RECONCILIATION_UNCOMMITTED` | records the exact-commit matrix, current two-path continuation, and evidence boundaries without becoming final-head proof |

No current diff directly closes the audited DUPLA subprocess failure, the
Apple hosted-adapter trust failure, or the TESSELLA `1.7x` performance failure.
Those families remain open; absence of a mapped edit is not evidence that they
are fixed.

### Terminal hosted evidence at `7db2414d...`

The complete read-only audit is
`/private/tmp/pr170-run-32994300065-audit.md`. Its evidence root is
`/private/tmp/pr170-run-32994300065-audit.EEEtHl`; all 38 advertised artifacts
and the complete 106-entry log archive were inspected.

| Evidence | Exact result | Status for the current worktree |
|---|---|---|
| Terminal jobs | 74 total: 24 failed, 29 succeeded, 21 conditionally skipped, 0 cancelled/queued/running | `DIAGNOSTIC_PREDECESSOR` |
| JUnit aggregate | 32,728 tests, 42 failures, 6 errors, 2 skips, counting the duplicated Apple infrastructure result once | `RED / DIAGNOSTIC_PREDECESSOR` |
| Successful physical families | exact-head F3DZ 39/39, SUBSTRATIA 4/4 plus verifier `PASS`, and HELIOS 83/83 plus verifier `PASS` on RTX 3070/Vulkan | `PROVEN_AT_7DB2414D_ONLY` |
| Apple acceptance | `Apple Paravirtual device`, Metal, IntegratedGpu, vendor/device `0/0`, `software_fallback=false`; physical trust failed closed before the suite ran | `BLOCKED / NOT_PROVEN` |
| NVIDIA aggregate acceptance | TESSELLA, visual, M-06, and ANAMNESIS production were red even though narrower NVIDIA families passed | `FAILED / NOT_PROVEN` |
| Artifact self-identification | only 10 of 38 artifact directories contained the full 40-character head SHA in content; 28 were content-unbound | `FAILED_AT_7DB2414D` |
| Local exact-head bundle | `target/pr170-exact-final-7db2414d-20260826-evidence/` exists and is locally reviewed evidence for `7db2414d...` only | `STALE_AFTER_TRACKED_CHANGES` |

The 21 skipped jobs are accounted by `scope=full`, reusable-workflow family
selection, or optional-diagnostic conditions. This does not erase the two
unexpected runtime skips in M-06, and it does not make the failed full summary
acceptable.

### Open failure and acceptance ledger

| Family or gate | Current disposition |
|---|---|
| Rust 1.98 lint drift | correction implemented and component-approved; current combined/Standards/Spec review approved; hosted Rust proof remains `NOT_PROVEN` |
| HELIOS Windows/macOS conservative proof | candidate correction implemented; cross-platform causality `NOT_PROVEN` |
| Linux Python collection | candidate correction implemented; Linux 3.10-3.13 full and slow reruns pending |
| Windows CRLF source tests | candidate correction implemented; Windows rerun pending |
| Windows certificate helper | candidate correction implemented; exact Windows attack-suite rerun pending |
| DUPLA on Windows/macOS | operative child error and shared cause remain `NOT_PROVEN`; no direct current correction identified |
| Windows sdist decoding | candidate correction implemented; supported-Windows matrix pending |
| Python 3.10 `tomllib` | candidate correction implemented; Linux/Windows/macOS 3.10 proof pending |
| Apple Metal hosted trust | blocked on authoritative proof of physical Apple hardware or an approved real Apple Silicon runner; unknown identity must continue to fail closed |
| TESSELLA | canonical residency warm-up and repeated timing samples are removed; cold single-sample structure and component checks are `APPROVED / LOCALLY_PROVEN`, with the required speedup unchanged at `>= 1.7`; exact physical NVIDIA/Vulkan cold timing remains `NOT_PROVEN` |
| ANAMNESIS production | candidate deterministic-route correction exists; portability child causality and required physical rerun remain `NOT_PROVEN` |
| Rainier visual parity | candidate correction exists; exact NVIDIA/Vulkan SSIM and remaining visual records are `NOT_PROVEN` |
| M-06 | fixture and selection corrections exist; physical execution and zero-skip proof are pending |
| Artifact binding and Full Acceptance Summary | binder implementation exists; a fresh exact-head dispatch, complete artifact inspection, and green dependency summary are pending |

### Final acceptance boundary

The current worktree is not a final candidate. The `7db2414d...` wheel,
native-library hash, local JUnits, physical results, reviews, and hosted run are
predecessor evidence only. Combined, Ponytail-disposition, Standards, and Spec
review is closed for the exact current two-path candidate, and the TERMINUS
source remediation also has focused independent approval. Those reviews do not
prove exact-current execution or publication gates. Before PR #170 can leave draft, the
continuation contract still requires:

- a committed exact candidate and an entirely fresh locked release-LTO wheel,
  isolated venv, local matrix, JUnit/accounting bundle, and physical evidence;
- exact-head hosted reruns that close every failure family, bind every required
  artifact to the full SHA, and produce a green `Full Acceptance Summary`;
- physical Apple and required NVIDIA/Vulkan proof without fallback, skips,
  retries, warm-ups, stale results, or unresolved
  `NOT_PROVEN` claims; and
- live readback proving the tested SHA equals the PR head, the PR is no longer
  draft, and GitHub reports `MERGEABLE/CLEAN`.

Until those conditions are proven, PR #170 remains incomplete and must remain
draft. The production secret-backed certificate signer is not claimed to have
run and is not required unless certificates genuinely rotate.

### Current component checkpoint (2026-08-27)

This checkpoint supersedes only the statements above that the current diff has
no DUPLA correction and no approved Apple-runner route. It does not reclassify
run `32994300065`, claim a current commit or remote head, or turn component proof
into final-candidate acceptance.

| Component | Exact current evidence | Review and proof status |
|---|---|---|
| DUPLA Apple selection and count | `tests/test_determinism_hash.py` marks `test_dupla_dd_demo_is_backend_pinned_and_byte_identical` as `apple_metal_physical`; `tests/apple_metal_acceptance.toml` owns 229 semantic physical cases, two fresh TV20 processes, and 35 contract cases, exactly 266 executions; `tests/test_no_silent_degradation.py` locks 266. The focused lock passed 1/1, and targeted collection selected 229 physical tests with DUPLA exactly once. | independent review after three rounds: `APPROVED`, zero findings; selection/count contract `LOCALLY_PROVEN`; the complete 266-execution physical matrix remains `NOT_PROVEN` |
| Dedicated ephemeral Apple runner contract | `.github/workflows/ci.yml` routes only manual `scope=full` on `codex/refactor-forge3d-20260812` to the unique `forge3d-pr170-apple-metal` label, proves Apple Silicon host identity before checkout, and excludes schedules/default labels; `tests/test_no_silent_degradation.py` and `docs/ci-validation.md` lock the route and lifecycle. Focused live route/parser proof passed 5/5; five negative mutations covering strict shell, Apple identity, default-label rejection, and duplicate `.yml`/`.yaml` ownership were all rejected. | independent review after three rounds: `APPROVED`, zero findings; workflow contract `LOCALLY_PROVEN`; the runner is not registered and the workflow has not run, so hosted Apple execution is `NOT_PROVEN` |
| Apple authoritative adapter identity | `src/core/gpu.rs`, `src/py_functions/diagnostics.rs`, `scripts/run_apple_metal_acceptance.py`, and `tests/engine_info_api_compat.rs` expose authoritative `vendor=0x106b` and `device=<nonzero MTLDevice.registryID>` for an attributable Apple Metal device while preserving raw wgpu `0/0` as `raw_vendor` and `raw_device`; parent and render subprocesses must match all four fields, and the public Rust `EngineInfo` layout remains unchanged. Focused Rust, Python, API-compatibility, and check proof is green. | implementation checks `LOCALLY_PROVEN`; this host returns `no_adapter`, so physical exact-current identity and full Apple acceptance remain `NOT_PROVEN`; the prior r2 Apple M4 artifact is `SUPERSEDED_COMPONENT_PREDECESSOR` only |

The current identity implementation's final locked release-LTO wheel SHA-256 is
`28e264e7b877f9a7456a6e21d2c41f1e03417e3b73f1def3676ce67cdf368da5`;
its installed `_forge3d.abi3.so` SHA-256 is
`f8548a7d2401c2e704bbbd865906c58a18b755797530e310571d354b94bf141b`.
Those hashes bind the green implementation checks, not a physical run: the
exact-current installed wheel reports `no_adapter`. The older
`/private/tmp/forge3d-apple-identity-r2.tfRS02/adapter-before.json` and its wheel
and native hashes are superseded component-predecessor evidence only.

| Review surface | Exact current state | Status |
|---|---|---|
| Combined `gpt-5.6-sol:xhigh` audit | reviewed the exact current two-path candidate after the TERMINUS remediation and ledger reconciliation | `APPROVED`, zero contract defects; execution and publication remain separate |
| Final Ponytail review | reviewed the exact current two-path candidate for unnecessary complexity | `Lean already. Ship.`; net `-0` necessary lines |
| Code review — Standards | final review of the exact current two-path candidate | `APPROVED`, zero findings |
| Code review — Spec | final review of the exact current two-path candidate, including the checked `<= 32,768` `u32 -> u16 -> f32` route and its two f32 boundary regressions | `APPROVED`, zero findings; parity P1 and TERMINUS ratchet closed; execution gates remain separate |
| Scoped TERMINUS remediation review | reviewed only the live removal of the two ratchet-breaking `.expect` conversions after the exact full lane exposed them | `APPROVED`, zero findings; corroborates the whole-current reviews |

The combined, Ponytail, Standards, and Spec axes were scheduled serially only
because the platform agent-thread limit prevented parallel dispatch. Each ran
in a separate blind context; serial scheduling did not merge their evidence or
review judgments.

The audited hosted evidence remains red: run `32994300065` completed with
failure at predecessor head `7db2414d...`, and its `Fast Contract` job failed.
A frozen committed-candidate release-LTO wheel, the complete 266-execution matrix,
NVIDIA/Vulkan acceptance, green exact-head hosted checks, a final commit and
push, tested-head/PR-head equality, readiness, and live `MERGEABLE/CLEAN`
readback all remain `NOT_PROVEN`.

### Post-combined-audit remediation checkpoint (2026-08-27)

The combined `gpt-5.6-sol:xhigh` audit reported three P1 findings. The component
evidence below records their remediations; owner authorization applies to the
exact terrain pin. The final combined re-audit reran after the later Apple
identity and Standards-policy changes and `APPROVED` the current production
bytes.

| Remediation | Exact current evidence | Review and proof status |
|---|---|---|
| Exact terrain source pin | The owner authorized the exact current terrain-source pin in `src/verify/ir/engine.rs`; the complete source-lock suite passed 4/4. | exact pin `LOCALLY_PROVEN`; owner authorization recorded; current combined audit `APPROVED` |
| TESSELLA cold acceptance semantics | `tests/test_hzb_culling.py` removes repeated seven-sample timing and median selection, takes one baseline and one culled sample, and locks the cold single-sample shape; `tests/test_flythrough_popping.py` removes canonical residency warm-up and records zero warm-up steps. The existing required speedup remains unchanged at `>= 1.7`. | independent component review `APPROVED`; source/contract remediation `LOCALLY_PROVEN`; physical NVIDIA/Vulkan cold timing remains `NOT_PROVEN` |
| Rainier premium rendering contract | `python/forge3d/presets.py` restores and retains premium PCSS shadows and IBL; the Rainier route and exact terrain pin are locked by `tests/test_lighting_alignment.py`, `tests/test_mapscene_presets.py`, `tests/test_terrain_demo_preset_integration.py`, and `src/verify/ir/engine.rs`. Focused Rainier/pin proof passed 11 tests. | independent Rainier review `APPROVED`; focused source/preset contract `LOCALLY_PROVEN`; exact NVIDIA/Vulkan Rainier SSIM remains `NOT_PROVEN` |

Run `32994300065` and its `Fast Contract` remain red predecessor evidence. A
frozen committed-candidate release-LTO wheel, complete 266-execution Apple
matrix, physical Apple and NVIDIA/Vulkan acceptance, green exact-head hosted
checks, final commit and push, tested-head/PR-head equality, readiness, and live
`MERGEABLE/CLEAN` readback all remain `NOT_PROVEN`.

### Authoritative current changed-file claim map (2026-08-27)

This map supersedes the abbreviated current implementation map above only for
lossless changed-file ownership. Every path in the 50-path commit delta
from `7db2414d...` to `a3260432...` appears once; the two live modified paths are members
of that same set. A component or focused proof status does not imply physical
acceptance, hosted acceptance, publication, or merge readiness.

<!-- CURRENT-CHANGED-FILE-MAP-BEGIN -->
| Explicit path | Claim / lane owner | Proof / status |
|---|---|---|
| `.claude/rules/build-and-ci.md` | Code review Standards policy remediation | `COMMITTED_AT_57F4B43E`; focused Standards and final Spec `APPROVED`; hosted/final acceptance `NOT_PROVEN` |
| `.github/actions/bind-artifact-head/action.yml` | Exact-head artifact self-identification action | `COMMITTED_AT_57F4B43E`; focused contract green; hosted artifact proof `NOT_PROVEN` |
| `.github/workflows/build-wheel.yml` | Wheel artifact exact-head binding | `COMMITTED_AT_57F4B43E`; focused contract green; hosted wheel proof `NOT_PROVEN` |
| `.github/workflows/ci.yml` | Acceptance orchestration: dedicated Apple runner, artifact binding, platform and physical-family selection | AETHER exact-`always()`, positive-probe, and exact-head-bound upload remediation independently `APPROVED`; exact-current hosted execution and final summary `NOT_PROVEN` |
| `.github/workflows/determinism-matrix.yml` | Hosted determinism diagnostics and artifact binding | `COMMITTED_AT_57F4B43E`; focused contract green; hosted execution `NOT_PROVEN` |
| `.github/workflows/test-python-wheel.yml` | Python-wheel JUnit and artifact binding | `COMMITTED_AT_57F4B43E`; focused contract green; hosted matrix `NOT_PROVEN` |
| `Cargo.lock` | Apple native Metal identity dependency lock | locked release-LTO build green; current combined/Standards/Spec `APPROVED`; final acceptance `NOT_PROVEN` |
| `Cargo.toml` | Apple native Metal identity dependency plus Rust 1.98 lint contract | focused build/check green; current combined/Standards/Spec `APPROVED`; final acceptance `NOT_PROVEN` |
| `docs/ci-validation.md` | Dedicated Apple runner selection and lifecycle documentation | focused route contract green; runner registration/execution and hosted acceptance `NOT_PROVEN` |
| `docs/refactor-forge3d.md` | Current evidence ledger and Spec changed-file-map remediation | committed through `a3260432...`, then modified only for exact-matrix, TERMINUS-remediation, and evidence-boundary reconciliation; new exact-current committed proof `NOT_PROVEN` |
| `python/forge3d/map_scene.py` | Rainier premium MapScene sampling and lighting route | independent component `APPROVED`; focused contract `LOCALLY_PROVEN`; NVIDIA/Vulkan SSIM `NOT_PROVEN` |
| `python/forge3d/presets.py` | Rainier premium PCSS, IBL, and nearest-height preset | independent component `APPROVED`; focused contract `LOCALLY_PROVEN`; NVIDIA/Vulkan SSIM `NOT_PROVEN` |
| `python/forge3d/terrain_demo.py` | Rainier terrain-demo sampling parity | independent component `APPROVED`; focused contract `LOCALLY_PROVEN`; NVIDIA/Vulkan SSIM `NOT_PROVEN` |
| `python/forge3d/terrain_params.py` | Rainier height-filter API contract | independent component `APPROVED`; focused contract `LOCALLY_PROVEN`; physical parity `NOT_PROVEN` |
| `scripts/run_apple_metal_acceptance.py` | Authoritative Apple vendor/device identity and exact subprocess binding | focused Python/API checks green; exact-current physical identity `NOT_PROVEN` because host reports `no_adapter` |
| `src/core/gpu.rs` | Native Apple Metal registry identity and preserved raw wgpu IDs | focused Rust/check proof `LOCALLY_PROVEN`; exact-current physical identity `NOT_PROVEN` |
| `src/geometry/mod.rs` | Rust 1.98 lint portability | component and current combined audit `APPROVED / LOCALLY_PROVEN`; hosted Rust proof `NOT_PROVEN` |
| `src/path_tracing/adjudication.rs` | Windows CRLF-insensitive adjudication source contract | current combined audit `APPROVED`; exact Windows proof `NOT_PROVEN` |
| `src/path_tracing/hybrid_compute/terrain_heightfield.rs` | HELIOS conservative proof portability and exact runtime sentinel contract | `u32::MAX` runtime sentinel assertion, focused test, and both clippy contracts green; scoped fix independently `APPROVED`; Windows/macOS causality and hosted proof `NOT_PROVEN` |
| `src/py_functions/diagnostics.rs` | Python authoritative Apple identity exposure with public API compatibility | focused Python/API/check proof `LOCALLY_PROVEN`; exact-current physical identity `NOT_PROVEN` |
| `src/py_types/aov.rs` | Rust 1.98 lint portability | component and current combined audit `APPROVED / LOCALLY_PROVEN`; hosted Rust proof `NOT_PROVEN` |
| `src/shader_sources.rs` | Rainier selectable height reconstruction source locks | independent Rainier component `APPROVED`; focused source contract `LOCALLY_PROVEN`; NVIDIA/Vulkan parity `NOT_PROVEN` |
| `src/shaders/terrain_pbr_pom.wgsl` | Rainier nearest-height premium terrain shader route | independent Rainier component `APPROVED`; focused source contract `LOCALLY_PROVEN`; NVIDIA/Vulkan SSIM `NOT_PROVEN` |
| `src/shaders/terrain_shadow_depth.wgsl` | Rainier matching nearest-height shadow caster | independent Rainier component `APPROVED`; focused source contract `LOCALLY_PROVEN`; NVIDIA/Vulkan SSIM `NOT_PROVEN` |
| `src/terrain/analysis/viewshed.rs` | Windows CRLF-insensitive FXC source contract | current combined audit `APPROVED`; exact Windows proof `NOT_PROVEN` |
| `src/terrain/render_params/decode_lighting.rs` | Rainier height-filter decode | independent Rainier component `APPROVED`; focused contract `LOCALLY_PROVEN`; physical parity `NOT_PROVEN` |
| `src/terrain/render_params/native_lighting.rs` | Rainier native height-filter propagation | independent Rainier component `APPROVED`; focused contract `LOCALLY_PROVEN`; physical parity `NOT_PROVEN` |
| `src/terrain/renderer/draw/execute.rs` | Rainier/visibility-buffer execution route | independent Rainier component `APPROVED`; focused contract `LOCALLY_PROVEN`; NVIDIA/Vulkan parity `NOT_PROVEN` |
| `src/terrain/renderer/shadows/setup.rs` | Rainier nearest-height shadow setup | independent Rainier component `APPROVED`; focused contract `LOCALLY_PROVEN`; NVIDIA/Vulkan SSIM `NOT_PROVEN` |
| `src/terrain/renderer/upload.rs` | Rainier height-filter upload | independent Rainier component `APPROVED`; focused contract `LOCALLY_PROVEN`; physical parity `NOT_PROVEN` |
| `src/terrain/renderer/visibility_buffer.rs` | Rainier height reconstruction and visibility-buffer parity | committed `a3260432...` checks each extent at `<= 32,768`, converts through exact `u16 -> f32`, and preserves WGSL f32 clamp/multiply/floor/min semantics, but its two checked-conversion `.expect` calls made the exact full lane red. The sole live remediation retains those assertions, uses guarded lossless `as u16` conversions with no fallback, restores TERMINUS `panic=1` / `unwrap=31` / `expect=15`, and removes this path from the source allowlist inventory. Ratchet 5/5 including ablation, world inventory 11/11, Rust boundaries 2/2, shader locks 2/2, focused visibility/Rainier 9/9, formatting, and both clippy gates are green; scoped independent, combined, Ponytail, Standards, and Spec reviews are all `APPROVED` for the exact current candidate. The fresh exact-current wheel/full lane and NVIDIA/Vulkan SSIM remain `NOT_PROVEN` |
| `src/verify/ir/engine.rs` | Owner-authorized exact terrain source pin | source-lock 4/4 `LOCALLY_PROVEN`; current combined audit `APPROVED` |
| `tests/apple_metal_acceptance.toml` | Apple 229 semantic / 266 total execution manifest | selection/count component `APPROVED / LOCALLY_PROVEN`; complete physical 266 matrix `NOT_PROVEN` |
| `tests/engine_info_api_compat.rs` | Public Rust `EngineInfo` layout compatibility | committed at `57f4b43e...`; focused Rust/API check green; combined/Standards/Spec `APPROVED`; physical acceptance `NOT_PROVEN` |
| `tests/test_api_contracts.py` | Python 3.10 TOML and API compatibility contracts | current combined/Standards/Spec `APPROVED`; exact hosted Python matrix `NOT_PROVEN` |
| `tests/test_astro_night_golden.py` | Generic collection boundary plus strict Apple identity consumer | focused collection/identity checks green; hosted Linux and physical Apple execution `NOT_PROVEN` |
| `tests/test_certificate_verifier.py` | Windows certificate-helper portability | pre-identity-change combined audit passed; exact Windows attack-suite proof `NOT_PROVEN` |
| `tests/test_ci_artifact_binding.py` | Exact-head artifact-binding negative and topology contracts | committed at `57f4b43e...`; focused contract and final Spec `APPROVED`; hosted artifact proof `NOT_PROVEN` |
| `tests/test_ci_lfs_fanout.py` | M-06 Rainier fixture fanout contract | focused static contract green; physical M-06 execution and zero-skip proof `NOT_PROVEN` |
| `tests/test_determinism_hash.py` | DUPLA Apple physical-family ownership | component `APPROVED`; targeted selection `LOCALLY_PROVEN`; complete 266 matrix `NOT_PROVEN` |
| `tests/test_flythrough_popping.py` | TESSELLA zero-warm-up 600-frame acceptance semantics | component `APPROVED`; cold structure `LOCALLY_PROVEN`; physical NVIDIA/Vulkan timing `NOT_PROVEN` |
| `tests/test_hzb_culling.py` | TESSELLA cold single-sample `>= 1.7` gate | component `APPROVED`; cold structure `LOCALLY_PROVEN`; physical NVIDIA/Vulkan timing `NOT_PROVEN` |
| `tests/test_lighting_alignment.py` | Rainier premium PCSS/IBL/height-filter locks | independent Rainier component `APPROVED`; focused contract `LOCALLY_PROVEN`; NVIDIA/Vulkan SSIM `NOT_PROVEN` |
| `tests/test_mapscene_presets.py` | Rainier MapScene premium-preset locks | independent Rainier component `APPROVED`; focused contract `LOCALLY_PROVEN`; NVIDIA/Vulkan SSIM `NOT_PROVEN` |
| `tests/test_no_silent_degradation.py` | Cross-lane Apple identity, dedicated-runner, count, and fail-closed acceptance locks | focused checks and current combined/Standards/Spec `APPROVED`; hosted physical acceptance `NOT_PROVEN` |
| `tests/test_recipe_goldens.py` | Apple physical recipe identity and owner-boundary locks | focused contract green; exact-current physical recipe execution and hosted proof `NOT_PROVEN` |
| `tests/test_sdist_manifest.py` | Windows strict UTF-8 sdist contract | pre-identity-change combined audit passed; exact hosted Windows matrix `NOT_PROVEN` |
| `tests/test_terrain_demo_preset_integration.py` | Rainier MapScene/terrain-demo premium parity locks | independent Rainier component `APPROVED`; focused contract `LOCALLY_PROVEN`; NVIDIA/Vulkan SSIM `NOT_PROVEN` |
| `tests/test_terrain_vt_pbr_families.py` | Apple adapter binding plus visibility/clipmap allocation contracts | focused contract green; exact-current physical Apple/NVIDIA execution `NOT_PROVEN` |
| `tests/test_visibility_buffer.py` | Rainier visibility-buffer geometry and sampling locks | independent Rainier component `APPROVED`; focused contract `LOCALLY_PROVEN`; NVIDIA/Vulkan SSIM `NOT_PROVEN` |
<!-- CURRENT-CHANGED-FILE-MAP-END -->

Verification uses the newline-terminated, locale-sorted path column only. The
canonical 50-path set is
`git diff --name-only 7db2414d816fd02b4fce776fe0ed37d4d4506802 a3260432341585eea1e486bccff4fa898c824f4a | LC_ALL=C sort`; the map set is the
first backtick-delimited field from each data row between the two markers above,
also sorted under `LC_ALL=C`. Both contain 50 rows, have no duplicate path, and
have SHA-256 `ebfe7893bb3066d6fa77f9909e9e4378cc3f14e84eab7a2d66bb24269a6dc3f0`.
The full ordered 50-row map SHA-256 is
`7e1553ab027d4daad333d5ccbef1855571865ab56164441bcefe6167f756130f` and covers each
complete Markdown data-row byte sequence, including owner and proof/status.

### Final local validation and AETHER remediation checkpoint (2026-08-27)

This checkpoint records local execution against the uncommitted 50-entry
worktree. It does not create a final candidate commit or supersede any physical,
hosted, or publication boundary above.

| Evidence phase | Exact result | Status and boundary |
|---|---|---|
| Initial final-local attempt | At unchanged local HEAD `7db2414d816fd02b4fce776fe0ed37d4d4506802`, `cargo forge3d-clippy` and `cargo forge3d-clippy-acceptance` failed on the constant-folded `assert!(PROOF_OUTPUT_SENTINEL > 1)` at `src/path_tracing/hybrid_compute/terrain_heightfield.rs:2103`. The sole isolated dependency install failed on sandbox DNS while resolving `geopandas`; the sole wheel command produced Cargo `dev`, not release-LTO. Its diagnostic-only wheel/native SHA-256 values were `8d96d8ac4c835e4352dd713a167c2733a723a983306d10ac5d0224b2c7433e5b` and `4daa5355d11677af758a005cf60987cbd40ba941ba2142aaed37b05caeb8936e`. | `RED / INVALID_FINAL_PROOF`; no Python or physical lane was reclassified |
| Scoped sentinel repair | The test now runtime-observes `PROOF_OUTPUT_SENTINEL` through `std::hint::black_box` and asserts exact equality to `u32::MAX`. Formatting passed, the focused contract passed 1/1, and both routine and acceptance clippy passed. | scoped fix independently `APPROVED / LOCALLY_PROVEN` |
| Fresh final-r2 Rust and wheel component | `/private/tmp/pr170-final-r2.y7I19R` used a prepared isolated Python 3.13 environment, a new empty external Cargo target, and explicit `maturin 1.15.0 build --profile release-lto --locked`. The log reported `release-lto`; exactly one wheel was built. Wheel SHA-256: `f778d8f6ef8d19fc402fd97e58fbb27e404dfc0d141ea74e8d68803f89c0ee50`; installed native SHA-256: `b052e519eb1106816971d6011837d0383c07d90234522fae81373f327f33ff1a`. Formatting, full-feature check, workspace tests, doctests, routine clippy, and acceptance clippy passed; workspace aggregation was 1,448 passed, zero failed, nine ignored, and four workflow-filtered, while the explicit doctest lane was 12 passed and six ignored. | exact r2 component proof `LOCALLY_PROVEN`; after the subsequent `.github/workflows/ci.yml` and ledger changes these wheel/native bytes are `PREDECESSOR_EVIDENCE`, not final exact-candidate proof |
| Python fast/full/slow | Fast executed 744 outcomes: 738 passed, six failed, zero skipped/xfail, with 28 deselected. Five failures were the pre-commit inventory guard correctly rejecting untracked `tests/test_ci_artifact_binding.py`; one was the AETHER exact-`always()` workflow contract. Full and slow each failed before collection on the same untracked-inventory guard, so neither produced JUnit or a selection ledger. | fast `RED`; full/slow `BLOCKED_PRECOLLECTION`; complete Python matrix and zero-skip proof `NOT_PROVEN` |
| Apple physical acceptance | The authoritative runner was invoked once without adapter warm-up or retry. Metal initialization produced one `no_adapter` infrastructure error and zero skips; no physical phase executed and no adapter identity record was created. | zero warm-ups and zero retries observed; exact physical identity and all 266 executions `NOT_PROVEN` |
| AETHER workflow remediation | The physical evidence binder and uploader retain the positive-probe predicate, both upload blocks have the exact single-line `if: always()` prefix, and uploads require their exact-head binder's `bound == 'true'`; the marker path also preserves its ABSENT-existence predicate. YAML parsing passed; focused AETHER plus complete artifact-binding proof passed 17/17; related no-silent/cost-control proof passed 3/3. The reviewer P1 restoration received final independent approval. | `APPROVED / LOCALLY_PROVEN`; no hosted AETHER artifact or exact-current wheel execution exists |

The r2 release-LTO hashes prove only the source state before the later workflow
and ledger edits. A final committed candidate still requires a new empty-target
locked release-LTO build, isolated complete Rust/Python matrix with accounting,
the full 266-case physical Apple matrix, required NVIDIA/Vulkan proof, green
exact-head hosted checks and artifacts, commit/push and tested-head/PR-head
equality, readiness, and live `MERGEABLE/CLEAN`. All remain `NOT_PROVEN`.

### Stable 50-entry final review closure (2026-08-27)

The final review axes ran after the sentinel and AETHER corrections and the
preceding ledger reconciliation. The reviewed inventory is the unchanged local
HEAD/tree `7db2414d816fd02b4fce776fe0ed37d4d4506802` /
`b738a245de8ea9b76bad453b54a4c5a96abb33d2` plus 50 uncommitted porcelain
entries: porcelain SHA-256
`bc6aba2eecf2b97b19b36f96d95ee514a118df9dc4f790b8bb7576b38442b655`,
47-path tracked-diff SHA-256
`44241ddef7f2a229bd05eb976c9fc1c32d0fb2e34b4cecec22474caf429723bd`,
canonical 50-path map-set SHA-256
`ebfe7893bb3066d6fa77f9909e9e4378cc3f14e84eab7a2d66bb24269a6dc3f0`,
and ordered 50-row map SHA-256
`2589ca360281d0b0573046df129c8e5733f216be467c1bf9c645273ebb75a7ec`.
Those hashes bind inventory and map serialization; no commit yet binds the
uncommitted content.

| Final axis | Exact outcome |
|---|---|
| Combined `gpt-5.6-sol:xhigh` | `APPROVED`, zero defects |
| Ponytail | `Lean already. Ship.`; net `-0` necessary lines |
| Code review — Standards | `APPROVED`, zero findings |
| Code review — Spec | `APPROVED`, zero findings; verified complete 275-path union coverage and classified the full/slow untracked-inventory failures as expected pre-commit behavior, without turning them into green execution |

This self-describing reconciliation changes only the ledger after those review
verdicts; it does not change reviewed source or workflow bytes. Review closure
was therefore satisfied for the 50-path candidate later committed as
`57f4b43e...`. The subsequent final nearest-sampling parity remediation received
new combined, Standards, and Spec approval with zero findings after closing the
f64-semantics P1 and was later committed as `a3260432...`. Ponytail proposed
consolidating its two axis-specific cap checks for net `-6` lines; that finding
was MSW-`REJECTED` because it closed no contract or proof gap. The later
TERMINUS `.expect` removal has focused independent approval plus exact-current
combined, Ponytail, Standards, and Spec approval. A commit of
that remediation, fresh exact-committed wheel and complete matrix, physical
Apple and NVIDIA/Vulkan proof, hosted green checks, publication,
tested-head/PR-head equality, readiness, and live `MERGEABLE/CLEAN` remain
`NOT_PROVEN`.

### Exact commit matrix and nearest-sampling remediation checkpoint (2026-08-27)

The fresh evidence root `/private/tmp/pr170-final-commit.I7YD15` binds the sole
matrix attempt to clean commit
`57f4b43efa5ce8fd5425fdabe6c96b39f03716d6` and tree
`cedc79736eba8b5d8ddb30f1fbea3d2cfdc34e48`. Its newline-terminated artifact
manifest, `/private/tmp/pr170-final-commit.I7YD15/artifact-sha256.txt`, has
SHA-256 `dbda50e0a806636fc0d5727a85b696a8767df7bf6e8195e67a324c4605c48b5b`.

| Exact-commit evidence | Exact result | Status and boundary |
|---|---|---|
| Rust matrix | Formatting, full-feature check, routine clippy, and acceptance clippy passed. The serialized workspace invocation executed 1,436 non-doc tests passed, zero failed, three ignored, and four workflow-filtered, followed by its embedded Cargo doctest block with 12 passed, zero failed, and six ignored. The separate explicit `cargo test --doc` lane independently executed 12 passed, zero failed, and six ignored. | exact `57f4b43e...` Rust bytes `LOCALLY_PROVEN` |
| Fresh locked release-LTO wheel | A new empty external Cargo target and explicit `maturin 1.15.0 build --profile release-lto --locked` produced exactly one wheel. Wheel SHA-256: `7f5192556a04f6d1fd80d5af304f0fea10f5fbb202c98be0721bd524ab6f6fc7`; isolated installed native SHA-256: `1de61254d792e9997223a1f754ae792f74f7bfb2b9f63a435963440eb45bf903`. Import provenance resolved to that isolated installation. | exact `57f4b43e...` artifact `LOCALLY_PROVEN`; after the later source remediation it is `PREDECESSOR_EVIDENCE`, not an exact-current wheel |
| Python fast | 744 passed, zero failed, zero skipped/xfail, with 28 deselected; zero-skip assertion passed. | exact `57f4b43e...` fast lane `LOCALLY_PROVEN` |
| Python slow | two passed, zero failed, zero skipped/xfail, with 4,526 deselected; zero-skip assertion passed. | exact `57f4b43e...` slow lane `LOCALLY_PROVEN` |
| Python full | 4,038 passed, four failed, 39 skipped, and 447 deselected. All 39 skips were runtime loopback tests blocked by host `EPERM`. The four failures were the complete `tests/test_world_coord_f32_gate.py` inventory family: actual 1,557 conversions and digest `ef3c73cee53740ae544c5d707877bb7854b4a6d163201f68e543bb2595cca54e`, versus the frozen 1,555 inventory, caused exactly by `dims.0 as f32` and `dims.1 as f32` in `sample_height_nearest`. | exact `57f4b43e...` full lane `RED`; zero-skip acceptance not met |
| Apple physical acceptance | The single no-warm-up/no-retry attempt ended with one `no_adapter` infrastructure error, zero skips, and no attributable adapter identity; none of the 266 manifest executions ran. | physical identity and complete 266-case matrix `NOT_PROVEN` |

The first subsequent source attempt changed only
`src/terrain/renderer/visibility_buffer.rs::sample_height_nearest` to widen the
clamped `f32` coordinate and `u32` extent to `f64`. It removed the two inventory
sites and passed its scoped checks, but combined and Spec review correctly
rejected it because f64 multiplication can select a different texel than WGSL's
f32 multiplication. That attempt is superseded diagnostic predecessor history,
not the current implementation or current proof.

The implementation later committed at `a3260432...` kept the same source scope
and no frozen inventory, threshold, golden, or Python test changes. It
fail-closed when either texture extent exceeded Forge3D's accepted WebGPU limit
of 32,768 texels, converted exactly through `u16 -> f32`, and then used the same
f32 clamp/multiply/floor/min sequence as `terrain_pbr_pom.wgsl` and
`terrain_shadow_depth.wgsl`. Its normalized-coordinate and boundary tests prove
texel 7 for `uv=0.7, dims=10` and texel 1 for
`uv=1.0_f32/257.0_f32, dims=257`. Combined, Standards, and Spec review
`APPROVED` those committed parity bytes with zero findings and closed the parity
P1; Ponytail's optional net-`-6` cap-check consolidation was MSW-`REJECTED`.
The later exact full lane exposed that its two checked conversions added two
`.expect` sites, so those reviews do not by themselves approve the current
post-TERMINUS bytes.

Because production source changed after both the `57f4b43e...` and
`a3260432...` exact-commit builds, their wheel, native-library, and matrix
artifacts remain lossless predecessor evidence only. A new commit containing
the scoped TERMINUS remediation, a fresh empty-target locked release-LTO build,
and a complete exact-commit full lane are required. Physical
Apple and NVIDIA/Vulkan acceptance, green exact-head hosted checks and bound
artifacts, commit/push, tested-head/PR-head equality, readiness, and live
`MERGEABLE/CLEAN` all remain `NOT_PROVEN`.

### Exact `a3260432...` local matrix and TERMINUS remediation checkpoint (2026-08-27)

The phase-A evidence root `/private/tmp/pr170-final-a326.huWAhp` froze a clean
worktree at commit `a3260432341585eea1e486bccff4fa898c824f4a` and tree
`97eb7a74612f2c12b645e124dff37c5c13524997`. Its 20-row phase-A artifact
manifest, `/private/tmp/pr170-final-a326.huWAhp/artifact-sha256.txt`, has
SHA-256 `a22312fad39c204f10224e929aafc1e513db7e5adc365a514bebed1c0c9d4be3`.
The later root-owned full-lane files are separately hashed below and are not
silently treated as members of that earlier manifest.

| Exact `a3260432...` evidence | Exact result | Status and boundary |
|---|---|---|
| Rust phase A | Formatting, portable full-feature check, routine clippy, and acceptance clippy passed. The serialized workspace invocation executed 1,436 non-doc tests passed, zero failed, three ignored, and four workflow-filtered; its embedded Cargo doctest block executed 12 passed, zero failed, and six ignored. The separate explicit doctest lane independently executed 12 passed, zero failed, and six ignored. | exact committed Rust bytes `LOCALLY_PROVEN` |
| Fresh locked release-LTO wheel | One new empty external Cargo target and venv `maturin 1.15.0 build --profile release-lto --locked` produced exactly one wheel; the log explicitly reports `release-lto`. Wheel SHA-256: `015a364dc21423a90066bff4e30bea02e52da50a87b5c739d0f98c166d45033d`; isolated installed native SHA-256: `9e44efa6f78a4742696c7cc4fc2e4bc913a672c68a3b0d4be57a063cf39da7e5`; import provenance resolved to that isolated venv. | exact committed artifact `LOCALLY_PROVEN`; after the later source edit it is `PREDECESSOR_EVIDENCE` |
| Generic fast | 744 passed, zero failed, zero skipped/xfail, and 28 deselected; its 744-selected/28-deselected ledger and JUnit exist, and the zero-skip verifier passed. | exact committed fast lane `LOCALLY_PROVEN` |
| Generic slow | Two passed, zero failed, zero skipped/xfail, and 4,526 deselected; its two-selected/4,526-deselected ledger and JUnit exist, and the zero-skip verifier passed. | exact committed slow lane `LOCALLY_PROVEN` |
| Apple physical acceptance | The authoritative runner was invoked once without prior probe, warm-up, or retry. It produced one `no_adapter` infrastructure error, zero failures, zero skips, no attributable adapter record, and executed zero of the required 266 cases. | exact physical identity and all 266 executions `NOT_PROVEN` |
| Root-owned loopback-enabled generic full | The exact committed lane ran once outside the restrictive sandbox: 4,078 passed, three failed, zero skipped, and 447 deselected. The selection ledger exists. The zero-skip verifier was red only because the JUnit contained three failures, not because of any skip. Full log SHA-256 `50ac5f3cb99ddca1c5260aa2759ea7c28d5e904314ff7e86c335c733d359cb56`; verifier log `884916c2a7790af1587aec00b01c07a024c38d4cf89555026409edcef7795423`; JUnit `1582fdfc2519f3b0b9d19b9b3953475463e45b2235521ede3a35a5b0149a160b`; selection ledger `5755126eb0b279e9ec8a42f69fb4fa751bb78c45004f526081ee7b6b5b8c7105`. | exact committed full lane `RED`; zero-skip execution accounting is complete, but clean acceptance is not proven |

All three full-lane failures belong to one exact TERMINUS cause introduced by
the two checked `u16::try_from(...)` conversions in
`src/terrain/renderer/visibility_buffer.rs`: terrain `.expect` count was 17
against the ratchet limit 15, the file newly appeared outside the source
allowlist, and the ablation prerequisite consequently also failed. No test,
ratchet, threshold, allowlist, inventory, or golden was changed.

The sole subsequent production edit removes those two `.expect` calls. The
existing axis-specific `dims <= 32,768` assertions remain authoritative; after
that proven bound, `dims.0 as u16` and `dims.1 as u16` are mathematically
lossless, introduce no fallback, and retain exact `f32::from(u16)` plus the
unchanged WGSL-parity clamp/multiply/floor/min sequence. TERMINUS is restored to
`panic=1`, `unwrap=31`, and `expect=15`, and the visibility-buffer path is again
absent from the source allowlist inventory. The full ratchet including ablation
passed 5/5; world-coordinate inventory passed 11/11; exact Rust boundary tests
passed 2/2; shader locks passed 2/2; focused nonphysical visibility/Rainier
contracts passed 9/9; formatting and both clippy gates passed. Independent
scoped review `APPROVED` this remediation with zero findings.

Because source changed after every `a3260432...` artifact and matrix lane, its
wheel, native library, JUnits, and selection ledgers are now exact committed
predecessor evidence, not exact-current proof. A new commit, empty-target locked
release-LTO build, and complete exact-current matrix including one clean full
lane remain required. Physical Apple and NVIDIA/Vulkan acceptance, green
exact-head hosted checks and bound artifacts, push, tested-head/PR-head
equality, readiness, and live `MERGEABLE/CLEAN` remain `NOT_PROVEN`.
