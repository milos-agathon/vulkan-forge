# MENSURA M-06 Option 2 — Full Geospatial Viewer Anchoring

Status: implementation complete on `codex/m06-fgv-audit-remediation`; merge
acceptance remains pending until the required NVIDIA/Vulkan PR job is green with
zero skipped M-06 tests. No signed recipe certificate refresh is required: the
target recipe passes on local Metal with the fresh release extension and its
shared signed shader hashes are unchanged.

The authoritative contract is
[`mensura-m06-full-geospatial-viewer-spec.md`](./mensura-m06-full-geospatial-viewer-spec.md).
This note describes the implementation and the evidence that the PR must
produce. It does not substitute local preflight results for the required CI
result.

## Runtime contract

The interactive viewer owns one persistent `camera_anchor`. At the beginning of
each frame it selects one complete camera pose using terrain, point-cloud, then
general-camera precedence. Terrain rebases against the horizontal camera target;
the other camera modes rebase against the eye. The selected `FrameCamera` copies
the anchor and supplies view, projection, screen render, snapshot, motion blur,
labels, point clouds, fog, GI, sky, objects, and picking for that frame.

Absolute world positions remain `f64` until subtraction from the copied anchor.
`Anchor::narrow` in `src/camera/anchor.rs` is the only world-coordinate
`f64 -> f32` implementation. Render vertices, colors, normalized UV extents,
local density-volume coordinates, and local scatter transforms remain `f32`
because they are not absolute world coordinates.

The production rebase site is `Viewer::prepare_frame_anchor`. A rebase performs
one deterministic refresh before encoding:

- point-cloud, vector, label, object, and picking caches are repacked from their
  persistent absolute or local sources against the new anchor;
- point-cloud and vector GPU buffers are updated through existing `COPY_DST`
  allocations;
- vector BVHs are rebuilt from the actual render vertices and feature IDs;
- TAA, SSGI, SSAO, SSR, fog, motion, and terrain histories are invalidated in
  place; and
- the previous view-projection state is reset to the current copied frame.

SSGI and TAA history reset APIs are infallible and allocation-free. The resource
ledger, rebase count, history-invalidation count, active camera, effect flags,
point-cloud CPU/render/device bytes, and host-visible budget state are exposed by
`get_stats` for live adjudication.

## Coordinate ownership

| Source | Persistent representation | Render boundary |
|---|---|---|
| General, terrain, and point-cloud cameras | `DVec3` eye/target | copied `FrameCamera` through `Anchor::view_look_at` |
| Object transforms | local mesh plus `DVec3` translation | one anchor-relative model translation; rotation/scale remain local |
| Terrain | `RasterInfo`, `DVec2` origin/span | physical footprint `(c,-f)` and `(a*w,-e*h)` packed from the copied anchor |
| Point cloud | LAS-derived `Vec<PointSource3D>` with `DVec3` positions | persistent f32 render cache and existing `COPY_DST` instance buffer |
| Vector overlay | typed 8-lane rows: XYZ `f64`, RGBA `f32`, ID `u32` | anchor-packed render vertices; drape UVs use the physical terrain footprint |
| Labels/callouts | `DVec3` point or `Vec<DVec3>` polyline | anchor-packed projection cache |
| CityJSON | transformed positions and tessellation in `f64` | local direction conversion through `Anchor` |
| Picks | render-frame intersections | restored to absolute `f64` with the exact copied frame anchor |
| Scatter/density volumes | terrain-local `f32` contract | scaled through the same non-square physical origin/span as terrain |

IPC absolute-world fields use `f64`. Vector rows reject any length other than
eight, non-finite XYZ/RGBA, non-integral IDs, negative IDs, and IDs above
`u32::MAX`. Commands are stable-partitioned so frame-establishing loads/camera
changes run before content commands in the same batch. Camera, content, object,
point-cloud, vector, label, and scene-review changes validate against one copied
prospective frame before publishing state.

## Terrain metadata policy

Only a genuinely absent GeoTIFF transform selects the legacy local footprint:
origin `(0,0)`, span `(width,height)`, and no invented CRS. A present transform
must be finite, north-up, axis-aligned, positive in map X, negative in map Y, and
produce finite positive spans. Rotation, shear, mirroring, south-up axes, zero
span, malformed metadata, and metadata-read failures are typed errors.

`load_terrain` validates metadata synchronously at the IPC trust boundary before
the command is enqueued, before a terrain scene is constructed, and before GPU
allocation. Simple relief, PBR, CSM, vector draping, density volumes, scatter,
DoF, screen rendering, and snapshots use the same physical footprint and copied
anchor.

## Atomicity and bounds

Earth-scale coordinates are valid. The safety boundary applies only to the
anchor-relative residual and rejects any camera or content point more than
1,000,000 m from its prospective frame. Non-finite data is always rejected.

Scene-review installation is transactional. It validates the complete effective
scene, stages all new raster/vector/label/scatter runtime objects, rolls the
staged IDs back on any failure, removes the previous runtime only after the stage
succeeds, and publishes the registry snapshot last. A rejected finite-but-distant
label or vector therefore cannot partially update the runtime or query surface.

## Required evidence

The required CI job is `M-06 Full Geospatial Viewer (NVIDIA Vulkan)` in
`.github/workflows/ci.yml`. It runs on the self-hosted NVIDIA Windows runner with
`WGPU_BACKEND=vulkan`, checks out LFS data, installs the current wheel, builds a
fresh release `interactive_viewer`, requires a real hardware terrain probe, and
runs the source gates plus `tests/test_m06_full_geospatial_viewer.py`. A standard
JUnit parser fails the job for zero collected tests, any failure/error, or any
skip. `Full Acceptance Summary` requires this job whenever M-06 is selected;
`PR Core Success` deliberately makes no physical-hardware claim.

The live file records backend, images, frame metrics, resource counts, and viewer
logs under `tests/artifacts/m06`. It measures:

- pixel parity between missing-transform local terrain and the same terrain at
  translated Earth-scale coordinates;
- fail-closed signed-transform rejection with unchanged GPU ledger counts;
- non-square PBR/CSM/DoF/volumetric/scatter rendering with enabled
  TAA/SSGI-temporal/SSR/fog telemetry;
- a full terrain orbit with zero rebase/history churn plus consecutive-frame
  no-flash SSIM/MAE;
- object-transform and 500,000-point coexistence under terrain camera ownership,
  exact source/render/device byte reporting, and ten allocation-free rebases;
- live one-millimetre red/blue separation at Earth magnitude with a zero-distance
  negative control; and
- scene-review rollback after a render-frame residual rejection.

Source gates independently reject hidden component/index/helper narrowing,
multiline or duplicate matrix producers, additional production rebase sites,
raw GPU allocations, and historical local-frame enforcement patterns. Unit tests
also cover arbitrary/missing CRS, malformed metadata, non-square scatter and
density mapping, exact typed-vector parsing, vector precision across repeated
rebases, point-cloud buffer reuse, active-camera precedence, snapshot
composition, mouse picking, and transaction ordering.

## Local preflight evidence

The predecessor pre-PR Metal preflight recorded the following results with an
ABI3 wheel and release viewer. They remain supplementary historical regression
evidence; the final fail-closed M-06 entry point now rejects Metal before live
collection and only the required NVIDIA/Vulkan job can establish release
acceptance:

- local/translated-terrain parity: SSIM `0.9999999993148123`, MAE
  `7.978e-09`, maximum channel difference one byte; the signed-transform
  negative control was rejected without changing the `1047` buffer / `26`
  texture ledger;
- all required effects reported enabled; a full orbit produced exactly one
  initial rebase/history invalidation and no additional churn, steady resources
  at `1075` buffers / `54` textures, and consecutive-frame no-flash SSIM `1.0`
  with MAE `0.0`;
- host-visible peak was `656404` bytes and the tracked total was approximately
  `325.96` MB, both within policy;
- the 500,000-point case reported exact source/render/device totals of
  `24000000` bytes each, retained terrain camera ownership, and held resource
  counts at `1051` buffers / `26` textures through ten rebases;
- the live millimetre case rendered one red and one blue pixel exactly one pixel
  apart, while the zero-distance control produced no red separation; and
- the scene-review residual rejection preserved the prior runtime and registry
  exactly.

The signed `mapscene_terrain_raster` recipe golden also passes against the fresh
release extension on Metal. Its built-in IBL resolution is hermetic: optional
untracked demo HDRIs cannot change signed-recipe pixels. Viewer shadow depth now
has its own 128-byte camera ABI shader, while the offscreen signed recipe keeps
its original shared 112-byte shader. The shared
`terrain.shadow_depth.shader` and `terrain_pbr_pom.shader` hashes therefore
remain unchanged and no protected certificate-refresh run is needed.

## Audit remediation reconciliation

The source audits did not assign stable `Axx` identifiers. This implementation
ledger assigns `A01` through `A37` in audit order so every distinct finding can
be tracked without collapsing it into a theme. `FULL` means the production path
and locally admissible proof are complete. NVIDIA/Vulkan-only rows remain
`NOT_PROVEN`; local Metal is never substituted for them.

| ID | Requirement / original audit gap | Status | Production and test evidence | Remaining deficiency |
|---|---|---|---|---|
| A01 | Exact world-coordinate f64-to-f32 inventory | FULL | `test_world_coord_f32_gate.py` inventories operations and rejecting probes; `Anchor::narrow` is the sole sanctioned implementation. | None. |
| A02 | Exact matrix-producer inventory | FULL | `test_m06_viewer_matrix_contract.py` keys file/function/operation/ordinal and rejects multiline, alias, inverse, duplicate, and previous-VP probes. | None. |
| A03 | One persistent viewer anchor and one frame-start rebase | FULL | `test_m06_single_rebase_contract.py` inventories storage, construction, aliases, delegates, and the sole mutation. | None. |
| A04 | Object transform applies translation/rotation/scale exactly once | FULL | Absolute translation remains f64; vertices stay local; `anchored_object_model` is the sole render transform and is shared by visible, fog-shadow, and pick paths. | None. |
| A05 | One terrain > point-cloud > general camera for the whole frame | FULL | Frozen `FrameCamera` drives geometry, terrain, labels, sky/fog, GI, points, snapshots, and picking; precedence/permutation tests are green. | None. |
| A06 | Point-cloud snapshot composes over terrain | FULL | Snapshot point pass loads the existing color target; composition tests assert terrain is not cleared. | None. |
| A07 | Exact GeoTIFF transform decoding and typed rejection | FULL | Exact 16-value ModelTransformation and scale/tie parsing reject partial, malformed, non-finite, signed, rotated, sheared, and zero-span inputs. | None. |
| A08 | One synchronous metadata ingress before enqueue/allocation | FULL | `load_terrain` uses one preflight/parser; metadata failures cannot become missing-transform fallback. | None. |
| A09 | Complete non-square physical footprint across terrain consumers | FULL | PBR, simple, shadow, drape, scatter, density, DoF, snapshot, and picking use one origin/span contract. | None. |
| A10 | Terrain Rust/WGSL sizes, offsets, alignment, and bindings | FULL | Compile-time 160/128/256-byte assertions plus WGSL field-order and minimum-binding gates are green. | None. |
| A11 | Allocation-free deterministic temporal reset | FULL | Shared validity state resets TAA/SSGI/SSAO/SSR/fog without resource creation; first post-rebase blend is bypassed. | NVIDIA visual no-flash evidence is tracked separately in A35. |
| A12 | Generic-object fog shadows and separate 128-byte viewer ABI | FULL | Viewer shadow uniforms contain light VP plus anchored object model; terrain-only scenes do not unwrap a missing geometry buffer; signed 112-byte shader is untouched. | NVIDIA visual shadow evidence is tracked separately in A35. |
| A13 | Prospective-frame residual validation for all content | FULL | Camera, terrain, object, label, vector, and point content validate the whole prospective batch before state publication. | None. |
| A14 | Stable frame-establishing command order | FULL | `command_batch.rs` stable-partitions terrain/point/camera before content; all 24 Rust permutations pass. | None. |
| A15 | Correlated IPC completion and rendered-frame fencing | FULL | Command IDs and applied/rendered revisions prevent enqueue acknowledgements from masquerading as completed snapshots or picks. | None. |
| A16 | Transactional scene-review installation | FULL | Complete scene staging includes labels, vectors, rasters, BVHs, and scatter; failpoints roll back IDs/runtime and publish registry last. | NVIDIA rollback evidence is tracked separately in A36. |
| A17 | Label, line, curved-label, and callout f64 retention | FULL | Rust serde/storage and Python payload tests preserve sub-millimetre Earth-scale values to the anchor boundary. | None. |
| A18 | Exact eight-lane vector schema and ID validation | FULL | XYZ f64, RGBA f32, ID u32; wrong lanes, non-finite values, fractional/negative/overflow IDs reject synchronously. | None. |
| A19 | Vector repack and BVH use actual render vertices and feature IDs | FULL | Rebase repacks through existing `COPY_DST`; BVH rebuild/failpoint and repeated-rebase precision tests pass. | None. |
| A20 | Point-cloud f64 source and reusable render/device buffers | FULL | Source positions remain f64; render cache is anchor-packed; rebases rewrite existing buffers. | None. |
| A21 | Exact point/vector/terrain/temporal telemetry | FULL | Source, render, device, tracked, and owner/resource IDs report real allocations rather than estimates. | NVIDIA memory evidence is tracked separately in A36. |
| A22 | Point-cloud culling, visibility, and picking share the frame | FULL | Active-camera frustum, screen/snapshot visibility, and absolute pick restoration are exercised together. | NVIDIA 500k coexistence is tracked separately in A36. |
| A23 | Mouse/IPC picks use frozen-frame matrices and return absolute f64 | FULL | `pick_at` and queued mouse picks execute exactly once against the rendered revision and restore the copied anchor. | NVIDIA millimetre output is tracked separately in A36. |
| A24 | CityJSON transforms and normals retain precision | FULL | Transformed positions remain f64; normals derive from local direction differences and tests cover Earth-scale transforms. | None. |
| A25 | 3D Tiles hierarchy/content transforms retain f64 and exact bytes | FULL | World transforms/bounds and PNTS RTC offsets remain f64; render packing occurs after scene anchoring and telemetry includes f64 source bytes. | None. |
| A26 | Density/scatter identity is independent of render-origin metadata | FULL | Immutable voxel/content identity excludes `render_origin_span`; rebases update only metadata/uniforms without allocation. | None. |
| A27 | JUnit zero-skip verifier is fail closed | FULL | Nested/root suites, malformed counters/XML, contradictions, zero tests, failures, errors, skips, and xfails have rejecting fixtures. | None. |
| A28 | CI preserves pytest and verifier failures and uploads evidence | FULL | Workflow records both exit codes, always verifies an existing XML, always uploads evidence, and fails if either command fails. | Runtime artifact contents are tracked separately in A37. |
| A29 | CI tests the exact installed wheel, not repo-local Python/native files | FULL | Installed-wheel path gate requires `FORGE3D_NO_BOOTSTRAP=1` and rejects repo-local package or extension resolution. | Exact required-lane execution is tracked separately in A37. |
| A30 | Hardware probe fails closed on adapter absence/crash | FULL | Probe distinguishes ABSENT from CRASH; the required lane requires physical NVIDIA/Vulkan and a fresh release viewer. | Actual required adapter execution is tracked separately in A37. |
| A31 | Signed recipe, shader hashes, and certificates remain protected | FULL | Target Metal recipe passes; certificate signature verifies; deterministic built-in IBL ignores optional local HDRIs; no golden/certificate refresh occurred. | Cross-backend determinism remains a required-lane concern. |
| A32 | Neighboring terrain PBR and vector-overlay behavior | FULL | Local Metal integration completed 24 tests with zero skips after the fresh release build. | Metal is supplementary, not release proof. |
| A33 | Live M-06 harness and adversarial negative controls | FULL | Native raster writer plus rasterio oracle, SSIM/MAE controls, cast-before-subtract control, typed transform failures, and zero-skip artifact schema are implemented and source-tested. | Hardware results are tracked in A34-A37. |
| A34 | Live translated/rebased terrain equivalence on NVIDIA/Vulkan | NOT_PROVEN | Required test is wired fail closed. | The exact published remediation head must complete the required external job green. |
| A35 | Live effects, full orbit, CSM/fog/temporal no-flash on NVIDIA/Vulkan | NOT_PROVEN | Required assertions and artifacts are wired. | The exact published remediation head must complete the required external job green. |
| A36 | Live 500k points, 1 mm separation, resource stability, and rollback on NVIDIA/Vulkan | NOT_PROVEN | Required assertions and negative controls are wired. | The exact published remediation head must complete the required external job green. |
| A37 | Exact-head required CI: zero skips/failures/errors and uploaded evidence | NOT_PROVEN | `Full Acceptance Summary` depends on `test-m06-full-geospatial-viewer` whenever selected; JUnit and evidence validators are fail closed. | Refresh the selected exact-head job after publication; local evidence is not release proof. |

| Specification task | Status | Local closure | Remaining deficiency |
|---|---|---|---|
| M06-FGV-00 | FULL | Enforcement, allocation, inventory, JUnit, workflow, and negative-control gates are implemented and green. | None. |
| M06-FGV-01 | FULL | Metadata, fallback, arbitrary/missing CRS, typed rejection, and height-tag contracts are green. | None. |
| M06-FGV-02 | FULL | Single owner/rebase, active camera, frozen frame, and pre-encode refresh contracts are green. | None. |
| M06-FGV-03 | FULL | Uniform/WGSL/pipeline contracts and protected local Metal recipe parity are green. | NVIDIA runtime is included in M06-FGV-09. |
| M06-FGV-04 | FULL | Exact matrix inventory and all named subsystem producers are classified and frame-bound. | None. |
| M06-FGV-05 | FULL | Persistent precision, repack/BVH, picking, point buffers, telemetry, and temporal resets are green. | NVIDIA runtime is included in M06-FGV-09. |
| M06-FGV-06 | FULL | Rust serde, exact vector schema, scene-review precision, and transaction tests are green. | None. |
| M06-FGV-07 | FULL | Python payload validation, public API, docs, and stubs are green. | None. |
| M06-FGV-08 | FULL | Prospective residual boundary, no-mutation errors, and sole-narrowing gates are green. | None. |
| M06-FGV-09 | NOT_PROVEN | Required NVIDIA/Vulkan workflow and live harness are fully wired. | Exact-head hardware job must complete green with zero skips and valid evidence. |
| M06-FGV-10 | FULL | Compatibility, signed-recipe protection, neighboring integration, implementation plan, and handoff documentation are complete locally. | Release verdict remains gated by M06-FGV-09. |

## Certificate policy

This change does not weaken recipe certificate verification and does not write a
private signing key or unsigned replacement certificates. The protected
certificate-refresh workflow remains the only authorized signing path. The
packaged target recipe verification above establishes that this remediation did
not change its signed dependency hashes. Any future reviewed WGSL dependency
change that does alter a signed recipe hash remains blocked until the protected
workflow produces and verifies the refreshed certificate set.
