# Fable 5 Moonshot Prompts for forge3d — Round 2

Date: 2026-07-08

Ten new **implementation** prompts (`13`–`22`), written after 01–12 and after
AEQUITAS (#01) and SUTURA (#10) landed. They are disjoint from the first twelve:
where round 1 asked *"can forge3d be physically correct?"*, round 2 asks four
harder questions the tree's current state makes unavoidable:

1. **Can it be geodetically correct?** (`13`, `15`, `22`) — today a polygon's area
   is reported in square degrees, ECEF is truncated to `f32` at Earth radius, a
   failed reprojection returns an all-nodata raster with a success status, and a
   lake 200 km across is drawn as a plane 785 m above its own far shore.
2. **Can it stop lying about itself?** (`14`, `17`) — `required_features:
   Features::empty()` structurally guarantees the 450-line GPU-timing subsystem
   can never fire; `check_budget` has one caller and warns by default while roughly
   190 raw `create_buffer` sites bypass the tracker; 12 of 243 pytest files run in CI;
   four Cargo features passed to `cargo test` are referenced nowhere in `src/`.
3. **Can it render what it does not know?** (`16`) — every DEM is a random field
   and every pixel is currently rendered as if its elevation were exact.
4. **Can it reach a reader?** (`18`, `21`) — labels are drawn with
   `(ord(char) >> ((row*5+col) % 7)) & 1` when Pillow is absent, and a legend's
   colours are hashed, so nothing survives a deuteranope or an offset press.

Every prompt follows the house style (Objective · Context — verified anchors ·
Operating rules · What to build · Public API/shader changes · Definition of done ·
Tests & validation · Non-goals · Verification). Every anchor was verified against
the live tree on 2026-07-08. Every prompt carries a **measurable win** with hard
thresholds, an **ablation** that proves the hard part is load-bearing (so it
cannot be quietly implemented as a no-op), and a CI gate.

Here, a "CI gate" means a validation claim assigned to the appropriate tier.
It does not make every moonshot's exhaustive or physical proof a routine
pull-request requirement; `docs/censor-validation-policy.md` defines that
boundary.

> **Verify before you trust.** Anchors drift. Re-run
> [`00-codex-audit-prompt.md`](00-codex-audit-prompt.md) against the current tree
> before executing any of these.

## The portfolio

| # | Codename | Track | The moonshot | Measurable win |
|---|----------|-------|--------------|----------------|
| 13 | [MENSURA](13-mensura.md) | Hybrid | Datum/unit/height errors made **uncompilable**; `f64` world + geoid + geodesics | 4326→UTM→3857→ECEF→4326 closure < 1e-9°; EPSG G7-2 to 1 mm; EGM96 < 0.5 m; 5 `compile_fail` doctests |
| 14 | [CENSOR](14-censor.md) | Hybrid | Signed `RenderCertificate` proving **no fallback ran** | current raw allocations → 0 untracked allocations; budget enforce-by-default; real `gpu_ms`; `degradations == []`; 0 dead features |
| 15 | [HELIOS](15-helios.md) | Singularity | Shadows on a curved, refracting Earth, **validated against the sky** | NREL SPA to 0.0003°; viewshed IoU ≥ 0.98 vs GRASS; flat-earth ablation IoU ≤ 0.96 |
| 16 | [DUBIUM](16-dubium.md) | Singularity | The map renders its own **error bars** | Analytic σ within 5% of a 1000-realization MC; `p_lit` Brier < 0.01; ≤ 3× cost |
| 17 | [ANAMNESIS](17-anamnesis.md) | Singularity | Content-addressed incremental rendering — **Bazel for maps** | Incremental == cold, byte-for-byte; exact recompute set; ≥ 20×; 10k-mutation hermeticity at 1.000 |
| 18 | [LITTERA](18-littera.md) | Core Supremacy | Real OpenType shaping + true MSDF + outline vector export, **no HarfBuzz** | 100% `hb-shape` glyph/advance match; `BidiCharacterTest` 100%; three surfaces agree at ΔE2000 < 2 |
| 19 | [TESSELLA](19-tessella.md) | Core Supremacy | GPU-driven, HZB-culled, **visibility-buffer** terrain over a disk-backed BC7 VT | ≥ 256 GiB virtual texture, 0 fallback texels, < 512 MiB; culled render **bitwise identical** |
| 20 | [NEPHELE](20-nephele.md) | Core Supremacy | Participating media closed against **ratio tracking** | Energy closes to 1e-3; froxel vs unbiased reference ΔE2000 < 2.5; `textureSampleCompare` in compute = 0 |
| 21 | [CHROMA](21-chroma.md) | Singularity | CVD-safe palettes + **ICC/CMYK separations** with a hard ink limit | ICC round trip ΔE2000 < 0.5 mean; TAC ≤ 300% with 0 violations; min pairwise ΔE2000 ≥ 15 under 6 conditions |
| 22 | [FLUVIUS](22-fluvius.md) | Singularity | Hydrologic terrain + **geopotential lake surfaces** | Bit-exact parallel flow accumulation; stream F1 ≥ 0.95 vs GRASS; lake deviates 785 ± 8 m from its plane |

## Dependency-aware build order

Two keystones gate the rest, exactly as PROMETHEUS and AEQUITAS did in round 1.

1. **CENSOR (14)** — landed architectural substrate. It deleted
   `Features::empty()`, which silently invalidated timestamps, bindless, and BC
   compression. TESSELLA (19) consumes its negotiated capabilities; ANAMNESIS
   (17) consumes its shader hashes and capability fingerprint; other prompts use
   its degradation/allocation/certificate interfaces. That dependency does not
   require rerunning CENSOR's full matrix, signing, goldens, physical-GPU proof,
   or scratch red proof on every downstream pull request.
2. **MENSURA (13)**, in parallel — the `f64`/geoid/geodesic substrate. HELIOS (15),
   DUBIUM (16), and FLUVIUS (22) are all blocked on it and should not hand-roll
   a second ellipsoid.
3. **LITTERA (18)** and **NEPHELE (20)**, independent of the above, can run any
   time; NEPHELE wants PROMETHEUS (#02)'s `terrain_trace`, which has landed.
4. **TESSELLA (19)** consumes CENSOR. **ANAMNESIS (17)** consumes CENSOR and
   TERRA-DETERMINATA (#04) — a cache without bit-exactness is unsound.
5. **The crowns** — HELIOS (15), DUBIUM (16), CHROMA (21), FLUVIUS (22) — once the
   geodesy is typed, the execution is certified, and the metrics exist.

## Why these are impossible elsewhere

Round 1's moat was five pillars: *geospatial correctness · reproducibility ·
WebGPU portability · a real GPU path tracer · deterministic cartography*. Round 2
aims at the **pairwise intersections** nobody else occupies:

- **geodesy × ray-marched heightfield** → HELIOS's curved refracting shadows and
  FLUVIUS's geopotential lakes. QGIS has the geodesy and no renderer. Blender has
  the renderer and no datum.
- **geodesy × the type system** → MENSURA. GDAL has a datum *string*; nothing has
  a datum *type* that refuses to add a metre to a degree.
- **determinism × content addressing** → ANAMNESIS. Only bit-exactness makes a
  render cache sound, and only forge3d has a determinism matrix with zero-byte
  tolerance.
- **path tracer × uncertainty** → DUBIUM. Stochastic visibility through a min-max
  pyramid is a probability integral no rasterizer can pose.
- **cartography × colour science** → CHROMA. Mapbox will not separate to CMYK;
  Illustrator will not path-trace a mountain.
- **honesty × CI** → CENSOR. Every engine has degraded paths. None of them signs
  a certificate saying so.

## Held back

Three more were drafted and cut, to keep the ten sharp. Re-add for a fifteen:
**SIMULACRUM** (physically-based sensor simulation: multi-return LiDAR waveforms,
SAR backscatter, thermal IR, validated against real ALS returns by a KS test),
**GENERALIS** (scale-dependent cartographic generalization — Douglas–Peucker /
Visvalingam by zoom, contour–label coupling, hachures, Swiss-style multidirectional
relief), and **EXITUS** (forge3d can read glTF, CityJSON, COPC, and 3D Tiles and
can write only OBJ and STL — close the loop with semantic round-trip guarantees;
note `src/io/mod.rs:3` and `src/lib.rs:132` both advertise a PLY pipeline that does
not exist in any form).
