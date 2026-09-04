# Fable 5 Moonshot Prompts for forge3d — Round 3

Date: 2026-07-12

Twenty new **implementation** prompts (`23`–`42`), written after rounds 1–2 and
after PROMETHEUS (#02), AEQUITAS (#01), TERRA-DETERMINATA (#04), SUTURA (#10),
MENSURA (#13), and CENSOR (#14) landed. Rounds 1–2 made forge3d physically
correct, geodetically typed, deterministic, and architecturally required to
report how it rendered. Round 3 asks six questions that only become
askable *on top of* that substrate — each one aimed where no other engine can
follow:

1. **Can it simulate the world it draws?** (`23`, `24`, `25`) — today water,
   sound, and geological time exist in forge3d only as pixels. A flood is a
   raster somebody else computed; a noise map is an import; a mountain has no
   past. Physics enters the map as keyframes, never as conservation laws.
2. **Can it sense beyond light, and map beyond Earth?** (`26`, `27`, `28`,
   `29`) — the engine renders exactly one signal (visible light), on exactly
   one body (Earth), for exactly one hemisphere of reality (above the ground).
   LiDAR waveforms, SAR layover, the Moon's own datum, the actual night sky,
   and everything below z = 0 are all invisible to it.
3. **Can it prove, not merely test?** (`30`, `31`, `32`, `33`, `34`) — every
   guarantee in the tree today is empirical: a golden, a fuzz run, a threshold.
   Nothing is *proven*: not the absence of NaN in a shader, not the sign of an
   orientation predicate under cancellation, not pixel coverage, not the ulp
   error of planet-scale coordinates on an f32 GPU, not survival at the
   antimeridian.
4. **Can it keep time under oath?** (`35`, `36`) — CENSOR certified *what*
   rendered but not *when*; a frame may take 8 ms or 80 ms and the certificate
   is equally green. And every DEM byte is stored as if bandwidth were free.
5. **Can it face the reader without lying?** (`37`, `38`) — a scale bar drawn
   on a Mercator view at 60°N overstates ground distance by ~2× and forge3d
   will happily render it; a coastline drawn at 1:10M carries 1:10k vertices.
   The engine that certifies its own execution still lets the *map* lie.
6. **Can it close the loop?** (`39`, `40`, `41`, `42`) — forge3d reads five
   scene formats and writes two; `src/io` has advertised a PLY pipeline that
   has never existed in any form; scenes are single-author; cameras are
   hand-keyed; and the path tracer re-learns the same light transport from
   zero every single frame.

Every prompt follows the house style (Objective · Context — verified anchors ·
Operating rules · What to build · Public API/shader changes · Definition of
done · Tests & validation · Non-goals · Verification). Every anchor was
verified against the live tree on 2026-07-12. Every prompt carries a
**measurable win** with hard thresholds, an **ablation** that proves the hard
part is load-bearing, and a validation gate assigned to its appropriate CI or
acceptance tier rather than automatically to routine PR gating. Six of the
twenty (`26`, `38`, `39`, `40`, `41`, `42`) promote ideas rounds 1–2 explicitly held back
(SIMULACRUM, GENERALIS, EXITUS, CONFLUENCE, AUTOCHIRON, AUGUR) from one-line
mentions to full specifications; the other fourteen are new fronts.

> **Verify before you trust.** Anchors drift. Re-run
> [`00-codex-audit-prompt.md`](00-codex-audit-prompt.md) against the current
> tree before executing any of these.

## The portfolio

| # | Codename | Track | The moonshot | Measurable win |
|---|----------|-------|--------------|----------------|
| 23 | [SONITUS](23-sonitus.md) | Singularity | Ray-traced outdoor acoustics — regulation-grade noise maps | Free-field closed forms < 0.1 dB; Maekawa knife-edge ±1 dB; binary-occlusion ablation ≥ 10 dB wrong in shadow zones |
| 24 | [DILUVIUM](24-diluvium.md) | Singularity | GPU shallow-water flooding on real DEMs | Ritter dam-break L2 < 1%; lake-at-rest drift < 1e-6 m over 10k steps; mass closes to 1e-6; bit-exact reruns |
| 25 | [OROGENESIS](25-orogenesis.md) | Singularity | A million years of erosion as a render parameter | Slope–area exponent within 2% of m/n; analytic hillslope profile < 1%; 1 Myr @ 2048² ≤ 120 s, bit-exact |
| 26 | [SIMULACRUM](26-simulacrum.md) | Singularity | Physically based sensor simulation: LiDAR waveforms, SAR, thermal | Waveform vs real ALS: KS D ≤ 0.15; SAR layover/shadow IoU ≥ 0.98 and ≤ 1 px vs analytic; thermal balance < 1 W/m² |
| 27 | [SELENE](27-selene.md) | Hybrid | Planetary cartography — the Moon and Mars as first-class datums | Lunar/Mars round-trip closure < 1e-9°; Mars geodesics vs reference < 1 mm; body mix-ups uncompilable; WGS84-on-Mars ablation > 2,900 km |
| 28 | [SIDERA](28-sidera.md) | Singularity | The astronomically correct night sky | vs JPL Horizons: Sun < 10″, Moon < 30″, planets < 1′; no-precession ablation > 20′; deterministic night golden |
| 29 | [SUBTERRA](29-subterra.md) | Singularity | The map that goes underground — volumetric geology | DVR vs analytic line integrals < 1e-4; isosurface Hausdorff < 0.5 voxel + watertight; honest 512³-f32 budget refusal |
| 30 | [DUPLA](30-dupla.md) | Hybrid | Double-double WGSL — f64 truth on f32 GPUs | Measured error ≤ cited Joldeș–Muller–Popescu bounds over 10⁸ adversarial vectors; mm-at-Everest jitter < 0.01 px vs f32 ≥ 1 px |
| 31 | [LIMES](31-limes.md) | Core Supremacy | Analytic-coverage vector rasterization — exact-area AA | Mean coverage error < 1e-3 vs 4096× supersampling; max < 0.5/255; crack-free joins; MSAA ablation fails the gate |
| 32 | [EUCLIDEA](32-euclidea.md) | Core Supremacy | Exact geometric predicates + snap-rounded booleans, no GEOS | 0 sign errors on 1e7 random + 1e6 adversarial cases vs exact oracle; 1e5 fuzzed booleans 100% valid topology |
| 33 | [PROBATUM](33-probatum.md) | Singularity | Formal shader verification — prove no-NaN, not test it | ≥ 10 core WGSL modules proven NaN/Inf/OOB-free over declared input ranges; seeded-bug ablation caught; CI proof gate |
| 34 | [TERMINUS](34-terminus.md) | Hybrid | The torture atlas — antimeridian, poles, and degenerate inputs | 500-case adversarial corpus: 100% render-or-structured-raise, 0 panics, 0 hangs; in-tree property fuzzer + shrinker |
| 35 | [PULSUS](35-pulsus.md) | Core Supremacy | The frame-pacing contract — latency under oath | 1 h fuzzed soak: p99.9 ≤ vsync budget; 0 unexplained misses; every miss certified with a per-pass timing culprit |
| 36 | [COMPENDIUM](36-compendium.md) | Core Supremacy | A certified-max-error DEM codec with GPU decode | ≥ 30% smaller than the tuned deflate baseline at equal max-error; per-tile error bound machine-verified; CPU/GPU decode bit-identical; ≥ 1 Gpx/s |
| 37 | [ARBITER](37-arbiter.md) | Singularity | The cartographic honesty linter — the map that refuses to lie | 40/40 lying maps caught, 0/40 honest false positives; scale-bar ground-truth error > 5% blocks in strict mode |
| 38 | [GENERALIS](38-generalis.md) | Core Supremacy | Scale-dependent generalization with topology guarantees | Shared-boundary mosaic: 0 gaps/overlaps at every zoom; Hausdorff ≤ declared tolerance; 0 self-intersections |
| 39 | [EXITUS](39-exitus.md) | Hybrid | Close the I/O loop — writers with semantic round-trip proofs | PLY/glTF/CityJSON/3D Tiles writers; import→export→import attribute fidelity 100%, geometry Hausdorff ≈ 0; external validators pass |
| 40 | [CONFLUENCE](40-confluence.md) | Singularity | CRDT collaborative editing with pixel convergence | 10k randomized concurrent-op merges: state hashes equal on all replicas; post-merge renders byte-identical |
| 41 | [AUTOCHIRON](41-autochiron.md) | Singularity | Intent-driven cinematic camera choreography | 0 terrain intersections with min-clearance; framing constraints held ≥ 95% of frames; jerk-bounded C² path; lerp ablation collides |
| 42 | [AUGUR](42-augur.md) | Singularity | An online neural radiance cache that forgets moving shadows | ≥ 4× variance reduction at equal time; bias ΔE2000 mean < 1 vs converged no-cache PT; ghost shadows decay < 2% within 32 frames (frozen-cache ablation > 10%); bit-deterministic training |

## Dependency-aware build order

Round 3 assumes the round-1/2 keystones (PROMETHEUS, AEQUITAS,
TERRA-DETERMINATA, MENSURA, CENSOR) are in the tree — they are. Within round 3:

1. **EUCLIDEA (32)** first among the vector prompts — GENERALIS (38) consumes
   its exact predicates directly, while ARBITER (37) and EXITUS (39) reuse its
   topology validation wherever their workflows inspect geometry rather than
   hand-rolling a second predicate stack. LIMES (31) is independent of it (coverage is
   analytic integration, not topology) and can run in parallel.
2. **TERMINUS (34)** and **PROBATUM (33)** early — they are honesty
   infrastructure in the CENSOR lineage. Later prompts may reuse their fuzz
   corpus and proof infrastructure when relevant; dependency alone does not
   inherit heavyweight gates. PROBATUM needs CENSOR's preprocessed-WGSL hash
   seam, which exists.
3. **The physics trio** — FLUVIUS (#22) has NOT landed (verified 2026-07-12:
   zero flow-routing code in-tree), so OROGENESIS (25) builds the shared
   D8/priority-flood flow-graph substrate FLUVIUS will later adopt; DILUVIUM
   (24) integrates the PDE and needs no flow graph at all. SONITUS (23) needs
   only MENSURA + the extruded-buildings mesh path and can run any time.
4. **SELENE (27)** before **SIDERA (28)** — SIDERA's ephemerides want
   SELENE's body-parameter registry so the Moon exists exactly once.
   SIMULACRUM (26) after PROMETHEUS (landed) — it rides `terrain_trace`.
   Note: HELIOS (#15) and AETHER (#11) have NOT landed (verified 2026-07-12:
   no SPA/`src/geo/solar.rs`, no spectral atmosphere) — SIDERA and SONITUS
   are written to stand without them.
5. **DUPLA (30)** before ORBIS (#05) resumes — absolute-coordinate GPU work
   at planetary extents is unsound without it.
6. **The crowns** — CONFLUENCE (40), AUTOCHIRON (41), AUGUR (42) — last:
   CONFLUENCE needs SUTURA bundles + TERRA-DETERMINATA hashes; AUTOCHIRON
   needs `terrain_trace` for clearance; AUGUR needs AEQUITAS as its bias
   oracle and CENSOR to certify the cache as a declared approximation.

PULSUS (35) adds its own required `pulsus-soak` self-hosted real-GPU CI job;
its 60-minute ladder-on/off acceptance pair is gated there, not treated as a
manual benchmark.

## Why these are impossible elsewhere

Rounds 1–2 built the five pillars and their pairwise intersections. Round 3
occupies the **triple points** — capabilities that need three pillars at once:

- **simulation × geodesy × determinism** → DILUVIUM's floods run on a geoid,
  not a plane, and replay bit-exactly in court. HEC-RAS has the physics and no
  renderer; Houdini has the solver and no datum; neither can sign the run.
- **sensing × path tracer × honesty** → SIMULACRUM's SAR image carries a
  CENSOR certificate naming its scattering model. No remote-sensing simulator
  certifies its own approximations.
- **formal methods × WGSL × CI** → PROBATUM. Shader compilers prove type
  safety; nobody proves *value* safety (no-NaN) of shipping map shaders on
  every commit.
- **cartography × metrology × refusal** → ARBITER. Every GIS draws whatever
  scale bar you ask for. An engine with MENSURA's typed geodesy is the first
  that can *know* the bar is wrong — and CENSOR taught it how to refuse.
- **precision × the GPU's own limits** → DUPLA does on an f32 lane what
  vendors reserve for f64 silicon, with proven error bounds — the difference
  between "looks stable" and "is stable" at planet scale.
- **collaboration × bit-exactness** → CONFLUENCE. Figma merges documents;
  only a deterministic renderer can promise two artists the *same pixels*
  after a merge.

## Held back

Three were drafted and cut to keep the twenty sharp. Re-add for a twenty-five:
**VENTUS** (mass-consistent wind fields over terrain validated against the
Askervein Hill dataset, driving scatter/particle advection), **SPECULUM**
(MapLibre style-spec conformance: an expression VM + the official render-test
fixture corpus as a CI gate), and **LEGIO** (split-frame multi-adapter
rendering, byte-identical to single-GPU at ≥ 1.7× scaling — TERRA-DETERMINATA
across devices in one frame).
