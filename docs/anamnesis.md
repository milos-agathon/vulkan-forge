# ANAMNESIS content-addressed renderer

`forge3d.anamnesis` is the deterministic offline build system for frame
sequences. `render_sequence(recipe, cache=...)` declares a Merkle pass graph and
keys every output from:

- the stable pass label;
- the complete pipeline/copy descriptor bytes;
- the exact uploaded uniform bytes, including initialized padding;
- role-separated input keys such as `heightmap@0` and `water_mask@1`;
- negotiated features, codegen limits, backend, DX12 compiler policy, and Naga
  capability policy;
- crate version, full Git SHA, locked Naga version, and the preprocessed WGSL
  tree identity.

Unknown opaque-renderer state is never guessed. A `render_frame` callback must
provide both `render_frame_fingerprint` and `render_frame_context`, and its
output key includes the complete pixel-affecting recipe projection. Structured
`pass_executors` must provide a fingerprint and captured-input context for
every executor; this is the supported boundary for exact pass-level
invalidation.

The prediction is independent of execution. ANAMNESIS first compiles every key,
then compares the resulting manifest with the prior manifest before reading the
store or invoking an encoder. Changed keys are forced to execute; unchanged
keys must restore. A missing unchanged entry is reported as a prediction
mismatch instead of quietly redefining the prediction from the cache lookup.

`cache=None` performs no cache I/O and follows the original render path. A
cache path never permits a hit from incomplete material: unsupported native
terrain state conservatively recomputes.

## Store and diagnostics

The Rust and Python implementations use the same
`forge3d.anamnesis.store/1` metadata and LRU clock. Entries retain the canonical
key material needed to reconstruct the key. Blob bytes, metadata, prefix and
entry directories, control files, and quarantine all count toward the hard
`max_bytes` on-disk bound. Corrupt data is quarantined and its build manifest
is invalidated before the next render.

Warm sequence builds keep a bounded, signature-checked in-process L1 for stores
they just populated. A changed blob or metadata timestamp invalidates that L1
entry and routes through the normal integrity/quarantine path. Unchanged
ancestors committed by a restored descendant key are elided; only terminal
outputs and the direct inputs of actual misses are materialized. LRU touches
are appended once per sequence to the shared `access.log` journal, which both
the Python and Rust garbage collectors consult. This avoids thousands of
per-hit metadata rewrites without weakening cross-language eviction order.

The store defaults to `.forge3d/cache`. Use:

```text
python -m forge3d.anamnesis explain <key>
python -m forge3d.anamnesis verify
python -m forge3d.anamnesis gc <max_bytes>
python -m forge3d.anamnesis render-sequence recipe.json --frames 600
```

`explain` reconstructs and checks each pass key and recursively follows its
named inputs. `verify` uses the native store reader when the extension is
present; bidirectional interoperability is contract-tested.

## Native framegraph restoration

Offline renderers construct graphs only through `RendererGraphBuilder`.
Diagnostics reads the last compiled production report and does not create a
parallel synthetic graph. Each compiled pass owns its actual resource
descriptions, declaration bytes, deterministic order, and transition plan.

`GraphScheduler` walks that compiled graph bottom-up. A hit invokes the
resource-restoration callback under the graph's transition contract; a miss
executes the encoder and stores its output. The shared offscreen forward path
contains the production proof: cached tightly packed HDR bytes are uploaded
into the actual tracked color texture, and the declared readback consumer then
runs with the same compiled color transition as a cold producer. Borrowed wgpu
pipelines are cacheable only when their owner supplies a
`ForwardCacheDeclaration` containing the reconstructible pipeline, uniform,
external-resource, capability, and engine bytes.

The terrain draw and AOV phases execute through the compiled graph. Native
terrain caching remains deliberately disabled while virtual-texture,
streaming, scatter, and every other mutable binding cannot all be serialized;
this is a sound false miss, not a partial-key hit. The interactive viewer is
explicitly out of scope.

## Portability and acceptance

The required physical acceptance lane renders 600 real TerrainRenderer frames,
changes one visible label, and requires:

- 600/600 byte-identical outputs against a cold modified render;
- an exact three-pass prediction/observation set (`label.compile`,
  `label.composite`, and `frame.output`);
- zero incremental native-terrain encodes;
- at least 20× cold-to-incremental wall-clock speedup.

The physical portability lane seeds a production TERRA pass on a protected
NVIDIA/Vulkan runner, transfers the unified store to a separate Windows/NVIDIA
DX12 runner, requires at least 99% hits, independently
re-renders the committed TERRA golden, and rehydrates the cached raw RGBA pass
as a GPU texture before byte comparison. A changed compatibility profile must
produce 0% hits. Metal may be exercised by a separately selected diagnostic,
but it is not part of the portability or acceptance contract.
