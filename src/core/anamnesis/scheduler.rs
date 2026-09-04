//! Bottom-up incremental scheduler over byte-addressable pass outputs.
//!
//! GPU integrations restore a cached blob into the declared output resource
//! before dependents execute. Barrier planning remains a framegraph concern:
//! cached producers are still present in the compiled graph and their output
//! resources therefore retain the same declared state transitions.

use super::key::{pass_key, InputKey, PassKey};
use super::report::CacheReport;
use super::store::ContentStore;
use crate::core::framegraph_impl::{PassDesc, RendererGraphPlan, ResourceBarrier, ResourceHandle};
use std::collections::BTreeMap;
use std::io::Result;
use std::io::{Error, ErrorKind};
use std::time::Instant;

pub struct PassRequest<'a> {
    pub label: &'a str,
    pub pipeline_descriptor_bytes: &'a [u8],
    pub uniform_bytes: &'a [u8],
    pub input_keys: &'a [InputKey],
    pub capability_fingerprint_bytes: &'a [u8],
    pub engine_fingerprint_bytes: &'a [u8],
    pub estimated_wall_ms: f64,
}

pub struct Scheduler {
    store: ContentStore,
    report: CacheReport,
}

/// Bottom-up scheduler for a compiled real-resource framegraph.
///
/// Both `execute` and `restore` receive the exact barrier list preceding the
/// pass. A cache hit therefore cannot bypass the transition contract: the
/// caller must materialize the serialized texture/buffer before dependents
/// run, under the same barriers as a freshly encoded producer.
pub struct GraphScheduler {
    store: Option<ContentStore>,
    report: CacheReport,
    capability_fingerprint_bytes: Vec<u8>,
    engine_fingerprint_bytes: Vec<u8>,
}

pub enum GraphPassAction<'a> {
    Execute {
        barriers: &'a [&'a ResourceBarrier],
        capture_output: bool,
    },
    Restore {
        blob: &'a [u8],
        barriers: &'a [&'a ResourceBarrier],
    },
}

pub enum GraphPassOutcome {
    Executed(Vec<u8>),
    Restored,
}

impl GraphScheduler {
    pub fn new(
        store: ContentStore,
        capability_fingerprint_bytes: Vec<u8>,
        engine_fingerprint_bytes: Vec<u8>,
    ) -> Self {
        Self {
            store: Some(store),
            report: CacheReport::default(),
            capability_fingerprint_bytes,
            engine_fingerprint_bytes,
        }
    }

    /// Execute the same graph substrate without store participation.
    ///
    /// This is the opt-in inertness path: passes execute in graph order, but
    /// the cache report remains empty and no filesystem is opened.
    pub fn disabled(
        capability_fingerprint_bytes: Vec<u8>,
        engine_fingerprint_bytes: Vec<u8>,
    ) -> Self {
        Self {
            store: None,
            report: CacheReport::default(),
            capability_fingerprint_bytes,
            engine_fingerprint_bytes,
        }
    }

    pub fn execute_graph<F, R>(
        &mut self,
        graph: &RendererGraphPlan,
        leaf_keys: &BTreeMap<ResourceHandle, PassKey>,
        mut execute: F,
        mut restore: R,
    ) -> Result<BTreeMap<ResourceHandle, PassKey>>
    where
        F: FnMut(&PassDesc, &[&ResourceBarrier]) -> Result<Vec<u8>>,
        R: FnMut(&PassDesc, &[u8], &[&ResourceBarrier]) -> Result<()>,
    {
        let mut resource_keys = leaf_keys.clone();
        for pass in graph.ordered_passes() {
            let inputs = pass
                .reads
                .iter()
                .map(|handle| {
                    let resource = graph.resource(*handle).ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidInput,
                            format!("framegraph pass {:?} reads an unknown resource", pass.name),
                        )
                    })?;
                    let key = resource_keys.get(handle).copied().ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidInput,
                            format!(
                                "framegraph pass {:?} reads resource {:?} before it has a key",
                                pass.name, resource.desc.name
                            ),
                        )
                    })?;
                    Ok(InputKey::new(resource.desc.name.clone(), key))
                })
                .collect::<Result<Vec<_>>>()?;
            let (key, derivation) = pass_key(
                &pass.name,
                &pass.pipeline_descriptor_bytes,
                &pass.uniform_bytes,
                &inputs,
                &self.capability_fingerprint_bytes,
                &self.engine_fingerprint_bytes,
            );
            let barriers = graph.barriers_before(&pass.name);
            let cacheable = pass.cache_disabled_reason.is_none();
            let blob = match (cacheable, self.store.as_ref()) {
                (true, Some(store)) => match store.get(key)? {
                    Some((blob, metadata)) => {
                        restore(pass, &blob, &barriers)?;
                        self.report.hits.push(pass.name.clone());
                        self.report.bytes_read += blob.len() as u64;
                        self.report.wall_ms_saved += metadata.measured_wall_ms.max(0.0);
                        blob
                    }
                    None => {
                        let started = Instant::now();
                        let blob = execute(pass, &barriers)?;
                        store.put_measured(
                            key,
                            &blob,
                            derivation,
                            None,
                            started.elapsed().as_secs_f64() * 1000.0,
                        )?;
                        self.report.misses.push(pass.name.clone());
                        self.report.bytes_written += blob.len() as u64;
                        blob
                    }
                },
                _ => {
                    let blob = execute(pass, &barriers)?;
                    if self.store.is_some() {
                        self.report.misses.push(pass.name.clone());
                    }
                    blob
                }
            };
            let _ = blob;
            for output in &pass.writes {
                resource_keys.insert(*output, key);
            }
        }
        Ok(resource_keys)
    }

    /// Single-callback graph traversal for stateful native renderers.
    ///
    /// Rust renderers usually need the same mutable scene and resource owner
    /// for both recompute and restore. A single callback avoids manufacturing
    /// disjoint mutable borrows while preserving the exact scheduler policy.
    pub fn execute_graph_with<F>(
        &mut self,
        graph: &RendererGraphPlan,
        leaf_keys: &BTreeMap<ResourceHandle, PassKey>,
        mut run: F,
    ) -> Result<BTreeMap<ResourceHandle, PassKey>>
    where
        F: FnMut(&PassDesc, GraphPassAction<'_>) -> Result<GraphPassOutcome>,
    {
        let mut resource_keys = leaf_keys.clone();
        for pass in graph.ordered_passes() {
            let inputs = pass
                .reads
                .iter()
                .map(|handle| {
                    let resource = graph.resource(*handle).ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidInput,
                            format!("framegraph pass {:?} reads an unknown resource", pass.name),
                        )
                    })?;
                    let key = resource_keys.get(handle).copied().ok_or_else(|| {
                        Error::new(
                            ErrorKind::InvalidInput,
                            format!(
                                "framegraph pass {:?} reads resource {:?} before it has a key",
                                pass.name, resource.desc.name
                            ),
                        )
                    })?;
                    Ok(InputKey::new(resource.desc.name.clone(), key))
                })
                .collect::<Result<Vec<_>>>()?;
            let (key, derivation) = pass_key(
                &pass.name,
                &pass.pipeline_descriptor_bytes,
                &pass.uniform_bytes,
                &inputs,
                &self.capability_fingerprint_bytes,
                &self.engine_fingerprint_bytes,
            );
            let barriers = graph.barriers_before(&pass.name);
            let cacheable = pass.cache_disabled_reason.is_none();
            if let (true, Some(store)) = (cacheable, self.store.as_ref()) {
                if let Some((blob, metadata)) = store.get(key)? {
                    match run(
                        pass,
                        GraphPassAction::Restore {
                            blob: &blob,
                            barriers: &barriers,
                        },
                    )? {
                        GraphPassOutcome::Restored => {}
                        GraphPassOutcome::Executed(_) => {
                            return Err(Error::new(
                                ErrorKind::InvalidData,
                                format!(
                                    "graph pass {:?} executed during a cache restoration",
                                    pass.name
                                ),
                            ))
                        }
                    }
                    self.report.hits.push(pass.name.clone());
                    self.report.bytes_read += blob.len() as u64;
                    self.report.wall_ms_saved += metadata.measured_wall_ms.max(0.0);
                } else {
                    let started = Instant::now();
                    let blob = match run(
                        pass,
                        GraphPassAction::Execute {
                            barriers: &barriers,
                            capture_output: true,
                        },
                    )? {
                        GraphPassOutcome::Executed(blob) => blob,
                        GraphPassOutcome::Restored => {
                            return Err(Error::new(
                                ErrorKind::InvalidData,
                                format!(
                                    "graph pass {:?} restored without a cached blob",
                                    pass.name
                                ),
                            ))
                        }
                    };
                    store.put_measured(
                        key,
                        &blob,
                        derivation,
                        None,
                        started.elapsed().as_secs_f64() * 1000.0,
                    )?;
                    self.report.misses.push(pass.name.clone());
                    self.report.bytes_written += blob.len() as u64;
                }
            } else {
                match run(
                    pass,
                    GraphPassAction::Execute {
                        barriers: &barriers,
                        capture_output: false,
                    },
                )? {
                    GraphPassOutcome::Executed(_) => {}
                    GraphPassOutcome::Restored => {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            format!("uncached graph pass {:?} did not execute", pass.name),
                        ))
                    }
                }
                if self.store.is_some() {
                    self.report.misses.push(pass.name.clone());
                }
            }
            for output in &pass.writes {
                resource_keys.insert(*output, key);
            }
        }
        Ok(resource_keys)
    }

    pub fn report(&self) -> &CacheReport {
        &self.report
    }

    pub fn into_report(self) -> CacheReport {
        self.report
    }
}

impl Scheduler {
    pub fn new(store: ContentStore) -> Self {
        Self {
            store,
            report: CacheReport::default(),
        }
    }

    pub fn execute<F>(&mut self, request: PassRequest<'_>, encode: F) -> Result<(PassKey, Vec<u8>)>
    where
        F: FnOnce() -> Result<Vec<u8>>,
    {
        let (key, derivation) = pass_key(
            request.label,
            request.pipeline_descriptor_bytes,
            request.uniform_bytes,
            request.input_keys,
            request.capability_fingerprint_bytes,
            request.engine_fingerprint_bytes,
        );
        if let Some((blob, metadata)) = self.store.get(key)? {
            self.report.hits.push(request.label.to_string());
            self.report.bytes_read += blob.len() as u64;
            self.report.wall_ms_saved += metadata
                .measured_wall_ms
                .max(request.estimated_wall_ms)
                .max(0.0);
            return Ok((key, blob));
        }
        let started = Instant::now();
        let blob = encode()?;
        self.store.put_measured(
            key,
            &blob,
            derivation,
            None,
            started.elapsed().as_secs_f64() * 1000.0,
        )?;
        self.report.misses.push(request.label.to_string());
        self.report.bytes_written += blob.len() as u64;
        let _elapsed = started.elapsed();
        Ok((key, blob))
    }

    pub fn report(&self) -> &CacheReport {
        &self.report
    }
    pub fn into_report(self) -> CacheReport {
        self.report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::anamnesis::EngineFingerprint;
    use std::cell::Cell;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn verified_hit_skips_the_encoder() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("forge3d-anamnesis-scheduler-{nonce}"));
        // Linux allocates 4 KiB per directory; leave enough room for the
        // self-describing entry plus root, quarantine, and hash-prefix dirs.
        let store = ContentStore::new(&root, 64 * 1024, true).unwrap();
        let engine = EngineFingerprint::current().canonical_bytes();
        let encoded = Cell::new(0u32);
        let execute = |scheduler: &mut Scheduler| {
            scheduler
                .execute(
                    PassRequest {
                        label: "test.output",
                        pipeline_descriptor_bytes: b"pipeline",
                        uniform_bytes: b"uniform",
                        input_keys: &[],
                        capability_fingerprint_bytes: b"caps",
                        engine_fingerprint_bytes: &engine,
                        estimated_wall_ms: 4.5,
                    },
                    || {
                        encoded.set(encoded.get() + 1);
                        Ok(b"pixels".to_vec())
                    },
                )
                .unwrap()
        };

        let mut cold = Scheduler::new(store.clone());
        let (_, cold_blob) = execute(&mut cold);
        assert_eq!(encoded.get(), 1);
        assert_eq!(cold.report().misses, ["test.output"]);

        let mut warm = Scheduler::new(store);
        let (_, warm_blob) = execute(&mut warm);
        assert_eq!(encoded.get(), 1, "warm hit must not invoke the encoder");
        assert_eq!(warm.report().hits, ["test.output"]);
        assert_eq!(warm.report().bytes_read, 6);
        assert_eq!(warm.report().wall_ms_saved, 4.5);
        assert_eq!(cold_blob, warm_blob);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn graph_hits_restore_resources_under_the_compiled_barriers() {
        use crate::core::framegraph_impl::{
            PassType, RendererGraphBuilder, ResourceDesc, ResourceType,
        };
        use wgpu::{Extent3d, TextureFormat, TextureUsages};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("forge3d-anamnesis-graph-{nonce}"));
        let store = ContentStore::new(&root, 64 * 1024, true).unwrap();
        let mut builder = RendererGraphBuilder::new();
        let desc = |name: &str, is_transient: bool| ResourceDesc {
            name: name.into(),
            resource_type: ResourceType::ColorAttachment,
            format: Some(TextureFormat::Rgba8Unorm),
            extent: Some(Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            }),
            size: None,
            usage: Some(TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING),
            can_alias: false,
            is_transient,
        };
        let leaf = builder.add_resource(desc("leaf", false));
        let middle = builder.add_resource(desc("middle", true));
        let output = builder.add_resource(desc("output", true));
        builder
            .add_pass("first", PassType::Graphics, |pass| {
                pass.read(leaf)
                    .write(middle)
                    .pipeline_descriptor(b"first-pipeline".to_vec())
                    .uniform_bytes(b"first-uniform".to_vec());
                Ok(())
            })
            .unwrap();
        builder
            .add_pass("second", PassType::Graphics, |pass| {
                pass.read(middle)
                    .write(output)
                    .pipeline_descriptor(b"second-pipeline".to_vec())
                    .uniform_bytes(b"second-uniform".to_vec());
                Ok(())
            })
            .unwrap();
        let graph = builder.compile().unwrap();
        let leaf_keys = BTreeMap::from([(leaf, super::super::leaf_key(b"leaf"))]);
        let engine = EngineFingerprint::current().canonical_bytes();

        let mut cold = GraphScheduler::new(store.clone(), b"caps".to_vec(), engine.clone());
        let mut cold_barriers = Vec::new();
        cold.execute_graph(
            &graph,
            &leaf_keys,
            |pass, barriers| {
                cold_barriers.push((pass.name.clone(), barriers.len()));
                Ok(pass.name.as_bytes().to_vec())
            },
            |_, _, _| panic!("cold graph cannot restore"),
        )
        .unwrap();
        assert_eq!(cold.report().misses, ["first", "second"]);

        let mut warm = GraphScheduler::new(store, b"caps".to_vec(), engine);
        let mut restored = Vec::new();
        warm.execute_graph(
            &graph,
            &leaf_keys,
            |_, _| panic!("warm graph cannot execute an encoder"),
            |pass, blob, barriers| {
                restored.push((pass.name.clone(), blob.to_vec(), barriers.len()));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(warm.report().hits, ["first", "second"]);
        assert_eq!(restored[0].0, "first");
        assert_eq!(restored[1].0, "second");
        assert!(
            restored[1].2 > 0,
            "dependent cached resource must retain its transition barrier"
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn all_miss_invalidation_reports_zero_bytes_read() {
        use crate::core::framegraph_impl::{
            PassType, RendererGraphBuilder, ResourceDesc, ResourceType,
        };
        use wgpu::{Extent3d, TextureFormat, TextureUsages};

        let build_graph = |first_uniform: &[u8]| {
            let mut builder = RendererGraphBuilder::new();
            let desc = |name: &str, is_transient: bool| ResourceDesc {
                name: name.into(),
                resource_type: ResourceType::ColorAttachment,
                format: Some(TextureFormat::Rgba8Unorm),
                extent: Some(Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                }),
                size: None,
                usage: Some(TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING),
                can_alias: false,
                is_transient,
            };
            let leaf = builder.add_resource(desc("leaf", false));
            let middle = builder.add_resource(desc("middle", true));
            let output = builder.add_resource(desc("output", true));
            builder
                .add_pass("first", PassType::Graphics, |pass| {
                    pass.read(leaf)
                        .write(middle)
                        .pipeline_descriptor(b"first-pipeline".to_vec())
                        .uniform_bytes(first_uniform.to_vec());
                    Ok(())
                })
                .unwrap();
            builder
                .add_pass("second", PassType::Graphics, |pass| {
                    pass.read(middle)
                        .write(output)
                        .pipeline_descriptor(b"second-pipeline".to_vec())
                        .uniform_bytes(b"second-uniform".to_vec());
                    Ok(())
                })
                .unwrap();
            (builder.compile().unwrap(), leaf)
        };

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("forge3d-anamnesis-all-miss-{nonce}"));
        let store = ContentStore::new(&root, 64 * 1024, true).unwrap();
        let engine = EngineFingerprint::current().canonical_bytes();

        let (cold_graph, cold_leaf) = build_graph(b"first-uniform");
        let cold_leaf_keys = BTreeMap::from([(cold_leaf, super::super::leaf_key(b"leaf"))]);
        let mut cold = GraphScheduler::new(store.clone(), b"caps".to_vec(), engine.clone());
        cold.execute_graph_with(&cold_graph, &cold_leaf_keys, |pass, action| match action {
            GraphPassAction::Execute { capture_output, .. } => {
                assert!(capture_output);
                Ok(GraphPassOutcome::Executed(pass.name.as_bytes().to_vec()))
            }
            GraphPassAction::Restore { .. } => panic!("cold graph cannot restore"),
        })
        .unwrap();

        let (changed_graph, changed_leaf) = build_graph(b"changed-first-uniform");
        let changed_leaf_keys = BTreeMap::from([(changed_leaf, super::super::leaf_key(b"leaf"))]);
        let mut changed = GraphScheduler::new(store, b"caps".to_vec(), engine);
        changed
            .execute_graph_with(
                &changed_graph,
                &changed_leaf_keys,
                |pass, action| match action {
                    GraphPassAction::Execute { capture_output, .. } => {
                        assert!(capture_output);
                        Ok(GraphPassOutcome::Executed(pass.name.as_bytes().to_vec()))
                    }
                    GraphPassAction::Restore { .. } => {
                        panic!("changed graph cannot restore a stale pass")
                    }
                },
            )
            .unwrap();
        assert_eq!(changed.report().hits, Vec::<String>::new());
        assert_eq!(changed.report().misses, ["first", "second"]);
        assert_eq!(changed.report().bytes_read, 0);
        assert!(changed.report().bytes_written > 0);
        fs::remove_dir_all(root).unwrap();
    }
}
