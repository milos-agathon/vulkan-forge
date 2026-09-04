use super::coverage::{decode_coverage_scene, ingest_coverage_scene};
use crate::core::error::RenderError;
use crate::vector::coverage::{compile_coverage, CoverageCompiledScene};
use std::sync::{Arc, Mutex, OnceLock};

struct CachedCoverageScene {
    source_json: String,
    canonical_json: String,
    compiled: Arc<CoverageCompiledScene>,
}

static COVERAGE_SCENE_CACHE: OnceLock<Mutex<Option<CachedCoverageScene>>> = OnceLock::new();

pub(super) struct CompiledCoverageSceneLookup {
    pub(super) compiled: Arc<CoverageCompiledScene>,
    pending: Option<CachedCoverageScene>,
}

impl CompiledCoverageSceneLookup {
    pub(super) fn commit(&mut self) -> Result<(), RenderError> {
        let Some(entry) = self.pending.take() else {
            return Ok(());
        };
        let mut cached = cache().lock().map_err(|_| cache_poisoned())?;
        *cached = Some(entry);
        Ok(())
    }
}

pub(super) fn compiled_coverage_scene(
    scene_json: &str,
) -> Result<CompiledCoverageSceneLookup, RenderError> {
    {
        let cached = cache().lock().map_err(|_| cache_poisoned())?;
        if let Some(entry) = cached.as_ref() {
            if entry.source_json == scene_json {
                return Ok(cache_hit(entry));
            }
        }
    }

    let input = decode_coverage_scene(scene_json)?;
    let canonical_json = serde_json::to_string(&input).map_err(|error| {
        RenderError::Upload(format!("vector_coverage_scene_canonical_json: {error}"))
    })?;
    {
        let mut cached = cache().lock().map_err(|_| cache_poisoned())?;
        if let Some(entry) = cached.as_mut() {
            if entry.canonical_json == canonical_json {
                entry.source_json = scene_json.to_owned();
                return Ok(cache_hit(entry));
            }
        }
    }

    let compiled = Arc::new(compile_coverage(ingest_coverage_scene(input)?)?);
    Ok(CompiledCoverageSceneLookup {
        compiled: compiled.clone(),
        pending: Some(CachedCoverageScene {
            source_json: scene_json.to_owned(),
            canonical_json,
            compiled,
        }),
    })
}

fn cache() -> &'static Mutex<Option<CachedCoverageScene>> {
    COVERAGE_SCENE_CACHE.get_or_init(|| Mutex::new(None))
}

fn cache_hit(entry: &CachedCoverageScene) -> CompiledCoverageSceneLookup {
    CompiledCoverageSceneLookup {
        compiled: entry.compiled.clone(),
        pending: None,
    }
}

fn cache_poisoned() -> RenderError {
    RenderError::Render("vector_coverage_scene_cache: lock poisoned".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIRST: &str = r#"{
        "width": 4,
        "height": 4,
        "layers": [{
            "name": "one",
            "quality": "analytic",
            "fill_rule": "nonzero",
            "color": [1.0, 1.0, 1.0, 1.0]
        }]
    }"#;
    const EQUIVALENT: &str = r#"{"layers":[{"color":[1,1,1,1],"fill_rule":"nonzero","name":"one","quality":"analytic","polygons":[],"polylines":[],"polygon_grid":null}],"height":4,"width":4}"#;
    const CHANGED: &str = r#"{"height":4,"layers":[{"color":[1,1,1,1],"fill_rule":"nonzero","name":"one","quality":"analytic"}],"width":5}"#;

    #[test]
    fn cache_is_canonical_content_keyed_and_single_entry() {
        *cache().lock().unwrap() = None;

        let mut first = compiled_coverage_scene(FIRST).unwrap();
        assert!(first.pending.is_some());
        let first_compiled = first.compiled.clone();
        first.commit().unwrap();

        let equivalent = compiled_coverage_scene(EQUIVALENT).unwrap();
        assert!(equivalent.pending.is_none());
        assert!(Arc::ptr_eq(&first_compiled, &equivalent.compiled));

        let mut changed = compiled_coverage_scene(CHANGED).unwrap();
        assert!(changed.pending.is_some());
        assert!(!Arc::ptr_eq(&first_compiled, &changed.compiled));
        changed.commit().unwrap();

        let original_after_replacement = compiled_coverage_scene(FIRST).unwrap();
        assert!(original_after_replacement.pending.is_some());
        assert!(!Arc::ptr_eq(
            &first_compiled,
            &original_after_replacement.compiled
        ));
    }
}
