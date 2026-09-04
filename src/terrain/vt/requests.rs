use crate::terrain::vt_family_residency::{TileKey, VT_FAMILY_COUNT};
use std::collections::HashSet;
use std::ops::{Index, IndexMut};

/// Device-independent feedback retention state. A not-ready map is an
/// observation, never a request-set transition.
#[derive(Default)]
pub(crate) struct RetainedRequestSet {
    buckets: [HashSet<TileKey>; VT_FAMILY_COUNT],
}

impl RetainedRequestSet {
    #[cfg_attr(not(feature = "extension-module"), allow(dead_code))]
    pub(crate) fn iter(&self) -> impl Iterator<Item = &HashSet<TileKey>> {
        self.buckets.iter()
    }

    pub(crate) fn on_not_ready(&mut self) {
        // Intentionally preserve every bucket until a resident upload
        // succeeds and removes the corresponding key.
    }

    /// Retain a resolved feedback key only while its physical cache tile is
    /// absent. Sparse stores can resolve an unavailable fine key to an already
    /// resident ancestor; re-inserting that ancestor would prevent the request
    /// set from ever converging to zero.
    pub(crate) fn retain_if_nonresident(&mut self, key: TileKey, is_resident: bool) -> bool {
        if is_resident {
            return false;
        }
        self.buckets[key.family_slot as usize].insert(key)
    }
}

impl Index<usize> for RetainedRequestSet {
    type Output = HashSet<TileKey>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.buckets[index]
    }
}

impl IndexMut<usize> for RetainedRequestSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.buckets[index]
    }
}

#[cfg(test)]
mod tests {
    use super::RetainedRequestSet;
    use crate::terrain::vt_family_residency::TileKey;

    #[test]
    fn thirty_not_ready_frames_preserve_requests_until_resident() {
        let key = TileKey {
            family_slot: 0,
            material_index: 0,
            x: 17,
            y: 9,
            mip_level: 3,
        };
        let mut retained = RetainedRequestSet::default();
        retained[0].insert(key);
        for _ in 0..30 {
            retained.on_not_ready();
            assert!(retained[0].contains(&key));
        }
        retained[0].remove(&key);
        assert!(retained[0].is_empty());
    }

    #[test]
    fn resident_resolved_ancestor_is_not_retained_but_missing_fine_page_is() {
        let resident_ancestor = TileKey {
            family_slot: 0,
            material_index: 0,
            x: 1,
            y: 0,
            mip_level: 6,
        };
        let missing_fine = TileKey {
            family_slot: 0,
            material_index: 0,
            x: 8,
            y: 3,
            mip_level: 3,
        };
        let mut retained = RetainedRequestSet::default();

        assert!(!retained.retain_if_nonresident(resident_ancestor, true));
        assert!(retained[0].is_empty());

        assert!(retained.retain_if_nonresident(missing_fine, false));
        assert!(retained[0].contains(&missing_fine));
    }
}
