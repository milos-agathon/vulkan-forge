use crate::core::error::RenderError;
use crate::core::memory_tracker::{calculate_texture_size, global_tracker, is_host_visible_usage};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use wgpu::util::DeviceExt;
use wgpu::{BufferDescriptor, BufferUsages, TextureDescriptor, TextureFormat};

/// Resource handle that automatically unregisters on drop
#[derive(Debug)]
pub enum ResourceHandle {
    Buffer {
        size: u64,
        is_host_visible: bool,
        ledger_id: u64,
    },
    Texture {
        size: u64,
        ledger_id: u64,
    },
}

impl ResourceHandle {
    fn ledger_id(&self) -> u64 {
        match self {
            Self::Buffer { ledger_id, .. } | Self::Texture { ledger_id, .. } => *ledger_id,
        }
    }
}

impl Drop for ResourceHandle {
    fn drop(&mut self) {
        let tracker = global_tracker();
        match self {
            ResourceHandle::Buffer {
                size,
                is_host_visible,
                ledger_id,
            } => {
                ledger().remove(*ledger_id);
                tracker.free_ledger_allocation(*size, *is_host_visible);
                tracker.free_buffer_allocation(*size, *is_host_visible);
            }
            ResourceHandle::Texture { size, ledger_id } => {
                ledger().remove(*ledger_id);
                tracker.free_ledger_allocation(*size, false);
                tracker.free_texture_allocation_bytes(*size);
            }
        }
    }
}

#[track_caller]
fn caller_label() -> String {
    let loc = std::panic::Location::caller();
    format!("{}:{}", loc.file().replace('\\', "/"), loc.line())
}

fn register_buffer_with_ledger(
    size: u64,
    is_host_visible: bool,
    label: String,
    call_site: String,
) -> Result<ResourceHandle, RenderError> {
    let tracker = global_tracker();
    tracker.track_buffer_allocation_labeled(size, is_host_visible, &label)?;
    tracker.track_ledger_allocation(size, is_host_visible);
    let ledger_id = ledger().insert(
        label,
        size,
        is_host_visible,
        LedgerCategory::Buffer,
        call_site,
    );
    Ok(ResourceHandle::Buffer {
        size,
        is_host_visible,
        ledger_id,
    })
}

fn register_texture_with_ledger(size: u64, label: String, call_site: String) -> ResourceHandle {
    let tracker = global_tracker();
    tracker.track_texture_allocation_bytes(size);
    tracker.track_ledger_allocation(size, false);
    let ledger_id = ledger().insert(label, size, false, LedgerCategory::Texture, call_site);
    ResourceHandle::Texture { size, ledger_id }
}

/// Register a buffer allocation and return a handle that will unregister on drop
#[track_caller]
pub fn register_buffer(size: u64, usage: BufferUsages) -> Result<ResourceHandle, RenderError> {
    let is_host_visible = is_host_visible_usage(usage);
    let call_site = caller_label();
    register_buffer_with_ledger(size, is_host_visible, call_site.clone(), call_site)
}

/// Register a texture allocation and return a handle that will unregister on drop
#[track_caller]
pub fn register_texture(width: u32, height: u32, format: TextureFormat) -> ResourceHandle {
    register_texture_bytes(calculate_texture_size(width, height, format))
}

/// Register an exactly sized texture allocation.
#[track_caller]
pub fn register_texture_bytes(size: u64) -> ResourceHandle {
    let call_site = caller_label();
    register_texture_with_ledger(size, call_site.clone(), call_site)
}

/// Register a buffer allocation with explicit host-visible flag
#[track_caller]
pub fn register_buffer_explicit(
    size: u64,
    is_host_visible: bool,
) -> Result<ResourceHandle, RenderError> {
    let call_site = caller_label();
    register_buffer_with_ledger(size, is_host_visible, call_site.clone(), call_site)
}

/// Register a scoped non-wgpu host allocation after enforcing the global
/// host-visible budget. This is used for transient upload encodings whose
/// backing `Vec` is otherwise invisible to the GPU resource wrappers.
#[track_caller]
pub fn tracked_host_allocation(size: u64, label: &str) -> Result<ResourceHandle, RenderError> {
    let call_site = caller_label();
    register_buffer_with_ledger(size, true, label.to_owned(), call_site)
}

// ---------------------------------------------------------------------------
// CENSOR: total allocation ledger
// ---------------------------------------------------------------------------

/// Classification of a tracked allocation for the ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerCategory {
    Buffer,
    Texture,
}

static NEXT_ALLOCATION_OWNER_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static ACTIVE_ALLOCATION_OWNER: Cell<Option<u64>> = const { Cell::new(None) };
}

#[derive(Clone, Debug)]
pub struct AllocationOwner {
    id: u64,
}

impl AllocationOwner {
    pub fn new() -> Self {
        Self {
            id: NEXT_ALLOCATION_OWNER_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn activate(&self) -> AllocationOwnerGuard {
        let previous = ACTIVE_ALLOCATION_OWNER.with(|owner| owner.replace(Some(self.id)));
        AllocationOwnerGuard { previous }
    }
}

pub struct AllocationOwnerGuard {
    previous: Option<u64>,
}

impl Drop for AllocationOwnerGuard {
    fn drop(&mut self) {
        ACTIVE_ALLOCATION_OWNER.with(|owner| owner.set(self.previous));
    }
}

fn active_allocation_owner() -> Option<u64> {
    ACTIVE_ALLOCATION_OWNER.with(Cell::get)
}

#[derive(Clone, Debug)]
struct LedgerEntry {
    label: String,
    bytes: u64,
    host_visible: bool,
    #[allow(dead_code)]
    category: LedgerCategory,
    #[allow(dead_code)]
    call_site: String,
    owner_id: Option<u64>,
}

struct LedgerCapture {
    ids: HashSet<u64>,
    current_host_visible: u64,
    current_device_local: u64,
    peak_host_visible: u64,
    peak_device_local: u64,
    current_by_label: BTreeMap<String, u64>,
    peak_by_label: BTreeMap<String, u64>,
}

impl LedgerCapture {
    fn new(entries: &HashMap<u64, LedgerEntry>, owner_ids: &[u64]) -> Self {
        let owners: HashSet<u64> = owner_ids.iter().copied().collect();
        let mut capture = Self {
            ids: HashSet::new(),
            current_host_visible: 0,
            current_device_local: 0,
            peak_host_visible: 0,
            peak_device_local: 0,
            current_by_label: BTreeMap::new(),
            peak_by_label: BTreeMap::new(),
        };
        for (&id, entry) in entries {
            if entry.owner_id.is_some_and(|owner| owners.contains(&owner)) {
                capture.add(id, entry);
            }
        }
        capture
    }

    fn add(&mut self, id: u64, entry: &LedgerEntry) {
        if !self.ids.insert(id) {
            return;
        }
        let current = self
            .current_by_label
            .entry(entry.label.clone())
            .or_insert(0);
        *current += entry.bytes;
        let peak = self.peak_by_label.entry(entry.label.clone()).or_insert(0);
        *peak = (*peak).max(*current);
        if entry.host_visible {
            self.current_host_visible += entry.bytes;
            self.peak_host_visible = self.peak_host_visible.max(self.current_host_visible);
        } else {
            self.current_device_local += entry.bytes;
            self.peak_device_local = self.peak_device_local.max(self.current_device_local);
        }
    }

    fn remove(&mut self, id: u64, entry: &LedgerEntry) {
        if !self.ids.remove(&id) {
            return;
        }
        if let Some(bytes) = self.current_by_label.get_mut(&entry.label) {
            *bytes = bytes.saturating_sub(entry.bytes);
            if *bytes == 0 {
                self.current_by_label.remove(&entry.label);
            }
        }
        if entry.host_visible {
            self.current_host_visible = self.current_host_visible.saturating_sub(entry.bytes);
        } else {
            self.current_device_local = self.current_device_local.saturating_sub(entry.bytes);
        }
    }

    fn report(self) -> LedgerReport {
        LedgerReport {
            peak_host_visible_bytes: self.peak_host_visible,
            peak_device_local_bytes: self.peak_device_local,
            current_host_visible_bytes: self.current_host_visible,
            current_device_local_bytes: self.current_device_local,
            by_label: self.peak_by_label,
        }
    }
}

struct OwnerCapture {
    capture: LedgerCapture,
}

impl OwnerCapture {
    fn new(entries: &HashMap<u64, LedgerEntry>, owner_id: u64) -> Self {
        Self {
            capture: LedgerCapture::new(entries, &[owner_id]),
        }
    }

    fn add(&mut self, id: u64, entry: &LedgerEntry) {
        self.capture.add(id, entry);
    }

    fn remove(&mut self, id: u64, entry: &LedgerEntry) {
        self.capture.remove(id, entry);
    }

    fn report(self) -> OwnerLedgerReport {
        let report = self.capture.report();
        OwnerLedgerReport {
            peak_host_visible_bytes: report.peak_host_visible_bytes,
            peak_device_local_bytes: report.peak_device_local_bytes,
            current_host_visible_bytes: report.current_host_visible_bytes,
            current_device_local_bytes: report.current_device_local_bytes,
            by_label: report.by_label,
        }
    }
}

/// Global ledger recording every live tracked allocation.
///
/// Counters are only ever mutated while the `entries` mutex is held, so
/// [`AllocationLedger::snapshot`] can read a consistent view (and assert the
/// sum-of-entries == counters invariant in debug builds).
pub struct AllocationLedger {
    entries: Mutex<HashMap<u64, LedgerEntry>>,
    next_id: AtomicU64,
    current_host_visible: AtomicU64,
    current_device_local: AtomicU64,
    peak_host_visible: AtomicU64,
    peak_device_local: AtomicU64,
    capture: Mutex<Option<LedgerCapture>>,
    owner_captures: Mutex<BTreeMap<u64, OwnerCapture>>,
}

impl AllocationLedger {
    fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            current_host_visible: AtomicU64::new(0),
            current_device_local: AtomicU64::new(0),
            peak_host_visible: AtomicU64::new(0),
            peak_device_local: AtomicU64::new(0),
            capture: Mutex::new(None),
            owner_captures: Mutex::new(BTreeMap::new()),
        }
    }

    fn insert(
        &self,
        label: String,
        bytes: u64,
        host_visible: bool,
        category: LedgerCategory,
        call_site: String,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut map = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let entry = LedgerEntry {
            label,
            bytes,
            host_visible,
            category,
            call_site,
            owner_id: active_allocation_owner(),
        };
        map.insert(id, entry.clone());
        if host_visible {
            let cur = self
                .current_host_visible
                .fetch_add(bytes, Ordering::Relaxed)
                + bytes;
            self.peak_host_visible.fetch_max(cur, Ordering::Relaxed);
        } else {
            let cur = self
                .current_device_local
                .fetch_add(bytes, Ordering::Relaxed)
                + bytes;
            self.peak_device_local.fetch_max(cur, Ordering::Relaxed);
        }
        if let Some(capture) = self
            .capture
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_mut()
        {
            capture.add(id, &entry);
        }
        if let Some(owner_id) = entry.owner_id {
            if let Some(capture) = self
                .owner_captures
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .get_mut(&owner_id)
            {
                capture.add(id, &entry);
            }
        }
        id
    }

    fn begin_owner_capture(&self, owner_id: u64) -> OwnerCaptureGuard<'_> {
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        self.owner_captures
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(owner_id, OwnerCapture::new(&entries, owner_id));
        OwnerCaptureGuard {
            ledger: self,
            owner_id,
            active: true,
        }
    }

    fn finish_owner_capture(&self, owner_id: u64) -> OwnerLedgerReport {
        self.owner_captures
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&owner_id)
            .map(OwnerCapture::report)
            .unwrap_or_default()
    }

    fn abort_owner_capture(&self, owner_id: u64) {
        self.owner_captures
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&owner_id);
    }

    fn begin_capture(&self, owner_ids: &[u64]) {
        // ponytail: render capture is process-global and serialized; use capture
        // IDs only if concurrent renders become a supported public contract.
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        *self.capture.lock().unwrap_or_else(|p| p.into_inner()) =
            Some(LedgerCapture::new(&entries, owner_ids));
    }

    fn extend_capture(&self, owner_ids: &[u64]) {
        if owner_ids.is_empty() {
            return;
        }
        let owners: HashSet<u64> = owner_ids.iter().copied().collect();
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let mut capture = self.capture.lock().unwrap_or_else(|p| p.into_inner());
        let Some(capture) = capture.as_mut() else {
            return;
        };
        for (&id, entry) in entries.iter() {
            if entry.owner_id.is_some_and(|owner| owners.contains(&owner)) {
                capture.add(id, entry);
            }
        }
    }

    fn finish_capture(&self) -> LedgerReport {
        self.capture
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
            .map(LedgerCapture::report)
            .unwrap_or_default()
    }

    fn abort_capture(&self) {
        self.capture
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
    }

    fn remove(&self, id: u64) {
        let mut map = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(entry) = map.remove(&id) {
            if entry.host_visible {
                let _ = self.current_host_visible.fetch_update(
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                    |current| Some(current.saturating_sub(entry.bytes)),
                );
            } else {
                let _ = self.current_device_local.fetch_update(
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                    |current| Some(current.saturating_sub(entry.bytes)),
                );
            }
            if let Some(capture) = self
                .capture
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .as_mut()
            {
                capture.remove(id, &entry);
            }
            if let Some(owner_id) = entry.owner_id {
                if let Some(capture) = self
                    .owner_captures
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .get_mut(&owner_id)
                {
                    capture.remove(id, &entry);
                }
            }
        }
    }

    fn snapshot(&self) -> LedgerReport {
        let map = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let mut by_label: BTreeMap<String, u64> = BTreeMap::new();
        let mut sum_host_visible = 0u64;
        let mut sum_device_local = 0u64;
        for entry in map.values() {
            *by_label.entry(entry.label.clone()).or_insert(0) += entry.bytes;
            if entry.host_visible {
                sum_host_visible += entry.bytes;
            } else {
                sum_device_local += entry.bytes;
            }
        }
        let current_host_visible_bytes = self.current_host_visible.load(Ordering::Relaxed);
        let current_device_local_bytes = self.current_device_local.load(Ordering::Relaxed);
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                sum_host_visible, current_host_visible_bytes,
                "ledger host-visible sum-of-entries must equal the running counter"
            );
            debug_assert_eq!(
                sum_device_local, current_device_local_bytes,
                "ledger device-local sum-of-entries must equal the running counter"
            );
        }
        let _ = (sum_host_visible, sum_device_local);
        LedgerReport {
            peak_host_visible_bytes: self.peak_host_visible.load(Ordering::Relaxed),
            peak_device_local_bytes: self.peak_device_local.load(Ordering::Relaxed),
            current_host_visible_bytes,
            current_device_local_bytes,
            by_label,
        }
    }
}

/// Immutable snapshot of the [`AllocationLedger`].
#[derive(Clone, Debug, Default)]
pub struct LedgerReport {
    pub peak_host_visible_bytes: u64,
    pub peak_device_local_bytes: u64,
    pub current_host_visible_bytes: u64,
    pub current_device_local_bytes: u64,
    /// Sum of live allocation bytes per label.
    pub by_label: BTreeMap<String, u64>,
}

impl LedgerReport {
    /// The `n` labels consuming the most bytes, largest first (ties broken by label).
    pub fn top_consumers(&self, n: usize) -> Vec<(String, u64)> {
        let mut ranked: Vec<(String, u64)> = self
            .by_label
            .iter()
            .map(|(label, &bytes)| (label.clone(), bytes))
            .collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(n);
        ranked
    }
}

#[derive(Clone, Debug, Default)]
pub struct OwnerLedgerReport {
    pub peak_host_visible_bytes: u64,
    pub peak_device_local_bytes: u64,
    pub current_host_visible_bytes: u64,
    pub current_device_local_bytes: u64,
    pub by_label: BTreeMap<String, u64>,
}

#[must_use = "an owner capture must be finished or retained until the render exits"]
pub struct OwnerCaptureGuard<'a> {
    ledger: &'a AllocationLedger,
    owner_id: u64,
    active: bool,
}

impl OwnerCaptureGuard<'_> {
    pub fn finish(mut self) -> OwnerLedgerReport {
        self.active = false;
        self.ledger.finish_owner_capture(self.owner_id)
    }
}

impl Drop for OwnerCaptureGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            self.ledger.abort_owner_capture(self.owner_id);
        }
    }
}

static LEDGER: OnceLock<AllocationLedger> = OnceLock::new();

/// Access the process-global allocation ledger.
pub fn ledger() -> &'static AllocationLedger {
    LEDGER.get_or_init(AllocationLedger::new)
}

/// Snapshot the global allocation ledger.
pub fn ledger_snapshot() -> LedgerReport {
    ledger().snapshot()
}

pub fn begin_owner_capture(owner_id: u64) -> OwnerCaptureGuard<'static> {
    ledger().begin_owner_capture(owner_id)
}

pub fn finish_owner_capture(owner_id: u64) -> OwnerLedgerReport {
    ledger().finish_owner_capture(owner_id)
}

/// Start render-local peak accounting from the allocations currently alive.
pub fn begin_ledger_capture(owner_ids: &[u64]) {
    ledger().begin_capture(owner_ids);
}

/// Add renderer owners discovered by a nested render to the active capture.
pub fn extend_ledger_capture(owner_ids: &[u64]) {
    ledger().extend_capture(owner_ids);
}

/// CENSOR audit F-07: cross-channel accounting invariant.
///
/// Every `ResourceHandle` updates both the ledger and the registry's exact
/// wrapper subset as one lifetime-owned allocation, so both axes must match.
pub fn ledger_registry_cross_check() -> Result<(), String> {
    let snap = ledger().snapshot();
    let (registry_host_visible, registry_device_local) = global_tracker().ledger_totals();
    if snap.current_host_visible_bytes != registry_host_visible {
        return Err(format!(
            "ledger host-visible total ({}) differs from the registry counter ({})",
            snap.current_host_visible_bytes, registry_host_visible
        ));
    }
    if snap.current_device_local_bytes != registry_device_local {
        return Err(format!(
            "ledger device-local total ({}) differs from the registry counter ({registry_device_local})",
            snap.current_device_local_bytes
        ));
    }
    Ok(())
}

/// Finish render-local peak accounting and return the captured ledger report.
///
/// Debug builds assert the ledger/registry cross-invariant here — once per
/// render capture, not merely on the budget-error path — and `snapshot()`
/// inside the check also re-runs the ledger's own sum-of-entries == counters
/// assertion at the same cadence.
pub fn finish_ledger_capture() -> LedgerReport {
    #[cfg(debug_assertions)]
    if let Err(violation) = ledger_registry_cross_check() {
        panic!("allocation accounting invariant violated at render-capture finish: {violation}");
    }
    ledger().finish_capture()
}

pub fn abort_ledger_capture() {
    ledger().abort_capture();
}

/// Render the top-`n` ledger consumers as a `"label=bytes, ..."` string for
/// budget-error messages. Returns `"(none)"` when the ledger is empty.
pub fn ledger_top_consumers_string(n: usize) -> String {
    let top = ledger_snapshot().top_consumers(n);
    if top.is_empty() {
        return "(none)".to_string();
    }
    top.iter()
        .map(|(label, bytes)| format!("{label}={bytes}"))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// CENSOR: RAII tracked GPU resource wrappers
// ---------------------------------------------------------------------------

/// A `wgpu::Buffer` whose lifetime is tracked in the global registry + ledger.
///
/// `Deref`s to the inner buffer; dropping removes the ledger entry (the inner
/// `ResourceHandle` frees the registry accounting via its own `Drop`).
#[derive(Debug)]
pub struct TrackedBuffer {
    inner: wgpu::Buffer,
    _registry: ResourceHandle,
}

impl TrackedBuffer {
    /// Explicit accessor for the wrapped buffer (equivalent to `&*self`).
    pub fn inner(&self) -> &wgpu::Buffer {
        &self.inner
    }

    /// Stable ledger handle for this exact tracked allocation.
    pub fn ledger_id(&self) -> u64 {
        self._registry.ledger_id()
    }
}

impl std::ops::Deref for TrackedBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// A `wgpu::Texture` whose lifetime is tracked in the global registry + ledger.
#[derive(Debug)]
pub struct TrackedTexture {
    inner: wgpu::Texture,
    _registry: ResourceHandle,
}

impl TrackedTexture {
    /// Explicit accessor for the wrapped texture (equivalent to `&*self`).
    pub fn inner(&self) -> &wgpu::Texture {
        &self.inner
    }

    /// Stable ledger handle for this exact tracked allocation.
    pub fn ledger_id(&self) -> u64 {
        self._registry.ledger_id()
    }
}

impl std::ops::Deref for TrackedTexture {
    type Target = wgpu::Texture;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Format a `#[track_caller]` location as a cross-platform-stable `file:line`.
///
/// Must be called directly inside a `#[track_caller]` function (there is no
/// automatic propagation through an intermediate helper), so this is a macro.
macro_rules! caller_site {
    () => {{
        let loc = ::std::panic::Location::caller();
        format!("{}:{}", loc.file().replace('\\', "/"), loc.line())
    }};
}

/// Create a buffer, enforcing the host-visible budget policy and recording it
/// in the registry + allocation ledger.
#[track_caller]
pub fn tracked_create_buffer(
    device: &wgpu::Device,
    desc: &BufferDescriptor<'_>,
) -> Result<TrackedBuffer, RenderError> {
    let host_visible = is_host_visible_usage(desc.usage);
    let call_site = caller_site!();
    let label = desc
        .label
        .map(|s| s.to_string())
        .unwrap_or_else(|| call_site.clone());
    let registry = register_buffer_with_ledger(desc.size, host_visible, label, call_site)?;
    let buffer = device.create_buffer(desc);
    Ok(TrackedBuffer {
        inner: buffer,
        _registry: registry,
    })
}

/// Create a buffer with initial contents (like `DeviceExt::create_buffer_init`),
/// enforcing the host-visible budget policy and recording it in the ledger.
#[track_caller]
pub fn tracked_create_buffer_init(
    device: &wgpu::Device,
    desc: &wgpu::util::BufferInitDescriptor<'_>,
) -> Result<TrackedBuffer, RenderError> {
    let host_visible = is_host_visible_usage(desc.usage);
    let call_site = caller_site!();
    let label = desc
        .label
        .map(|s| s.to_string())
        .unwrap_or_else(|| call_site.clone());
    // Match wgpu's DeviceExt padding: round contents up to COPY_BUFFER_ALIGNMENT.
    let unpadded = desc.contents.len() as u64;
    let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
    let size = ((unpadded + align_mask) & !align_mask).max(wgpu::COPY_BUFFER_ALIGNMENT);
    let registry = register_buffer_with_ledger(size, host_visible, label, call_site)?;
    let buffer = device.create_buffer_init(desc);
    Ok(TrackedBuffer {
        inner: buffer,
        _registry: registry,
    })
}

/// Create a texture and record it in the registry + allocation ledger.
///
/// Textures are device-local; the 512 MiB host-visible budget does not apply,
/// so no budget check is performed (only registry/ledger accounting).
#[track_caller]
pub fn tracked_create_texture(
    device: &wgpu::Device,
    desc: &TextureDescriptor<'_>,
) -> Result<TrackedTexture, RenderError> {
    let call_site = caller_site!();
    let label = desc
        .label
        .map(|s| s.to_string())
        .unwrap_or_else(|| call_site.clone());
    let size = calculate_texture_descriptor_size(desc);
    let texture = device.create_texture(desc);
    let registry = register_texture_with_ledger(size, label, call_site);
    Ok(TrackedTexture {
        inner: texture,
        _registry: registry,
    })
}

/// Exact byte size of every subresource described by a texture descriptor.
pub fn calculate_texture_descriptor_size(desc: &TextureDescriptor<'_>) -> u64 {
    let (block_width, block_height) = desc.format.block_dimensions();
    let bytes_per_block = desc
        .format
        .block_copy_size(None)
        .map(u64::from)
        .unwrap_or_else(|| calculate_texture_size(block_width, block_height, desc.format));
    let mut total = 0u64;
    for mip in 0..desc.mip_level_count {
        let width = (desc.size.width >> mip).max(1);
        let height = match desc.dimension {
            wgpu::TextureDimension::D1 => 1,
            _ => (desc.size.height >> mip).max(1),
        };
        let layers_or_depth = match desc.dimension {
            wgpu::TextureDimension::D3 => (desc.size.depth_or_array_layers >> mip).max(1),
            _ => desc.size.depth_or_array_layers,
        };
        let blocks = u64::from(width.div_ceil(block_width))
            .saturating_mul(u64::from(height.div_ceil(block_height)))
            .saturating_mul(u64::from(layers_or_depth));
        total = total.saturating_add(blocks.saturating_mul(bytes_per_block));
    }
    total.saturating_mul(u64::from(desc.sample_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory_tracker::ResourceRegistry;

    #[test]
    fn test_resource_handle_cleanup() {
        // Test with isolated registry (can't easily test global one)
        let registry = ResourceRegistry::new();

        // Test buffer handle
        {
            global_tracker()
                .track_buffer_allocation(1024, true)
                .expect("test allocation fits budget");
            global_tracker().track_ledger_allocation(1024, true);
            let handle = ResourceHandle::Buffer {
                size: 1024,
                is_host_visible: true,
                ledger_id: ledger().insert(
                    "test-resource-handle".to_string(),
                    1024,
                    true,
                    LedgerCategory::Buffer,
                    "test".to_string(),
                ),
            };

            // Manually track allocation to simulate what register_buffer does
            registry
                .track_buffer_allocation(1024, true)
                .expect("test allocation fits budget");

            let metrics = registry.get_metrics();
            assert_eq!(metrics.buffer_count, 1);
            assert_eq!(metrics.buffer_bytes, 1024);
            assert_eq!(metrics.host_visible_bytes, 1024);

            // Now drop the handle (but it will call global_tracker, not our local registry)
            drop(handle);
        }
    }

    #[test]
    fn test_register_buffer_helper() {
        let usage = BufferUsages::COPY_DST | BufferUsages::MAP_READ;
        let initial_metrics = global_tracker().get_metrics();

        {
            let _handle = register_buffer(2048, usage).expect("test allocation fits budget");
            let after_alloc_metrics = global_tracker().get_metrics();

            // Should have increased by our allocation
            assert_eq!(
                after_alloc_metrics.buffer_count,
                initial_metrics.buffer_count + 1
            );
            assert_eq!(
                after_alloc_metrics.buffer_bytes,
                initial_metrics.buffer_bytes + 2048
            );
            assert_eq!(
                after_alloc_metrics.host_visible_bytes,
                initial_metrics.host_visible_bytes + 2048
            );
        }

        // After handle drop, should return to initial state
        let final_metrics = global_tracker().get_metrics();
        assert_eq!(final_metrics.buffer_count, initial_metrics.buffer_count);
        assert_eq!(final_metrics.buffer_bytes, initial_metrics.buffer_bytes);
        assert_eq!(
            final_metrics.host_visible_bytes,
            initial_metrics.host_visible_bytes
        );
    }

    #[test]
    fn test_register_texture_helper() {
        let initial_metrics = global_tracker().get_metrics();

        {
            let _handle = register_texture(512, 512, TextureFormat::Rgba8Unorm);
            let after_alloc_metrics = global_tracker().get_metrics();

            // Should have increased by our allocation (512*512*4 = 1,048,576 bytes)
            assert_eq!(
                after_alloc_metrics.texture_count,
                initial_metrics.texture_count + 1
            );
            assert_eq!(
                after_alloc_metrics.texture_bytes,
                initial_metrics.texture_bytes + 1_048_576
            );
        }

        // After handle drop, should return to initial state
        let final_metrics = global_tracker().get_metrics();
        assert_eq!(final_metrics.texture_count, initial_metrics.texture_count);
        assert_eq!(final_metrics.texture_bytes, initial_metrics.texture_bytes);
    }

    /// Restore the budget policy at the end of a test that mutated it.
    struct PolicyGuard(&'static str);
    impl Drop for PolicyGuard {
        fn drop(&mut self) {
            let _ = global_tracker().set_budget_policy(self.0);
        }
    }
    fn save_policy() -> PolicyGuard {
        PolicyGuard(global_tracker().get_budget_policy())
    }

    #[test]
    fn test_enforce_error_names_label_and_top_consumers() {
        let _guard = save_policy();
        let _ = global_tracker().set_budget_policy("enforce");

        // A 600 MiB host-visible request exceeds the 512 MiB limit even with an
        // otherwise-empty budget, so this trips regardless of concurrent state.
        let bytes = 600u64 * 1024 * 1024;
        let err = global_tracker()
            .check_budget_labeled(bytes, "unit-test-blob")
            .expect_err("600 MiB host-visible request must exceed the budget under enforce");
        let msg = err.to_string();
        assert!(
            msg.contains("Memory budget exceeded"),
            "message keeps the legacy prefix: {msg}"
        );
        assert!(
            msg.contains("unit-test-blob"),
            "message names the offending label: {msg}"
        );
        assert!(
            msg.contains("top consumers"),
            "message names the top consumers: {msg}"
        );
    }

    #[test]
    fn test_drop_removes_ledger_entry_and_decrements_counters() {
        let device = match crate::core::gpu::create_device_for_test() {
            Some(d) => d,
            None => {
                eprintln!("skipping: no GPU adapter");
                return;
            }
        };

        let before = ledger_snapshot();
        let size = 4096u64;
        {
            let _buf = tracked_create_buffer(
                &device,
                &BufferDescriptor {
                    label: Some("drop-test-buffer"),
                    size,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            )
            .expect("small host-visible allocation must succeed");

            let during = ledger_snapshot();
            assert_eq!(
                during.current_host_visible_bytes,
                before.current_host_visible_bytes + size,
                "ledger host-visible counter increments while the buffer is live"
            );
            assert_eq!(
                during.by_label.get("drop-test-buffer").copied(),
                Some(size),
                "ledger records the labeled entry"
            );
        }

        let after = ledger_snapshot();
        assert_eq!(
            after.current_host_visible_bytes, before.current_host_visible_bytes,
            "dropping the TrackedBuffer decrements the ledger counter"
        );
        assert!(
            !after.by_label.contains_key("drop-test-buffer"),
            "dropping the TrackedBuffer removes the ledger entry"
        );

        device.poll(wgpu::Maintain::Wait);
        std::mem::forget(device);
    }

    #[test]
    fn test_invariant_holds_after_interleaved_alloc_free() {
        let device = match crate::core::gpu::create_device_for_test() {
            Some(d) => d,
            None => {
                eprintln!("skipping: no GPU adapter");
                return;
            }
        };

        let mk = |label: &'static str, size: u64| {
            tracked_create_buffer(
                &device,
                &BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                },
            )
            .expect("allocation must succeed")
        };

        let before = ledger_snapshot();
        let a = mk("inv-a", 1024);
        let b = mk("inv-b", 2048);
        // snapshot() debug-asserts the sum==counter invariant internally.
        let mid = ledger_snapshot();
        assert_eq!(
            mid.current_host_visible_bytes,
            before.current_host_visible_bytes + 1024 + 2048
        );
        drop(a);
        let c = mk("inv-c", 512);
        let _ = ledger_snapshot();
        drop(b);
        drop(c);

        let after = ledger_snapshot();
        assert_eq!(
            after.current_host_visible_bytes, before.current_host_visible_bytes,
            "counters return to baseline after all frees"
        );

        device.poll(wgpu::Maintain::Wait);
        std::mem::forget(device);
    }

    #[test]
    fn test_none_label_falls_back_to_call_site() {
        let device = match crate::core::gpu::create_device_for_test() {
            Some(d) => d,
            None => {
                eprintln!("skipping: no GPU adapter");
                return;
            }
        };

        let buf = tracked_create_buffer(
            &device,
            &BufferDescriptor {
                label: None,
                size: 256,
                usage: BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
        .expect("device-local allocation must succeed");

        let snap = ledger_snapshot();
        // The fallback label is this file's path (forward-slashed) + a line number.
        let has_call_site = snap
            .by_label
            .keys()
            .any(|k| k.contains("resource_tracker.rs:") && !k.contains('\\'));
        assert!(
            has_call_site,
            "None-label allocation falls back to a normalized call-site label; labels: {:?}",
            snap.by_label.keys().collect::<Vec<_>>()
        );

        drop(buf);
        device.poll(wgpu::Maintain::Wait);
        std::mem::forget(device);
    }

    #[test]
    fn capture_peak_starts_at_live_allocation_total() {
        let ledger = AllocationLedger::new();
        let owner = AllocationOwner::new();
        let _scope = owner.activate();
        let persistent = ledger.insert(
            "persistent".to_string(),
            1024,
            true,
            LedgerCategory::Buffer,
            "test:1".to_string(),
        );

        ledger.begin_capture(&[owner.id()]);
        let lazy = ledger.insert(
            "lazy".to_string(),
            32 * 1024,
            true,
            LedgerCategory::Buffer,
            "test:2".to_string(),
        );
        let first = ledger.finish_capture();

        ledger.begin_capture(&[owner.id()]);
        let second = ledger.finish_capture();

        assert_eq!(first.peak_host_visible_bytes, 33 * 1024);
        assert_eq!(
            second.peak_host_visible_bytes,
            first.peak_host_visible_bytes
        );

        ledger.remove(lazy);
        ledger.remove(persistent);
    }

    #[test]
    fn texture_descriptor_size_counts_mips_layers_and_samples() {
        let desc = TextureDescriptor {
            label: Some("sized-texture"),
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 2,
            },
            mip_level_count: 3,
            sample_count: 4,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        assert_eq!(calculate_texture_descriptor_size(&desc), 2688);
    }

    #[test]
    fn texture_descriptor_size_uses_format_blocks_and_3d_mips() {
        let compressed = TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 7,
                height: 5,
                depth_or_array_layers: 3,
            },
            mip_level_count: 2,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Bc1RgbaUnorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        assert_eq!(calculate_texture_descriptor_size(&compressed), 120);

        let volume = TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 4,
            },
            mip_level_count: 3,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        assert_eq!(calculate_texture_descriptor_size(&volume), 73);
    }

    #[test]
    fn capture_excludes_unrelated_allocation_owner() {
        let ledger = AllocationLedger::new();
        let owner_a = AllocationOwner::new();
        let owner_b = AllocationOwner::new();

        let allocation_a = {
            let _scope = owner_a.activate();
            ledger.insert(
                "owner-a".to_string(),
                1024,
                true,
                LedgerCategory::Buffer,
                "test:a".to_string(),
            )
        };
        let allocation_b = {
            let _scope = owner_b.activate();
            ledger.insert(
                "owner-b".to_string(),
                2048,
                true,
                LedgerCategory::Buffer,
                "test:b".to_string(),
            )
        };

        ledger.begin_capture(&[owner_a.id()]);
        let report = ledger.finish_capture();
        assert_eq!(report.peak_host_visible_bytes, 1024);
        assert_eq!(report.by_label.get("owner-a"), Some(&1024));
        assert!(!report.by_label.contains_key("owner-b"));

        ledger.remove(allocation_a);
        ledger.remove(allocation_b);
    }

    #[test]
    fn nested_owner_extension_adds_persistent_renderer_allocations() {
        let ledger = AllocationLedger::new();
        let outer_owner = AllocationOwner::new();
        let nested_owner = AllocationOwner::new();

        let outer_allocation = {
            let _scope = outer_owner.activate();
            ledger.insert(
                "outer-owner".to_string(),
                1024,
                true,
                LedgerCategory::Buffer,
                "test:outer".to_string(),
            )
        };
        let nested_allocation = {
            let _scope = nested_owner.activate();
            ledger.insert(
                "nested-owner".to_string(),
                4096,
                false,
                LedgerCategory::Texture,
                "test:nested".to_string(),
            )
        };

        ledger.begin_capture(&[outer_owner.id()]);
        ledger.extend_capture(&[nested_owner.id()]);
        let report = ledger.finish_capture();

        assert_eq!(report.peak_host_visible_bytes, 1024);
        assert_eq!(report.peak_device_local_bytes, 4096);
        assert_eq!(report.by_label.get("outer-owner"), Some(&1024));
        assert_eq!(report.by_label.get("nested-owner"), Some(&4096));

        ledger.remove(outer_allocation);
        ledger.remove(nested_allocation);
    }

    #[test]
    fn capture_retains_peak_by_label_after_temporary_allocation_drops() {
        let ledger = AllocationLedger::new();
        ledger.begin_capture(&[]);
        let allocation = ledger.insert(
            "temporary-readback".to_string(),
            8192,
            true,
            LedgerCategory::Buffer,
            "test:temporary".to_string(),
        );
        ledger.remove(allocation);

        let report = ledger.finish_capture();
        assert_eq!(report.current_host_visible_bytes, 0);
        assert_eq!(report.peak_host_visible_bytes, 8192);
        assert_eq!(report.by_label.get("temporary-readback"), Some(&8192));
    }

    #[test]
    fn capture_includes_mid_capture_ownerless_allocations() {
        // Allocations made while a capture is active are counted even when no
        // AllocationOwner is active (owner_id == None): a mid-render ownerless
        // allocation must show up in the certificate evidence rather than
        // silently under-counting (CENSOR audit F-07). Only PRE-EXISTING
        // ownerless entries (ambient process state) stay excluded.
        let ledger = AllocationLedger::new();
        let ambient = ledger.insert(
            "ambient-ownerless".to_string(),
            1024,
            true,
            LedgerCategory::Buffer,
            "test:ambient".to_string(),
        );
        ledger.begin_capture(&[]);
        let mid_render = ledger.insert(
            "mid-render-ownerless".to_string(),
            4096,
            true,
            LedgerCategory::Buffer,
            "test:mid".to_string(),
        );
        let report = ledger.finish_capture();

        assert_eq!(report.peak_host_visible_bytes, 4096);
        assert_eq!(report.by_label.get("mid-render-ownerless"), Some(&4096));
        assert!(!report.by_label.contains_key("ambient-ownerless"));

        ledger.remove(ambient);
        ledger.remove(mid_render);
    }

    #[test]
    fn owner_captures_are_independent_under_root_capture() {
        let ledger = AllocationLedger::new();
        let owner_a = AllocationOwner::new();
        let owner_b = AllocationOwner::new();

        ledger.begin_capture(&[]);
        let capture_a = ledger.begin_owner_capture(owner_a.id());
        let capture_b = ledger.begin_owner_capture(owner_b.id());

        let allocation_a = {
            let _scope = owner_a.activate();
            ledger.insert(
                "owner-a-temporary".to_string(),
                1024,
                true,
                LedgerCategory::Buffer,
                "test:owner-a".to_string(),
            )
        };
        let allocation_b = {
            let _scope = owner_b.activate();
            ledger.insert(
                "owner-b-persistent".to_string(),
                4096,
                false,
                LedgerCategory::Texture,
                "test:owner-b".to_string(),
            )
        };
        ledger.remove(allocation_a);

        let report_a = capture_a.finish();
        let report_b = capture_b.finish();
        let root_report = ledger.finish_capture();

        assert_eq!(report_a.peak_host_visible_bytes, 1024);
        assert_eq!(report_a.current_host_visible_bytes, 0);
        assert_eq!(report_a.by_label.get("owner-a-temporary"), Some(&1024));
        assert_eq!(report_a.peak_device_local_bytes, 0);
        assert_eq!(report_b.peak_device_local_bytes, 4096);
        assert_eq!(report_b.current_device_local_bytes, 4096);
        assert_eq!(report_b.by_label.get("owner-b-persistent"), Some(&4096));
        assert_eq!(report_b.peak_host_visible_bytes, 0);
        assert_eq!(root_report.peak_host_visible_bytes, 1024);
        assert_eq!(root_report.peak_device_local_bytes, 4096);
        assert_eq!(root_report.by_label.get("owner-a-temporary"), Some(&1024));
        assert_eq!(root_report.by_label.get("owner-b-persistent"), Some(&4096));

        ledger.remove(allocation_b);
    }

    #[test]
    fn owner_capture_guard_aborts_on_early_error() {
        fn fail_after_allocation(
            ledger: &AllocationLedger,
            owner: &AllocationOwner,
            allocation: &mut Option<u64>,
        ) -> Result<(), ()> {
            let _capture = ledger.begin_owner_capture(owner.id());
            let _scope = owner.activate();
            *allocation = Some(ledger.insert(
                "early-error".to_string(),
                2048,
                true,
                LedgerCategory::Buffer,
                "test:early-error".to_string(),
            ));
            Err(())
        }

        let ledger = AllocationLedger::new();
        let owner = AllocationOwner::new();
        let mut allocation = None;
        let result = fail_after_allocation(&ledger, &owner, &mut allocation);

        assert!(result.is_err());
        assert!(!ledger
            .owner_captures
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .contains_key(&owner.id()));

        ledger.remove(allocation.expect("the failing path allocated a resource"));
        let report = ledger.begin_owner_capture(owner.id()).finish();
        assert_eq!(report.peak_host_visible_bytes, 0);
        assert_eq!(report.current_host_visible_bytes, 0);
    }

    #[test]
    fn global_ledger_registry_cross_check_matches_both_axes() {
        let before_ledger = ledger_snapshot();
        let before_registry = global_tracker().ledger_totals();
        let _handle = register_buffer_explicit(4096, true).expect("test allocation fits budget");
        let ledger = ledger_snapshot();
        let registry = global_tracker().ledger_totals();
        assert_eq!(ledger.current_host_visible_bytes, registry.0);
        assert_eq!(ledger.current_device_local_bytes, registry.1);
        assert_eq!(
            ledger.current_host_visible_bytes,
            before_ledger.current_host_visible_bytes + 4096
        );
        assert_eq!(registry.0, before_registry.0 + 4096);
        assert!(ledger_registry_cross_check().is_ok());
    }

    #[test]
    fn mixed_host_visible_paths_atomically_enforce_aggregate_budget() {
        let _policy_guard = save_policy();
        global_tracker()
            .set_budget_policy("enforce")
            .expect("valid policy");
        let tracker = global_tracker();
        let before = tracker.get_metrics().host_visible_bytes;
        let available = tracker
            .get_budget_limit()
            .checked_sub(before)
            .expect("tracker starts within budget");
        let request = available / 2 + 1;
        let start = std::sync::Arc::new(std::sync::Barrier::new(3));
        let scoped_start = std::sync::Arc::clone(&start);
        let scoped = std::thread::spawn(move || {
            scoped_start.wait();
            tracked_host_allocation(request, "mixed.atomic.scoped")
        });
        let ordinary_start = std::sync::Arc::clone(&start);
        let ordinary = std::thread::spawn(move || {
            ordinary_start.wait();
            register_buffer_explicit(request, true)
        });
        start.wait();
        let scoped_result = scoped.join().expect("scoped path completes");
        let ordinary_result = ordinary.join().expect("ordinary path completes");
        let success_count =
            usize::from(scoped_result.is_ok()) + usize::from(ordinary_result.is_ok());
        assert_eq!(success_count, 1, "exactly one racing reservation must fit");
        let failure_count = usize::from(matches!(&scoped_result, Err(RenderError::Budget(_))))
            + usize::from(matches!(&ordinary_result, Err(RenderError::Budget(_))));
        assert_eq!(
            failure_count, 1,
            "the losing path returns a typed budget error"
        );
        assert_eq!(tracker.get_metrics().host_visible_bytes, before + request);
        drop(scoped_result);
        drop(ordinary_result);
        assert_eq!(tracker.get_metrics().host_visible_bytes, before);
    }

    #[test]
    fn global_ledger_registry_cross_check_rejects_an_unpaired_entry() {
        let id = ledger().insert(
            "unpaired-ledger-entry".to_string(),
            1,
            false,
            LedgerCategory::Buffer,
            "test:cross".to_string(),
        );
        assert!(ledger_registry_cross_check().is_err());
        ledger().remove(id);
        assert!(ledger_registry_cross_check().is_ok());
    }
}
