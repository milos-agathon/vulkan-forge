use glam::Vec2;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::terrain::tiling::{TileBounds, TileData, TileId};

use super::common::{is_descendant_of, CoalescePolicy};
use super::readers::HeightReader;

pub struct AsyncTileLoader {
    req_tx: SyncSender<TileId>,
    done_rx: Receiver<TileData>,
    _dispatcher: thread::JoinHandle<()>,
    _workers: Vec<thread::JoinHandle<()>>,
    _worker_txs: Vec<Sender<TileId>>,
    pool_size: usize,
    // E1e: dedup + backpressure
    pending: Mutex<HashSet<TileId>>, // TileIds currently in-flight
    max_in_flight: usize,            // Backpressure limit
    // E1e/E1f: cancellation
    cancelled: Mutex<HashSet<TileId>>, // Requests canceled by the main thread; results will be dropped
    policy: CoalescePolicy,
    // E1m: counters
    c_requests: AtomicUsize,          // attempts
    c_enqueued: AtomicUsize,          // successfully sent to dispatcher
    c_dropped_by_policy: AtomicUsize, // not enqueued due to coalescing
    c_canceled: AtomicUsize,          // cancel() marked
    c_send_fail: AtomicUsize,         // channel try_send failed
    c_completed: AtomicUsize,         // results delivered by drain_completed
}

impl AsyncTileLoader {
    pub fn new_with_reader(
        root_bounds: TileBounds,
        tile_size: Vec2,
        tile_resolution: u32,
        max_in_flight: usize,
        pool_size: usize,
        reader: Arc<dyn HeightReader>,
        policy: CoalescePolicy,
    ) -> Self {
        let (req_tx, req_rx) = mpsc::sync_channel::<TileId>(max_in_flight.max(1));
        let (done_tx, done_rx) = mpsc::channel::<TileData>();

        // Worker pool
        let mut worker_txs = Vec::new();
        let mut workers = Vec::new();
        let n = pool_size.max(1);
        for _ in 0..n {
            let (wtx, wrx) = mpsc::channel::<TileId>();
            worker_txs.push(wtx.clone());
            let done_tx_clone = done_tx.clone();
            let rb = root_bounds.clone();
            let ts = tile_size;
            let reader = reader.clone();
            let handle = thread::spawn(move || {
                while let Ok(id) = wrx.recv() {
                    let w = tile_resolution;
                    let h = tile_resolution;
                    let heights = reader.read(&rb, ts, id, w, h);
                    let td = TileData::new(id, heights, w, h);
                    let _ = done_tx_clone.send(td);
                }
            });
            workers.push(handle);
        }

        // Dispatcher
        let worker_txs_for_dispatcher: Vec<Sender<TileId>> = worker_txs.iter().cloned().collect();
        let dispatcher = thread::spawn(move || {
            let mut idx: usize = 0;
            while let Ok(id) = req_rx.recv() {
                if worker_txs_for_dispatcher.is_empty() {
                    break;
                }
                let _ = worker_txs_for_dispatcher[idx % worker_txs_for_dispatcher.len()].send(id);
                idx = idx.wrapping_add(1);
            }
        });

        Self {
            req_tx,
            done_rx,
            _dispatcher: dispatcher,
            _workers: workers,
            _worker_txs: worker_txs,
            pool_size: n,
            pending: Mutex::new(HashSet::new()),
            max_in_flight: max_in_flight.max(1),
            cancelled: Mutex::new(HashSet::new()),
            policy,
            c_requests: AtomicUsize::new(0),
            c_enqueued: AtomicUsize::new(0),
            c_dropped_by_policy: AtomicUsize::new(0),
            c_canceled: AtomicUsize::new(0),
            c_send_fail: AtomicUsize::new(0),
            c_completed: AtomicUsize::new(0),
        }
    }

    pub fn request(&self, id: TileId) -> bool {
        // E1e: deduplicate and apply backpressure + E1k: LOD coalescing (policy-configurable)
        if let Ok(mut pend) = self.pending.lock() {
            self.c_requests.fetch_add(1, Ordering::Relaxed);
            // If previously canceled, clear that state when re-requested
            if let Ok(mut can) = self.cancelled.lock() {
                can.remove(&id);
            }
            match self.policy {
                CoalescePolicy::PreferCoarse => {
                    // Drop child if any ancestor pending
                    let mut p = id;
                    while let Some(parent) = p.parent() {
                        if pend.contains(&parent) {
                            self.c_dropped_by_policy.fetch_add(1, Ordering::Relaxed);
                            return false;
                        }
                        p = parent;
                    }
                    if pend.contains(&id) {
                        return false;
                    }
                    if pend.len() >= self.max_in_flight {
                        return false;
                    }
                    // Insert and cancel descendants
                    pend.insert(id);
                    let mut to_cancel: Vec<TileId> = Vec::new();
                    for &d in pend.iter() {
                        if d != id && is_descendant_of(d, id) {
                            to_cancel.push(d);
                        }
                    }
                    if self.req_tx.try_send(id).is_err() {
                        pend.remove(&id);
                        self.c_send_fail.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                    self.c_enqueued.fetch_add(1, Ordering::Relaxed);
                    if !to_cancel.is_empty() {
                        if let Ok(mut can) = self.cancelled.lock() {
                            for d in &to_cancel {
                                pend.remove(d);
                                can.insert(*d);
                            }
                            self.c_canceled
                                .fetch_add(to_cancel.len(), Ordering::Relaxed);
                        }
                    }
                    true
                }
                CoalescePolicy::PreferFine => {
                    // If any descendant pending, skip this coarser request
                    for &d in pend.iter() {
                        if is_descendant_of(d, id) {
                            self.c_dropped_by_policy.fetch_add(1, Ordering::Relaxed);
                            return false;
                        }
                    }
                    if pend.contains(&id) {
                        return false;
                    }
                    if pend.len() >= self.max_in_flight {
                        return false;
                    }
                    // Tentatively insert
                    pend.insert(id);
                    // Try-send; if fails, roll back and keep ancestors
                    if self.req_tx.try_send(id).is_err() {
                        pend.remove(&id);
                        self.c_send_fail.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                    self.c_enqueued.fetch_add(1, Ordering::Relaxed);
                    // Cancel ancestors (post-send)
                    if let Ok(mut can) = self.cancelled.lock() {
                        let mut p = id;
                        while let Some(parent) = p.parent() {
                            if pend.remove(&parent) {
                                can.insert(parent);
                            }
                            p = parent;
                        }
                        self.c_canceled.fetch_add(1, Ordering::Relaxed);
                    }
                    true
                }
            }
        } else {
            false
        }
    }

    pub fn drain_completed(&self, limit: usize) -> Vec<TileData> {
        let mut out = Vec::new();
        while out.len() < limit {
            match self.done_rx.try_recv() {
                Ok(td) => {
                    // E1e: mark request as no longer pending
                    if let Ok(mut pend) = self.pending.lock() {
                        pend.remove(&td.tile_id);
                    }
                    // If canceled, drop the result silently
                    if let Ok(mut can) = self.cancelled.lock() {
                        if can.remove(&td.tile_id) {
                            continue;
                        }
                    }
                    self.c_completed.fetch_add(1, Ordering::Relaxed);
                    out.push(td)
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        out
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        let pending_len = self.pending.lock().map(|p| p.len()).unwrap_or(0);
        (pending_len, self.max_in_flight, self.pool_size)
    }

    pub fn counters(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.c_requests.load(Ordering::Relaxed),
            self.c_enqueued.load(Ordering::Relaxed),
            self.c_dropped_by_policy.load(Ordering::Relaxed),
            self.c_canceled.load(Ordering::Relaxed),
            self.c_send_fail.load(Ordering::Relaxed),
            self.c_completed.load(Ordering::Relaxed),
        )
    }

    /// Cancel a list of pending/in-flight requests. Returns number of IDs marked canceled.
    pub fn cancel(&self, ids: &[TileId]) -> usize {
        let mut n = 0usize;
        if let Ok(mut pend) = self.pending.lock() {
            if let Ok(mut can) = self.cancelled.lock() {
                for id in ids {
                    if pend.remove(id) {
                        can.insert(*id);
                        n += 1;
                    }
                }
                if n > 0 {
                    self.c_canceled.fetch_add(n, Ordering::Relaxed);
                }
            }
        }
        n
    }
}
