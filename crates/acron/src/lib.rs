#![feature(trait_alias)]
#![allow(clippy::len_without_is_empty)]

use base::index::*;
use base::operator::*;
use base::scalar::F32;
use base::search::*;
use bytemuck::{Pod, Zeroable};
use common::dir_ops::sync_dir;
use common::mmap_array::MmapArray;
use crc32fast::hash as crc32;
use parking_lot::{Mutex, RwLock};
use quantization::operator::OperatorQuantization;
use quantization::Quantization;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::fs::create_dir;
use std::ops::RangeInclusive;
use std::path::Path;
use std::sync::Arc;
use storage::operator::OperatorStorage;
use storage::StorageCollection;

pub trait OperatorAcron = Operator + OperatorQuantization + OperatorStorage;
pub struct Acron<O: OperatorAcron> {
    mmap: AcronMmap<O>,
}

impl<O: OperatorAcron> Acron<O> {
    pub fn create<S: Source<O>>(path: &Path, options: IndexOptions, source: &S) -> Self {
        create_dir(path).unwrap();
        let ram = make(path, options, source);
        let mmap = save(ram, path);
        sync_dir(path);
        Self { mmap }
    }

    pub fn open(path: &Path, options: IndexOptions) -> Self {
        let mmap = open(path, options);
        Self { mmap }
    }

    pub fn basic(
        &self,
        vector: Borrowed<'_, O>,
        opts: &SearchOptions,
        filter: impl Filter,
    ) -> BinaryHeap<Reverse<Element>> {
        basic(&self.mmap, vector, opts.hnsw_ef_search, filter)
    }

    pub fn vbase<'a>(
        &'a self,
        vector: Borrowed<'a, O>,
        opts: &'a SearchOptions,
        filter: impl Filter + 'a,
    ) -> (Vec<Element>, Box<(dyn Iterator<Item = Element> + 'a)>) {
        vbase(&self.mmap, vector, opts.hnsw_ef_search, filter)
    }

    pub fn len(&self) -> u32 {
        self.mmap.storage.len()
    }

    pub fn vector(&self, i: u32) -> Borrowed<'_, O> {
        self.mmap.storage.vector(i)
    }

    pub fn payload(&self, i: u32) -> Payload {
        self.mmap.storage.payload(i)
    }
}

unsafe impl<O: OperatorAcron> Send for Acron<O> {}
unsafe impl<O: OperatorAcron> Sync for Acron<O> {}

pub struct AcronRam<O: OperatorAcron> {
    storage: Arc<StorageCollection<O>>,
    quantization: Quantization<O, StorageCollection<O>>,
    // ----------------------
    m: u32,
    m_beta: u32,
    // ----------------------
    graph: AcronRamGraph,
    // ----------------------
    visited: VisitedPool,
}

struct AcronRamGraph {
    vertexs: Vec<AcronRamVertex>,
}

struct AcronRamVertex {
    layers: Vec<RwLock<AcronRamLayer>>,
}

impl AcronRamVertex {
    fn levels(&self) -> u8 {
        self.layers.len() as u8 - 1
    }
}

struct AcronRamLayer {
    edges: Vec<AcronEdge>,
}

pub struct AcronMmap<O: OperatorAcron> {
    storage: Arc<StorageCollection<O>>,
    quantization: Quantization<O, StorageCollection<O>>,
    // ----------------------
    m: u32,
    m_beta: u32,
    // ----------------------
    edges: MmapArray<AcronEdge>,
    by_layer_id: MmapArray<usize>,
    by_vertex_id: MmapArray<usize>,
    // ----------------------
    visited: VisitedPool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
struct AcronEdge(F32, u32);
// we may convert a memory-mapped graph to a memory graph
// so that it speeds merging sealed segments

unsafe impl Pod for AcronEdge {}
unsafe impl Zeroable for AcronEdge {}

unsafe impl<O: OperatorAcron> Send for AcronMmap<O> {}
unsafe impl<O: OperatorAcron> Sync for AcronMmap<O> {}

pub fn make<O: OperatorAcron, S: Source<O>>(
    path: &Path,
    options: IndexOptions,
    source: &S,
) -> AcronRam<O> {
    log::warn!("Creating Acron index at {:?}", path);
    let AcronIndexingOptions {
        m,
        m_beta,
        gamma,
        quantization: quantization_opts,
    } = options.indexing.clone().unwrap_acron();
    let m_beta = m_beta as usize;
    let storage = Arc::new(StorageCollection::create(&path.join("raw"), source));
    rayon::check();
    let quantization = Quantization::create(
        &path.join("quantization"),
        options.clone(),
        quantization_opts,
        &storage,
        (0..storage.len()).collect::<Vec<_>>(),
    );
    rayon::check();
    let n = storage.len();
    let graph = AcronRamGraph {
        vertexs: (0..n)
            .into_par_iter()
            .map(|i| AcronRamVertex {
                layers: (0..count_layers_of_a_vertex(m, i))
                    .map(|_| RwLock::new(AcronRamLayer { edges: Vec::new() }))
                    .collect(),
            })
            .collect(),
    };
    let entry = RwLock::<Option<u32>>::new(None);
    let visited = VisitedPool::new(storage.len());
    (0..n).into_par_iter().for_each(|i| {
        fn fast_search<O: OperatorAcron>(
            quantization: &Quantization<O, StorageCollection<O>>,
            graph: &AcronRamGraph,
            levels: RangeInclusive<u8>,
            u: u32,
            target: Borrowed<'_, O>,
        ) -> u32 {
            let mut u = u;
            let mut u_dis = quantization.distance(target, u);
            for i in levels.rev() {
                let mut changed = true;
                while changed {
                    changed = false;
                    let guard = graph.vertexs[u as usize].layers[i as usize].read();
                    for &AcronEdge(_, v) in guard.edges.iter() {
                        let v_dis = quantization.distance(target, v);
                        if v_dis < u_dis {
                            u = v;
                            u_dis = v_dis;
                            changed = true;
                        }
                    }
                }
            }
            u
        }
        #[allow(clippy::too_many_arguments)]
        fn local_search<O: OperatorAcron>(
            quantization: &Quantization<O, StorageCollection<O>>,
            graph: &AcronRamGraph,
            visited: &mut VisitedGuard,
            vector: Borrowed<'_, O>,
            m_beta: usize,
            s: u32,
            k: usize,
            i: u8,
        ) -> Vec<AcronEdge> {
            let mut visited = visited.fetch();
            let mut candidates = BinaryHeap::<Reverse<AcronEdge>>::new();
            let mut results = BinaryHeap::new();
            let s_dis = quantization.distance(vector, s);
            visited.mark(s);
            candidates.push(Reverse(AcronEdge(s_dis, s)));
            results.push(AcronEdge(s_dis, s));
            while let Some(Reverse(AcronEdge(u_dis, u))) = candidates.pop() {
                if !(results.len() < k || u_dis < results.peek().unwrap().0) {
                    break;
                }
                let edges = &graph.vertexs[u as usize].layers[i as usize].read().edges;
                let stage1_iter = edges.iter().map(|&AcronEdge(_, v)| v).take(m_beta);
                let stage2_iter = edges
                    .iter()
                    .map(|&AcronEdge(_, v)| v)
                    .skip(m_beta)
                    .flat_map(|v| {
                        let edges = &graph.vertexs[v as usize].layers[i as usize].read().edges;
                        std::iter::once(v)
                            .chain(edges.iter().map(|&AcronEdge(_, v)| v))
                            .collect::<Vec<_>>()
                    });
                for v in stage1_iter.chain(stage2_iter) {
                    if !visited.check(v) {
                        continue;
                    }
                    visited.mark(v);
                    let v_dis = quantization.distance(vector, v);
                    if results.len() < k || v_dis < results.peek().unwrap().0 {
                        candidates.push(Reverse(AcronEdge(v_dis, v)));
                        results.push(AcronEdge(v_dis, v));
                        if results.len() > k {
                            results.pop();
                        }
                    }
                }
            }
            results.into_sorted_vec()
        }
        fn select(graph: &AcronRamGraph, input: &mut Vec<AcronEdge>, layer: u8, m_beta: usize) {
            if input.len() <= m_beta {
                return;
            }
            let mut res = Vec::new();
            res.extend_from_slice(&input[..m_beta]);
            let mut vertex_set = HashSet::new();
            for &val in input.iter().skip(m_beta) {
                if vertex_set.contains(&val.1) {
                    continue;
                }
                res.push(val);
                let edges = graph.vertexs[val.1 as usize].layers[layer as usize].read();
                vertex_set.extend(edges.edges.iter().map(|&AcronEdge(_, v)| v));
            }
            *input = res;
        }
        rayon::check();
        let mut visited = visited.fetch();
        let target = storage.vector(i);
        let levels = graph.vertexs[i as usize].levels();
        let local_entry;
        let update_entry;
        {
            let check = |global: Option<u32>| {
                if let Some(u) = global {
                    graph.vertexs[u as usize].levels() < levels
                } else {
                    true
                }
            };
            let read = entry.read();
            if check(*read) {
                drop(read);
                let write = entry.write();
                if check(*write) {
                    local_entry = *write;
                    update_entry = Some(write);
                } else {
                    local_entry = *write;
                    update_entry = None;
                }
            } else {
                local_entry = *read;
                update_entry = None;
            }
        };
        let Some(mut u) = local_entry else {
            if let Some(mut write) = update_entry {
                *write = Some(i);
            }
            return;
        };
        let top = graph.vertexs[u as usize].levels();
        if top > levels {
            u = fast_search(&quantization, &graph, levels + 1..=top, u, target);
        }
        let mut result = Vec::with_capacity(1 + std::cmp::min(levels, top) as usize);
        for j in (0..=std::cmp::min(levels, top)).rev() {
            let mut edges = local_search(
                &quantization,
                &graph,
                &mut visited,
                target,
                m_beta,
                u,
                (m * gamma) as usize,
                j,
            );
            edges.sort();
            select(&graph, &mut edges, j, m_beta);
            u = edges.first().unwrap().1;
            result.push(edges);
        }
        for j in 0..=std::cmp::min(levels, top) {
            let mut write = graph.vertexs[i as usize].layers[j as usize].write();
            write.edges = result.pop().unwrap();
            let edges = write.edges.clone();
            drop(write);
            for &AcronEdge(n_dis, n) in edges.iter() {
                let mut write = graph.vertexs[n as usize].layers[j as usize].write();
                let element = AcronEdge(n_dis, i);
                let (Ok(index) | Err(index)) = write.edges.binary_search(&element);
                write.edges.insert(index, element);
                let mut edges = write.edges.clone();
                drop(write);
                let origin = crc32(bytemuck::cast_slice(&edges));
                select(&graph, &mut edges, j, m_beta);
                write = graph.vertexs[n as usize].layers[j as usize].write();
                let new = crc32(bytemuck::cast_slice(&edges));
                if origin == new {
                    write.edges = edges;
                }
            }
        }
        if let Some(mut write) = update_entry {
            *write = Some(i);
        }
    });
    AcronRam {
        storage,
        quantization,
        m,
        m_beta: m_beta as u32,
        graph,
        visited,
    }
}

pub fn save<O: OperatorAcron>(mut ram: AcronRam<O>, path: &Path) -> AcronMmap<O> {
    let edges = MmapArray::create(
        &path.join("edges"),
        ram.graph
            .vertexs
            .iter_mut()
            .flat_map(|v| v.layers.iter_mut())
            .flat_map(|v| &v.get_mut().edges)
            .copied(),
    );
    rayon::check();
    let by_layer_id = MmapArray::create(&path.join("by_layer_id"), {
        let iter = ram.graph.vertexs.iter_mut();
        let iter = iter.flat_map(|v| v.layers.iter_mut());
        let iter = iter.map(|v| v.get_mut().edges.len());
        caluate_offsets(iter)
    });
    rayon::check();
    let by_vertex_id = MmapArray::create(&path.join("by_vertex_id"), {
        let iter = ram.graph.vertexs.iter_mut();
        let iter = iter.map(|v| v.layers.len());
        caluate_offsets(iter)
    });
    rayon::check();
    AcronMmap {
        storage: ram.storage,
        quantization: ram.quantization,
        m: ram.m,
        m_beta: ram.m_beta,
        edges,
        by_layer_id,
        by_vertex_id,
        visited: ram.visited,
    }
}

pub fn open<O: OperatorAcron>(path: &Path, options: IndexOptions) -> AcronMmap<O> {
    let idx_opts = options.indexing.clone().unwrap_acron();
    let storage = Arc::new(StorageCollection::open(&path.join("raw"), options.clone()));
    let quantization = Quantization::open(
        &path.join("quantization"),
        options.clone(),
        idx_opts.quantization,
        &storage,
    );
    let edges = MmapArray::open(&path.join("edges"));
    let by_layer_id = MmapArray::open(&path.join("by_layer_id"));
    let by_vertex_id = MmapArray::open(&path.join("by_vertex_id"));
    let n = storage.len();
    AcronMmap {
        storage,
        quantization,
        m: idx_opts.m,
        m_beta: idx_opts.m_beta,
        edges,
        by_layer_id,
        by_vertex_id,
        visited: VisitedPool::new(n),
    }
}

pub fn basic<O: OperatorAcron>(
    mmap: &AcronMmap<O>,
    vector: Borrowed<'_, O>,
    ef_search: usize,
    filter: impl Filter,
) -> BinaryHeap<Reverse<Element>> {
    let Some(s) = entry(mmap, filter.clone()) else {
        return BinaryHeap::new();
    };
    let levels = count_layers_of_a_vertex(mmap.m, s) - 1;
    let u = fast_search(mmap, 1..=levels, s, vector, filter.clone());
    local_search_basic(mmap, ef_search, u, vector, filter).into_reversed_heap()
}

pub fn vbase<'a, O: OperatorAcron>(
    mmap: &'a AcronMmap<O>,
    vector: Borrowed<'a, O>,
    range: usize,
    filter: impl Filter + 'a,
) -> (Vec<Element>, Box<(dyn Iterator<Item = Element> + 'a)>) {
    let Some(s) = entry(mmap, filter.clone()) else {
        return (Vec::new(), Box::new(std::iter::empty()));
    };
    let levels = count_layers_of_a_vertex(mmap.m, s) - 1;
    let u = fast_search(mmap, 1..=levels, s, vector, filter.clone());
    let mut iter = local_search_vbase(mmap, u, vector, filter.clone());
    let mut queue = BinaryHeap::<Element>::with_capacity(1 + range);
    let mut stage1 = Vec::new();
    for x in &mut iter {
        if queue.len() == range && queue.peek().unwrap().distance < x.distance {
            stage1.push(x);
            break;
        }
        if queue.len() == range {
            queue.pop();
        }
        queue.push(x);
        stage1.push(x);
    }
    (stage1, Box::new(iter))
}

pub fn entry<O: OperatorAcron>(mmap: &AcronMmap<O>, mut filter: impl Filter) -> Option<u32> {
    let m = mmap.m;
    let n = mmap.storage.len();
    let mut shift = 1u64;
    let mut count = 0u64;
    while shift * m as u64 <= n as u64 {
        shift *= m as u64;
    }
    while shift != 0 {
        let mut i = 1u64;
        while i * shift <= n as u64 {
            let e = (i * shift - 1) as u32;
            if i % m as u64 != 0 {
                if filter.check(mmap.storage.payload(e)) {
                    return Some(e);
                }
                count += 1;
                if count >= 10000 {
                    return None;
                }
            }
            i += 1;
        }
        shift /= m as u64;
    }
    None
}

pub fn fast_search<O: OperatorAcron>(
    mmap: &AcronMmap<O>,
    levels: RangeInclusive<u8>,
    u: u32,
    vector: Borrowed<'_, O>,
    mut filter: impl Filter,
) -> u32 {
    let mut u = u;
    let mut u_dis = mmap.quantization.distance(vector, u);
    for i in levels.rev() {
        let mut changed = true;
        while changed {
            changed = false;
            let edges = find_neighbors(mmap, u, i, mmap.m_beta as usize)
                .into_iter()
                .filter(|&&AcronEdge(_, v)| filter.check(mmap.storage.payload(v)));
            for &AcronEdge(_, v) in edges {
                let v_dis = mmap.quantization.distance(vector, v);
                if v_dis < u_dis {
                    u = v;
                    u_dis = v_dis;
                    changed = true;
                }
            }
        }
    }
    u
}

pub fn local_search_basic<O: OperatorAcron>(
    mmap: &AcronMmap<O>,
    k: usize,
    s: u32,
    vector: Borrowed<'_, O>,
    mut filter: impl Filter,
) -> ElementHeap {
    let mut visited = mmap.visited.fetch();
    let mut visited = visited.fetch();
    let mut candidates = BinaryHeap::<Reverse<(F32, u32)>>::new();
    let mut results = ElementHeap::new(k);
    visited.mark(s);
    let s_dis = mmap.quantization.distance(vector, s);
    candidates.push(Reverse((s_dis, s)));
    results.push(Element {
        distance: s_dis,
        payload: mmap.storage.payload(s),
    });
    while let Some(Reverse((u_dis, u))) = candidates.pop() {
        if !results.check(u_dis) {
            break;
        }
        let edges = find_neighbors(mmap, u, 0, mmap.m_beta as usize)
            .into_iter()
            .filter(|&&AcronEdge(_, v)| filter.check(mmap.storage.payload(v)));
        for &AcronEdge(_, v) in edges {
            if !visited.check(v) {
                continue;
            }
            visited.mark(v);
            let v_dis = mmap.quantization.distance(vector, v);
            if !results.check(v_dis) {
                continue;
            }
            candidates.push(Reverse((v_dis, v)));
            results.push(Element {
                distance: v_dis,
                payload: mmap.storage.payload(v),
            });
        }
    }
    results
}

pub fn local_search_vbase<'a, O: OperatorAcron>(
    mmap: &'a AcronMmap<O>,
    s: u32,
    vector: Borrowed<'a, O>,
    mut filter: impl Filter + 'a,
) -> impl Iterator<Item = Element> + 'a {
    let mut visited = mmap.visited.fetch2();
    let mut candidates = BinaryHeap::<Reverse<(F32, u32)>>::new();
    visited.mark(s);
    let s_dis = mmap.quantization.distance(vector, s);
    candidates.push(Reverse((s_dis, s)));
    std::iter::from_fn(move || {
        let Reverse((u_dis, u)) = candidates.pop()?;
        {
            let edges = find_neighbors(mmap, u, 0, mmap.m_beta as usize)
                .into_iter()
                .filter(|&&AcronEdge(_, v)| filter.check(mmap.storage.payload(v)));
            for &AcronEdge(_, v) in edges {
                if !visited.check(v) {
                    continue;
                }
                visited.mark(v);
                let v_dis = mmap.quantization.distance(vector, v);
                candidates.push(Reverse((v_dis, v)));
            }
        }
        Some(Element {
            distance: u_dis,
            payload: mmap.storage.payload(u),
        })
    })
}

fn count_layers_of_a_vertex(m: u32, i: u32) -> u8 {
    let mut x = i + 1;
    let mut ans = 1;
    while x % m == 0 {
        ans += 1;
        x /= m;
    }
    ans
}

fn caluate_offsets(iter: impl Iterator<Item = usize>) -> impl Iterator<Item = usize> {
    let mut offset = 0usize;
    let mut iter = std::iter::once(0).chain(iter);
    std::iter::from_fn(move || {
        let x = iter.next()?;
        offset += x;
        Some(offset)
    })
}

fn find_edges<O: OperatorAcron>(mmap: &AcronMmap<O>, u: u32, level: u8) -> &[AcronEdge] {
    let offset = u as usize;
    let index = mmap.by_vertex_id[offset]..mmap.by_vertex_id[offset + 1];
    let offset = index.start + level as usize;
    let index = mmap.by_layer_id[offset]..mmap.by_layer_id[offset + 1];
    &mmap.edges[index]
}

fn find_neighbors<O: OperatorAcron>(
    mmap: &AcronMmap<O>,
    u: u32,
    level: u8,
    m_beta: usize,
) -> impl IntoIterator<Item = &AcronEdge> + '_ {
    let edges = find_edges(mmap, u, level);
    let stage1_iter = edges.iter().take(m_beta);
    let stage2_iter = edges.iter().skip(m_beta).flat_map(move |v| {
        let edges = find_edges(mmap, v.1, level);
        std::iter::once(v).chain(edges.iter()).collect::<Vec<_>>()
    });
    stage1_iter.chain(stage2_iter)
}

struct VisitedPool {
    n: u32,
    locked_buffers: Mutex<Vec<VisitedBuffer>>,
}

impl VisitedPool {
    pub fn new(n: u32) -> Self {
        Self {
            n,
            locked_buffers: Mutex::new(Vec::new()),
        }
    }
    pub fn fetch(&self) -> VisitedGuard {
        let buffer = self
            .locked_buffers
            .lock()
            .pop()
            .unwrap_or_else(|| VisitedBuffer::new(self.n as _));
        VisitedGuard { buffer, pool: self }
    }

    fn fetch2(&self) -> VisitedGuardChecker {
        let mut buffer = self
            .locked_buffers
            .lock()
            .pop()
            .unwrap_or_else(|| VisitedBuffer::new(self.n as _));
        {
            buffer.version = buffer.version.wrapping_add(1);
            if buffer.version == 0 {
                buffer.data.fill(0);
            }
        }
        VisitedGuardChecker { buffer, pool: self }
    }
}

struct VisitedGuard<'a> {
    buffer: VisitedBuffer,
    pool: &'a VisitedPool,
}

impl<'a> VisitedGuard<'a> {
    fn fetch(&mut self) -> VisitedChecker<'_> {
        self.buffer.version = self.buffer.version.wrapping_add(1);
        if self.buffer.version == 0 {
            self.buffer.data.fill(0);
        }
        VisitedChecker {
            buffer: &mut self.buffer,
        }
    }
}

impl<'a> Drop for VisitedGuard<'a> {
    fn drop(&mut self) {
        let src = VisitedBuffer {
            version: 0,
            data: Vec::new(),
        };
        let buffer = std::mem::replace(&mut self.buffer, src);
        self.pool.locked_buffers.lock().push(buffer);
    }
}

struct VisitedChecker<'a> {
    buffer: &'a mut VisitedBuffer,
}

impl<'a> VisitedChecker<'a> {
    fn check(&mut self, i: u32) -> bool {
        self.buffer.data[i as usize] != self.buffer.version
    }
    fn mark(&mut self, i: u32) {
        self.buffer.data[i as usize] = self.buffer.version;
    }
}

struct VisitedGuardChecker<'a> {
    buffer: VisitedBuffer,
    pool: &'a VisitedPool,
}

impl<'a> VisitedGuardChecker<'a> {
    fn check(&mut self, i: u32) -> bool {
        self.buffer.data[i as usize] != self.buffer.version
    }
    fn mark(&mut self, i: u32) {
        self.buffer.data[i as usize] = self.buffer.version;
    }
}

impl<'a> Drop for VisitedGuardChecker<'a> {
    fn drop(&mut self) {
        let src = VisitedBuffer {
            version: 0,
            data: Vec::new(),
        };
        let buffer = std::mem::replace(&mut self.buffer, src);
        self.pool.locked_buffers.lock().push(buffer);
    }
}

struct VisitedBuffer {
    version: usize,
    data: Vec<usize>,
}

impl VisitedBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            version: 0,
            data: bytemuck::zeroed_vec(capacity),
        }
    }
}

pub struct ElementHeap {
    binary_heap: BinaryHeap<Element>,
    k: usize,
}

impl ElementHeap {
    pub fn new(k: usize) -> Self {
        assert!(k != 0);
        Self {
            binary_heap: BinaryHeap::new(),
            k,
        }
    }
    pub fn check(&self, distance: F32) -> bool {
        self.binary_heap.len() < self.k || distance < self.binary_heap.peek().unwrap().distance
    }
    pub fn push(&mut self, element: Element) -> Option<Element> {
        self.binary_heap.push(element);
        if self.binary_heap.len() == self.k + 1 {
            self.binary_heap.pop()
        } else {
            None
        }
    }
    pub fn into_reversed_heap(self) -> BinaryHeap<Reverse<Element>> {
        self.binary_heap.into_iter().map(Reverse).collect()
    }
}
