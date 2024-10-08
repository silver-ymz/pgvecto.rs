use base::index::*;
use pgrx::guc::{GucContext, GucFlags, GucRegistry, GucSetting};

static FLAT_SQ_RERANK_SIZE: GucSetting<i32> =
    GucSetting::<i32>::new(SearchOptions::default_flat_sq_rerank_size() as i32);

static FLAT_SQ_FAST_SCAN: GucSetting<bool> =
    GucSetting::<bool>::new(SearchOptions::default_flat_sq_fast_scan());

static FLAT_PQ_RERANK_SIZE: GucSetting<i32> =
    GucSetting::<i32>::new(SearchOptions::default_flat_pq_rerank_size() as i32);

static FLAT_PQ_FAST_SCAN: GucSetting<bool> =
    GucSetting::<bool>::new(SearchOptions::default_flat_pq_fast_scan());

static IVF_SQ_RERANK_SIZE: GucSetting<i32> =
    GucSetting::<i32>::new(SearchOptions::default_ivf_sq_rerank_size() as i32);

static IVF_SQ_FAST_SCAN: GucSetting<bool> =
    GucSetting::<bool>::new(SearchOptions::default_ivf_sq_fast_scan());

static IVF_PQ_RERANK_SIZE: GucSetting<i32> =
    GucSetting::<i32>::new(SearchOptions::default_ivf_pq_rerank_size() as i32);

static IVF_PQ_FAST_SCAN: GucSetting<bool> =
    GucSetting::<bool>::new(SearchOptions::default_ivf_pq_fast_scan());

static IVF_NPROBE: GucSetting<i32> =
    GucSetting::<i32>::new(SearchOptions::default_ivf_nprobe() as i32);

static HNSW_EF_SEARCH: GucSetting<i32> =
    GucSetting::<i32>::new(SearchOptions::default_hnsw_ef_search() as i32);

static RABITQ_NPROBE: GucSetting<i32> =
    GucSetting::<i32>::new(SearchOptions::default_rabitq_nprobe() as i32);

static RABITQ_EPSILON: GucSetting<f64> =
    GucSetting::<f64>::new(SearchOptions::default_rabitq_epsilon() as f64);

static RABITQ_FAST_SCAN: GucSetting<bool> =
    GucSetting::<bool>::new(SearchOptions::default_rabitq_fast_scan());

static DISKANN_EF_SEARCH: GucSetting<i32> =
    GucSetting::<i32>::new(SearchOptions::default_diskann_ef_search() as i32);

pub unsafe fn init() {
    GucRegistry::define_int_guc(
        "vectors.flat_sq_rerank_size",
        "Scalar quantization reranker size.",
        "https://docs.pgvecto.rs/usage/search.html",
        &FLAT_SQ_RERANK_SIZE,
        0,
        65535,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        "vectors.flat_sq_fast_scan",
        "Enables fast scan or not.",
        "https://docs.pgvecto.rs/usage/search.html",
        &FLAT_SQ_FAST_SCAN,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vectors.flat_pq_rerank_size",
        "Product quantization reranker size.",
        "https://docs.pgvecto.rs/usage/search.html",
        &FLAT_PQ_RERANK_SIZE,
        0,
        65535,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        "vectors.flat_pq_fast_scan",
        "Enables fast scan or not.",
        "https://docs.pgvecto.rs/usage/search.html",
        &FLAT_PQ_FAST_SCAN,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vectors.ivf_sq_rerank_size",
        "Scalar quantization reranker size.",
        "https://docs.pgvecto.rs/usage/search.html",
        &IVF_SQ_RERANK_SIZE,
        0,
        65535,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        "vectors.ivf_sq_fast_scan",
        "Enables fast scan or not.",
        "https://docs.pgvecto.rs/usage/search.html",
        &IVF_SQ_FAST_SCAN,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vectors.ivf_pq_rerank_size",
        "Product quantization reranker size.",
        "https://docs.pgvecto.rs/usage/search.html",
        &IVF_PQ_RERANK_SIZE,
        0,
        65535,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        "vectors.ivf_pq_fast_scan",
        "Enables fast scan or not.",
        "https://docs.pgvecto.rs/usage/search.html",
        &IVF_PQ_FAST_SCAN,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vectors.ivf_nprobe",
        "`nprobe` argument of IVF algorithm.",
        "https://docs.pgvecto.rs/usage/search.html",
        &IVF_NPROBE,
        1,
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vectors.hnsw_ef_search",
        "`ef_search` argument of HNSW algorithm.",
        "https://docs.pgvecto.rs/usage/search.html",
        &HNSW_EF_SEARCH,
        1,
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vectors.rabitq_nprobe",
        "`nprobe` argument of RaBitQ algorithm.",
        "https://docs.pgvecto.rs/usage/search.html",
        &RABITQ_NPROBE,
        1,
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_float_guc(
        "vectors.rabitq_epsilon",
        "`epsilon` argument of RaBitQ algorithm.",
        "https://docs.pgvecto.rs/usage/search.html",
        &RABITQ_EPSILON,
        1.0,
        4.0,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_bool_guc(
        "vectors.rabitq_fast_scan",
        "Enables fast scan or not.",
        "https://docs.pgvecto.rs/usage/search.html",
        &RABITQ_FAST_SCAN,
        GucContext::Userset,
        GucFlags::default(),
    );
    GucRegistry::define_int_guc(
        "vectors.diskann_ef_search",
        "`ef_search` argument of DiskANN algorithm.",
        "https://docs.pgvecto.rs/usage/search.html",
        &DISKANN_EF_SEARCH,
        1,
        u16::MAX as _,
        GucContext::Userset,
        GucFlags::default(),
    );
}

pub fn search_options() -> SearchOptions {
    SearchOptions {
        flat_sq_rerank_size: FLAT_SQ_RERANK_SIZE.get() as u32,
        flat_sq_fast_scan: FLAT_SQ_FAST_SCAN.get(),
        flat_pq_rerank_size: FLAT_PQ_RERANK_SIZE.get() as u32,
        flat_pq_fast_scan: FLAT_PQ_FAST_SCAN.get(),
        ivf_sq_rerank_size: IVF_SQ_RERANK_SIZE.get() as u32,
        ivf_sq_fast_scan: IVF_SQ_FAST_SCAN.get(),
        ivf_pq_rerank_size: IVF_PQ_RERANK_SIZE.get() as u32,
        ivf_pq_fast_scan: IVF_PQ_FAST_SCAN.get(),
        ivf_nprobe: IVF_NPROBE.get() as u32,
        hnsw_ef_search: HNSW_EF_SEARCH.get() as u32,
        rabitq_nprobe: RABITQ_NPROBE.get() as u32,
        rabitq_epsilon: RABITQ_EPSILON.get() as f32,
        rabitq_fast_scan: RABITQ_FAST_SCAN.get(),
        diskann_ef_search: DISKANN_EF_SEARCH.get() as u32,
    }
}
