use crate::search::MultiColumnData;
use crate::vector::*;
use crate::{distance::*, search::Strategy};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;
use validator::{Validate, ValidationError};

#[must_use]
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum CreateError {
    #[error("Invalid index options.")]
    InvalidIndexOptions { reason: String },
}

#[must_use]
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum DropError {
    #[error("Index not found.")]
    NotExist,
}

#[must_use]
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum FlushError {
    #[error("Index not found.")]
    NotExist,
}

#[must_use]
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum InsertError {
    #[error("Index not found.")]
    NotExist,
    #[error("Invalid vector.")]
    InvalidVector,
}

#[must_use]
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum DeleteError {
    #[error("Index not found.")]
    NotExist,
}

#[must_use]
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum BasicError {
    #[error("Index not found.")]
    NotExist,
    #[error("Invalid vector.")]
    InvalidVector,
    #[error("Invalid search options.")]
    InvalidSearchOptions { reason: String },
}

#[must_use]
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum VbaseError {
    #[error("Index not found.")]
    NotExist,
    #[error("Invalid vector.")]
    InvalidVector,
    #[error("Invalid search options.")]
    InvalidSearchOptions { reason: String },
}

#[must_use]
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum ListError {
    #[error("Index not found.")]
    NotExist,
}

#[must_use]
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum StatError {
    #[error("Index not found.")]
    NotExist,
}

#[must_use]
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum AlterError {
    #[error("Setting key {key} is not exist.")]
    BadKey { key: String },
    #[error("Setting key {key} has a wrong value {value}.")]
    BadValue { key: String, value: String },
    #[error("Index not found.")]
    NotExist,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct IndexFlexibleOptions {
    #[serde(default = "IndexFlexibleOptions::default_optimizing_threads")]
    #[validate(range(min = 1, max = 65535))]
    pub optimizing_threads: u16,
}

impl IndexFlexibleOptions {
    pub fn default_optimizing_threads() -> u16 {
        1
    }
}

impl Default for IndexFlexibleOptions {
    fn default() -> Self {
        Self {
            optimizing_threads: Self::default_optimizing_threads(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "IndexOptions::validate_index_options"))]
pub struct IndexOptions {
    #[validate]
    pub vector: VectorOptions,
    #[validate]
    pub segment: SegmentsOptions,
    #[validate]
    pub optimizing: OptimizingOptions,
    #[validate]
    pub indexing: IndexingOptions,
    #[validate]
    pub multicolumn: MultiColumnOptions,
}

impl IndexOptions {
    fn validate_index_options(options: &IndexOptions) -> Result<(), ValidationError> {
        if options.vector.v != VectorKind::SVecf32 && options.vector.v != VectorKind::BVecf32 {
            return Ok(());
        }
        let is_trivial = match &options.indexing {
            IndexingOptions::Flat(x) => matches!(x.quantization, QuantizationOptions::Trivial(_)),
            IndexingOptions::Ivf(x) => matches!(x.quantization, QuantizationOptions::Trivial(_)),
            IndexingOptions::Hnsw(x) => matches!(x.quantization, QuantizationOptions::Trivial(_)),
            IndexingOptions::Acron(x) => matches!(x.quantization, QuantizationOptions::Trivial(_)),
        };
        if !is_trivial {
            return Err(ValidationError::new(
                "Quantization is not supported for svector and bvector.",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "Self::validate_0"))]
#[validate(schema(function = "Self::validate_dims"))]
pub struct VectorOptions {
    #[validate(range(min = 1, max = 1_048_575))]
    #[serde(rename = "dimensions")]
    pub dims: u32,
    #[serde(rename = "vector")]
    pub v: VectorKind,
    #[serde(rename = "distance")]
    pub d: DistanceKind,
}

impl VectorOptions {
    // Jaccard distance is only supported for bvector.
    pub fn validate_0(&self) -> Result<(), ValidationError> {
        if self.v != VectorKind::BVecf32 && self.d == DistanceKind::Jaccard {
            return Err(ValidationError::new(
                "Jaccard distance is only supported for bvector.",
            ));
        }
        Ok(())
    }

    pub fn validate_dims(&self) -> Result<(), ValidationError> {
        if self.v != VectorKind::SVecf32 && self.dims > 65535 {
            return Err(ValidationError::new(
                "Except svector, the maximum number of dimensions is 65535.",
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "Self::validate_0"))]
pub struct SegmentsOptions {
    #[serde(default = "SegmentsOptions::default_max_growing_segment_size")]
    #[validate(range(min = 1, max = 4_000_000_000u32))]
    pub max_growing_segment_size: u32,
    #[serde(default = "SegmentsOptions::default_max_sealed_segment_size")]
    #[validate(range(min = 1, max = 4_000_000_000u32))]
    pub max_sealed_segment_size: u32,
}

impl SegmentsOptions {
    fn default_max_growing_segment_size() -> u32 {
        20_000
    }
    fn default_max_sealed_segment_size() -> u32 {
        1_000_000
    }
    // max_growing_segment_size <= max_sealed_segment_size
    fn validate_0(&self) -> Result<(), ValidationError> {
        if self.max_growing_segment_size > self.max_sealed_segment_size {
            return Err(ValidationError::new(
                "`max_growing_segment_size` must be less than or equal to `max_sealed_segment_size`",
            ));
        }
        Ok(())
    }
}

impl Default for SegmentsOptions {
    fn default() -> Self {
        Self {
            max_growing_segment_size: Self::default_max_growing_segment_size(),
            max_sealed_segment_size: Self::default_max_sealed_segment_size(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct OptimizingOptions {
    #[serde(default = "OptimizingOptions::default_sealing_secs")]
    #[validate(range(min = 1, max = 60))]
    pub sealing_secs: u64,
    #[serde(default = "OptimizingOptions::default_sealing_size")]
    #[validate(range(min = 1, max = 4_000_000_000u32))]
    pub sealing_size: u32,
    #[serde(default = "OptimizingOptions::default_delete_threshold")]
    #[validate(range(min = 0.01, max = 1.00))]
    pub delete_threshold: f64,
}

impl OptimizingOptions {
    fn default_sealing_secs() -> u64 {
        60
    }
    fn default_sealing_size() -> u32 {
        1
    }
    fn default_delete_threshold() -> f64 {
        0.2
    }
}

impl Default for OptimizingOptions {
    fn default() -> Self {
        Self {
            sealing_secs: Self::default_sealing_secs(),
            sealing_size: Self::default_sealing_size(),
            delete_threshold: Self::default_delete_threshold(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum IndexingOptions {
    Flat(FlatIndexingOptions),
    Ivf(IvfIndexingOptions),
    Hnsw(HnswIndexingOptions),
    Acron(AcronIndexingOptions),
}

impl IndexingOptions {
    pub fn unwrap_flat(self) -> FlatIndexingOptions {
        let IndexingOptions::Flat(x) = self else {
            unreachable!()
        };
        x
    }
    pub fn unwrap_ivf(self) -> IvfIndexingOptions {
        let IndexingOptions::Ivf(x) = self else {
            unreachable!()
        };
        x
    }
    pub fn unwrap_hnsw(self) -> HnswIndexingOptions {
        let IndexingOptions::Hnsw(x) = self else {
            unreachable!()
        };
        x
    }
    pub fn unwrap_acron(self) -> AcronIndexingOptions {
        let IndexingOptions::Acron(x) = self else {
            unreachable!()
        };
        x
    }
}

impl Default for IndexingOptions {
    fn default() -> Self {
        Self::Hnsw(Default::default())
    }
}

impl Validate for IndexingOptions {
    fn validate(&self) -> Result<(), validator::ValidationErrors> {
        match self {
            Self::Flat(x) => x.validate(),
            Self::Ivf(x) => x.validate(),
            Self::Hnsw(x) => x.validate(),
            Self::Acron(x) => x.validate(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct FlatIndexingOptions {
    #[serde(default)]
    #[validate]
    pub quantization: QuantizationOptions,
}

impl Default for FlatIndexingOptions {
    fn default() -> Self {
        Self {
            quantization: QuantizationOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct IvfIndexingOptions {
    #[serde(default = "IvfIndexingOptions::default_least_iterations")]
    #[validate(range(min = 1, max = 1_000_000))]
    pub least_iterations: u32,
    #[serde(default = "IvfIndexingOptions::default_iterations")]
    #[validate(range(min = 1, max = 1_000_000))]
    pub iterations: u32,
    #[serde(default = "IvfIndexingOptions::default_nlist")]
    #[validate(range(min = 1, max = 1_000_000))]
    pub nlist: u32,
    #[serde(default = "IvfIndexingOptions::default_nsample")]
    #[validate(range(min = 1, max = 1_000_000))]
    pub nsample: u32,
    #[serde(default)]
    #[validate]
    pub quantization: QuantizationOptions,
}

impl IvfIndexingOptions {
    fn default_least_iterations() -> u32 {
        16
    }
    fn default_iterations() -> u32 {
        500
    }
    fn default_nlist() -> u32 {
        1000
    }
    fn default_nsample() -> u32 {
        65536
    }
}

impl Default for IvfIndexingOptions {
    fn default() -> Self {
        Self {
            least_iterations: Self::default_least_iterations(),
            iterations: Self::default_iterations(),
            nlist: Self::default_nlist(),
            nsample: Self::default_nsample(),
            quantization: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct HnswIndexingOptions {
    #[serde(default = "HnswIndexingOptions::default_m")]
    #[validate(range(min = 4, max = 128))]
    pub m: u32,
    #[serde(default = "HnswIndexingOptions::default_ef_construction")]
    #[validate(range(min = 10, max = 2000))]
    pub ef_construction: usize,
    #[serde(default)]
    #[validate]
    pub quantization: QuantizationOptions,
}

impl HnswIndexingOptions {
    fn default_m() -> u32 {
        12
    }
    fn default_ef_construction() -> usize {
        300
    }
}

impl Default for HnswIndexingOptions {
    fn default() -> Self {
        Self {
            m: Self::default_m(),
            ef_construction: Self::default_ef_construction(),
            quantization: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[validate(schema(function = "AcronIndexingOptions::validate_0"))]
#[serde(deny_unknown_fields)]
pub struct AcronIndexingOptions {
    #[serde(default = "AcronIndexingOptions::default_m")]
    #[validate(range(min = 4, max = 128))]
    pub m: u32,
    #[serde(default = "AcronIndexingOptions::default_m_beta")]
    #[validate(range(min = 4, max = 2000))]
    pub m_beta: u32,
    #[serde(default = "AcronIndexingOptions::default_gamma")]
    #[validate(range(min = 1, max = 1000))]
    pub gamma: u32,
    #[serde(default)]
    #[validate]
    pub quantization: QuantizationOptions,
}

impl AcronIndexingOptions {
    fn default_m() -> u32 {
        12
    }
    fn default_m_beta() -> u32 {
        24
    }
    fn default_gamma() -> u32 {
        10
    }
    fn validate_0(&self) -> Result<(), ValidationError> {
        if self.m_beta > self.m * self.gamma {
            return Err(ValidationError::new(
                "`m_beta` must be less than or equal to `m * gamma`",
            ));
        }
        Ok(())
    }
}

impl Default for AcronIndexingOptions {
    fn default() -> Self {
        Self {
            m: Self::default_m(),
            m_beta: Self::default_m_beta(),
            gamma: Self::default_gamma(),
            quantization: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationOptions {
    Trivial(TrivialQuantizationOptions),
    Scalar(ScalarQuantizationOptions),
    Product(ProductQuantizationOptions),
}

impl Validate for QuantizationOptions {
    fn validate(&self) -> Result<(), validator::ValidationErrors> {
        match self {
            Self::Trivial(x) => x.validate(),
            Self::Scalar(x) => x.validate(),
            Self::Product(x) => x.validate(),
        }
    }
}

impl Default for QuantizationOptions {
    fn default() -> Self {
        Self::Trivial(Default::default())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct TrivialQuantizationOptions {}

impl Default for TrivialQuantizationOptions {
    fn default() -> Self {
        Self {}
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct ScalarQuantizationOptions {}

impl Default for ScalarQuantizationOptions {
    fn default() -> Self {
        Self {}
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct ProductQuantizationOptions {
    #[serde(default = "ProductQuantizationOptions::default_sample")]
    #[validate(range(min = 1, max = 1_000_000))]
    pub sample: u32,
    #[serde(default)]
    pub ratio: ProductQuantizationOptionsRatio,
}

impl ProductQuantizationOptions {
    fn default_sample() -> u32 {
        65535
    }
}

impl Default for ProductQuantizationOptions {
    fn default() -> Self {
        Self {
            sample: Self::default_sample(),
            ratio: Default::default(),
        }
    }
}

#[repr(u16)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum ProductQuantizationOptionsRatio {
    X4 = 1,
    X8 = 2,
    X16 = 4,
    X32 = 8,
    X64 = 16,
}

impl Default for ProductQuantizationOptionsRatio {
    fn default() -> Self {
        Self::X4
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MultiColumnOptions {
    pub data_type: MultiColumnDataType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MultiColumnDataType {
    None,
    I32,
    I64,
    F32,
    F64,
}

impl Default for MultiColumnOptions {
    fn default() -> Self {
        Self {
            data_type: MultiColumnDataType::None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SearchOptions {
    pub prefilter_enable: bool,
    #[validate(range(min = 1, max = 65535))]
    pub hnsw_ef_search: usize,
    #[validate(range(min = 1, max = 1_000_000))]
    pub ivf_nprobe: u32,
    pub acron_filter: Vec<(Strategy, MultiColumnData)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexStat {
    pub indexing: bool,
    pub segments: Vec<SegmentStat>,
    pub options: IndexOptions,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SegmentStat {
    pub id: Uuid,
    #[serde(rename = "type")]
    pub typ: String,
    pub length: usize,
    pub size: u64,
}
