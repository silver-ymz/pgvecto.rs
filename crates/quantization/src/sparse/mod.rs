pub mod operator;
use base::index::{IndexOptions, QuantizationOptions, SparseQuantizationOptions};
use base::operator::*;
use base::scalar::F32;
use base::search::Collection;
use common::dir_ops::sync_dir;
use common::mmap_array::MmapArray;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use self::operator::OperatorSparseQuantization;

pub struct SparseQuantization<O: OperatorSparseQuantization, C: Collection<O>> {
    bitmap_lens: u16,
    indexes: MmapArray<usize>,
    values: MmapArray<Scalar<O>>,
    offsets: MmapArray<usize>,
    _maker: PhantomData<fn(C) -> C>,
}

unsafe impl<O: OperatorSparseQuantization, C: Collection<O>> Send for SparseQuantization<O, C> {}
unsafe impl<O: OperatorSparseQuantization, C: Collection<O>> Sync for SparseQuantization<O, C> {}

impl<O: OperatorSparseQuantization, C: Collection<O>> SparseQuantization<O, C> {
    pub fn create(
        path: &Path,
        _: IndexOptions,
        quantization_options: QuantizationOptions,
        collection: &Arc<C>,
        permutation: Vec<u32>, // permutation is the mapping from placements to original ids
    ) -> Self {
        std::fs::create_dir(path).unwrap();
        let QuantizationOptions::Sparse(SparseQuantizationOptions { bitmap_lens }) =
            quantization_options
        else {
            unreachable!()
        };
        let mut indexes = vec![];
        let mut values = vec![];
        let mut offsets = vec![0usize];
        let n = collection.len();
        for i in 0..n {
            let vector = collection.vector(permutation[i as usize]);
            let (index, value) = O::make_quantized_sparse_vector(bitmap_lens, vector);
            indexes.extend_from_slice(&index);
            values.extend_from_slice(&value);
            offsets.push(offsets.last().unwrap() + value.len());
        }
        let indexes = MmapArray::create(&path.join("indexes"), indexes.into_iter());
        let values = MmapArray::create(&path.join("values"), values.into_iter());
        let offsets = MmapArray::create(&path.join("offsets"), offsets.into_iter());
        sync_dir(path);
        Self {
            bitmap_lens,
            indexes,
            values,
            offsets,
            _maker: PhantomData,
        }
    }

    pub fn open(
        path: &Path,
        _: IndexOptions,
        quantization_options: QuantizationOptions,
        _: &Arc<C>,
    ) -> Self {
        let QuantizationOptions::Sparse(SparseQuantizationOptions { bitmap_lens }) =
            quantization_options
        else {
            unreachable!()
        };
        let indexes = MmapArray::open(&path.join("indexes"));
        let values = MmapArray::open(&path.join("values"));
        let offsets = MmapArray::open(&path.join("offsets"));
        Self {
            bitmap_lens,
            indexes,
            values,
            offsets,
            _maker: PhantomData,
        }
    }

    pub fn distance(&self, lhs: Borrowed<'_, O>, rhs: u32) -> F32 {
        let (ref lhs_index, ref lhs_value) = O::make_quantized_sparse_vector(self.bitmap_lens, lhs);
        let cnt = ((self.bitmap_lens + 63) / usize::BITS as u16) as usize;
        let rhs_index = &self.indexes[rhs as usize * cnt..rhs as usize * cnt + cnt];
        let rhs_value = &self.values[self.offsets[rhs as usize]..self.offsets[rhs as usize + 1]];
        O::sparse_quantization_distance(
            self.bitmap_lens,
            lhs_index,
            lhs_value,
            rhs_index,
            rhs_value,
        )
    }

    pub fn distance2(&self, lhs: u32, rhs: u32) -> F32 {
        let cnt = ((self.bitmap_lens + 63) / usize::BITS as u16) as usize;
        let lhs_index = &self.indexes[lhs as usize * cnt..lhs as usize * cnt + cnt];
        let lhs_value = &self.values[self.offsets[lhs as usize]..self.offsets[lhs as usize + 1]];
        let rhs_index = &self.indexes[rhs as usize * cnt..rhs as usize * cnt + cnt];
        let rhs_value = &self.values[self.offsets[rhs as usize]..self.offsets[rhs as usize + 1]];
        O::sparse_quantization_distance(
            self.bitmap_lens,
            lhs_index,
            lhs_value,
            rhs_index,
            rhs_value,
        )
    }
}
