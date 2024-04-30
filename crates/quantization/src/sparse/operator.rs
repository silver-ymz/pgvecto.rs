use base::operator::*;
use base::scalar::*;
use num_traits::Zero;

pub trait OperatorSparseQuantization: Operator {
    fn sparse_quantization_distance(
        bitmap_lens: u16,
        lhs_index: &[usize],
        lhs_value: &[Scalar<Self>],
        rhs_index: &[usize],
        rhs_value: &[Scalar<Self>],
    ) -> F32;
    fn make_quantized_sparse_vector(
        bitmap_lens: u16,
        vector: Borrowed<'_, Self>,
    ) -> (Vec<usize>, Vec<Scalar<Self>>);
}

impl OperatorSparseQuantization for SVecf32Dot {
    fn sparse_quantization_distance(
        bitmap_lens: u16,
        lhs_index: &[usize],
        lhs_value: &[Scalar<Self>],
        rhs_index: &[usize],
        rhs_value: &[Scalar<Self>],
    ) -> F32 {
        let mut lhs_i = 0;
        let mut rhs_i = 0;
        let mut result = F32::zero();
        for i in 0..bitmap_lens {
            let lhs_bit =
                lhs_index[(i / usize::BITS as u16) as usize] & (1 << (i % usize::BITS as u16)) != 0;
            let rhs_bit =
                rhs_index[(i / usize::BITS as u16) as usize] & (1 << (i % usize::BITS as u16)) != 0;
            if lhs_bit && rhs_bit {
                result += lhs_value[lhs_i] * rhs_value[rhs_i];
            }
            lhs_i += lhs_bit as usize;
            rhs_i += rhs_bit as usize;
        }
        result * F32(-1.)
    }

    fn make_quantized_sparse_vector(
        bitmap_lens: u16,
        vector: Borrowed<'_, Self>,
    ) -> (Vec<usize>, Vec<Scalar<Self>>) {
        let bits = usize::BITS as usize;
        let mut index = vec![0usize; (bitmap_lens as usize + bits - 1) / bits];
        let mut value = vec![<Scalar<Self>>::zero(); bitmap_lens as usize];
        for (idx, val) in vector.indexes().iter().zip(vector.values().iter()) {
            let idx = (idx % bitmap_lens as u32) as usize;
            index[idx / bits] |= 1 << (idx % bits);
            value[idx] = *val;
        }
        let value = value
            .into_iter()
            .enumerate()
            .filter(|(i, _)| index[i / bits] & (1 << (i % bits)) != 0)
            .map(|(_, v)| v)
            .collect();
        (index, value)
    }
}

macro_rules! unimplemented_operator_sparse_quantization {
    ($t:ty) => {
        impl OperatorSparseQuantization for $t {
            fn sparse_quantization_distance(
                _bitmap_lens: u16,
                _lhs_index: &[usize],
                _lhs_value: &[Scalar<Self>],
                _rhs_index: &[usize],
                _rhs_value: &[Scalar<Self>],
            ) -> F32 {
                unimplemented!()
            }
            fn make_quantized_sparse_vector(
                _bitmap_lens: u16,
                _vector: Borrowed<'_, Self>,
            ) -> (Vec<usize>, Vec<Scalar<Self>>) {
                unimplemented!()
            }
        }
    };

    ($($t:ty),*) => {
        $(unimplemented_operator_sparse_quantization!($t);)*
    };
}

unimplemented_operator_sparse_quantization!(
    BVecf32Cos,
    BVecf32Dot,
    BVecf32Jaccard,
    BVecf32L2,
    Vecf32Cos,
    Vecf32Dot,
    Vecf32L2,
    Vecf16Cos,
    Vecf16Dot,
    Vecf16L2,
    SVecf32Cos,
    SVecf32L2,
    Veci8Cos,
    Veci8Dot,
    Veci8L2
);
