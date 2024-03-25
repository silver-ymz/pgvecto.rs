use super::{VectorBorrowed, VectorKind, VectorOwned};
use crate::scalar::{ScalarLike, F16, F32};
use num_traits::{Float, Zero};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Vecf16Owned(Vec<F16>);

impl Vecf16Owned {
    #[inline(always)]
    pub fn new(slice: Vec<F16>) -> Self {
        Self::new_checked(slice).unwrap()
    }
    #[inline(always)]
    pub fn new_checked(slice: Vec<F16>) -> Option<Self> {
        if !(1 <= slice.len() && slice.len() <= 65535) {
            return None;
        }
        Some(unsafe { Self::new_unchecked(slice) })
    }
    /// # Safety
    ///
    /// * `slice.len()` must not be zero.
    #[inline(always)]
    pub unsafe fn new_unchecked(slice: Vec<F16>) -> Self {
        Self(slice)
    }
    #[inline(always)]
    pub fn slice(&self) -> &[F16] {
        self.0.as_slice()
    }
    #[inline(always)]
    pub fn slice_mut(&mut self) -> &mut [F16] {
        self.0.as_mut_slice()
    }
}

impl VectorOwned for Vecf16Owned {
    type Scalar = F16;
    type Borrowed<'a> = Vecf16Borrowed<'a>;

    const VECTOR_KIND: VectorKind = VectorKind::Vecf16;

    fn dims(&self) -> u32 {
        self.0.len() as u32
    }

    fn for_borrow(&self) -> Vecf16Borrowed<'_> {
        Vecf16Borrowed(self.0.as_slice())
    }

    fn to_vec(&self) -> Vec<F16> {
        self.0.clone()
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Vecf16Borrowed<'a>(&'a [F16]);

impl<'a> Vecf16Borrowed<'a> {
    #[inline(always)]
    pub fn new(slice: &'a [F16]) -> Self {
        Self::new_checked(slice).unwrap()
    }
    #[inline(always)]
    pub fn new_checked(slice: &'a [F16]) -> Option<Self> {
        if !(1 <= slice.len() && slice.len() <= 65535) {
            return None;
        }
        Some(unsafe { Self::new_unchecked(slice) })
    }
    /// # Safety
    ///
    /// * `slice.len()` must not be zero.
    #[inline(always)]
    pub unsafe fn new_unchecked(slice: &'a [F16]) -> Self {
        Self(slice)
    }
    #[inline(always)]
    pub fn slice(&self) -> &[F16] {
        self.0
    }
}

impl<'a> VectorBorrowed for Vecf16Borrowed<'a> {
    type Scalar = F16;
    type Owned = Vecf16Owned;

    fn dims(&self) -> u32 {
        self.0.len() as u32
    }

    fn for_own(&self) -> Vecf16Owned {
        Vecf16Owned(self.0.to_vec())
    }

    fn to_vec(&self) -> Vec<F16> {
        self.0.to_vec()
    }
}

#[cfg(any(target_arch = "x86_64", doc))]
#[doc(cfg(target_arch = "x86_64"))]
unsafe fn cosine_v4_avx512fp16(lhs: &[F16], rhs: &[F16]) -> F32 {
    assert!(lhs.len() == rhs.len());
    let n = lhs.len();
    unsafe { c::v_f16_cosine_avx512fp16(lhs.as_ptr().cast(), rhs.as_ptr().cast(), n).into() }
}

#[cfg(target_arch = "x86_64")]
#[doc(cfg(target_arch = "x86_64"))]
unsafe fn cosine_v4(lhs: &[F16], rhs: &[F16]) -> F32 {
    assert!(lhs.len() == rhs.len());
    let n = lhs.len();
    unsafe { c::v_f16_cosine_v4(lhs.as_ptr().cast(), rhs.as_ptr().cast(), n).into() }
}

#[cfg(target_arch = "x86_64")]
#[doc(cfg(target_arch = "x86_64"))]
unsafe fn cosine_v3(lhs: &[F16], rhs: &[F16]) -> F32 {
    assert!(lhs.len() == rhs.len());
    let n = lhs.len();
    unsafe { c::v_f16_cosine_v3(lhs.as_ptr().cast(), rhs.as_ptr().cast(), n).into() }
}

#[detect::multiversion(v4_avx512fp16 = import, v4 = import, v3 = import, v2, neon, fallback = export)]
pub fn cosine(lhs: &[F16], rhs: &[F16]) -> F32 {
    assert!(lhs.len() == rhs.len());
    let n = lhs.len();
    let mut xy = F32::zero();
    let mut x2 = F32::zero();
    let mut y2 = F32::zero();
    for i in 0..n {
        xy += lhs[i].to_f() * rhs[i].to_f();
        x2 += lhs[i].to_f() * lhs[i].to_f();
        y2 += rhs[i].to_f() * rhs[i].to_f();
    }
    xy / (x2 * y2).sqrt()
}

pub fn dot(lhs: &[F16], rhs: &[F16]) -> F32 {
    #[detect::multiversion(v4, v3, v2, neon, fallback)]
    fn dot(lhs: &[F16], rhs: &[F16]) -> F32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut xy = F32::zero();
        for i in 0..n {
            xy += lhs[i].to_f() * rhs[i].to_f();
        }
        xy
    }
    #[cfg(target_arch = "x86_64")]
    if detect::v4_avx512fp16::detect() {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        unsafe {
            return c::v_f16_dot_avx512fp16(lhs.as_ptr().cast(), rhs.as_ptr().cast(), n).into();
        }
    }
    #[cfg(target_arch = "x86_64")]
    if detect::v4::detect() {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        unsafe {
            return c::v_f16_dot_v4(lhs.as_ptr().cast(), rhs.as_ptr().cast(), n).into();
        }
    }
    #[cfg(target_arch = "x86_64")]
    if detect::v3::detect() {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        unsafe {
            return c::v_f16_dot_v3(lhs.as_ptr().cast(), rhs.as_ptr().cast(), n).into();
        }
    }
    dot(lhs, rhs)
}

pub fn sl2(lhs: &[F16], rhs: &[F16]) -> F32 {
    #[detect::multiversion(v4, v3, v2, neon, fallback)]
    fn sl2(lhs: &[F16], rhs: &[F16]) -> F32 {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        let mut d2 = F32::zero();
        for i in 0..n {
            let d = lhs[i].to_f() - rhs[i].to_f();
            d2 += d * d;
        }
        d2
    }
    #[cfg(target_arch = "x86_64")]
    if detect::v4_avx512fp16::detect() {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        unsafe {
            return c::v_f16_sl2_avx512fp16(lhs.as_ptr().cast(), rhs.as_ptr().cast(), n).into();
        }
    }
    #[cfg(target_arch = "x86_64")]
    if detect::v4::detect() {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        unsafe {
            return c::v_f16_sl2_v4(lhs.as_ptr().cast(), rhs.as_ptr().cast(), n).into();
        }
    }
    #[cfg(target_arch = "x86_64")]
    if detect::v3::detect() {
        assert!(lhs.len() == rhs.len());
        let n = lhs.len();
        unsafe {
            return c::v_f16_sl2_v3(lhs.as_ptr().cast(), rhs.as_ptr().cast(), n).into();
        }
    }
    sl2(lhs, rhs)
}

#[detect::multiversion(v4 = export, v3 = export, v2 = export, neon = export, fallback = export)]
fn length(vector: &[F16]) -> F16 {
    let n = vector.len();
    let mut dot = F16::zero();
    for i in 0..n {
        dot += vector[i] * vector[i];
    }
    dot.sqrt()
}

#[detect::multiversion(v4 = export, v3 = export, v2 = export, neon = export, fallback = export)]
pub fn l2_normalize(vector: &mut [F16]) {
    let n = vector.len();
    let l = length(vector);
    for i in 0..n {
        vector[i] /= l;
    }
}

#[detect::multiversion(v4 = export, v3 = export, v2 = export, neon = export, fallback = export)]
pub fn xy_x2_y2(lhs: &[F16], rhs: &[F16]) -> (F32, F32, F32) {
    assert!(lhs.len() == rhs.len());
    let n = lhs.len();
    let mut xy = F32::zero();
    let mut x2 = F32::zero();
    let mut y2 = F32::zero();
    for i in 0..n {
        xy += lhs[i].to_f() * rhs[i].to_f();
        x2 += lhs[i].to_f() * lhs[i].to_f();
        y2 += rhs[i].to_f() * rhs[i].to_f();
    }
    (xy, x2, y2)
}

#[detect::multiversion(v4 = export, v3 = export, v2 = export, neon = export, fallback = export)]
pub fn xy_x2_y2_delta(lhs: &[F16], rhs: &[F16], del: &[F16]) -> (F32, F32, F32) {
    assert!(lhs.len() == rhs.len());
    let n = lhs.len();
    let mut xy = F32::zero();
    let mut x2 = F32::zero();
    let mut y2 = F32::zero();
    for i in 0..n {
        xy += lhs[i].to_f() * (rhs[i].to_f() + del[i].to_f());
        x2 += lhs[i].to_f() * lhs[i].to_f();
        y2 += (rhs[i].to_f() + del[i].to_f()) * (rhs[i].to_f() + del[i].to_f());
    }
    (xy, x2, y2)
}

#[detect::multiversion(v4 = export, v3 = export, v2 = export, neon = export, fallback = export)]
pub fn dot_delta(lhs: &[F16], rhs: &[F16], del: &[F16]) -> F32 {
    assert!(lhs.len() == rhs.len());
    let n: usize = lhs.len();
    let mut xy = F32::zero();
    for i in 0..n {
        xy += lhs[i].to_f() * (rhs[i].to_f() + del[i].to_f());
    }
    xy
}

#[detect::multiversion(v4 = export, v3 = export, v2 = export, neon = export, fallback = export)]
pub fn distance_squared_l2_delta(lhs: &[F16], rhs: &[F16], del: &[F16]) -> F32 {
    assert!(lhs.len() == rhs.len());
    let n = lhs.len();
    let mut d2 = F32::zero();
    for i in 0..n {
        let d = lhs[i].to_f() - (rhs[i].to_f() + del[i].to_f());
        d2 += d * d;
    }
    d2
}
