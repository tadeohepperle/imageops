use image::{ImageBuffer, Luma};
use rayon::prelude::*;

/// in db
pub fn psnr(
    original: &ImageBuffer<Luma<u8>, Vec<u8>>,
    noisy: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> f64 {
    assert_eq!(original.dimensions(), noisy.dimensions());

    let buf_1_par = original.as_raw().par_iter();
    let buf_2_par = noisy.as_raw().par_iter();
    let s: u64 = buf_1_par
        .zip(buf_2_par)
        .map(|(a, b)| ((*a as i32 - *b as i32) * (*a as i32 - *b as i32)) as u64)
        .sum();
    let (w, h) = original.dimensions();
    let mse = s as f64 / (w as f64 * h as f64);
    20.0 * (255.0 as f64).log10() - 10.0 * mse.log10()
}
