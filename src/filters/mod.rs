use std::cmp::{max, min};

use image::{DynamicImage, ImageBuffer, Luma, Rgb};

pub fn sobel(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let output_sobel_x = sobel_x(&img);
    let output_sobel_y = sobel_y(&img);
    sobel_add(output_sobel_x, output_sobel_y)
}

pub fn sobel_x(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (w, h) = img.dimensions();
    let mut output = ImageBuffer::new(w, h);
    for x in 1..(w - 1) {
        for y in 1..(h - 1) {
            let val: i16 = -(img.get_pixel(x - 1, y - 1).0[0] as i16)
                - (img.get_pixel(x - 1, y).0[0] as i16)
                - (img.get_pixel(x - 1, y).0[0] as i16)
                - (img.get_pixel(x - 1, y + 1).0[0] as i16)
                + (img.get_pixel(x + 1, y - 1).0[0] as i16)
                + (img.get_pixel(x + 1, y).0[0] as i16)
                + (img.get_pixel(x + 1, y).0[0] as i16)
                + (img.get_pixel(x + 1, y + 1).0[0] as i16);
            output.put_pixel(x, y, Luma([val.abs() as u8]));
        }
    }
    output
}

pub fn sobel_y(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (w, h) = img.dimensions();
    let mut output = ImageBuffer::new(w, h);
    for x in 1..(w - 1) {
        for y in 1..(h - 1) {
            let val: i16 = -(img.get_pixel(x - 1, y - 1).0[0] as i16)
                - (img.get_pixel(x, y - 1).0[0] as i16)
                - (img.get_pixel(x, y - 1).0[0] as i16)
                - (img.get_pixel(x + 1, y - 1).0[0] as i16)
                + (img.get_pixel(x - 1, y + 1).0[0] as i16)
                + (img.get_pixel(x, y + 1).0[0] as i16)
                + (img.get_pixel(x, y + 1).0[0] as i16)
                + (img.get_pixel(x + 1, y + 1).0[0] as i16);
            output.put_pixel(x, y, Luma([val.abs() as u8]));
        }
    }
    output
}

pub fn conv_3x3(
    img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    mask: [[i16; 3]; 3],
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (w, h) = img.dimensions();
    let mut output = ImageBuffer::new(w, h);

    for x in 1..(w - 1) {
        for y in 1..(h - 1) {
            let p1 = img.get_pixel(x - 1, y - 1).0[0] as i16 * mask[0][0];
            let p2 = img.get_pixel(x - 1, y).0[0] as i16 * mask[0][1];
            let p3 = img.get_pixel(x - 1, y + 1).0[0] as i16 * mask[0][2];
            let p4 = img.get_pixel(x, y - 1).0[0] as i16 * mask[1][0];
            let p5 = img.get_pixel(x, y).0[0] as i16 * mask[1][1];
            let p6 = img.get_pixel(x, y + 1).0[0] as i16 * mask[1][2];
            let p7 = img.get_pixel(x + 1, y - 1).0[0] as i16 * mask[2][0];
            let p8 = img.get_pixel(x + 1, y).0[0] as i16 * mask[2][1];
            let p9 = img.get_pixel(x + 1, y + 1).0[0] as i16 * mask[2][2];
            let val: u8 = (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9).abs() as u8;
            output.put_pixel(x, y, Luma([val]));
        }
    }
    output
}

fn sobel_add(
    img1: ImageBuffer<Luma<u8>, Vec<u8>>,
    img2: ImageBuffer<Luma<u8>, Vec<u8>>,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    assert!(img1.dimensions() == img2.dimensions());
    let (w, h) = (img1.width(), img1.height());
    let mut output = ImageBuffer::new(w, h);
    for x in 0..w {
        for y in 0..h {
            let p1 = img1.get_pixel(x, y).0[0] as f64;
            let p2 = img2.get_pixel(x, y).0[0] as f64;

            let val = (p1 * p1 + p2 * p2).sqrt();

            output.put_pixel(x, y, Luma([val as u8]));
        }
    }

    return output;
}

pub fn contrast_enhancement(input: &DynamicImage, factor: f64) -> DynamicImage {
    match input {
        DynamicImage::ImageRgb8(rgb_img) => {
            let mapping = contrast_enhancement_mapping(factor);
            let output = apply_mapping_rgb(rgb_img, mapping);
            DynamicImage::ImageRgb8(output)
        }
        _ => panic!("image type not implemented"),
    }
}

pub fn histogram_specification(input: &DynamicImage, reference: &DynamicImage) -> DynamicImage {
    let reference_as_luma8 = reference.grayscale().to_luma8();
    let hist_ref = calculate_histogram(&reference_as_luma8);

    match input {
        DynamicImage::ImageLuma8(input) => {
            let hist_input = calculate_histogram(&input);
            let mapping = histogram_to_histogram_mapping(hist_input, hist_ref);
            DynamicImage::ImageLuma8(apply_mapping(&input, mapping))
        }
        DynamicImage::ImageRgb8(input) => {
            let mut hsv_image = HsvImage::from_rgb(&input);
            let hist_input = calculate_histogram(&hsv_image.buffer_value);
            let mapping = histogram_to_histogram_mapping(hist_input, hist_ref);
            hsv_image.buffer_value = apply_mapping(&hsv_image.buffer_value, mapping);
            let rgb_image = hsv_image.to_rgb();
            DynamicImage::ImageRgb8(rgb_image)
        }
        _ => panic!("image type not implemented"),
    }
}

pub fn histogram_normalization(input: DynamicImage) -> DynamicImage {
    match input {
        DynamicImage::ImageLuma8(input) => {
            let histogram = calculate_histogram(&input);
            let mapping = histogram_to_equal_mapping(histogram);
            DynamicImage::ImageLuma8(apply_mapping(&input, mapping))
        }
        DynamicImage::ImageRgb8(input) => {
            let mut hsv_image = HsvImage::from_rgb(&input);
            let histogram = calculate_histogram(&hsv_image.buffer_value);
            let mapping = histogram_to_equal_mapping(histogram);
            hsv_image.buffer_value = apply_mapping(&hsv_image.buffer_value, mapping);
            let rgb_image = hsv_image.to_rgb();
            DynamicImage::ImageRgb8(rgb_image)
        }
        _ => panic!("image type not implemented"),
    }
}

pub fn apply_mapping(
    input: &ImageBuffer<Luma<u8>, Vec<u8>>,
    mapping: [u8; 256],
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (w, h) = input.dimensions();
    let mut output: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(w, h);
    for x in 0..w {
        for y in 0..h {
            let val: u8 = mapping[input.get_pixel(x, y).0[0] as usize];
            output.put_pixel(x, y, Luma([val]));
        }
    }
    output
}

pub fn apply_mapping_rgb(
    input: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    mapping: [u8; 256],
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (w, h) = input.dimensions();
    let mut output: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w, h);
    for x in 0..w {
        for y in 0..h {
            let rgb = input.get_pixel(x, y).0;
            let rgb_mapped = [
                mapping[rgb[0] as usize],
                mapping[rgb[1] as usize],
                mapping[rgb[2] as usize],
            ];
            output.put_pixel(x, y, Rgb(rgb_mapped));
        }
    }
    output
}

fn contrast_enhancement_mapping(factor: f64) -> [u8; 256] {
    let mut mapping: [u8; 256] = [0; 256];
    for i in 0..256 {
        mapping[i] = ((i as f64 - 127.5) * factor).clamp(0.0, 255.0) as u8
    }
    mapping
}

fn histogram_to_equal_mapping(histogram: [f64; 256]) -> [u8; 256] {
    let cumulative = cumulative_histogram(histogram);
    let mut mapping: [u8; 256] = [0; 256];
    for i in 0..256 {
        mapping[i] = (cumulative[i] * 255.0) as u8;
    }
    mapping
}

fn cumulative_histogram(histogram: [f64; 256]) -> [f64; 256] {
    let mut cumulative: [f64; 256] = [0.0; 256];
    cumulative[0] = histogram[0];
    for i in 1..256 {
        cumulative[i] = cumulative[i - 1] + histogram[i]
    }
    cumulative
}

fn histogram_to_histogram_mapping(origin: [f64; 256], destination: [f64; 256]) -> [u8; 256] {
    let mut mapping: [u8; 256] = [0; 256];
    let cum1 = cumulative_histogram(origin);
    let cum2 = cumulative_histogram(destination);
    // pixel => q1 in cum1 => at what pixel val do we go above this quantile in cum2? => return this value
    let p_map = |p: u8| {
        let q1 = cum1[p as usize] as f64;
        let val_at_q2 = cum2
            .iter()
            .enumerate()
            .find(|(_, cum)| **cum >= q1)
            .unwrap_or((255, &1.0))
            .0;
        return val_at_q2 as u8;
    };
    mapping
        .iter_mut()
        .enumerate()
        .for_each(|(i, e)| *e = p_map(i as u8));
    mapping
}

fn calculate_histogram(input: &ImageBuffer<Luma<u8>, Vec<u8>>) -> [f64; 256] {
    let mut hist: [f64; 256] = [0.0; 256];
    let mut counter: usize = 0;
    for p in input.pixels() {
        hist[p.0[0] as usize] += 1.0;
        counter += 1
    }
    let counter = counter as f64;
    for i in 0..hist.len() {
        hist[i] /= counter;
    }
    hist
}

pub struct HsvImage {
    pub buffer_hue: ImageBuffer<Luma<u16>, Vec<u16>>,
    pub buffer_sat: ImageBuffer<Luma<f64>, Vec<f64>>,
    pub buffer_value: ImageBuffer<Luma<u8>, Vec<u8>>,
}
impl HsvImage {
    pub fn to_rgb(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (w, h) = self.buffer_value.dimensions();
        let mut buffer_rgb: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(w, h);
        for x in 0..w {
            for y in 0..h {
                let (h, s, v) = (
                    self.buffer_hue.get_pixel(x, y).0[0],
                    self.buffer_sat.get_pixel(x, y).0[0],
                    self.buffer_value.get_pixel(x, y).0[0],
                );
                let rgb = hsv_to_rgb(Hsv(h, s, v));
                buffer_rgb.put_pixel(x, y, Rgb(rgb));
            }
        }
        buffer_rgb
    }

    pub fn from_rgb(rgb_image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Self {
        let (w, h) = rgb_image.dimensions();
        let mut buffer_hue: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(w, h);
        let mut buffer_sat: ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(w, h);
        let mut buffer_value: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(w, h);

        for (x, y, pix) in rgb_image.enumerate_pixels() {
            let Hsv(h, s, v) = rgb_to_hsv(pix.0);
            buffer_hue.put_pixel(x, y, Luma([h]));
            buffer_sat.put_pixel(x, y, Luma([s]));
            buffer_value.put_pixel(x, y, Luma([v]));
        }
        HsvImage {
            buffer_hue,
            buffer_sat,
            buffer_value,
        }
    }
}

#[derive(Debug)]
pub struct Hsv(u16, f64, u8);
pub fn rgb_to_hsv(rgb: [u8; 3]) -> Hsv {
    // let oneby255: f64 = 1 / 255;
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];
    let rf: f64 = r as f64 / 256.0;
    let gf: f64 = g as f64 / 256.0;
    let bf: f64 = b as f64 / 256.0;
    let cmin = min(min(r, g), b);
    let cmax = max(max(r, g), b);
    let del: f64 = (cmax - cmin) as f64 / 256.0;
    let hue: u16 = match cmax {
        _ if cmax == cmin => 0.0,
        _ if cmax == r => 60.0 * (((gf - bf) / del) % 6.0),
        _ if cmax == g => 60.0 * ((bf - rf) / del + 2.0),
        _ if cmax == b => 60.0 * ((rf - gf) / del + 4.0),
        _ => panic!("should not happen"),
    } as u16;

    let sat: f64 = match cmax {
        0 => 0.0,
        _ => del * 256.0 / (cmax as f64),
    };
    let value = cmax;
    Hsv(hue, sat, value)
}

pub fn hsv_to_rgb(hsv: Hsv) -> [u8; 3] {
    let Hsv(h, s, v) = hsv;
    let c = v as f64 * s;

    let x = (c * (1.0 - ((h as f64 / 60.0) % 2.0 - 1.0).abs())) as u8;
    let c = c as u8;
    let m = v - c;
    let rgb: [u8; 3] = match h % 360 {
        0..=59 => [c, x, 0],
        60..=119 => [x, c, 0],
        120..=179 => [0, c, x],
        180..=239 => [0, x, c],
        240..=299 => [x, 0, c],
        300..=360 => [c, 0, x],
        _ => panic!("should not happen"),
    };
    [rgb[0] + m, rgb[1] + m, rgb[2] + m]
}
