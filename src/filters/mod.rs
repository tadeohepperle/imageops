use image::{ImageBuffer, Luma};

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

pub fn histogram_specification(
    input: &ImageBuffer<Luma<u8>, Vec<u8>>,
    reference: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let h1 = calculate_histogram(input);
    let h2 = calculate_histogram(reference);
    let mapping = histogram_to_histogram_mapping(h1, h2);
    apply_mapping(input, mapping)
}

pub fn histogram_normalization(
    input: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let histogram = calculate_histogram(input);
    let mapping = histogram_to_equal_mapping(histogram);
    apply_mapping(input, mapping)
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
