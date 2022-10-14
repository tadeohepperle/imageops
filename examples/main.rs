use std::{
    env, fs,
    time::{Duration, Instant},
};

use image::{io::Reader as ImageReader, DynamicImage, ImageBuffer, Luma};
use imageops::{
    filters::{self, hsv_to_rgb, rgb_to_hsv, HsvImage},
    measures,
};

use std::collections::HashMap;
type ImageOperation = fn(&DynamicImage, file_name: &str) -> (DynamicImage, Duration);

// cargo run --example main sobel
// cargo run --example main hist_norm_grey
// cargo run --example main hist_norm_color
// cargo run --example main hist_spec_color
// cargo run --example main hist_spec_color2
// cargo run --example main to_and_from_hsv
// cargo run --example main contrast_enhancement_05
// cargo run --example main contrast_enhancement_30
// cargo run --example main average_smoothing
// cargo run --example main gaussian_smoothing
// cargo run --example main median_smoothing
// cargo run --example main weighted_median_smoothing

// fn(&DynamicImage, &str) -> (DynamicImage, Duration)

fn apply_smoothing(
    smoothing: fn(&ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>>,
) -> impl Fn(&DynamicImage, &str) -> (DynamicImage, Duration) {
    return move |img: &DynamicImage, file_name: &str| {
        let noisy_image = image::open(format!("examples/images/input/noisy/{}", file_name))
            .unwrap()
            .grayscale()
            .to_luma8();
        let greyscale_image = img.grayscale().to_luma8();
        let pnsr_original_noise = measures::psnr(&greyscale_image, &noisy_image);
        println!("[original -  noise] pcnr: {:2}dB", pnsr_original_noise);
        let timer = Instant::now();
        let output = smoothing(&greyscale_image);

        let duration = timer.elapsed();
        let pnsr_original_smoothed = measures::psnr(&greyscale_image, &output);
        println!("[original - smooth] pcnr: {:2}dB", pnsr_original_smoothed);
        return (DynamicImage::ImageLuma8(output), duration);
    };
}

pub fn main() {
    let rgb: [u8; 3] = [170, 131, 0];
    println!("Rgb: {:?}", rgb);
    let hsv = rgb_to_hsv(rgb);
    println!("Hsv: {:?}", hsv);
    let rgb = hsv_to_rgb(hsv);
    println!("Rgb after: {:?}", rgb);
    // load refrence

    // define operations
    let mut operations: HashMap<String, ImageOperation> = HashMap::new();
    operations.insert("sobel".to_owned(), |img, _file_name| {
        let greyscale_image = img.grayscale().to_luma8();
        let timer = Instant::now();
        let sobel_image = filters::sobel(&greyscale_image);
        (DynamicImage::ImageLuma8(sobel_image), timer.elapsed())
    });
    operations.insert("sobel_x".to_owned(), |img, _file_name| {
        let greyscale_image = img.grayscale().to_luma8();
        let timer = Instant::now();
        let sobel_image = filters::sobel_x(&greyscale_image);
        (DynamicImage::ImageLuma8(sobel_image), timer.elapsed())
    });
    operations.insert("sobel_y".to_owned(), |img, _file_name| {
        let greyscale_image = img.grayscale().to_luma8();
        let timer = Instant::now();
        let sobel_image = filters::sobel_y(&greyscale_image);
        (DynamicImage::ImageLuma8(sobel_image), timer.elapsed())
    });
    operations.insert("average_smoothing".to_owned(), |img, file_name| {
        apply_smoothing(imageops::filters::average_smoothing)(img, file_name)
    });
    operations.insert("gaussian_smoothing".to_owned(), |img, file_name| {
        apply_smoothing(imageops::filters::gaussian_smoothing)(img, file_name)
    });

    operations.insert("median_smoothing".to_owned(), |img, file_name| {
        apply_smoothing(imageops::filters::median_smoothing)(img, file_name)
    });

    operations.insert("weighted_median_smoothing".to_owned(), |img, file_name| {
        apply_smoothing(imageops::filters::weighted_median_smoothing)(img, file_name)
    });

    operations.insert("hist_norm_grey".to_owned(), |img, _file_name| {
        let greyscale_image = DynamicImage::ImageLuma8(img.grayscale().to_luma8());
        let timer = Instant::now();
        let hist_image_grey = filters::histogram_normalization(greyscale_image);
        (hist_image_grey, timer.elapsed())
    });

    operations.insert("hist_norm_color".to_owned(), |img, _file_name| {
        let rgb_image = DynamicImage::ImageRgb8(img.to_rgb8());
        let timer = Instant::now();
        let hist_normalized = filters::histogram_normalization(rgb_image);
        (hist_normalized, timer.elapsed())
    });
    operations.insert("hist_spec_color".to_owned(), |img, _file_name| {
        let reference_img = ImageReader::open("examples/images/reference.jpg")
            .unwrap()
            .decode()
            .unwrap();

        let rgb_img = DynamicImage::ImageRgb8(img.to_rgb8());
        let timer = Instant::now();
        let hist_normalized = filters::histogram_specification(&rgb_img, &reference_img);
        (hist_normalized, timer.elapsed())
    });
    operations.insert("hist_spec_color2".to_owned(), |img, _file_name| {
        let reference_img = ImageReader::open("examples/images/reference2.jpg")
            .unwrap()
            .decode()
            .unwrap();
        let rgb_img = DynamicImage::ImageRgb8(img.to_rgb8());
        let timer = Instant::now();
        let hist_normalized = filters::histogram_specification(&rgb_img, &reference_img);
        (hist_normalized, timer.elapsed())
    });
    operations.insert("contrast_enhancement_05".to_owned(), |img, _file_name| {
        let rgb_img = DynamicImage::ImageRgb8(img.to_rgb8());
        let timer = Instant::now();
        let output = filters::contrast_enhancement(&rgb_img, 0.5);
        (output, timer.elapsed())
    });
    operations.insert("contrast_enhancement_30".to_owned(), |img, _file_name| {
        let rgb_img = DynamicImage::ImageRgb8(img.to_rgb8());
        let timer = Instant::now();
        let output = filters::contrast_enhancement(&rgb_img, 3.0);
        (output, timer.elapsed())
    });
    operations.insert("to_and_from_hsv".to_owned(), |img, _file_name| {
        let rgb_image = img.to_rgb8();
        let timer = Instant::now();
        let hsv: HsvImage = HsvImage::from_rgb(&rgb_image);
        let rgb_image_2 = hsv.to_rgb();
        (DynamicImage::ImageRgb8(rgb_image_2), timer.elapsed())
    });
    // operations.insert("hist_specific_grey".to_owned(), |img, file_name | {
    //     let reference_img = ImageReader::open("examples/images/reference.jpg")
    // .unwrap()
    //         .decode()
    //         .unwrap()
    //         .grayscale()
    //         .to_luma8();

    //     let timer = Instant::now();
    //     (
    //         filters::histogram_specification(&img, &reference_img),
    //         timer,
    //     )
    // });

    // execution
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first() {
        None => {
            println!("no operation given.")
        }
        Some(op_string) => {
            println!("{}", op_string);
            match operations.get(op_string) {
                None => println!("operation {} was not found", op_string),
                Some(image_op) => {
                    println!("Performing {} on example images...", op_string);
                    let file_names = fs::read_dir("./examples/images/input")
                        .unwrap()
                        .map(|e| e.unwrap())
                        .map(|e| e.file_name().into_string().unwrap())
                        .filter(|e| e.contains("."))
                        .map(|e| e)
                        .collect::<Vec<String>>();

                    for file_name in file_names {
                        let image_path = format!("examples/images/input/{}", file_name);
                        let img = ImageReader::open(&image_path).unwrap().decode().unwrap();

                        println!(
                            "Applying {} on {} (width: {}, height: {})",
                            op_string,
                            image_path,
                            img.width(),
                            img.height()
                        );

                        let (output, duration) = image_op(&img, &file_name);

                        println!("{:.3?}", duration);

                        output
                            .save(output_image_path(&file_name, op_string))
                            .unwrap();
                    }
                }
            }
        }
    }
}

fn output_image_path(file_name: &str, op_string: &str) -> String {
    let parts = file_name.split(".").collect::<Vec<&str>>();
    if parts.len() != 2 {
        panic!("path {file_name} is invalid, contains too many dots, 2 expected");
    }
    format!(
        "examples/images/output/{}_{}.{}",
        parts[0], op_string, parts[1]
    )
}
