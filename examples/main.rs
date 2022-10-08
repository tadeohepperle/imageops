use std::{env, fs, time::Instant};

use image::{io::Reader as ImageReader, ImageBuffer, Luma};
use imageops::filters;

use std::collections::HashMap;
type ImageOperation =
    fn(&ImageBuffer<Luma<u8>, Vec<u8>>) -> (ImageBuffer<Luma<u8>, Vec<u8>>, Instant);

// cargo run --example main sobel
// cargo run --example main hist_norm_grey
pub fn main() {
    // load refrence

    // define operations
    let mut operations: HashMap<String, ImageOperation> = HashMap::new();
    operations.insert("sobel".to_owned(), |img| {
        let timer = Instant::now();
        (filters::sobel(img), timer)
    });
    operations.insert("sobel_x".to_owned(), |img| {
        let timer = Instant::now();
        (filters::sobel_x(img), timer)
    });
    operations.insert("sobel_y".to_owned(), |img| {
        let timer = Instant::now();
        (filters::sobel_y(img), timer)
    });
    operations.insert("hist_norm_grey".to_owned(), |img| {
        let timer = Instant::now();
        (filters::histogram_normalization(img), timer)
    });
    operations.insert("hist_specific_grey".to_owned(), |img| {
        let reference_img = ImageReader::open("examples/images/reference.jpg")
            .unwrap()
            .decode()
            .unwrap()
            .grayscale()
            .to_luma8();

        let timer = Instant::now();
        (
            filters::histogram_specification(&img, &reference_img),
            timer,
        )
    });

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
                        .map(|e| e.unwrap().file_name().into_string().unwrap())
                        .map(|e| e)
                        .collect::<Vec<String>>();

                    for file_name in file_names {
                        let image_path = format!("examples/images/input/{}", file_name);

                        let img = ImageReader::open(&image_path)
                            .unwrap()
                            .decode()
                            .unwrap()
                            .grayscale()
                            .to_luma8();
                        println!(
                            "Applying {} on {} (width: {}, height: {})",
                            op_string,
                            image_path,
                            img.width(),
                            img.height()
                        );

                        let (output, timer) = image_op(&img);

                        println!("{:.3?}", timer.elapsed());

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
