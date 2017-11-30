# -*-coding=utf-8-*-
import argparse
import os
from PIL import Image

def resize_image(image, size):
    # resize an image to the given size
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    print ('total_length: ',num_images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if i % 100 == 0:
            print("The %d of %d"%(i,num_images))

def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images(image_dir, output_dir, image_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str,
                        default="sub_training_set/",
                        help="directory for training picture")
    parser.add_argument("--output_dir", type=str,
                        default="resize_sub_training_set/",
                        help="directory for saving picture")
    parser.add_argument("--image_size", type=int,
                        default=224,
                        help="size for image after processing")
    args = parser.parse_args()
    main(args)