import os, sys
import argparse
import cv2
from tqdm import tqdm


def resize_image(image, size):
    """Resize an image to the given size."""
    return cv2.resize(image, size, interpolation = cv2.INTER_AREA)


def resize_images(input_dir, output_dir, size):
    """Resize the images in 'input_dir' and save into 'output_dir'."""
    for idir in os.scandir(input_dir):
        if not idir.is_dir():
            continue
        if not os.path.exists(output_dir+'/'+idir.name):
            os.makedirs(output_dir+'/'+idir.name)    
        images = os.listdir(idir.path)
        n_images = len(images)
        for iimage, image in tqdm(enumerate(images)):
            f = os.path.join(idir.path, image)
            img = cv2.imread(f)
            img = resize_image(img, size)
            cv2.imwrite(os.path.join(output_dir+'/'+idir.name, image),img)
            if (iimage+1) % 10000 == 0:
                sys.stdout.write("[{}/{}] resized and saved.".format(iimage+1, n_images))
            
            
def main(args):

    input_dir = args.input_dir
    output_dir = args.output_dir
    image_size = (args.image_size, args.image_size)
    resize_images(input_dir, output_dir, image_size)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='datasets/Images',
                        help='directory for input images (unresized images)')

    parser.add_argument('--output_dir', type=str, default='datasets/Resized_Images',
                        help='directory for output images (resized images)')

    parser.add_argument('--image_size', type=int, default=224,
                        help='size of images after resizing')

    args = parser.parse_args()

    main(args)
