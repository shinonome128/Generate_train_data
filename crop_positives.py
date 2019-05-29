import configparser
import sys
import numpy as np
from skimage import io, transform
from glob import glob
# import pdb; pdb.set_trace()

"""
Main process
"""
def main():

    # Read config file
    config_file = configparser.ConfigParser()
    config_file.read('./crop_positives.conf', 'UTF-8')

    # Set parm
    input_dir = config_file.get('directory', 'INPUT')
    output_dir = config_file.get('directory', 'OUTPUT')
    window_size = int(config_file.get('file', 'WINDOW'))
    base_image_size = int(config_file.get('file', 'BASE'))

    # Get input images list
    image_list = glob(input_dir + '*')

    for i in image_list:

        # Read image, and check image size, image must be larger than window size (224, 224)
        image = transform.resize(io.imread(i), (base_image_size, base_image_size))

        # Display image
        # io.imshow(image)
        # io.show()

        # Slied window
        for y in range(0, image.shape[0] - window_size, 50):

            # Skip croping if there is no tire in the window
            if y <= 300 or y >= 500:
                continue

            for x in range(0, image.shape[1] - window_size, 50):

                # Skip croping if there is no tire in the window
                if x <= 200 or x >= 550:
                    continue

                # crop image
                cropped = image[y:y + window_size, x:x + window_size]
                io.imsave(output_dir + i.strip(input_dir).strip('.jpg') + '_y' + str(y).zfill(3) + 'x' + str(x).zfill(3) + '.jpg', cropped)

"""
This script is not executed when called from outside
"""
if __name__ == "__main__":
    main()
