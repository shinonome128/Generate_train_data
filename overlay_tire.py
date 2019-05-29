import sys
import configparser
from glob import iglob
import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
# import pdb; pdb.set_trace()

def main():
    config_file = configparser.ConfigParser()
    config_file.read('./overlay_tire.conf', 'UTF-8')
    cropped_dir = config_file.get('dir','TRIMMED')
    bg_dir = config_file.get('dir','BG')
    overlay_dir = config_file.get('dir','OVERLAY')

    for i, cropped_path in enumerate(iglob('%s/*.png' % cropped_dir)):
        cropped = cv2.imread(cropped_path, cv2.IMREAD_UNCHANGED)
        cropped = cropped.transpose(1,0,2)[:,::-1]
        for j, bg_path in enumerate(iglob('%s/*jpeg' % bg_dir)):
            bg = cv2.resize(cv2.imread(bg_path), (cropped.shape[1], cropped.shape[0]))
            cropped = cv2.GaussianBlur(cropped, (3,3), 0)
            # display_bgr_rgb(cropped)
            overlay = overlay_tire(cropped, bg)
            lin = overlay_dir + '/' + str(i) + '_' + str(j) + '.JPG'
            cv2.imwrite(lin, overlay)

def overlay_tire(cropped, bg):
    mask = cropped[:,:,3]
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    width, height = cropped.shape[:2]
    cropped = cropped[:,:,:3]
    overlay = np.where(mask==255, cropped, bg)
    return overlay

def display_bgr_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    io.imshow(img)
    io.show()

def display_gray_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    io.imshow(img)
    io.show()

if __name__ == "__main__":
    main()
