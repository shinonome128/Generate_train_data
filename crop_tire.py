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
    config_file.read('./crop_tire.conf', 'UTF-8')
    input_dir = config_file.get('dir','INPUT')
    output_dir = config_file.get('dir','OUTPUT')
    for i, image_path in enumerate(iglob('%s/*.JPG' % input_dir)):
        tire = cv2.imread(image_path)
        cropped = crop_tire(tire)
        cv2.imwrite(image_path.replace(input_dir, output_dir).replace('.JPG','.png'),cropped)

def crop_tire(tire):
    filtered = cv2.GaussianBlur(tire, (15, 15), 0)
    grayed = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(grayed, 150, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(binary[1])
    contours = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx=[]
    for c in contours[0]:
        if cv2.contourArea(c) < 100000:
            continue
        epsilon = 0.0001 * cv2.arcLength(c, True)
        approx.append(cv2.approxPolyDP(c, epsilon, True))
    tire_and_contours = np.copy(tire)
    tire_and_contours = cv2.drawContours(tire_and_contours, approx, -1, (0,0,0),50)
    mask = np.zeros_like(tire)
    mask = cv2.drawContours(mask, approx, -1, (255, 255, 255), -1)
    b_channel, g_channel, r_channel = cv2.split(mask)
    mask_alpha = cv2.merge((b_channel, g_channel, r_channel, cv2.bitwise_not(b_channel)))
    tire_alpha = Create_alpha(tire, 0)
    bg_alpha = np.zeros((tire.shape[0], tire.shape[1], 4), np.uint8)
    bg_alpha.fill(255)
    cropped = np.where(mask_alpha==255, tire_alpha, bg_alpha)
    return cropped

def Create_alpha(img, alpha):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA

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
