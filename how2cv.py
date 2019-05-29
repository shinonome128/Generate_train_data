import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
# import pdb; pdb.set_trace()

def main():

    # Read image and convert from BGR to RGB
    img = cv2.imread('./DATA/TIRE/RIMG2982.JPG')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    io.imshow(img_rgb)
    io.show()

    # Convert from BGR to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    io.imshow(img_gray)
    io.show()

    """
    _, threshold_img = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY)
    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
    io.imshow(threshold_img)
    io.show()
    """

    """
    img = cv2.imread('./DATA/TIRE/RIMG2982.JPG')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tire_min = np.array([50, 50, 50], np.uint8)
    tire_max = np.array([130, 130, 130], np.uint8)
    threshold_tire_img = cv2.inRange(img_hsv, tire_min, tire_max)
    threshold_tire_img = cv2.cvtColor(threshold_tire_img, cv2.COLOR_GRAY2RGB)
    io.imshow(threshold_tire_img)
    io.show()
    """


    """
    tire = cv2.imread('./DATA/TIRE/RIMG2982.JPG')
    tire_hsv = cv2.cvtColor(tire, cv2.COLOR_BGR2HSV)
    tire_rgb = cv2.cvtColor(tire_hsv, cv2.COLOR_HSV2RGB)
    # io.imshow(tire_rgb)
    # io.show()
    tire_min = np.array([50, 50, 50], np.uint8)
    tire_max = np.array([130, 130, 130], np.uint8)
    mask = cv2.inRange(tire, tire_min, tire_max)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # io.imshow(mask_rgb)
    # io.show()
    mask_inverse = cv2.bitwise_not(mask)
    mask_inverse_rgb = cv2.cvtColor(mask_inverse, cv2.COLOR_GRAY2RGB)
    # io.imshow(mask_inverse_rgb)
    # io.show()
    masked_tire = cv2.bitwise_and(tire, mask_rgb)
    # io.imshow(masked_tire)
    # io.show()
    masked_replace_white = cv2.addWeighted(masked_tire, 1, mask_inverse_rgb, 1, 0)
    masked_replace_white_rgb = cv2.cvtColor(masked_replace_white, cv2.COLOR_BGR2RGB)
    io.imshow(masked_replace_white_rgb)
    io.show()
    """

    """
    tire = cv2.imread('./DATA/TIRE/RIMG2982.JPG')
    tire_hsv = cv2.cvtColor(tire, cv2.COLOR_BGR2HSV)
    io.imshow(cv2.cvtColor(tire_hsv, cv2.COLOR_HSV2RGB))
    io.show()
    img_blur_small = cv2.GaussianBlur(tire_hsv, (15,15), 0)
    io.imshow(cv2.cvtColor(img_blur_small, cv2.COLOR_HSV2RGB))
    io.show()
    """

    """
    # get binary image and apply Gaussian blur
    tire = cv2.imread('./DATA/TIRE/RIMG2982.JPG')
    tire_gray = cv2.cvtColor(tire, cv2.COLOR_BGR2GRAY)
    tire_preprocessed = cv2.GaussianBlur(tire_gray, (7, 7), 0)
    _, tire_binary = cv2.threshold(tire_preprocessed, 140, 255, cv2.THRESH_BINARY)
    tire_binary = cv2.bitwise_not(tire_binary)
    io.imshow(cv2.cvtColor(tire_binary, cv2.COLOR_GRAY2RGB))
    io.show()
    tire_contours, _ = cv2.findContours(tire_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # make copy of image
    tire_and_contours = np.copy(tire)
    # find contours of large enough area
    min_tire_area = 1000000
    large_contours = [cnt for cnt in tire_contours if cv2.contourArea(cnt) > min_tire_area]
    # draw contours
    img = cv2.drawContours(tire_and_contours, large_contours, -1, (0,255,0),30)
    # print number of contours
    print('number of tire: %d' % len(large_contours))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    io.imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
    io.show()
    """

    """
    cups = cv2.imread('./DATA/TIRE/RIMG2982.JPG')
    # preprocess by blurring and grayscale
    cups_preprocessed  = cv2.cvtColor(cv2.GaussianBlur(cups, (7,7), 0), cv2.COLOR_BGR2GRAY)
    # find binary image with thresholding
    _, cups_thresh = cv2.threshold(cups_preprocessed, 80, 255, cv2.THRESH_BINARY)
    # _, cups_thresh = cv2.threshold(cups_preprocessed, 200, 255, cv2.THRESH_BINARY)
    io.imshow(cv2.cvtColor(cups_thresh, cv2.COLOR_GRAY2RGB))
    io.show()
    import pdb; pdb.set_trace()
    """

    """
    cups = cv2.imread('./DATA/TIRE/RIMG2982.JPG')
    cups_preprocessed  = cv2.cvtColor(cv2.GaussianBlur(cups, (7,7), 0), cv2.COLOR_BGR2GRAY)
    cups_edges = cv2.Canny(cups_preprocessed, threshold1=140, threshold2=255)
    io.imshow(cv2.cvtColor(cups_edges, cv2.COLOR_GRAY2RGB))
    io.show()
    """

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
