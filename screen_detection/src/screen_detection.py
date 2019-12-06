import argparse
from os import listdir
from os.path import join
import cv2
import imutils
import numpy as np

from constants import NEW_HEIGHT
from constants import d, sigmaColor, sigmaSpace # bilateralFilter
from constants import threshold1, threshold2, apertureSize, L2gradient # Canny
from constants import approx_value


def create_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data', help='path to images')
    return parser


def show(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)


def find_contours(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.bilateralFilter(gray_image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    edged_image = cv2.Canny(filtered_image, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient)

    contours = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    return contours


def find_screen_conturs(image, contours):
    screen_conturs = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approximated_countur = cv2.approxPolyDP(contour, approx_value * peri, True)
        if len(approximated_countur) == 4:
            screen_conturs = approximated_countur
            break
    return screen_conturs


def create_rectangle(points):
    rectangle = np.zeros((4, 2), dtype='float32')

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rectangle[0] = points[np.argmin(s)]
    rectangle[2] = points[np.argmax(s)]
    rectangle[1] = points[np.argmin(diff)]
    rectangle[3] = points[np.argmax(diff)]
    return rectangle


def find_max_height_and_width(rectangle):
    (top_left, top_right, bottom_right, bl) = rectangle

    bottom_width = np.sqrt(((bottom_right[0] - bl[0]) ** 2) + ((bottom_right[1] - bl[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))

    right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    left_height = np.sqrt(((top_left[0] - bl[0]) ** 2) + ((top_left[1] - bl[1]) ** 2))

    max_width = max(int(bottom_width), int(top_width))
    max_height = max(int(right_height), int(left_height))
    return max_height, max_width


def cut_screen(image, screen_contours, origin_image, ratio):
    points = screen_contours.reshape(4, 2)
    rectangle = create_rectangle(points)
    rectangle *= ratio

    max_height, max_width = find_max_height_and_width(rectangle)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rectangle, dst)
    screen_image = cv2.warpPerspective(origin_image, M, (max_width, max_height))
    return screen_image


def main(args):
    data_path = args.data_path
    images = listdir(data_path)
    for name in images:
        image_path = join(data_path, name)
        image = cv2.imread(image_path)

        image_width = image.shape[0]
        ratio = image_width / NEW_HEIGHT
        origin_image = image.copy()
        image = imutils.resize(image, height=NEW_HEIGHT)

        contours = find_contours(image)
        screen_contours = find_screen_conturs(image, contours)
        if screen_contours is not None:
            screen_image = cut_screen(image, screen_contours, origin_image, ratio)
            show(screen_image)
        else:
            show(image)


if __name__ == '__main__':
    argument_parser = create_argument_parser()
    arguments = argument_parser.parse_args()
    main(arguments)