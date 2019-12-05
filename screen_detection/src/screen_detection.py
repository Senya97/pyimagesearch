import argparse
from os import listdir
from os.path import join
import cv2
import imutils
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

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.bilateralFilter(gray_image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        edged_image = cv2.Canny(filtered_image, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient)

        cnts = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, approx_value * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
        if screenCnt is not None:
            cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
            show(image)
        else:
            show(image)


if __name__ == '__main__':
    argument_parser = create_argument_parser()
    arguments = argument_parser.parse_args()
    main(arguments)