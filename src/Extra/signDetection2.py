import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from skimage import feature as ft
import math


def normalize_rgb(img):
    """
    normalizes the RGB value to [0, 1]
    :param img: the image needs to be normalized
    :return: the image after normalization
    """
    normalize_img = np.copy(img).astype(np.float)
    sum_rgb = np.sum(img, axis=-1)
    for c in range(img.shape[-1]):  # traverse all the channel
        normalize_img[:, :, c] = img[:, :, c] / sum_rgb
    return normalize_img


def rgb_normalized_thresholding(img, ThR=0.4, ThG=0.3, ThB=0.4, ThY=0.85):
    """
    extracts the img based on the normalized rgb.
    :param img: the image that need to be extracted
    :param ThR: red lower threshold
    :param ThG: green upper threshold
    :param ThB: blue lower threshold
    :param ThY: yellow lower threshold
    :return: three channels of image divided by red, blue and yellow
    """
    blue = cv2.inRange(img, np.array([ThB, 0, 0], dtype=np.float), np.array([1, 1, 1], dtype=np.float))

    red = cv2.inRange(img, np.array([0, 0, ThR], dtype=np.float), np.array([1, ThG, 1], dtype=np.float))

    # yellow extraction
    yellow_value = img[:, :, 1] + img[:, :, 2]
    yellow = np.array(yellow_value >= ThY, dtype=np.uint8) * 255

    return red, blue, yellow


def detect(img):
    """
    detects the traffic sign from the image.
    :param img: the image needs to be detected
    :return: the ROIs of the image
    """
    img_area = img.shape[0] * img.shape[1]
    ROIs = []
    # normalize image
    norm_img = normalize_rgb(img)

    rgb_extraction = rgb_normalized_thresholding(norm_img)
    h, w = rgb_extraction[0].shape

    for i in range(len(rgb_extraction)):
        channel = np.zeros((h+2, w+2), dtype=np.uint8)
        channel[1:-1, 1:-1] = rgb_extraction[i]
        mask = np.zeros((h+4, w+4), dtype=np.uint8)

        # 用泛洪填充来填充中间的空白
        reverse = np.copy(channel)
        cv2.floodFill(reverse, mask, [0, 0], 255)
        reverse = np.array(reverse == 0, dtype=np.uint8) * 255

        channel = reverse + channel


        # 图像开运算：去除噪点
        erode = cv2.erode(channel, None, iterations=1)
        dilate = cv2.dilate(erode, None, iterations=1)



        # # 使区域闭合无空隙
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # closed = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

        # 连通量分割标志
        components_n, labels, rois, centroids = cv2.connectedComponentsWithStats(dilate)
        for roi in rois[1:]:
            # 比较横纵比（交通标志的外接矩形都接近正方形）
            if roi[2] / roi[3] <= 0.7 or roi[3] / roi[2] <= 0.7:
                continue
            # 比较面积，去除过小的roi
            elif roi[-1] < img_area * 0.005:
                continue
            else:
                ROIs.append(roi)

    # 合并三个通道的ROI，去除重复的区域
    blank = np.zeros(img.shape[:-1], dtype=np.uint8)
    for roi in ROIs:
        cv2.rectangle(blank, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), 255, -1)

    plt.imshow(blank, cmap='gray')
    plt.show()
    # 提取所有标志的ROI
    components_n, labels, total_rois, centroids = cv2.connectedComponentsWithStats(blank)
    for roi in total_rois:
        cv2.rectangle(img, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 0, 255), 5)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    return norm_img


if __name__ == '__main__':
    image = cv2.imread('../../Extra_data/No_noise2.jpg')
    img_bright = cv2.convertScaleAbs(image, alpha=2, beta=0)
    detect(image)
