import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature as ft
import math
import pandas as pd


from signDetection2 import detect


def generate_roi(img):
    w, h = img.shape[1], img.shape[0]
    default_roi = (0, 0, w, h)
    rois = detect(img=img)
    if len(rois) == 0:
        return default_roi
    elif len(rois) == 1:
        roi = rois[0]
        length = max([roi[2], roi[3]])
        return roi[0], roi[1], min(roi[0] + length, w), min(roi[1] + length, h)
    else:
        first_roi = rois[0]
        left, up, right, bottom = first_roi[0], first_roi[1], first_roi[0] + first_roi[2], first_roi[1] + first_roi[3]
        for roi in rois[1:]:
            if left > roi[0]:
                left = roi[0]
            if up > roi[1]:
                up = roi[1]
            if right < roi[0] + roi[2]:
                right = roi[0] + roi[2]
            if bottom < roi[1] + roi[3]:
                bottom = roi[1] + roi[3]
        width = right - left
        height = bottom - up
        length = max(width, height)

        return left, up, min(left + width, w), min(up + length, h)

if __name__ == '__main__':
    # Read in dataset
    image_root_path = r'E:/NUS/Project1/'
    train_file_path = r'E:\NUS\Project1\Test.csv'
    # test_file_path = r'E:\NUS\Project1\annotations.csv'

    # df_train = pd.read_csv(train_file_path)
    df_test = pd.read_csv(train_file_path)
    for idx, row in df_test.iterrows():
        # read image
        file_path = image_root_path + row['Path']
        image = cv2.imread(file_path)
        df_test.at[idx, 'Roi.X1'], df_test.at[idx, 'Roi.Y1'], df_test.at[idx, 'Roi.X2'], df_test.at[idx, 'Roi.Y2'] = generate_roi(img=image)
    df_test.to_csv(r'E:\NUS\Project1\Test_gen.csv', index=0)


