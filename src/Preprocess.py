import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from skimage import data, io, filters
from skimage import feature as ft
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from collections import Counter
import random
from skimage import util
from skimage import feature as ft


def resample(X, y):
    #   To avoid certain types of traffic signs having too few samples, by copying the images in the minorities
    #   that is, to make each types of traffic signs have similar size
    # Inputs:
    #   X: the original set of images
    #   y: the original set of labels
    # Outputs:
    #   newX: resampled set of images
    #   newy: resampled set of labels
    counter = {}
    for i in y:
        # List.count(i)统计列表元素对应的个数
        if y.count(i) > 0:
            counter[i] = y.count(i)
    valuelist = []
    for key, value in counter.items():
        valuelist.append(value)
    print("mean size of each type: ", np.mean(valuelist))
    print("median size of each type: ", np.median(valuelist))
    # 切片
    sliceList_X = []
    sliceList_y = []
    start = 0
    for i in range(0, len(valuelist)):
        sliceList_y.append(y[start:start + valuelist[i]])
        sliceList_X.append(X[start:start + valuelist[i]])
        start = start + valuelist[i]
    # 补齐至62项(中位数)
    for i in range(0, len(valuelist)):
        if valuelist[i] < 62:
            n = 62 - valuelist[i]
            k = 0
            for j in range(0, n):
                sliceList_y[i].append(sliceList_y[i][k])
                sliceList_X[i].append(sliceList_X[i][k])
                k = k + 1
                if k == valuelist[i]:
                    k = 0
    # 合并
    newy = []
    newX = []
    for i in range(0, len(valuelist)):
        newy.extend(sliceList_y[i])
        newX.extend(sliceList_X[i])
    print("size of X after resampling is: ", len(newX))
    print("size of y after resampling is: ", len(newy))
    return newX, newy


def resize(X, y):
    #   to resize image x to 48x48
    # Inputs:
    #   X: the original set of images
    #   y: the original set of labels
    # Outputs:
    #   temp_X: the set of resized images
    #   y: the original set of labels
    temp_X = []
    for x in X:
        temp_X.append(cv2.resize(x, (48, 48)))
    return temp_X, y


def grayscale(X, y):
    #   to change image X to grayscale
    # Inputs:
    #   X: the original set of images
    #   y: the original set of labels
    # Outputs:
    #   temp_X: the set of grayscaled images
    #   y: the original set of labels
    temp_X = []
    for x in X:
        temp_X.append(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
    return temp_X, y


def histogramEqualization(X, y):
    #   to do histogram equalization to image X
    # Inputs:
    #   X: the original set of images
    #   y: the original set of labels
    # Outputs:
    #   temp_X: the set of images after histogram equalization
    #   y: the original set of labels
    temp_X = []
    for x in X:
        temp_X.append(cv2.equalizeHist(x))
    return temp_X, y


def gaussianNoise(X, y):
    #   to add Gaussian noise to image X
    # Inputs:
    #   X: the original set of images
    #   y: the original set of labels
    # Outputs:
    #   X_noisy: the set of images added Gaussian noise
    #   y: the original set of labels
    X_noisy = []
    for x in X:
        X_noisy.append(util.random_noise(x, mode='gaussian').astype(np.float32))
    return X_noisy, y


def affine(X, y):
    #   to add affine transform to image X
    # Inputs:
    #   X: the original set of images
    #   y: the original set of labels
    # Outputs:
    #   X_affine: the set of images after affine transform
    #   y: the original set of labels
    def affineImage(image):
        # to affine one image
        rows, cols = image.shape[:2]
        point1 = np.float32([[20, 80], [300, 50], [80, 200]])
        point2 = np.float32([[10, 100], [300, 50], [100, 250]])
        M = cv2.getAffineTransform(point1, point2)
        image = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
        return image

    X_affine = []
    for x in X:
        X_affine.append(affineImage(x))
    return X_affine, y


def crop(X, y):
    #   to random crop each image X
    #   the min_ratio and max_ratio can be set in this function
    # Inputs:
    #   X: the original set of images
    #   y: the original set of labels
    # Output:
    #   X_crop: the set of images after random cropping
    #   y: the original set of labels
    def random_crop(image, min_ratio, max_ratio):
        #   to random crop one image
        # Arguments:
        #   min_ratio: the minima ratio of the cropped image to original image
        #   max_ratio: the maxima ratio of the cropped image to original image
        h, w = image.shape[:2]
        ratio = random.random()
        scale = min_ratio + ratio * (max_ratio - min_ratio)
        new_h = int(h * scale)
        new_w = int(w * scale)
        y = np.random.randint(0, h - new_h)
        x = np.random.randint(0, w - new_w)
        image = image[y:y + new_h, x:x + new_w, :]
        return image

    X_crop = []
    for x in X:
        X_crop.append(random_crop(x, 0.6, 1))
    return X_crop, y


def rotate(X, y):
    #   to rotate each images in X
    #   half of images rotate 15 degree clockwise, half of images rotate 15 degree counterclockwise
    # Inputs:
    #   X: the original set of images
    #   y: the original set of labels
    # Outputs:
    #   X_rotated: the set of images after rotation
    #   y: the original set of labels
    def rotateImage(image, angle):
        #   to rotate one image by certain degree
        (h, w) = image.shape[:2]  # 返回(高,宽,色彩通道数),此处取前两个值返回
        # 抓取旋转矩阵(应用角度的负值顺时针旋转)。参数1为旋转中心点;参数2为旋转角度,正的值表示逆时针旋转;参数3为各向同性的比例因子
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        # 计算图像的新边界维数
        newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
        newH = int((h * np.abs(M[0, 0])) + (w * np.abs(M[0, 1])))
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (newW - w) / 2
        M[1, 2] += (newH - h) / 2
        # 执行实际的旋转并返回图像
        return cv2.warpAffine(image, M, (newW, newH))  # borderValue 缺省，默认是黑色

    X_rotated = []
    for i in range(0, len(X) - 1):
        if i % 2 == 0:
            X_rotated.append(rotateImage(X[i], 15))
        if i % 2 == 1:
            X_rotated.append(rotateImage(X[i], -15))
    return X_rotated, y


def extractFeature(X, y):
    #   to extract HOG feature of each image
    # Inputs:
    #   X: the original set of images
    #   y: the original set of labels
    # Outputs:
    #   X_feature: the set of HOG features of corresponding images
    #   y: the original set of labels
    X_feature = []
    for x in X:
        x_feature = ft.hog(x, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1), visualize=False,
                           multichannel=False)
        X_feature.append(x_feature)
    return X_feature, y
