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
import Preprocess
from skimage import util

dataset_path = "F:\\NUS SOC 2021 phase2\\Project 1_ Traffic Sign Recogniti\\Dataset_1\\images"
X = []
y = []
for i in glob.glob(dataset_path + '\\*.png', recursive=True):
    label = i.split("images")[1][1:4]
    y.append(label)
    # write code to read ecah file i, and append it to list X
    X.append(cv2.imread(i))
# you should have X, y with 5998 entries on each.
print("size of y is: ", len(y))
print("size of X is : ", len(X))

X1, y1 = Preprocess.resample(X, y)
X2, y2 = Preprocess.resize(X1, y1)
X3, y3 = Preprocess.grayscale(X2, y2)
X4, y4 = Preprocess.histogramEqualization(X3, y3)

fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(X[1])
plt.title("original image")
fig.add_subplot(1, 2, 2)
plt.imshow(X4[1], cmap='gray')
plt.title("processed image")
plt.show()

# --------------------------------------------------------------

X_noisy, y_noisy = Preprocess.gaussianNoise(X, y)
X_affined, y_affined = Preprocess.affine(X, y)
X_rotated, y_rotated = Preprocess.rotate(X, y)
X_cropped, y_cropped = Preprocess.crop(X, y)

i = random.randint(0, len(y) - 1)
fig = plt.figure()

fig.add_subplot(2, 3, 1)
plt.imshow(X[i])
plt.title('original')

fig.add_subplot(2, 3, 2)
plt.imshow(X_noisy[i])
plt.title('noisy')

fig.add_subplot(2, 3, 3)
plt.imshow(X_affined[i])
plt.title('affined')

fig.add_subplot(2, 3, 4)
plt.imshow(X_rotated[i])
plt.title('rotated')

fig.add_subplot(2, 3, 5)
plt.imshow(X_cropped[i])
plt.title('cropped')

plt.show()
