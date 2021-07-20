import cv2
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from skimage import data, io, filters
from skimage import feature as ft
from sklearn.model_selection import train_test_split
from sklearn import svm
import time


# Read in dataset
image_root_path = r'E:/NUS/Project1/'
train_file_path = r'E:\NUS\Project1\Train.csv'
test_file_path = r'E:\NUS\Project1\Test.csv'


# Read in images and labels
train, train_labels, test, test_labels = [], [], [], []
df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

for idx, row in df_train.iterrows():
    # read image
    file_path = image_root_path + row['Path']
    train.append(cv2.imread(file_path))
    train_labels.append(row['ClassId'])

for idx, row in df_test.iterrows():
    # read image
    file_path = image_root_path + row['Path']
    test.append(cv2.imread(file_path))
    test_labels.append(row['ClassId'])

# Preprocessing
train_processed = []
test_processed = []
for x in train:
    # Write code to resize image x to 48x48 and store in temp_x
    temp_x = cv2.resize(x, (48, 48))
    # Write code to convert temp_x to grayscale
    temp_x_gray = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)
    # Append the converted image into X_processed
    train_processed.append(temp_x_gray)
for x in test:
    # Write code to resize image x to 48x48 and store in temp_x
    temp_x = cv2.resize(x, (48, 48))
    # Write code to convert temp_x to grayscale
    temp_x_gray = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)
    # Append the converted image into X_processed
    test_processed.append(temp_x_gray)

# Feature extraction
train_features = []
test_features = []
for x in train_processed:
    x_feature = ft.hog(x, orientations=8, pixels_per_cell=(10, 10),cells_per_block=(1, 1), visualize=False, multichannel=False)
    train_features.append(x_feature)

for x in test_processed:
    x_feature = ft.hog(x, orientations=8, pixels_per_cell=(10, 10),cells_per_block=(1, 1), visualize=False, multichannel=False)
    test_features.append(x_feature)

clf = svm.SVC()
clf.fit(train_features, train_labels)
score = clf.score(test_features, test_labels)
print(score)