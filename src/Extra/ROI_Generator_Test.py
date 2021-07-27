import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature as ft
import math
import pandas as pd

ground_true_file_path = r'E:\NUS\Project1\Train.csv'
test_file_path = r'E:\NUS\Project1\Train_gen.csv'

ground_true_df = pd.read_csv(ground_true_file_path)
test_df = pd.read_csv(test_file_path)

result = []

for idx, row in ground_true_df.iterrows():

    full_image = np.zeros((row['Height'], row['Width']), dtype=np.uint8)
    test_full_image = np.zeros((row['Height'], row['Width']), dtype=np.uint8)
    cv2.rectangle(full_image, (row['Roi.X1'], row['Roi.Y1']), (row['Roi.X2'], row['Roi.Y2']), 1, -1)
    cv2.rectangle(test_full_image, (test_df.at[idx, 'Roi.X1'], test_df.at[idx, 'Roi.Y1']), (test_df.at[idx, 'Roi.X2'], test_df.at[idx, 'Roi.Y2']), 1, -1)

    intersection = full_image & test_full_image
    union = full_image | test_full_image

    intersection = np.sum(intersection)
    union = np.sum(union)

    result.append(intersection / union)

ground_true_file_path = r'E:\NUS\Project1\Test.csv'
test_file_path = r'E:\NUS\Project1\Test_gen.csv'

ground_true_df = pd.read_csv(ground_true_file_path)
test_df = pd.read_csv(test_file_path)

for idx, row in ground_true_df.iterrows():

    full_image = np.zeros((row['Height'], row['Width']), dtype=np.uint8)
    test_full_image = np.zeros((row['Height'], row['Width']), dtype=np.uint8)
    cv2.rectangle(full_image, (row['Roi.X1'], row['Roi.Y1']), (row['Roi.X2'], row['Roi.Y2']), 1, -1)
    cv2.rectangle(test_full_image, (test_df.at[idx, 'Roi.X1'], test_df.at[idx, 'Roi.Y1']), (test_df.at[idx, 'Roi.X2'], test_df.at[idx, 'Roi.Y2']), 1, -1)

    intersection = full_image & test_full_image
    union = full_image | test_full_image

    intersection = np.sum(intersection)
    union = np.sum(union)

    result.append(intersection / union)

result = np.array(result)
print(np.mean(result))