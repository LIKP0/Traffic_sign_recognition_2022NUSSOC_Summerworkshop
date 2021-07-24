import glob
import cv2

#   This code is used to read images in dataset_3
#   that is, "BelgiumTSC_Training"
#   data source: https://people.ee.ethz.ch/~timofter/traffic_signs/
# OUTPUTS:
#   X: the set of images
#   y: the set of labels

X = []
y = []
# read images in folders "00000" to "00009"
for j in range(0, 10):
    dataset_path = "F:\\NUS SOC 2021 phase2\\Project 1_ Traffic Sign Recogniti\\BelgiumTSC_Training\\Training\\0000" + str(
        j)
    for i in glob.glob(dataset_path + '\\*.ppm', recursive=True):
        label = i.split("BelgiumTSC_Training\\Training\\")[1][0:5]
        y.append(label)
        X.append(cv2.imread(i))
# read images in floders "00010" to "00061"
for j in range(11, 62):
    dataset_path = "F:\\NUS SOC 2021 phase2\\Project 1_ Traffic Sign Recogniti\\BelgiumTSC_Training\\Training\\000" + str(
        j)
    for i in glob.glob(dataset_path + '\\*.ppm', recursive=True):
        label = i.split("BelgiumTSC_Training\\Training\\")[1][0:5]
        y.append(label)
        X.append(cv2.imread(i))

# print the size of this dataset
print(len(y))
print(len(X))