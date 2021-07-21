# NUS_CV_G7

**张柯远、徐承启、庄宇凡、李佳霖**

**SuSTech**

**2021 from July 12 to July 28**

## New Plan

| Time | Plan | Remarks |
| ----------- | ---------- | ---------- |
| 7.21      | More data agumentation  | larger size of added data |
| 7.21      | Object detection in large picture  | Color/Shape detect to get ROI |
| 7.21      | 细化模型开发  | 确定如何更好的测试模型，之后调参 |


## TimeTable

| Time | Achievement | Remarks |
| ----------- | ---------- | ---------- |
| 7.16      | Beginner level complete  ||
| 7.18      | Expert level complete  | simply add pre-processing method，Bonus CNN still has problem |
| 7.19      | new plan  | Bonus CNN complete，Bonus try no CNN method，try object detection in big scale |
| 7.20      | finish data agumentation  | different in CNN and traditional method |
| 7.21      | complete a try on large picture  | sliding window leads to a total fail  |

## Beginner level

[beginner.ipynb](https://github.com/LIKP0/NUS_CV_G7/blob/main/src/beginner.ipynb)

## Expert level

[Expert_differentPreprocessing.ipynb](https://github.com/LIKP0/NUS_CV_G7/blob/main/src/Expert_differentPreprocessing.ipynb)

[Expert_differentClassifier.ipynb](https://github.com/LIKP0/NUS_CV_G7/blob/main/src/Expert_differentClassifier.ipynb)

[Expert_augmentation.ipynb](https://github.com/LIKP0/NUS_CV_G7/blob/main/src/Expert_augmentation.ipynb)

- Pre-processing image

| Method | Result |
| ----------- | ---------- |
| resize to 48\*48     | 0.943  |
| Resample to make samples distribute uniformly   | 0.972  |
| Resample to make samples distribute uniformly (dataset2)  | 0.919  |
| Histogram Equalization   |0.935   |
| Resample + Histogram Equalization  |0.962  |
| add data augumentation    | done  |
| data strengthen by library (not plan to do temporarily)   |   |

- Method of data agumentation

**Traditional (based on SVM)**

![](./pic/different_augmentation_methods.png)

| Method | Remarks | Result |
| ----------- | ---------- | ---------- |
| rotate     | rotate +-10 degree,+-15 degree, each 5% of total data |0.889 |
| noise | add Gaussian White noise to 20% of the images|0.901 |
| affine  |  do affine transform to 20% of the images |0.895 |
| random crop  | randomly crop 20% of the images  | 0.885|
| mine hard negs   | collect failed test examples for retraining  | |
| multiscale   | resize?   | |
| fliplr  | impossible in real life   | |
| negative example   | picture with no signs, not collected yet  | |
| More train and test data   | not collected yet  | |

![](./pic/augmented_dataset1.png)

**For CNN**

Use dataloader in pytorch

| Method | Remarks | Result |
| ----------- | ---------- | ---------- |
| Random Gaussian Blur   | probability=0.3, gamma=3  | 98.01%|
| Random Affine transformation +  Random Gaussian Blur  | degree=15  | 97.89%|
| Random Affine transformation +  Random Gaussian Blur + Random brightness change   | brightness=(0.7-1.3)  | 97.92%|

- Extract features

| Method | Result |
| ----------- | ---------- |
| HOG     | 0.943  |
|    SIFT   |  |

- Classify model

| Method | Result |
| ----------- | ---------- |
| svm.SVC()     | 0.944  |
|    RandomForestClassifier()   | 0.975 |
|  GaussianNB()  | 0.781 |
|  KNeighborsClassifier(k= 1,3,5,7,9)  | max 0.968 |

## Bonus Level

[CNN.py](https://github.com/LIKP0/NUS_CV_G7/blob/main/src/CNN.py)

[CNN_withAugmentation.py](https://github.com/LIKP0/NUS_CV_G7/blob/main/src/CNN_withAugmentation.py)

[CNN_History_Attempt.md](https://github.com/LIKP0/NUS_CV_G7/blob/main/src/CNN_History_Attemp.md)


## Extra work

### Detection in a large scale

Example of error in detection of pictures of large scale

![](./pic/Error_example1.png)

| Method | Remarks | Result |
| ----------- | ---------- | ---------- |
| sliding window 48\*48 + max category proability sort   | directly give answers of all patches | very slow and not accurate |
| Object detection algorithm     | to crop the sign first  |  |

#### sliding window 48\*48 + max prob sort

[SlidingWindow_v1.ipynb](https://github.com/LIKP0/NUS_CV_G7/blob/main/src/SlidingWindow_v1.ipynb)

Use model clf trained in expert level (no negative samples)

1. Use sliding window to catch a patch
2. Use **clf.predict_proba(feature)** to get the probality of all categories and get the max
3. if max > max_threshold, collect the patch location
4. sort all the patch locations and show the top ones

![](./pic/Fail1.png)

#### Object detection algorithm 

1. Train a binary classification model with negative samples, to predict sign or not sign
2. Use sliding window and first detect if it is a sign
3. If it is a sign, use multi-classifier


