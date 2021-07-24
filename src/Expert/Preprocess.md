# Preprocess

functions used in Expert Level, including:

`resample(X, y)`:
avoid certain types of traffic signs having too few samples, by copying the images in the minorities

`resize(X, y)`:
to resize image x to 48x48

`grayscale(X, y)`:
to change image X to grayscale

`histogramEqualization(X, y)`:
to do histogram equalization to image X

`gaussianNoise(X, y)`:
to add Gaussian noise to image X

`affine(X, y)`:
to add affine transform to image X

`crop(X, y)`:
to random crop each image X

`rotate(X, y)`:
to rotate each images in X

`extractFeature(X, y)`:
to extract HOG feature of each image

`SVM(X_train, y_train, X_test, y_test)`:
train a SVM model and get its accuracy on test set

`randomForest(X_train, y_train, X_test, y_test)`:
train a Random Forest model and get its accuracy on test set

`gaussianNaiveBayes(X_train, y_train, X_test, y_test)`:
train a Gaussian Naive Bayes model and get its accuracy on test set

`kNearestNeighbours(X_train, y_train, X_test, y_test, k)`:
train a k-nearest Neighbours model and get its accuracy on test set
