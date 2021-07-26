# Sliding window

Use model trained in expert level (no negative samples)

1. Use sliding window to catch a patch (48\*48)
2. For each patch, use **svm.predict_proba(feature)** to get the probality of all categories and get the max
3. if max > max_threshold, collect the patch location and probability
4. sort all the legal probabilities and show the top ones
5. Resize the pictures to different sizes and work like above again, so that there can be a time the traffic sign is smaller than the window to be detected

Result:

It is very slow and not accurate.

Disadvantages:

1. Too slow: sliding window will take a lot of time each round, and we have to do many rounds for different picture sizes
2. Not accurate: it is very hard to fix the sign exactly in the window, which will lead to a failure in the classify part

![](../../pic/Fail1.png)
