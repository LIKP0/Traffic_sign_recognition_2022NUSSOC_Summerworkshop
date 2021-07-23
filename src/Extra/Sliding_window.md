# Sliding window

Use model clf trained in expert level (no negative samples)

1. Use sliding window to catch a patch
2. Use **clf.predict_proba(feature)** to get the probality of all categories and get the max
3. if max > max_threshold, collect the patch location
4. sort all the patch locations and show the top ones

![](./pic/Fail1.png)
