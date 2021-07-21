# CNN develop history

### Attempt 1
#### Preprocess: 
- Histogram Equalization
- Resize to 32*32

#### Classifier: 
Hierarchical CNN + 1 linear layer
#### Result: Max accuracy on test set 95.91%
![](../pic/result_cnn_2.png)

### Attempt 2
#### Preprocess: 
- Histogram Equalization
- Crop the picture according to Roi
- Resize to 32*32

#### Classifier: 
Hierarchical CNN + 1 linear layer
#### Result: Max accuracy on test set 98.37%
![](../pic/result_cnn_3.png)
