### Results of Different Classifiers on Dataset_1

|  classifier | SVM        | Random Forest |  Gaussian |  k-Nearest  |  
|:--------------:|:-----------:|:---------------:|:-----------:|:-------------:|
| accuracy     | 94.71% | 97.40% |78.62%|95.20%|
| training time /ms  |   907        |    3944        |    15    |     4     |
| classifyging time /ms   | 682  |     39      |  768     |  133      | 

*58 categories, 5998 images total\
*data from [Chinese Traffic Sign Recogntion Database (ia.ac.cn).](http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html)\
*tested by k_fold validation

### Results of Different Classifiers on Dataset_2

|  classifier | SVM        | Random Forest |  Gaussian |  k-Nearest  |  
|:--------------:|:-----------:|:---------------:|:-----------:|:-------------:|
| accuracy     | 92.33% | 88.66% |74.63%|91.67%|
| training time /ms  |   15867        |    45843        |    99    |     36     |
| classifyging time /ms   | 29784  |     325      |  416     |  4458      | 

*43 categories, 39209 images total\
*data from [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/)\
*tested by k_fold validation

### Results of Different Classifiers on Dataset_3

|  classifier | SVM        | Random Forest |  Gaussian |  k-Nearest  |  
|:--------------:|:-----------:|:---------------:|:-----------:|:-------------:|
| accuracy     | 90.78% | 91.11% |88.63%|97.73%|
| training time /ms  |   671        |    4491        |    13    |     4     |
| classifyging time /ms   | 496  |     33      | 17     |  83     | 

*62 categories, 4554 images total\
*data from [Belgium Traffic Sign dataset](https://people.ee.ethz.ch/~timofter/traffic_signs/)\
*tested by k_fold validation
