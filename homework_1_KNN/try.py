# 有点没明白前一个代码中的数组减法操作
import numpy as np
data_train = [[1,2,3,4],
              [3,4,3,2],
              [3,6,8,1],
              [3,5,7,1]]
data_test=[[1,4,6,7],
           [3,5,6,7],
           [1, 4, 6, 7],
           [3, 5, 6, 7]]
data_train = np.array(data_train)
data_test = np.array(data_test)
for i in range(4):
    y_pred = np.sum(np.abs(data_train-data_test[i,:]))   #原来还真是特么这么算的，data_train 把每个一维的拿出来和data_test的一维进行计算
    print(y_pred)

