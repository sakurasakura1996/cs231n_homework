import numpy as np
import pickle

# # 将训练数据分成5个batch 每个batch10000条数据
# train = open('./cifar-10-batches-py/data_batch_1','rb')   # rb 读模式和二进制模式
# dict = pickle.load(train,encoding='bytes')
#
# print(dict)
# print(dict[b'data'].shape)
#
# print(dict[b'batch_label'])
# print(len(dict[b'labels']))
#
# print(len(dict[b'filenames']))
# 因为KNN方法也没有所谓的训练过程，预测的时候就直接把它和所有已知分类的数据求一下距离然后找最近的k个数据，找其中出现次数最多的作为该预测分类
# 所以将这5个训练集合并起来
data_train = []
data_label = []
for i in range(1,6):
    train = open('./cifar-10-batches-py/data_batch_'+str(i),'rb')
    dict = pickle.load(train,encoding='bytes')
    for train_item in dict[b'data']:
        data_train.append(train_item)
    for label_item in dict[b'labels']:
        data_label.append(label_item)

data_train = np.array(data_train)
data_label = np.array(data_label)
print(data_train)
# print()
# print(len(data_train))

# 再获取一下测试集
data_test = []
test_label = []
train = open('./cifar-10-batches-py/test_batch','rb')
dict = pickle.load(train,encoding='bytes')
for train_item in dict[b'data']:
    data_test.append(train_item)
for label_item in dict[b'labels']:
    test_label.append(label_item)

data_test = np.array(data_test)
test_label = np.array(test_label)

#
# print(len(test_label))
class NearestNeighbor:
    def __init__(self):
        pass
    def train(self,X,y):
        self.Xtr = X
        self.ytr = y
    def predict(self,X):
        num_test = len(X)
        self.X = X
        Y_pred = np.zeros(num_test)
        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - self.X[i, :]), axis=1)  # 维度这个概念很重要 这个axis指定和不指定很不一样
            # print(type(distances[2]))
            # print(distances)
            # print(distances.shape)
            kmin=[]
            for j in range(12):
                min_index = np.argmin(distances)   # 找到最小的数，这里想改成knn，再观察一下正确率
                kmin.append(self.ytr[min_index])
                max_index = np.argmax(distances)
                # 这时候把其中最小的给去掉，再找最小的，就这样找到K个最小的值
                distances[min_index] = distances[max_index]
            # 找到出现次数最多的值
            b = np.argmax(np.bincount(kmin))
            Y_pred[i] = b
            if i % 10 == 0:
                print("第"+str(i)+"步")
        return Y_pred


nn = NearestNeighbor()
nn.train(data_train,data_label)
y_pred = nn.predict(data_test[:101])
accuracy = np.mean(test_label[:101]==y_pred)
print(accuracy)

