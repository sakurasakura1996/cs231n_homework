# 用KNN方法来做
import numpy as np
import pickle
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
# print(data_train)
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
            # print(distances)
            # print(distances.shape)
            min_index = np.argmin(distances)
            Y_pred[i] = self.ytr[min_index]
            if i % 10 == 0:
                print("第"+str(i)+"步")
        return Y_pred


nn = NearestNeighbor()
nn.train(data_train,data_label)
y_pred = nn.predict(data_test[:1001])
accuracy = np.mean(test_label[:1001]==y_pred)
print(accuracy)

