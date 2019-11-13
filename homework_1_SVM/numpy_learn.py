# 这里测试使用numpy ones和zeros的使用
import numpy as np
a = np.zeros(3)   # 默认为float64 位
b = np.zeros([2,3],int)  # int表示int 32位
c = np.zeros([3,3],dtype='float32')
d = np.zeros([3,4])
print(a,type(a[1]))
print(b,type(b[1][1]))
print(c,type(c[1][1]))
print(d,type(d[1][1]))

aa = np.ones(3)
print(aa,type(aa[1]))

bbn = np.ones([2,2],dtype='int32')
print(bbn,type(bbn[1][1]))

rs = np.array([[1,2,3],
               [2,4,7]])
rs1 = rs.reshape(-1,1)      #　内部所有数组成一列
rs2 = rs.reshape(1,-1)      # 内部所有数排成一行
# 相当于reshape(-1,1) or reshape(1,-1) 对内部数据的结构进行了转换
# reshape中的参数知道了，其实就是几行几列的意思，第一个如果为-1的话，那么表示的就是其中的元素个数列，也就是变成一列
# 如果是（1，-1）的话，也就是差不多，-1表示自动计算，但是计算出来要可以整除
print(rs1)
print(rs2)
rs3 = rs.reshape(-1,)   # 这里只写一个-1竟然变成了一个一维列表而不是二维的了
print(rs3)
rs4 = rs.reshape(2,-1)   #这里变换为2行，至于每行几列直接写一个 -1来让他自己计算，但是要能够整除
print(rs4)

add1 = np.array([[1],
                 [2],
                 [3]])
add2 = np.array([[1],
                 [4],
                 [3]])
print(add1-add2+1)  # 这里加一可以直接每个都加上去，之前在写knn的时候减法也是这样的
margin = np.maximum(0,add1-add2+1)   # 同理，这里的和0比较大小，也是分别去比较，好几把灵活啊
print(margin)

scores = np.array([[1,2,3,4]])
y = np.array([1,2,3
              ])
correct_class_score = scores[range(1), list(y)]
print(list(y))
print(str(y.shape))
print(correct_class_score)





