import math
import numpy as np

socres = np.array([[1,2,3],
                   [1,2,3]])
scores = np.array([1,2,3])
exp_scores = np.exp(socres)   #numpy 还是很灵活的，exp操作记住不需要math包，numpy直接有
print(socres)
print(exp_scores)

# 下面测试对numpy二维数组每行进行归一化
socres_normalize = socres / np.sum(socres)
print(socres_normalize)
a = np.array([1,2,3])
a_normalize = a / np.sum(a)
print(a_normalize)
print(-np.log(a_normalize))


# np.dot, np.outer
outer_1 = np.array([1,2,3])
outer_2 = np.array([2,2,3])
print(np.outer(outer_1,outer_2))   # 第一个参数的每个数和第二个向量分别相乘，这尼玛也太灵活了

print(np.dot(outer_1,outer_2))  # 内积
