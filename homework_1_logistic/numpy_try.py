import numpy as np
a = np.array([[1,2,2],
              [2,3,4],
              [2,5,6]])
print(a)
b = np.sum(a,axis=1)
print(b)
c = np.sum(a,axis=0)
print(c)
