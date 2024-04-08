import numpy as np

a = np.arange(40).reshape(10, 4)
print(a.shape)
print(a)
print(np.mean(a, axis=1))
print(a.reshape(5, 8))
print(np.mean(a.reshape(5, 8), axis=1))