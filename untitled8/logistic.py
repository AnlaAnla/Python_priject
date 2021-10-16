import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# x1 = np.mat([[0.3,0.2,0.1]])
# w1 = np.mat([[2,2,2],
#             [3,3,3],
#             [4,4,4]])
#
# w2 = np.mat([[2,2,2]])
#
# x2 = np.matmul(x1, w1.T)
#
# y  = np.matmul(x2, w2.T)
# print(y)

# x = np.linspace(-10,10,200)
# y = sigmoid(x)
#
# l1 = plt.plot(x, y)
#
# plt.show()
# print(y)

