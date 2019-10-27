import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from functions.dynamic_prog import DynamicProgFunction
from modules.dynamic_prog import DynamicProgModule

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.dp = DynamicProgModule()

    def forward(self, input1, input2):
        return self.dp(input1, input2)

def read_ucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    Y = list(map(lambda e: e-1, Y))
    Y = np.asarray(Y)
    X = data[:,1:]
    X = np.reshape(X, (X.shape[0], 1, -1)) # shape is (N, channel, L)
    return X, Y, X.shape[0], X.shape[2]


X, Y, n_sample, feature_len = read_ucr('../../../DATA')

model = MyNetwork()
y = torch.Tensor([3., 5, 3, 8, 3, 2, 4, 8, 4])
# print(X[0,:].shape)
y = torch.Tensor(np.reshape(X[0,:], (80,)))
x = torch.Tensor([1., 1, 2, 3, 1])
x = torch.Tensor(np.random.normal(size=(10,)))
input1, input2 = Variable(x, requires_grad=True), Variable(y, requires_grad=True)
print(model(input1, input2))

# print(input1 + input2)
# out = MyAddFunction(input1, input2)
# print(out.backward(x))
# print(input1.grad)
# 
# out = torch.add(input1, input2)
# print(out.backward())
# print(input1.grad)
# 
# if torch.cuda.is_available():
#     input1, input2, = input1.cuda(), input2.cuda()
#     print(model(input1, input2))
#     print(input1 + input2)
