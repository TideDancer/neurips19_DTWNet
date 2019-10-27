import torch
import torch.nn as nn
from torch.autograd import Variable
from functions.add import MyAddFunction
from modules.add import MyAddModule


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.add = MyAddModule()

    def forward(self, input1, input2):
        return self.add(input1, input2)

model = MyNetwork()
x = torch.Tensor([1.])
y = torch.Tensor([2.])
input1, input2 = Variable(x, requires_grad=True), Variable(y, requires_grad=True)
#print(model(input1, input2))
#print(input1 + input2)
out = MyAddFunction(input1, input2)
print(out.backward(x))
print(input1.grad)

out = torch.add(input1, input2)
print(out.backward())
print(input1.grad)

if torch.cuda.is_available():
    input1, input2, = input1.cuda(), input2.cuda()
    print(model(input1, input2))
    print(input1 + input2)
