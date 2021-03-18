import math
import torch
import torch.nn as nn
# we assume a 256-dimensional input and a 4-dimensional output for this 1-layer neural network
# hence, we initialize a 256x4 dimensional matrix filled with random values
weights = torch.randn(256, 4) / math.sqrt(256)
# we then ensure that the parameters of this neural network ar trainable, that is,
#the numbers in the 256x4 matrix can be tuned with the help of backpropagation of gradients
weights.requires_grad_()
# finally we also add the bias weights for the 4-dimensional output, and make these trainable too
bias = torch.zeros(4, requires_grad=True)
x = torch.randn(256)
alinear0 = torch.matmul(x, weights) + bias
print(alinear0)

alinear1 = nn.Linear(256, 4)

with torch.no_grad():
    alinear1.weight.copy_(weights.T)
    alinear1.bias.copy_(bias.T)
#print(alinear1.weight)
#print(alinear1.bias)

print(alinear1)
print(alinear1(x))