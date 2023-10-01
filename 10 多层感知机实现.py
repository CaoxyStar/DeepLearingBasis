import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from torch import nn

trans = transforms.ToTensor()
train_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)

batch_size = 256
train_DataLoader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
test_DataLoader = data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)

num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True) * 0.01)
params = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum() / len(y))

def net(X, params):
    X = X.reshape((-1, num_inputs))
    H = relu(torch.matmul(X, params[0]) + params[1])
    return torch.matmul(H, params[2]) + params[3]

num_epochs = 20
lr = 0.1

loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(params, lr=lr)

for epoch in range(num_epochs):
    for X, y in train_DataLoader:
        updater.zero_grad()
        y_hat = net(X, params)
        l = loss(y_hat, y)
        l.sum().backward()
        updater.step()

    with torch.no_grad():
        l = 0
        acc = 0
        temp = 0
        for X, y in test_DataLoader:
            y_hat = net(X, params)
            l += loss(y_hat, y).mean()
            acc += accuracy(y_hat, y)
            temp += 1
    print("epoch %d, accuracy %.5f, loss %.5f" % (epoch + 1, acc / temp, l / temp))



