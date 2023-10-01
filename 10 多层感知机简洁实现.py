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

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum() / len(y))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
net.apply(init_weights)

num_epoch = 20
lr = 0.1

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

for epoch in range(num_epoch):
    net.train()
    for X, y in train_DataLoader:
        optimizer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.mean().backward()
        optimizer.step()
    net.eval()
    with torch.no_grad():
        l = 0
        acc = 0
        temp = 0
        for X, y in test_DataLoader:
            y_hat = net(X)
            l += loss(y_hat, y)
            acc += accuracy(y_hat, y)
            temp += 1
        print("epoch %d, accuracy %.5f, loss %.5f" % (epoch + 1, acc / temp, l / temp))
