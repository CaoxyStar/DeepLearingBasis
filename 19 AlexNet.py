import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)

batch_size = 64
train_DataLoader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
test_DataLoader = data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)
net.to('cuda')

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum() / len(y))

num_epoch = 5
lr = 0.1

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

writer = SummaryWriter('logs')
i = 0
for epoch in range(num_epoch):
    net.train()
    for X, y in train_DataLoader:
        optimizer.zero_grad()
        X, y = X.to('cuda'), y.to('cuda')
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        writer.add_scalar('train_loss', l, i)
        i += 1
    net.eval()
    with torch.no_grad():
        l, acc, temp = 0, 0, 0
        for X, y in test_DataLoader:
            X, y = X.to('cuda'), y.to('cuda')
            y_hat = net(X)
            l += loss(y_hat, y)
            acc += accuracy(y_hat, y)
            temp += 1
        writer.add_scalar('test_loss', l/temp, epoch)
        print("epoch %d, accuracy %.5f, test loss %.5f" % (epoch + 1, acc / temp, l / temp))
writer.close()
torch.save(net.state_dict(), "nets/AlexNet.pth")




