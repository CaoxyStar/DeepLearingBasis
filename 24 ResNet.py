import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)

batch_size = 64
train_DataLoader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
test_DataLoader = data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(), nn.Linear(512, 10)
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

num_epoch = 10
lr = 0.05

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

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
torch.save(net.state_dict(), "nets/ResNet.pth")