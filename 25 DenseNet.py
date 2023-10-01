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
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels*i+input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    num_channels += num_convs*growth_rate
    if i!=len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels//2))
        num_channels = num_channels//2

net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10)
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
torch.save(net.state_dict(), "nets/DenseNet.pth")

