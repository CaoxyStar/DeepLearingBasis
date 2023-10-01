import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
import time

trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
test_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)

batch_size = 16
test_DataLoader = data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum() / len(y))

net = vgg(conv_arch)
net.load_state_dict(torch.load("nets/VGG.pth"))
net.to('cuda')

time1 = time.time()
with torch.no_grad():
    for X, y in test_DataLoader:
        X, y = X.to('cuda'), y.to('cuda')
        y_hat = net(X)

time2 = time.time()
print(time2-time1)
print("compute %f samples/sec." % (test_dataset.data.shape[0] / (time2 - time1)))
# 210 samples/sec
# 91.5 accuracy
