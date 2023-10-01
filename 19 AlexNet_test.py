import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
import time

trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
test_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)
print("dataset size: ", test_dataset.data.shape[0])

batch_size = 64
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

net.load_state_dict(torch.load("nets/AlexNet.pth"))
net.to('cuda')

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum() / len(y))

time1 = time.time()
with torch.no_grad():
    for X, y in test_DataLoader:
        X, y = X.to('cuda'), y.to('cuda')
        y_hat = net(X)

time2 = time.time()
print(time2-time1)
print("compute %f samples/sec." % (test_dataset.data.shape[0] / (time2 - time1)))
# 1752 samples/sec