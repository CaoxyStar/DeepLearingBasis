import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)

print("len of train_set: ", len(mnist_train))
print("len of test_set: ", len(mnist_test))

X, y = next(iter(data.DataLoader(mnist_train, batch_size=10)))
writer = SummaryWriter('logs')
for i in range(10):
    writer.add_image("show_dataset", X[i], i)
    print(y[i], "\t")
writer.close()

batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
train_iter2 = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=1)

timer_1 = time.time()
for X, y in train_iter:
    continue
timer_2 = time.time()
print("worker 4 cost %.4f" % (timer_2 - timer_1))
timer_1 = time.time()
for X, y in train_iter2:
    continue
timer_2 = time.time()
print("worker 4 cost %.4f" % (timer_2 - timer_1))
