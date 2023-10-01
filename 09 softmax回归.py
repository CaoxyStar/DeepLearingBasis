import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

# 数据集
batch_size = 256
trans = transforms.ToTensor()
train_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True)
train_Dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
test_Dataloader = data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)

# 网络模型
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

# 优化方法
def optimizer(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum() / len(y))

writer = SummaryWriter("logs")

# 训练
num_epochs = 20
for epoch in range(num_epochs):
    for X, y in train_Dataloader:
        y_hat = net(X)
        l = cross_entropy(y_hat, y)
        l.sum().backward()
        optimizer((W, b), 0.1, batch_size)
    with torch.no_grad():
        temp = 0
        acc = 0
        loss = 0
        for X, y in test_Dataloader:
            y_hat = net(X)
            loss += cross_entropy(y_hat, y).mean()
            acc += accuracy(y_hat, y)
            temp += 1
    print("epoch %d, accuracy %.5f, loss %.5f" % (epoch+1, acc/temp, loss/temp))
    writer.add_scalar("loss", loss/temp, epoch+1)
    writer.add_scalar("acc", acc/temp, epoch+1)
writer.close()

    

