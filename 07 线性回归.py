import random
import torch
from matplotlib import pyplot as plt

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3, 4], dtype=torch.float)
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

"""
#pyplot绘制散点图
fig = plt.figure(num=1)
ax = fig.add_subplot(111)
ax.scatter(features[:, 1], labels)
ax.set_title('Scatter')
plt.show()
"""

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

"""
batch_size = 10
for X,y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
"""

w = torch.normal(0, 0.01, size=(3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linear_reg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2/2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linear_reg
loss = squared_loss
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print("epoch %d, loss %.7f" % (epoch+1, train_l.mean()))
print(w)
print(b)
