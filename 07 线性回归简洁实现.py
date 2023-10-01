import torch
from torch.utils import data
from torch import nn

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3, 4], dtype=torch.float)
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 数据集获取，从dataset到dataloader
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 网络模型搭建
net =nn.Sequential(nn.Linear(3, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 设计损失函数
loss = nn.MSELoss()
# 选择优化方法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# 进行训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print("epoch %d, loss %.7f" % (epoch+1, l))
# 打印结果
print(net[0].weight.data)
print(net[0].bias.data)