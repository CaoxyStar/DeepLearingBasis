import pandas as pd
import torch
from torch import nn
import random
from torch.utils.tensorboard import SummaryWriter

# 数据集与预处理
train_dataset = pd.read_csv("data/kaggle_house_pred_train.csv")
test_dataset = pd.read_csv("data/kaggle_house_pred_test.csv")
all_features = pd.concat((train_dataset.iloc[:, 1:-1], test_dataset.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)
n_train = train_dataset.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_dataset.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def data_iter(batch_size, features, labels):
    num_examples = features.shape[0]
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

in_features = train_features.shape[1]

def get_net(num_inputs):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    net.apply(init_weights)
    return net

def log_rmse(y_hat, y):
    clipped_preds = torch.clamp(y_hat, 1, float('inf'))
    loss = nn.MSELoss()
    return torch.sqrt(loss(torch.log(clipped_preds), torch.log(y)))


def get_k_fold_data(k, i, X, y):
    assert k>1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j+1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j==i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def train(net, X_train, y_train, X_valid, y_valid,
          num_epochs, learning_rate, weight_decay, batch_size, k):
    writer = SummaryWriter("logs")
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = nn.MSELoss()
    for epoch in range(num_epochs):
        net.train()
        for X, y in data_iter(batch_size, X_train, y_train):
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
        if X_valid is not None:
            net.eval()
            with torch.no_grad():
                y_train_hat = net(X_train)
                train_loss = log_rmse(y_train_hat, y_train)
                y_valid_hat = net(X_valid)
                valid_loss = log_rmse(y_valid_hat, y_valid)
                writer.add_scalar("fold-" + str(k + 1) + " train_loss", train_loss, epoch)
                writer.add_scalar("fold-" + str(k + 1) + " valid_loss", valid_loss, epoch)
    writer.close()
    return train_loss, valid_loss

def k_fold(k, X_data, y_data, num_epochs, learning_rate,
           weight_decay, batch_size, num_inputs):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, X_data, y_data)
        net = get_net(num_inputs)
        train_loss, valid_loss = train(net, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate, weight_decay, batch_size, i)
        train_l_sum += train_loss
        valid_l_sum += valid_loss
        print("fold-%d: train_loss %.5f, valid_loss %.5f" % (i+1, train_loss, valid_loss))
    train_l_sum /= k
    valid_l_sum /= k
    print("%d-fold: train_loss_avg %.5f, valid_loss_avg %.5f" % (k, train_l_sum, valid_l_sum))

num_epochs = 600
learning_rate = 0.5
weight_decay = 0.2
batch_size = 64

k_fold(5, train_features, train_labels, num_epochs, learning_rate, weight_decay, batch_size, in_features)


