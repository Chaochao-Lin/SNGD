import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import torch

def batch_norm_1d(x, gamma, beta, is_training, moving_mean, moving_var, moving_momentum=0.1):
    eps = 1e-5
    x_mean = torch.mean(x, dim=0, keepdim=True) # 保留维度进行 broadcast
    x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
    if is_training:
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        moving_mean[:] = moving_momentum * moving_mean + (1. - moving_momentum) * x_mean
        moving_var[:] = moving_momentum * moving_var + (1. - moving_momentum) * x_var
    else:
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 数据预处理，标准化
    x = x.reshape((-1,))  # 拉平
    x = torch.from_numpy(x)
    return x


train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)  # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)


class multi_network(nn.Module):
    def __init__(self):
        super(multi_network, self).__init__()
        self.layer1 = nn.Linear(784, 100)
        self.relu = nn.ReLU(True)
        self.layer2 = nn.Linear(100, 10)

        self.gamma = nn.Parameter(torch.randn(100))
        self.beta = nn.Parameter(torch.randn(100))

        self.moving_mean = Variable(torch.zeros(100))
        self.moving_mean = self.moving_mean.cuda()
        self.moving_var = Variable(torch.zeros(100))
        self.moving_var = self.moving_var.cuda()

    def forward(self, x, is_train=True):
        x = self.layer1(x)
        x = batch_norm_1d(x, self.gamma, self.beta, is_train, self.moving_mean, self.moving_var)
        x = self.relu(x)
        x = self.layer2(x)
        return x


net = multi_network()
# 定义 loss 函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1

from datetime import datetime
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


# 定义训练函数
def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss += loss.item()
        train_acc += get_acc(output, label)

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    if valid_data is not None:
        valid_loss = 0
        valid_acc = 0
        net = net.eval()
        for im, label in valid_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda(), volatile=True)
                label = Variable(label.cuda(), volatile=True)
            else:
                im = Variable(im, volatile=True)
                label = Variable(label, volatile=True)
            output = net(im)
            loss = criterion(output, label)
            valid_loss += loss.item()
            valid_acc += get_acc(output, label)
        epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))
    else:
        epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                     (epoch, train_loss / len(train_data),
                      train_acc / len(train_data)))
    prev_time = cur_time
    print(epoch_str + time_str)


train(net, train_data, test_data, 10, optimizer, criterion)

print(net.moving_mean[:10])
no_bn_net = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(True),
    nn.Linear(100, 10)
)

optimizer = torch.optim.SGD(no_bn_net.parameters(), 1e-1) # 使用随机梯度下降，学习率 0.1
train(no_bn_net, train_data, test_data, 10, optimizer, criterion)