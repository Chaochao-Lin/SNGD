import torch
from torch.autograd import Variable
def train(net, train_data, valid_data, num_epochs, optimizer, criterion)
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            im = Variable(im.cuda())
            label = Variable(label.cuda())

            output = net(im)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
