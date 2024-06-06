import torch
import torch.nn as nn


class Net256(nn.Module):
    def __init__(self, n_classes):
        super(Net256, self).__init__()

        # input layer takes the first 256 principal components
        self.lin_in = nn.Linear(256, 256)
        nn.init.xavier_uniform_(self.lin_in.weight)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()

        # 1. hidden layer
        self.hidden1 = nn.Linear(256, 80)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.bn2 = nn.BatchNorm1d(80)
        self.relu2 = nn.ReLU()

        # 2. hidden layer
        self.hidden2 = nn.Linear(80, 40)
        nn.init.xavier_uniform_(self.hidden2.weight)
        self.bn3 = nn.BatchNorm1d(40)
        self.relu3 = nn.ReLU()

        # output layer
        self.lin_out = nn.Linear(40, n_classes)

    def forward(self, x):
        # input layer
        x = self.relu1(self.bn1(self.lin_in(x)))

        # first hidden layer
        x = self.relu2(self.bn2(self.hidden1(x)))

        # second hidden layer
        x = self.relu3(self.bn3(self.hidden2(x)))

        # output layer. We don't add a softmax layer because we weill use
        # cross entropy loss as a loss funtion
        out = self.lin_out(x)

        return out


def train_model(model, optimizer, loss_func, dataloader):
    # switch model in taining mode for training
    model.train()
    
    size = len(dataloader.dataset)

    training_loss = 0.0
    
    for i, (X, y) in enumerate(dataloader):
        output = model(X)
        optimizer.zero_grad()
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            current = i * dataloader.batch_size + len(X)
            print(f"Loss: {loss.item():>8f}  [{current:>5d}/{size:>5d}]")


def test_model(model, loss_func, dataloader):
    # switch model to evaluation mode for prediction
    model.eval()

    size = dataloader.batch_size
    num_batches = len(dataloader)
    test_loss = 0.0
    yhat_label = list()
    
    with torch.no_grad():
        for X, y in dataloader:
            yhat = model(X)
            test_loss += loss_func(yhat, y).item()
            yhat_label.extend(yhat.argmax(1))
    
    avg_loss = test_loss / num_batches
    
    return avg_loss, yhat_label

