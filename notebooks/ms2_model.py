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
        self.hidden1 = nn.Linear(256, 256)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.hidden2 = nn.Linear(256, 256)
        nn.init.xavier_uniform_(self.hidden2.weight)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        self.hidden3 = nn.Linear(256, 256)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()

        self.hidden4 = nn.Linear(256, 256)
        nn.init.xavier_uniform_(self.hidden4.weight)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        
        # 1. hidden layer
        self.hidden5 = nn.Linear(256, 128)
        nn.init.xavier_uniform_(self.hidden5.weight)
        self.bn6 = nn.BatchNorm1d(128)
        self.relu6 = nn.ReLU()

        # 2. hidden layer
        self.hidden6 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.hidden6.weight)
        self.bn7 = nn.BatchNorm1d(64)
        self.relu7 = nn.ReLU()

        # output layer
        self.lin_out = nn.Linear(64, n_classes)

    def forward(self, x):
        # input layer
        # x = self.relu1(self.bn1(self.lin_in(x)))

        # first hidden layer
        # x = self.relu2(self.bn2(self.hidden1(x)))

        # x = self.relu3(self.bn3(self.hidden2(x)))

        # x = self.relu4(self.bn4(self.hidden3(x)))
        x = self.relu5(self.bn5(self.hidden4(x)))
        x = self.relu6(self.bn6(self.hidden5(x)))
        # second hidden layer
        x = self.relu7(self.bn7(self.hidden6(x)))

        # output layer. We don't add a softmax layer because we weill use
        # cross entropy loss as a loss funtion
        out = self.lin_out(x)

        return out


def train_model(model, optimizer, loss_func, dataloader, device):
    # switch model in taining mode for training
    model.train()
    
    size = dataloader.batch_size
    num_batches = len(dataloader)
    training_loss = 0.0
    training_corr = 0.0
    
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        yhat = model(X)
        loss = loss_func(yhat, y)
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item() * X.size(0)
        _, pred = torch.max(yhat.data, 1)
        training_corr += (pred == y.argmax(1)).sum().item()
    
    return training_loss, training_corr


def valid_model(model, loss_func, dataloader, device):
    # switch model in taining mode for training
    model.eval()
    
    size = dataloader.batch_size
    num_batches = len(dataloader)
    validation_loss = 0.0
    validation_corr = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            yhat = model(X)
            loss = loss_func(yhat, y)
            
            validation_loss += loss.item() * X.size(0)
            validation_corr += (yhat.argmax(1) == y.argmax(1)).sum().item()
    
    return validation_loss, validation_corr


def test_model(model, loss_func, dataloader, device):
    # switch model to evaluation mode for prediction
    model.eval()

    size = dataloader.batch_size
    num_batches = len(dataloader)
    test_loss = 0.0
    yhat_label = list()
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            yhat = model(X)
            loss = loss_func(yhat, y)
            test_loss += loss.item() * X.size(0)
            yhat_label.extend(yhat.argmax(1))

    # copy list of prediction tensor back to device cpu for further
    # processing using scikit-learn metrics
    return test_loss, torch.tensor(yhat_label, device="cpu")

