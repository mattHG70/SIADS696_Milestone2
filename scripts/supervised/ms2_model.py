import torch
import torch.nn as nn

        
class Net256(nn.Module):
    def __init__(self, n_classes, input_size, hidden_size):
        super(Net256, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        out = self.linear_stack(x)
        return out


class Net3072(nn.Module):
    def __init__(self, n_classes):
        super(Net3072, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        out = self.linear_stack(x)
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

