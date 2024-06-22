"""
Python module defining the fee-forward neural network used to predict MoA 
based on image embeddin vectors. The neural net is implemnted in PyTorch.
In addition to the model class the module contains training, test and
validation functions.
"""
import torch
import torch.nn as nn


class Net3072(nn.Module):
    """
    Class implementing Pytroch Module. It defines the neural network used to prdict
    MoA. The input is the full embedding vector.
    The model is inspired by the fully connected layer of the VGG model.
    """
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
        """
        Forward path of the model.
        Params:
            -x (tensor): image embedding vector
        """
        out = self.linear_stack(x)
        return out


def train_model(model, optimizer, loss_func, dataloader, device):
    """
    Train a neural network using batches of image embedding vectors.
    Params:
        - model (PyTorch nn.Module): model which gets trained.
        - optimizer (PyTorch optimizer): optimizer used to calcualte the gardients.
        - loss_function (PyTroch loss function): loss function to be minimized.
        - dataloader (PyTorch Dataloader): dataloader containg the training dataset.
        - device (PyTroch device): device use in training, can be "cpu" or "gpu". 
    """
    # Switch model in taining mode for training
    model.train()
    
    training_loss = 0.0
    training_corr = 0.0
    
    # Train model on batches of images
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        # forward path
        yhat = model(X)
        loss = loss_func(yhat, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item() * X.size(0)
        _, pred = torch.max(yhat.data, 1)
        training_corr += (pred == y.argmax(1)).sum().item()
    
    return training_loss, training_corr


def valid_model(model, loss_func, dataloader, device):
    """
    Validate/test a neural network using batches of image embedding vectors.
    Params:
        - model (PyTorch nn.Module): model which gets trained.
        - loss_function (PyTroch loss function): loss function to be minimized.
        - dataloader (PyTorch Dataloader): dataloader containg the validation/test dataset.
        - device (PyTroch device): device use in training, can be "cpu" or "gpu". 
    """
    # Switch model in evaluation mode for doing predictions
    model.eval()
    
    validation_loss = 0.0
    validation_corr = 0.0

    # Disable gradient calculation in evaluation mode
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            # forward path
            yhat = model(X)
            loss = loss_func(yhat, y)
            
            validation_loss += loss.item() * X.size(0)
            validation_corr += (yhat.argmax(1) == y.argmax(1)).sum().item()
    
    return validation_loss, validation_corr


def test_model(model, loss_func, dataloader, device):
    """
    Special test function returning the pedicted labels (MoA). Can be used with
    scikit-learn metics functions.
    Params:
        - model (PyTorch nn.Module): model which gets trained.
        - loss_function (PyTroch loss function): loss function to be minimized.
        - dataloader (PyTorch Dataloader): dataloader containg the validation/test dataset.
        - device (PyTroch device): device use in training, can be "cpu" or "gpu". 
    """
    # switch model to evaluation mode for prediction
    model.eval()

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

