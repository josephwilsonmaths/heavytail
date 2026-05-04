import torch
import numpy as np
import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def train_loop(dataloader, model, loss_fn, optimizer, scheduler, early_stopping = None, train_mode=True):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    if train_mode:
        model.train()
    else:
        model.eval()
    device = next(model.parameters()).device
    size = len(dataloader.dataset)
    num_batches = len(dataloader)    
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluate metrics
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    if scheduler is not None:
        scheduler.step()

    train_loss /= num_batches
    correct /= size

    if early_stopping is not None:
        if train_loss < early_stopping:
            return train_loss, correct

    return train_loss, correct

def train_loop_binary(dataloader, model, loss_fn, optimizer, scheduler, train_mode=True):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    if train_mode:
        model.train()
    else:
        model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)    
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.to(dtype=torch.float64).unsqueeze(1))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluate metrics
        train_loss += loss.item()
        correct += (pred.sigmoid().round().squeeze(1) == y).type(torch.float).sum().item()

    if scheduler is not None:
        scheduler.step()

    train_loss /= num_batches
    correct /= size

    return train_loss, correct


def test_loop(dataloader, model, loss_fn, verbose=False):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    return test_loss, correct

def training(train_loader, 
             test_loader, 
             model, 
             loss_fn, 
             optimizer, 
             scheduler = None, 
             epochs: int = 50, 
             early_stopping = None, 
             verbose: bool = False, 
             progress_bar = True, 
             train_mode = True):
    '''
    Training function. Will train and test, and will report metrics.

    Outputs:
        - train_loss
        - train_acc
        - test_loss
        - test_acc
    '''
    
    if progress_bar:
        pbar = tqdm.trange(epochs)
    else:
        pbar = range(epochs)

    for epoch in pbar:
        train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer, scheduler, early_stopping=early_stopping, train_mode=train_mode)
        test_loss, test_acc = test_loop(test_loader, model, loss_fn)
        if verbose:
            if epoch % 1 == 0:
                print("Epoch {} of {}".format(epoch,epochs))
                print("Training loss = {:.4f}".format(train_loss))
                print("Train accuracy = {:.1f}%".format(100*train_acc))
                print("Test loss = {:.4f}".format(test_loss))
                print("Test accuracy = {:.1f}%".format(100*test_acc))
                print("\n -------------------------------------")
        if early_stopping is not None:
            if train_loss < early_stopping:
                print(f"Early stopping at epoch {epoch} with training loss {train_loss:.4f}!")
                return train_loss, train_acc, test_loss, test_acc
    if verbose:
        print("Done!")
        print("Final training loss = {:.4f}".format(train_loss))
        print("Final train accuracy = {:.1f}%".format(100*train_acc))
        print("Final test loss = {:.4f}".format(test_loss))
        print("Final test accuracy = {:.1f}%".format(100*test_acc))
    return train_loss, train_acc, test_loss, test_acc