from typing import Optional

import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

from utils.unpacking import basic_unpack


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def occurence_accuracy(pred, correct):
    return torch.argmax(pred, dim=1) == correct


def train_one_epoch(model, train_dataloader, criterion, optimizer, model_wrapper, device="cuda:0",
                    verbose=False) -> float:
    """
    Trains a model for a single run of the dataloader
    :param model: model to train
    :param train_dataloader: loader with training data
    :param criterion: loss criterion
    :param optimizer: weight optimizer
    :param model_wrapper: wrapper to get a loss from the model
    :param device: computation device
    :param verbose: whether to print tqdm bar
    :return: mean loss across all batches
    """
    model.train()
    losses = []

    for obj in tqdm(train_dataloader, disable=not verbose):
        optimizer.zero_grad()
        loss = model_wrapper(obj, device, model, criterion)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.mean(losses).item()


@torch.no_grad()
def predict(model, val_dataloader, criterion, device="cuda:0", verbose=False) -> (np.array, np.array):
    """
    Predicts the results for a loader
    :param model: model to use in prediction
    :param val_dataloader: dataloader with data
    :param criterion: loss to evaluate on a model
    :param device: computation device
    :param verbose: whether to print tqdm bar
    :return: losses for each batch, accuracies for each object
    """
    model.to(device)
    model.eval()
    losses = []
    accuracies = []

    for pic, task, arg, res in tqdm(val_dataloader, disable=not verbose):
        pic, task, arg, res = pic.to(device), task.to(device), arg.to(device), res.to(device)
        prediction = model(pic, task, arg)
        accuracies.extend(occurence_accuracy(prediction, res).cpu())
        losses.append(criterion(prediction, res).item())

    return np.array(losses), np.array(accuracies)


def train(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, model_wrapper=basic_unpack,
          device="cuda:0", n_epochs=10, scheduler=None, verbose=False, save_dir: Path = None, save_best=False,
          model_name: str = None, show_tqdm=False) -> (list[float], list[float], list[float]):
    """
    Train the model
    :param model: model to train
    :param train_dataloader: loader with training data
    :param val_dataloader: loader with validation data
    :param test_dataloader: loader with testing data
    :param criterion: loss criterion
    :param model_wrapper: wrapper function to get a single loss tensor from the model
    :param optimizer: weight optimizer
    :param device: computation device
    :param n_epochs: number of training epochs
    :param scheduler: scheduler for the learning rate (should be set to 'max' mode)
    :param verbose: whether to print end-of-epoch loss messages
    :param save_dir: directory to save model checkpoints to
    :param save_best: whether to save the best model
    :param model_name: name of the model for Tensorboard logging and saving
    :param show_tqdm: whether to show tqdm progress bars

    :returns: a tuple of train, validation, and test loss histories through the epochs
    """
    if model_name is None:
        model_name = type(model).__name__

    if save_best and save_dir is None:
        raise ValueError("Please provide a directory to save the models to")

    model.to(device)
    train_loss_history = []
    val_loss_history = []
    test_loss_history = []
    epoch_dict: {int: Path} = {}
    val_loss = -1
    model_save_dir = None

    if save_best:
        model_save_dir = save_dir / model_name
        model_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training of {model_name}")
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, model_wrapper, device, show_tqdm)
        val_losses, val_acc = predict(model, val_dataloader, criterion, device, show_tqdm)
        test_losses, test_acc = predict(model, test_dataloader, criterion, device, show_tqdm)

        val_loss = np.mean(val_losses).item()
        test_loss = np.mean(test_losses).item()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        test_loss_history.append(test_loss)

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss, 'val_acc': val_acc.mean(),
                   'test_acc': test_acc.mean()})
        if verbose:
            print(f"Epoch {epoch + 1}, Train loss: {train_loss:.4f}, Val loss/acc: {val_loss:.4f}/{val_acc.mean():.4f},"
                  f" Test loss/acc: {test_loss:.4f}/{test_acc.mean():.4f}")

        if scheduler:
            scheduler.step(val_loss)

        if save_best:
            torch.save(model.state_dict(), model_save_dir / f"state.e{epoch}.pt")
            epoch_dict[epoch] = model_save_dir / f"state.e{epoch}.pt"

    if save_best:
        best_epoch = np.argmin(val_loss_history)
        for epoch, path in epoch_dict.items():
            if epoch == best_epoch:
                path.rename(save_dir / f"{model_name}_best.pt")
            else:
                path.unlink()
        model_save_dir.rmdir()

    return train_loss_history, val_loss_history, test_loss_history
