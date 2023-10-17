import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from utils.unpacking import Unpack


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def timestamp():
    return str(datetime.now())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def occurence_accuracy(pred, correct):
    return torch.argmax(pred, dim=1) == correct


def binary_accuracy(pred, correct):
    return (pred >= 0.5) == correct


class TopKAccuracy:
    def __init__(self, k):
        self.k = k

    def __call__(self, pred, correct):
        dim = pred.shape[-1]
        thresh = torch.kthvalue(pred, dim=1, k=dim - self.k + 1, keepdim=True).values
        return ((pred >= thresh) * correct).sum(dim=1) / self.k


def train_one_epoch(model, train_dataloader, optimizer, model_wrapper, scaler, verbose=False) -> dict:
    """
    Trains a model for a single run of the dataloader
    :param model: model to train
    :param train_dataloader: loader with training data
    :param optimizer: weight optimizer
    :param model_wrapper: wrapper to get a loss from the model
    :param scaler: loss gradient scaler for mixed precision
    :param verbose: whether to print tqdm bar
    :return: metrics
    """
    model.train()
    metrics = None
    for obj in tqdm(train_dataloader, disable=not verbose):
        optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            total_loss, batch_metrics = model_wrapper(obj, model)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if metrics is None:
            metrics = {k: [] for k in batch_metrics.keys()}

        for k, v in batch_metrics.items():
            if isinstance(v, torch.Tensor):
                if v.ndim > 0:
                    metrics[k].extend(v.cpu())
                else:
                    metrics[k].append(v.item())
            else:
                metrics[k].append(v)

    return {k: np.mean(v).item() for k, v in metrics.items()}


@torch.no_grad()
def predict(model, val_dataloader, model_wrapper, verbose=False) -> dict:
    """
    Predicts the results for a loader
    :param model: model to use in prediction
    :param val_dataloader: dataloader with data
    :param model_wrapper: wrapper to get metrics from the model
    :param verbose: whether to print tqdm bar
    :return: metrics dictionary
    """
    model.eval()
    metrics = None

    for obj in tqdm(val_dataloader, disable=not verbose):
        batch_metrics = model_wrapper.evaluate(obj, model)

        if metrics is None:
            metrics = {k: [] for k in batch_metrics.keys()}

        for k, v in batch_metrics.items():
            if v.ndim > 0:
                metrics[k].extend(v.cpu())
            else:
                metrics[k].append(v.item())

    return {k: np.mean(v).item() for k, v in metrics.items()}


def train(model, train_dataloader, val_dataloader, test_dataloader, optimizer: torch.optim.Optimizer,
          model_wrapper: Unpack, device, n_epochs, scheduler=None, verbose=False, save_dir: Path = None,
          save_best=False, save_every=2, load_from: Path = None, model_name: str = None, show_tqdm=False,
          use_scaler=False):
    """
    Train the model
    :param model: model to train
    :param train_dataloader: loader with training data
    :param val_dataloader: loader with validation data
    :param test_dataloader: loader with testing data
    :param model_wrapper: wrapper function to get a single loss tensor from the model
    :param optimizer: weight optimizer
    :param device: computation device
    :param n_epochs: number of training epochs
    :param scheduler: scheduler for the learning rate (should be set to 'max' mode)
    :param verbose: whether to print end-of-epoch loss messages
    :param save_dir: directory to save model checkpoints to
    :param save_best: whether to save the best model
    :param load_from: loads the saved state from a path
    :param save_every: how often to save
    :param model_name: name of the model for Tensorboard logging and saving
    :param show_tqdm: whether to show tqdm progress bars
    :param use_scaler: whether to train in float16

    :returns: a tuple of train, validation, and test loss histories through the epochs
    """
    if model_name is None:
        model_name = type(model).__name__

    if save_best and save_dir is None:
        raise ValueError("Please provide a directory to save the models to")

    model.to(device)
    val_loss_history = []
    epoch_dict: {int: Path} = {}
    model_save_dir = None

    if save_best:
        model_save_dir = save_dir / model_name
        model_save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if load_from is not None:
        checkpoint = torch.load(load_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']

    scaler = GradScaler(enabled=use_scaler, growth_interval=200)

    print(f"Starting training of {model_name}")
    for epoch in range(start_epoch, start_epoch + n_epochs):

        train_metrics = train_one_epoch(model, train_dataloader, optimizer, model_wrapper, scaler, show_tqdm)
        val_metrics = predict(model, val_dataloader, model_wrapper, show_tqdm)
        test_metrics = predict(model, test_dataloader, model_wrapper, show_tqdm)

        wandb.log(
            {f"train_{k}": v for k, v in train_metrics.items()} | {
                f"val_{k}": v for k, v in val_metrics.items()} | {
                f"test_{k}": v for k, v in test_metrics.items()}
        )

        val_loss_history.append(val_metrics[model_wrapper.main_key])

        if verbose:
            train_string = "/".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            val_string = "/".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            test_string = "/".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
            print(
                f"[{timestamp()}] Epoch {epoch + 1}:\n\tTrain metrics: {train_string}\n"
                f"\tVal metrics: {val_string}\n\tTest metrics: {test_string}\n")

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(train_metrics[model_wrapper.main_key])
            else:
                scheduler.step()

        if save_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
            }, model_save_dir / f"state.e{epoch}.pt")
            epoch_dict[epoch] = model_save_dir / f"state.e{epoch}.pt"

    if save_best:
        torch.save({
            'epoch': n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
        }, save_dir / f"{model_name}_last.pt")
        best_epoch = np.argmin(val_loss_history[::save_every])
        for epoch, path in epoch_dict.items():
            if epoch == best_epoch:
                path.rename(save_dir / f"{model_name}_best.pt")
            else:
                path.unlink()
        model_save_dir.rmdir()
