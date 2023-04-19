import sys
import os
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, ChainedScheduler, CosineAnnealingWarmRestarts, \
    SequentialLR
from torchvision import transforms
from transformers import ViTConfig

sys.path.insert(1, str(Path(__file__).parents[1]))

from datasets.emnist import Emnist6LeftRight, Emnist24Directions
from modules.multitask import EncoderBUTD, EncDecBUTD, MixingBUTD
from utils.training import set_random_seed, train


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run(cfg: DictConfig):
    if cfg.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        print(OmegaConf.to_yaml(cfg))

    torch.autograd.set_detect_anomaly(cfg.detect_anomalies, check_nan=True)

    torch.backends.cuda.matmul.allow_tf32 = cfg.use_tf32
    torch.backends.cudnn.allow_tf32 = cfg.use_tf32

    set_random_seed(cfg.seed)

    model = create_model(cfg)

    train_data, val_data, test_data, loss = parse_task(cfg)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=cfg.pin_memory)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=cfg.pin_memory)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.001, end_factor=1, total_iters=cfg.warmup_epochs, verbose=True),
        CosineAnnealingWarmRestarts(optimizer, T_0=cfg.annealing_t0, verbose=True)
    ], milestones=[cfg.warmup_epochs])

    wandb.init(project="BU-TD Benchmark", entity="avagr", name=cfg.name, group=cfg.task.name,
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    wandb.define_metric("val_loss", summary="min")
    wandb.define_metric("test_loss", summary="min")
    wandb.define_metric("val_acc", summary="max")
    wandb.define_metric("test_acc", summary="max")
    wandb.watch(model)

    _, _, _ = train(model, train_loader, val_loader, test_loader, loss, optimizer, n_epochs=cfg.num_epochs,
                    scheduler=scheduler, verbose=cfg.print_epochs, save_dir=Path(cfg.save_directory),
                    save_best=cfg.save_best, model_name=cfg.name, show_tqdm=cfg.show_tqdm,
                    use_scaler=cfg.mixed_precision)


def create_model(cfg):
    match cfg.model.name:
        case "encoder":
            model = EncoderBUTD(cfg.task.num_tasks, cfg.task.num_classes, enc_config=ViTConfig(
                hidden_size=cfg.model.hidden_size, num_hidden_layers=cfg.model.num_layers,
                intermediate_size=cfg.model.intermediate_size
            ))

        case "enc-dec":
            model = EncDecBUTD(cfg.task.num_tasks, cfg.task.num_classes, enc_config=ViTConfig(
                hidden_size=cfg.model.encoder_hidden_size, num_hidden_layers=cfg.model.num_encoder_layers,
                intermediate_size=cfg.model.encoder_intermediate_size
            ), dec_config=ViTConfig(
                hidden_size=cfg.model.decoder_hidden_size, num_hidden_layers=cfg.model.num_decoder_layers,
                intermediate_size=cfg.model.decoder_intermediate_size
            ), use_butd=cfg.model.use_butd)

        case "mixing":
            model = MixingBUTD(cfg.task.num_tasks, cfg.task.num_classes, config=ViTConfig(
                hidden_size=cfg.model.hidden_size, num_hidden_layers=cfg.model.num_layers,
                intermediate_size=cfg.model.intermediate_size), total_token_size=cfg.model.hidden_size * 197,
                               use_self_attention=cfg.model.use_self_attention, mix_with=cfg.model.mix_layer)

        case _:
            raise ValueError(f"Model {cfg.model.name} is not supported")
    return model


def parse_task(cfg):
    match cfg.task.name:
        case "EMNIST-6":
            loss = nn.CrossEntropyLoss()
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            train_data = Emnist6LeftRight(os.path.join(cfg.task.root_path, "train"), cfg.task.num_classes, transform,
                                          cfg.task.dataset_size)
            val_data = Emnist6LeftRight(os.path.join(cfg.task.root_path, "val"), cfg.task.num_classes, transform)
            test_data = Emnist6LeftRight(os.path.join(cfg.task.root_path, "test"), cfg.task.num_classes, transform)

        case "EMNIST-24":
            loss = nn.CrossEntropyLoss()
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            train_data = Emnist24Directions(os.path.join(cfg.task.root_path, "train"), cfg.task.num_classes, transform,
                                            cfg.task.dataset_size)
            val_data = Emnist24Directions(os.path.join(cfg.task.root_path, "val"), cfg.task.num_classes, transform)
            test_data = Emnist24Directions(os.path.join(cfg.task.root_path, "test"), cfg.task.num_classes, transform)

        case _:
            raise ValueError(f"Dataset {cfg.task.name} is not supported")
    return train_data, val_data, test_data, loss


if __name__ == '__main__':
    run()
