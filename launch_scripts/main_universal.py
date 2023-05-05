import os
import sys
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, CosineAnnealingWarmRestarts, \
    SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTConfig

sys.path.insert(1, str(Path(__file__).parents[1]))

from datasets.emnist import Emnist6LeftRight, Emnist24Directions, EmnistExistence, EmnistLocation
from datasets.persons import PersonsClassification, PersonsClassificationOccurrence
from modules.multitask import EncoderBUTD, EncDecBUTD, MixingBUTD
from utils.training import set_random_seed, train, occurence_accuracy, TopKAccuracy, count_parameters
from utils.unpacking import BasicUnpack, TwoLossUnpack


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run(cfg: DictConfig):
    torch.autograd.set_detect_anomaly(cfg.detect_anomalies, check_nan=True)
    torch.backends.cuda.matmul.allow_tf32 = cfg.use_tf32
    torch.backends.cudnn.allow_tf32 = cfg.use_tf32

    set_random_seed(cfg.seed)

    model = create_model(cfg)

    cfg.model_size = count_parameters(model)

    if cfg.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        print(OmegaConf.to_yaml(cfg))

    train_data, val_data, test_data, wrapper = parse_task(cfg)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=cfg.pin_memory)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=cfg.pin_memory)

    optimizer, scheduler = create_optimizer(cfg, model)

    wandb.init(project="BU-TD Benchmark", entity="avagr", name=cfg.name, group=cfg.task.name,
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    wandb.watch(model)

    train(model, train_loader, val_loader, test_loader, optimizer, wrapper, n_epochs=cfg.num_epochs, device=cfg.device,
          scheduler=scheduler, verbose=cfg.print_epochs, save_dir=Path(cfg.save_directory), save_best=cfg.save_best,
          model_name=cfg.name, show_tqdm=cfg.show_tqdm, use_scaler=cfg.mixed_precision)


def create_optimizer(cfg, model):
    match cfg.optim:
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        case "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        case "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        case _:
            raise ValueError(f"Optimizer {cfg.optim} not supported")

    schedulers = []
    if cfg.warmup_epochs > 0:
        schedulers.append(
            LinearLR(optimizer, start_factor=0.001, end_factor=1, total_iters=cfg.warmup_epochs, verbose=True))
    if cfg.annealing_t0 > 0:
        schedulers.append(CosineAnnealingWarmRestarts(optimizer, T_0=cfg.annealing_t0, verbose=True))
    if cfg.scheduler_patience > 0:
        schedulers.append(ReduceLROnPlateau(optimizer, patience=cfg.scheduler_patience, verbose=True))
    match len(schedulers):
        case 0:
            scheduler = None
        case 1:
            scheduler = schedulers[0]
        case _:
            scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=[cfg.warmup_epochs])
    return optimizer, scheduler


def create_model(cfg):
    match cfg.model.name:
        case "encoder":
            model = EncoderBUTD(cfg.task.num_tasks, cfg.task.num_args, cfg.task.num_classes, enc_config=ViTConfig(
                hidden_size=cfg.model.hidden_size, num_hidden_layers=cfg.model.num_layers,
                intermediate_size=cfg.model.intermediate_size, num_channels=cfg.task.num_channels,
                patch_size=cfg.model.patch_size, num_attention_heads=cfg.model.num_heads,
                image_size=(cfg.task.image_h, cfg.task.image_w)),
                                use_sinusoidal=cfg.model.sinusoidal)

            if cfg.model.initialize_from is not None:
                state_dict = torch.load(cfg.model.initialize_from)
                # print(state_dict.keys())
                del [state_dict['classifier.weight'], state_dict['classifier.bias'],
                     state_dict['argument_embeddings.weight'], state_dict['argument_embeddings.bias'],
                     state_dict['task_embeddings.weight'], state_dict['task_embeddings.bias']]
                model.load_state_dict(state_dict, strict=False)

        case "enc-dec":
            model = EncDecBUTD(cfg.task.num_tasks, cfg.task.num_args, cfg.task.num_classes, enc_config=ViTConfig(
                hidden_size=cfg.model.encoder_hidden_size, num_hidden_layers=cfg.model.num_encoder_layers,
                intermediate_size=cfg.model.encoder_intermediate_size, num_channels=cfg.task.num_channels,
            ), dec_config=ViTConfig(
                hidden_size=cfg.model.decoder_hidden_size, num_hidden_layers=cfg.model.num_decoder_layers,
                intermediate_size=cfg.model.decoder_intermediate_size
            ), use_butd=cfg.model.use_butd, aux_head_size=cfg.model.aux_head_size)

        case "mixing":
            model = MixingBUTD(cfg.task.num_tasks, cfg.task.num_args, cfg.task.num_classes, config=ViTConfig(
                hidden_size=cfg.model.hidden_size, num_hidden_layers=cfg.model.num_layers,
                num_channels=cfg.task.num_channels, intermediate_size=cfg.model.intermediate_size),
                               total_token_size=cfg.model.hidden_size * 197,
                               use_self_attention=cfg.model.use_self_attention, mix_with=cfg.model.mix_layer)

        case "classifier":
            model = EncoderBUTD(cfg.task.num_tasks, cfg.task.num_args, cfg.task.num_classes, enc_config=ViTConfig(
                hidden_size=cfg.model.hidden_size, num_hidden_layers=cfg.model.num_layers,
                intermediate_size=cfg.model.intermediate_size, num_channels=cfg.task.num_channels,
            ), use_sinusoidal=cfg.model.sinusoidal)

        case "locator":
            model = EncoderBUTD(cfg.task.num_tasks, cfg.task.num_args, cfg.task.num_classes, enc_config=ViTConfig(
                hidden_size=cfg.model.hidden_size, num_hidden_layers=cfg.model.num_layers,
                intermediate_size=cfg.model.intermediate_size, num_channels=cfg.task.num_channels,
                patch_size=cfg.model.patch_size),
                                use_sinusoidal=cfg.model.sinusoidal)

        case _:
            raise ValueError(f"Model {cfg.model.name} is not supported")
    return model


def parse_task(cfg):
    match cfg.task.name:
        case "EMNIST-6":
            loss = nn.CrossEntropyLoss()
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
            wrapper = BasicUnpack(cfg.device, loss, occurence_accuracy)
            train_data = Emnist6LeftRight(os.path.join(cfg.task.root_path, "train"), cfg.task.num_classes, transform,
                                          cfg.task.dataset_size)
            val_data = Emnist6LeftRight(os.path.join(cfg.task.root_path, "val"), cfg.task.num_classes, transform)
            test_data = Emnist6LeftRight(os.path.join(cfg.task.root_path, "test"), cfg.task.num_classes, transform)

        case "EMNIST-24":
            loss = nn.CrossEntropyLoss()
            transform = transforms.Compose([
                # transforms.Resize((224, 224)), todo
                transforms.ToTensor(),
                # transforms.Normalize(mean=(cfg.task.mean,), std=(cfg.task.std,))
            ])
            wrapper = BasicUnpack(cfg.device, loss, occurence_accuracy)
            train_data = Emnist24Directions(os.path.join(cfg.task.root_path, "train"), cfg.task.num_classes, transform,
                                            cfg.task.dataset_size)
            val_data = Emnist24Directions(os.path.join(cfg.task.root_path, "val"), cfg.task.num_classes, transform)
            test_data = Emnist24Directions(os.path.join(cfg.task.root_path, "test"), cfg.task.num_classes, transform)

        case "EMNIST-24-Classification":
            loss = nn.BCEWithLogitsLoss()
            wrapper = BasicUnpack(cfg.device, loss, TopKAccuracy(k=24))
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                            transforms.Normalize(mean=(cfg.task.mean,), std=(cfg.task.std,))])
            train_data = EmnistExistence(os.path.join(cfg.task.root_path, "train"), cfg.task.num_classes, transform,
                                         size_limit=cfg.task.dataset_size, num_tasks=cfg.task.num_tasks)
            val_data = EmnistExistence(os.path.join(cfg.task.root_path, "val"), cfg.task.num_classes, transform,
                                       num_tasks=cfg.task.num_tasks)
            test_data = EmnistExistence(os.path.join(cfg.task.root_path, "test"), cfg.task.num_classes, transform,
                                        num_tasks=cfg.task.num_tasks)

        case "EMNIST-24-Location":
            loss = nn.CrossEntropyLoss()
            wrapper = BasicUnpack(cfg.device, loss, occurence_accuracy)
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                            transforms.Normalize(mean=(cfg.task.mean,), std=(cfg.task.std,))])
            train_data = EmnistLocation(os.path.join(cfg.task.root_path, "train"), cfg.task.num_classes, transform,
                                        size_limit=cfg.task.dataset_size, num_tasks=cfg.task.num_tasks)
            val_data = EmnistLocation(os.path.join(cfg.task.root_path, "val"), cfg.task.num_classes, transform,
                                      num_tasks=cfg.task.num_tasks)
            test_data = EmnistLocation(os.path.join(cfg.task.root_path, "test"), cfg.task.num_classes, transform,
                                       num_tasks=cfg.task.num_tasks)

        case "Persons":
            loss = nn.CrossEntropyLoss()
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            if not cfg.task.occurrence_loss:
                wrapper = BasicUnpack(cfg.device, loss, occurence_accuracy)
                train_data = PersonsClassification(os.path.join(cfg.task.root_path, "train"), cfg.task.num_tasks,
                                                   cfg.task.num_args, transform, size_limit=cfg.task.dataset_size)
                val_data = PersonsClassification(os.path.join(cfg.task.root_path, "val"), cfg.task.num_tasks,
                                                 cfg.task.num_args, transform, size_limit=cfg.task.dataset_size)
                test_data = PersonsClassification(os.path.join(cfg.task.root_path, "test"), cfg.task.num_tasks,
                                                  cfg.task.num_args, transform, size_limit=cfg.task.dataset_size)
            else:
                wrapper = TwoLossUnpack(cfg.device, loss, nn.BCEWithLogitsLoss(), cfg.task.loss_fraction,
                                        occurence_accuracy, TopKAccuracy(k=2), ["feat", "occ"])
                train_data = PersonsClassificationOccurrence(os.path.join(cfg.task.root_path, "train"),
                                                             cfg.task.num_tasks,
                                                             cfg.task.num_args, transform,
                                                             size_limit=cfg.task.dataset_size)
                val_data = PersonsClassificationOccurrence(os.path.join(cfg.task.root_path, "val"), cfg.task.num_tasks,
                                                           cfg.task.num_args, transform,
                                                           size_limit=cfg.task.dataset_size)
                test_data = PersonsClassificationOccurrence(os.path.join(cfg.task.root_path, "test"),
                                                            cfg.task.num_tasks,
                                                            cfg.task.num_args, transform,
                                                            size_limit=cfg.task.dataset_size)

        case _:
            raise ValueError(f"Dataset {cfg.task.name} is not supported")
    return train_data, val_data, test_data, wrapper


if __name__ == '__main__':
    run()
