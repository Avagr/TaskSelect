import os
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, SequentialLR
from torchvision import transforms
from transformers import ViTConfig, InstructBlipForConditionalGeneration, AutoProcessor

from datasets.cifar import CifarQuery, CifarQueryOccurrence
from datasets.emnist import Emnist6LeftRight, Emnist24Directions, Emnist24DirectionsOccurrence, EmnistExistence, \
    EmnistLocation
from datasets.persons import PersonsClassification, PersonsClassificationOccurrence
from datasets.vqa import GQA
from modules.embeddings import LinearEmbedding, TrickEmbedding
from modules.multitask import EncoderBUTD, EncDecBUTD, MixingBUTD
from utils.training import occurence_accuracy, TopKAccuracy, binary_accuracy
from utils.unpacking import BasicUnpack, TwoLossUnpack, KLUnpack, VQAUnpack, VQACollate


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
    if "num_tasks" in cfg.task:
        match cfg.task_embedding:
            case "linear":
                task_embedding = LinearEmbedding(cfg.task.num_tasks, cfg.model.hidden_size)
            case "trick":
                task_embedding = TrickEmbedding(cfg.task.num_tasks, cfg.model.hidden_size)
            case _:
                raise ValueError(f"Task embedding {cfg.task_embedding} is not supported")

        match cfg.arg_embedding:
            case "linear":
                arg_embedding = LinearEmbedding(cfg.task.num_args, cfg.model.hidden_size)
            case "trick":
                arg_embedding = TrickEmbedding(cfg.task.num_args, cfg.model.hidden_size)
            case _:
                raise ValueError(f"Argument embedding {cfg.arg_embedding} is not supported")
    else:
        task_embedding = None
        arg_embedding = None

    match cfg.model.name:
        case "encoder":
            model = EncoderBUTD(cfg.task.num_classes,
                                enc_config=ViTConfig(hidden_size=cfg.model.hidden_size,
                                                     num_hidden_layers=cfg.model.num_layers,
                                                     intermediate_size=cfg.model.intermediate_size,
                                                     num_channels=cfg.task.num_channels,
                                                     patch_size=cfg.model.patch_size,
                                                     num_attention_heads=cfg.model.num_heads,
                                                     image_size=(cfg.task.image_h, cfg.task.image_w)),
                                task_embedding=task_embedding,
                                argument_embedding=arg_embedding,
                                use_sinusoidal=cfg.model.sinusoidal,
                                aux_head_size=cfg.task.num_aux_classes if cfg.task.occurrence_loss else None)

            if cfg.model.initialize_from is not None:
                state_dict = torch.load(cfg.model.initialize_from)
                # print(state_dict.keys())
                del [state_dict['classifier.weight'], state_dict['classifier.bias'],
                     state_dict['argument_embeddings.weight'], state_dict['argument_embeddings.bias'],
                     state_dict['task_embeddings.weight'], state_dict['task_embeddings.bias']]
                model.load_state_dict(state_dict, strict=False)

        case "enc-dec":
            model = EncDecBUTD(cfg.task.num_classes,
                               enc_config=ViTConfig(
                                   hidden_size=cfg.model.encoder_hidden_size,
                                   num_hidden_layers=cfg.model.num_encoder_layers,
                                   intermediate_size=cfg.model.encoder_intermediate_size,
                                   num_channels=cfg.task.num_channels,
                                   patch_size=cfg.model.patch_size, image_size=(cfg.task.image_h, cfg.task.image_w)
                               ),
                               dec_config=ViTConfig(
                                   hidden_size=cfg.model.decoder_hidden_size,
                                   num_hidden_layers=cfg.model.num_decoder_layers,
                                   intermediate_size=cfg.model.decoder_intermediate_size
                               ),
                               task_embedding=task_embedding,
                               argument_embedding=arg_embedding,
                               use_butd=cfg.model.use_butd, use_sinusoidal=cfg.model.sinusoidal,
                               aux_head_size=cfg.task.num_aux_classes if cfg.task.occurrence_loss else None)

        case "mixing":
            num_patches = (cfg.task.image_h // cfg.model.patch_size) * (cfg.task.image_h // cfg.model.patch_size)
            model = MixingBUTD(cfg.task.num_classes,
                               config=ViTConfig(
                                   hidden_size=cfg.model.hidden_size,
                                   num_hidden_layers=cfg.model.num_layers,
                                   num_channels=cfg.task.num_channels,
                                   intermediate_size=cfg.model.intermediate_size,
                                   patch_size=cfg.model.patch_size,
                                   image_size=(cfg.task.image_h, cfg.task.image_w)),
                               task_embedding=task_embedding,
                               argument_embedding=arg_embedding,
                               total_token_size=cfg.model.hidden_size * (
                                       num_patches + (2 if cfg.task.occurrence_loss else 1)),
                               use_self_attention=cfg.model.use_self_attention, mix_with=cfg.model.mix_layer,
                               aux_head_size=cfg.task.num_aux_classes if cfg.task.occurrence_loss else None,
                               stretch=cfg.model.stretch_task, use_sinusoidal=cfg.model.sinusoidal)

        case "classifier":
            model = EncoderBUTD(cfg.task.num_classes,
                                enc_config=ViTConfig(
                                    hidden_size=cfg.model.hidden_size, num_hidden_layers=cfg.model.num_layers,
                                    intermediate_size=cfg.model.intermediate_size, num_channels=cfg.task.num_channels,
                                ),
                                task_embedding=task_embedding,
                                argument_embedding=arg_embedding,
                                use_sinusoidal=cfg.model.sinusoidal)

        case "locator":
            model = EncoderBUTD(cfg.task.num_classes,
                                enc_config=ViTConfig(
                                    hidden_size=cfg.model.hidden_size, num_hidden_layers=cfg.model.num_layers,
                                    intermediate_size=cfg.model.intermediate_size, num_channels=cfg.task.num_channels,
                                    patch_size=cfg.model.patch_size),
                                task_embedding=task_embedding,
                                argument_embedding=arg_embedding,
                                use_sinusoidal=cfg.model.sinusoidal)

        case "instructblip":
            model = InstructBlipForConditionalGeneration.from_pretrained(cfg.model.path,
                                                                         cache_dir=cfg.model.initialize_from)

        case _:
            raise ValueError(f"Model {cfg.model.name} is not supported")
    return model


def parse_task(cfg):
    collate_fn = None
    processor = None

    if cfg.model.processor is not None:
        processor = AutoProcessor.from_pretrained(cfg.model.processor)

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
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(cfg.task.mean,), std=(cfg.task.std,))
            ])
            if not cfg.task.occurrence_loss:
                wrapper = BasicUnpack(cfg.device, loss, occurence_accuracy)
                train_data = Emnist24Directions(os.path.join(cfg.task.root_path, "train"), cfg.task.num_classes,
                                                transform, cfg.task.dataset_size)
                val_data = Emnist24Directions(os.path.join(cfg.task.root_path, "val"), cfg.task.num_classes, transform)
                test_data = Emnist24Directions(os.path.join(cfg.task.root_path, "test"), cfg.task.num_classes,
                                               transform)
            else:
                wrapper = TwoLossUnpack(cfg.device, loss, nn.BCEWithLogitsLoss(), cfg.task.loss_fraction,
                                        occurence_accuracy, TopKAccuracy(k=24), ["feat", "occ"])
                train_data = Emnist24DirectionsOccurrence(os.path.join(cfg.task.root_path, "train"),
                                                          cfg.task.num_classes, transform, cfg.task.dataset_size)
                val_data = Emnist24DirectionsOccurrence(os.path.join(cfg.task.root_path, "val"), cfg.task.num_classes,
                                                        transform)
                test_data = Emnist24DirectionsOccurrence(os.path.join(cfg.task.root_path, "test"), cfg.task.num_classes,
                                                         transform)

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

        case "CIFAR10":
            transform = transforms.ToTensor()
            if not cfg.task.occurrence_loss:
                wrapper = BasicUnpack(cfg.device, nn.BCEWithLogitsLoss(), binary_accuracy)
                train_data = CifarQuery(cfg.task.root_path, True, transform)
                val_data = CifarQuery(cfg.task.root_path, False, transform)
                test_data = CifarQuery(cfg.task.root_path, False, transform)
            else:
                wrapper = TwoLossUnpack(cfg.device, nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(),
                                        cfg.task.loss_fraction, binary_accuracy, occurence_accuracy, ["feat", "occ"])
                train_data = CifarQueryOccurrence(cfg.task.root_path, True, transform)
                val_data = CifarQueryOccurrence(cfg.task.root_path, False, transform)
                test_data = CifarQueryOccurrence(cfg.task.root_path, False, transform)

        case "GQA":
            wrapper = VQAUnpack(cfg.device, nn.CrossEntropyLoss(), cfg.sampling_params, processor)
            collate_fn = VQACollate(processor)
            if cfg.task.log_mistakes:
                wrapper.set_logging(True)
            train_data = None if cfg.task.train_question_file is None else GQA(Path(cfg.task.img_dir),
                                                                               Path(cfg.task.train_question_file),
                                                                               cfg.task.prompt)
            val_data = None if cfg.task.val_question_file is None else GQA(Path(cfg.task.img_dir),
                                                                           Path(cfg.task.val_question_file),
                                                                           cfg.task.prompt)
            test_data = None if cfg.task.test_question_file is None else GQA(Path(cfg.task.img_dir),
                                                                             Path(cfg.task.test_question_file),
                                                                             cfg.task.prompt)

        case _:
            raise ValueError(f"Dataset {cfg.task.name} is not supported")

    if "num_tasks" in cfg.task and (cfg.task_embedding == "trick" or cfg.arg_embedding == "trick"):
        if isinstance(wrapper, BasicUnpack):
            wrapper = KLUnpack(cfg.device, wrapper.criterion, wrapper.accuracy_metric, cfg.kld_weight,
                               cfg.log_embeddings)

    return train_data, val_data, test_data, wrapper, collate_fn
