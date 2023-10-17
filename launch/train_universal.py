import os
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).parents[1]))

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from launch.launch_utils import create_optimizer, create_model, parse_task
from utils.training import set_random_seed, train, count_parameters


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

    train_data, val_data, test_data, wrapper, collate_fn = parse_task(cfg)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=cfg.pin_memory, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=cfg.pin_memory, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=cfg.pin_memory, collate_fn=collate_fn)

    optimizer, scheduler = create_optimizer(cfg, model)

    if cfg.resume_wandb_id is None:
        run_id = wandb.util.generate_id()
    else:
        run_id = cfg.resume_wandb_id
    print(f"Run id: {run_id}")
    wandb.init(id=run_id, resume="must" if cfg.resume_wandb_id is not None else "never",
               project=cfg.wandb_project, entity="avagr", name=cfg.name, group=cfg.task.name,
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    wandb.watch(model)

    train(model, train_loader, val_loader, test_loader, optimizer, wrapper, n_epochs=cfg.num_epochs, device=cfg.device,
          scheduler=scheduler, verbose=cfg.print_epochs, save_dir=Path(cfg.save_directory), save_best=cfg.save_best,
          model_name=cfg.name, show_tqdm=cfg.show_tqdm, use_scaler=cfg.mixed_precision, load_from=cfg.load_from)


if __name__ == '__main__':
    run()
