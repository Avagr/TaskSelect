import os
import random
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).parents[1]))

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from launch.launch_utils import create_model, parse_task
from utils.training import set_random_seed, predict, count_parameters


@hydra.main(config_path="configs", config_name="eval_config", version_base=None)
def run(cfg: DictConfig):
    torch.autograd.set_detect_anomaly(cfg.detect_anomalies, check_nan=True)
    torch.backends.cuda.matmul.allow_tf32 = cfg.use_tf32
    torch.backends.cudnn.allow_tf32 = cfg.use_tf32

    set_random_seed(cfg.seed)

    model = create_model(cfg).to(dtype=torch.bfloat16, device=cfg.device)

    cfg.model_size = count_parameters(model)

    if cfg.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        print(OmegaConf.to_yaml(cfg))

    train_data, val_data, test_data, wrapper, collate_fn = parse_task(cfg)

    match cfg.evaluate_fold:
        case "train":
            loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=cfg.pin_memory, collate_fn=collate_fn)
        case "val":
            loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=cfg.pin_memory, collate_fn=collate_fn)
        case "test":
            loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=cfg.pin_memory, collate_fn=collate_fn)
        case _:
            raise ValueError(f"Fold {cfg.evaluate_fold} is not supported")

    if cfg.resume_wandb_id is None:
        run_id = wandb.util.generate_id()
    else:
        run_id = cfg.resume_wandb_id

    print(f"Run id: {run_id}")
    wandb.init(id=run_id, resume="must" if cfg.resume_wandb_id is not None else "never",
               project=cfg.wandb_project, entity="avagr", name=cfg.name, group=cfg.task.name,
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    wandb.watch(model)
    metrics = predict(model, loader, wrapper, verbose=cfg.show_tqdm)
    wandb.run.summary.update(metrics)
    if cfg.task.log_mistakes:
        sampled = random.sample(wrapper.mistakes, min(cfg.task.mistakes_to_log, len(wrapper.mistakes)))
        wandb.log({
            "mistakes": [wandb.Image(f"{cfg.task.img_dir}/{img_id}.jpg",
                                     caption=f"{question}\nExpected: {gt}\n Got: {pred}") for
                         img_id, question, pred, gt in sampled]})


if __name__ == '__main__':
    run()
