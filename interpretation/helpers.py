import torch


def batch_process(model, img, task, args_dict):
    return {k: model(img, task, v).detach() for k, v in args_dict.items()}
