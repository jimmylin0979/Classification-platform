from typing import Optional
import torch
import os
import re

from utils import logger

def delete_previous_checkpoint(save_dir: str, checkpoint_prefix: str):
    """
    """
    
    re_pattern = checkpoint_prefix + r"_\d+.\d+.pt"
    logger.info(f"Delete previous checkpoint with pattern {re_pattern}")
    pattern = re.compile(re_pattern)
    for filename in os.listdir(save_dir):
        if pattern.match(filename):
            try:
                os.remove(os.path.join(save_dir, filename))
                logger.info(f"Delete previous checkpoint: {filename}")
            except EnvironmentError:
                pass

def get_previous_checkpoint_metrics(save_dir: str):
    """
    """
    
    checkpoint_prefixs = {"checkpoint_best": "best_model_metric", 
                            "checkpoint_ema_best": "best_model_ema_metric"}
    metrics = {}

    for checkpoint_prefix in checkpoint_prefixs:    
        re_pattern = checkpoint_prefix + r"_\d+.\d+.pt"
        logger.info(f"Restore from previous checkpoint with pattern {re_pattern}")
        pattern = re.compile(re_pattern)
        for filename in os.listdir(save_dir):
            if pattern.match(filename):
                try:
                    metric = filename.split("_")[-1][:-3]
                    metric = float(metric)
                    metrics[checkpoint_prefixs[checkpoint_prefix]] = metric
                    logger.info(f"Restore from previous checkpoint: {filename} with metric {metric}")
                except EnvironmentError:
                    pass
    
    return metrics

def save_checkpoint(epoch: int,
                    model: torch.nn.Module,
                    is_best: bool,
                    model_ema: torch.nn.Module,
                    is_best_ema: bool,
                    optimizer: torch.optim.Optimizer,
                    gradient_scaler: torch.cuda.amp.GradScaler,
                    save_dir: str,
                    *args, **kwargs) -> None:

    #
    model_state_dict = model.state_dict()
    model_ema_state_dict = model_ema.state_dict()
    optim_state_dict = optimizer.state_dict()
    gradient_scaler_state_dict = gradient_scaler.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "model_ema_state_dict": model_ema_state_dict,
        "optim_state_dict": optim_state_dict,
        "gradient_scaler_state_dict": gradient_scaler_state_dict
    }
    checkpoint_str = f"checkpoint.pt"
    checkpoint_path = f"{save_dir}/{checkpoint_str}"
    torch.save(checkpoint, checkpoint_path)

    # Add metric in the checkpoint file
    # Save the best model parameters (only parameters)
    if is_best:    
        #
        delete_previous_checkpoint(save_dir, "checkpoint_best")
        #
        assert "best_model_metric" in kwargs, "kwargs['best_model_metric'] should exist when is_best = True"
        metric = kwargs["best_model_metric"]
        checkpoint_path = f"{save_dir}/checkpoint_best_{metric:.4f}.pt"
        torch.save(model.state_dict(), checkpoint_path)

    # Save the best model ema parameters (only parameters)
    if is_best_ema:
        #
        delete_previous_checkpoint(save_dir, "checkpoint_ema_best")
        #
        assert "best_model_ema_metric" in kwargs, "kwargs['best_model_ema_metric'] should exist when is_best_ema = True"
        metric = kwargs["best_model_ema_metric"]    
        checkpoint_path = f"{save_dir}/checkpoint_ema_best_{metric:.4f}.pt"
        torch.save(model_ema.state_dict(), checkpoint_path)

def load_checkpoint(model: torch.nn.Module,
                    model_ema: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    gradient_scaler: torch.cuda.amp.GradScaler,
                    save_dir: str,
                    device_type: str,
                    *args, **kwargs):
    
    #
    checkpoint_path = f"{save_dir}/checkpoint.pt"
    logger.info("=" * 80)
    logger.info("Restore from previous checkpoint, and load it on device {}".format(device_type))
    if device_type == "cpu":
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(f"cpu"))
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(f"cuda"))

    #
    is_eval = False
    if "mode" in kwargs:
        logger.info(f"Start loading checkpoint from {save_dir} with mode Evaluate")
        is_eval = True

    #
    model.load_state_dict(checkpoint["model_state_dict"])
    model_ema.load_state_dict(checkpoint["model_ema_state_dict"])
    if not is_eval:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        gradient_scaler.load_state_dict(checkpoint["gradient_scaler_state_dict"])
    epoch = checkpoint["epoch"]

    # Return best metric of model and model_ema from checkpoint
    metrics = get_previous_checkpoint_metrics(save_dir)
    logger.info(f"Restore model checkpoint with metric {metrics['best_model_metric']}")
    logger.info(f"Restore model_ema checkpoint with metric {metrics['best_model_ema_metric']}")

    return model, model_ema, optimizer, gradient_scaler, epoch + 1, metrics

if __name__ == "__main__":
    
    #
    delete_previous_checkpoint("./results/convnext_1/", "checkpoint_best")
