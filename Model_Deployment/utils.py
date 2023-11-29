import torch
from torch import nn
import pathlib


def save_model(model: nn.Module, target_dir: str, model_name: str):
    DATA_PATH  = pathlib.Path(target_dir)
    SAVE_PATH = DATA_PATH/model_name 

    if SAVE_PATH.is_dir():
        print(f'{SAVE_PATH} already exists, skipping the creation part.')
    else:
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving the {model.__class__.__name__} in {SAVE_PATH}")
        torch.save(obj = model.state_dict(), f= SAVE_PATH)
