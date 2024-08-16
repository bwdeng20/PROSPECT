import warnings
import torch
from typing import Mapping


def load_th_or_pl_ckpt2md(model: torch.nn.Module, ckpt=None, map_location=None):
    if ckpt is None:
        warnings.warn(f"No checkpoint is loaded for {type(model)}!")
        return model
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = torch.load(ckpt, map_location=map_location)
    model_state_dict = loaded["state_dict"]
    is_ckpt_lit = "pytorch-lightning_version" in loaded
    if is_ckpt_lit:
        state_dict_wo_prefix = {
            k.split(".", 1)[1]: v for k, v in model_state_dict.items()
        }
    else:
        state_dict_wo_prefix = model_state_dict
    assert isinstance(state_dict_wo_prefix, Mapping)
    model.load_state_dict(state_dict_wo_prefix)
    return model
