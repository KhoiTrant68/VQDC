from omegaconf import OmegaConf
from pathlib import Path
import importlib


def mark_prepared(root):
    (Path(root) / ".ready").touch()  # More Pythonic way to join paths


def is_prepared(root):
    return (Path(root) / ".ready").exists()


def instantiate_from_config(config):
    if "target" not in config:  # More Pythonic way to check for key existence
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(
        **config.get("params", {})
    )  # Use {} for empty dict


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
