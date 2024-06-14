import torch 
from torch import nn, Tensor
import lightning as L

from einops import rearrange

from models.model_utils import disabled_train
from modules.dynamic.dynamic_utils import draw_dual_grain_256res_color