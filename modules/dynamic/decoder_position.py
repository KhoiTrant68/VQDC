import os

import sys
sys.path.append(os.getcwd())

import numpy as np

import torch
from torch import nn
from einops import rearrange

from modules.diffusion.model import (
    ResnetBlock,
    AttnBlock,
    Upsample,
    Normalize,
    nonlinearity,
)

from modules.dynamic.tools import trunc_normal_
from modules.dynamic.fourier_embedding import FourierPositionEmbedding