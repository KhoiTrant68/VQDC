import math
import torch
from functools import partial

# Step scheduler
def linear_warmup(warmup_steps, step):
    return step / max(1, warmup_steps) if step < warmup_steps else 1.0

scheduler_linear_warmup = partial(linear_warmup)

def linear_warmup_cosine_decay(warmup_steps, max_steps, multipler_min, step):
    if step < warmup_steps:
        return linear_warmup(warmup_steps, step)
    else:
        multipler = 0.5 * (math.cos((step - warmup_steps) / (max_steps - warmup_steps) * math.pi) + 1)
        return max(multipler, multipler_min)

scheduler_linear_warmup_cosine_decay = partial(linear_warmup_cosine_decay)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self