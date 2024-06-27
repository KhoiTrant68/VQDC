import torch 

def linear_warmup_cosine_decay(step, warmup_steps, max_steps, min_multiplier=0.0):
    """Combined linear warmup and cosine decay scheduler."""
    if step <= warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    multiplier = 0.5 * (1 + torch.cos(torch.pi * progress)) * (1 - min_multiplier) + min_multiplier
    return multiplier


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self