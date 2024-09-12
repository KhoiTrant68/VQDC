import torch
import torch.nn as nn


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        super().__init__()

        assert affine, "This implementation only supports affine ActNorm."
        self.logdet = logdet
        self.allow_reverse_init = allow_reverse_init
        self.initialized = False

        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

    def initialize(self, input):
        with torch.no_grad():
            # More efficient calculation of mean and std
            mean = input.mean(dim=(0, 2, 3), keepdim=True)
            std = input.std(dim=(0, 2, 3), keepdim=True)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if not self.initialized:
            if reverse and not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            self.initialize(input)
            self.initialized = True

        # Combine dimensions if needed (assuming NCHW format)
        if len(input.shape) == 2:
            input = input.unsqueeze(-1).unsqueeze(-1)
            squeeze = True
        else:
            squeeze = False

        if reverse:
            output = input / self.scale - self.loc
        else:
            output = self.scale * (input + self.loc)

        if squeeze:
            output = output.squeeze(-1).squeeze(-1)

        if self.logdet and not reverse:  # Only calculate logdet in forward pass
            log_abs = torch.log(torch.abs(self.scale))
            logdet = input.shape[2] * input.shape[3] * torch.sum(log_abs)
            return output, logdet * torch.ones_like(input[:, 0, 0, 0])

        return output
