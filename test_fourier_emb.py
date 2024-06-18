import torch
import torch.nn as nn
import numpy as np
import sys 
sys.path.append('VQDC/modules')
from modules.dynamic.fourier_embedding import FourierPositionEmbedding


def convert_to_coord_format(b, h, w, device="cpu", integer_values=False):
    if integer_values:
        y_channel, x_channel = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing="ij"
        )
    else:
        y_channel, x_channel = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing="ij"
        )
    x_channel = x_channel.unsqueeze(0).unsqueeze(0).expand(b, -1, -1, -1)
    y_channel = y_channel.unsqueeze(0).unsqueeze(0).expand(b, -1, -1, -1)
    return torch.cat((x_channel, y_channel), dim=1)


class ConLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super(ConLinear, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=bias)
        limit = np.sqrt(9 / ch_in) if is_first else np.sqrt(3 / ch_in)
        nn.init.uniform_(self.conv.weight, -limit, limit)

    def forward(self, x):
        return self.conv(x)


class FourierPositionEmbeddingNEW(nn.Module):
    def __init__(self, coord_size, hidden_size, integer_values=False):
        super(FourierPositionEmbeddingNEW, self).__init__()
        self.coord = convert_to_coord_format(1, coord_size, coord_size, "cpu", integer_values)
        self.lff = ConLinear(2, hidden_size, is_first=True)
        self.activation = torch.sin

    def forward(self, x):
        coord = self.coord.to(x.device)
        fourier_features = self.activation(self.lff(coord))
        return x + fourier_features


if __name__ == "__main__":
    import pprint
    x = torch.randn(10, 64, 32, 32)
    module = FourierPositionEmbedding(coord_size=32, hidden_size=64)
    out = module(x)
    pprint.print(out)
 

    module = FourierPositionEmbeddingNEW(coord_size=32, hidden_size=64)
    results = module(x)
    pprint.print(results)

    # print(out != results)


