from torch import nn

from utils.nn_modules import ActNorm


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the first conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm (bool) -- use ActNorm instead of BatchNorm
        """
        super().__init__()

        norm_layer = ActNorm if use_actnorm else nn.BatchNorm2d
        use_bias = use_actnorm or norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        layers = [
            nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw),
        ]

        self.main = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


def weights_init(m):
    """Initialize weights for Conv2d and BatchNorm2d layers."""
    if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
