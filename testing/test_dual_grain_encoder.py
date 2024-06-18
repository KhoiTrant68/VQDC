from modules.dynamic.encoder_dual_new import DualGrainEncoder
import unittest
import torch


class TestDualGrainEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = DualGrainEncoder(
            ch=64,
            ch_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            attn_resolutions=(16,),
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=3,
            resolution=256,
            z_channels=256,
            router_config={
                "target": "modules.dynamic.router_dual.DualGrainFeatureRouter",
                "params": {
                    "num_channels": 256,
                    "normalization_type": "group-32",
                    "gate_type": "2layer-fc-SiLu",
                },
            },
        )

    def test_forward_pass(self):
        x = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, 256x256 resolution
        x_entropy = torch.randn(1, 256)  # Batch size 1, matching z_channels

        outputs = self.encoder(x, x_entropy)

        self.assertIn("h_dual", outputs)
        self.assertIn("indices", outputs)
        self.assertIn("codebook_mask", outputs)
        self.assertIn("gate", outputs)

        h_dual = outputs["h_dual"]
        indices = outputs["indices"]
        codebook_mask = outputs["codebook_mask"]
        gate = outputs["gate"]

        print(h_dual.shape)
        print(indices.shape)
        print(codebook_mask.shape)
        print(gate.shape)



if __name__ == "__main__":
    unittest.main()
