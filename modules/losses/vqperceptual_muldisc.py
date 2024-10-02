import os
import sys

sys.path.append(os.getcwd())

import torch
from lpips import LPIPS
from torch import nn
from torch.nn import functional as F

from modules.discriminator.discriminator_model import weights_init
from utils.utils_modules import instantiate_from_config

# Define loss functions outside the class for reusability
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def hinge_g_loss(logits_fake):
    return -torch.mean(logits_fake)

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def bce_discr_loss(logits_real, logits_fake):
    return (-log(1 - torch.sigmoid(logits_fake)) - log(torch.sigmoid(logits_real))).mean()

def bce_gen_loss(logits_fake):
    return -log(torch.sigmoid(logits_fake)).mean()

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        disc_config,
        disc_init,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        disc_conditional=False,
        disc_adaptive_loss=True,
        disc_loss="hinge",
        disc_weight_max=None,
        budget_loss_config=None,
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "bce"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval() if perceptual_weight > 0 else None
        self.perceptual_weight = perceptual_weight

        self.discriminator_iter_start = disc_start
        self.discriminator = instantiate_from_config(disc_config)
        if disc_init:
            self.discriminator.apply(weights_init)

        self.loss_functions = {
            "hinge": (hinge_d_loss, hinge_g_loss),
            "vanilla": (vanilla_d_loss, hinge_g_loss),
            "bce": (bce_discr_loss, bce_gen_loss),
        }

        self.disc_loss_fn, self.gen_loss_fn = self.loss_functions.get(
            disc_loss, (None, None)
        )
        if self.disc_loss_fn is None:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.disc_adaptive_loss = disc_adaptive_loss
        self.disc_weight_max = disc_weight_max

        self.budget_loss = (
            instantiate_from_config(budget_loss_config) if budget_loss_config else None
        )

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = d_weight.clamp(0.0, 1e4).detach() * self.discriminator_weight
        if self.disc_weight_max is not None:
            d_weight.clamp_max_(self.disc_weight_max)
        return d_weight

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        gate=None,
    ):
        rec_loss = torch.abs(inputs - reconstructions)
        if self.perceptual_loss:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss.mean()

        if optimizer_idx == 0:
            if self.disc_conditional:
                assert cond is not None, "Conditional discriminator requires 'cond'."
                logits_fake = self.discriminator(
                    torch.cat((reconstructions, cond), dim=1)
                )
            else:
                logits_fake = self.discriminator(reconstructions)

            g_loss = self.gen_loss_fn(logits_fake)

            if self.disc_adaptive_loss:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(self.disc_weight_max)

            disc_factor = (
                self.disc_factor
                if global_step >= self.discriminator_iter_start
                else 0.0
            )
            loss = (
                nll_loss
                + d_weight * disc_factor * g_loss
                + self.codebook_weight * codebook_loss.mean()
            )

            if self.budget_loss:
                assert gate is not None, "Budget loss requires 'gate'."
                budget_loss = self.budget_loss(gate=gate)
                loss = loss + budget_loss

                log = {
                    f"{split}_total_loss": loss.detach().mean(),
                    f"{split}_quant_loss": codebook_loss.detach().mean(),
                    f"{split}_nll_loss": nll_loss.detach(),
                    f"{split}_rec_loss": rec_loss.detach().mean(),
                    f"{split}_p_loss": p_loss.detach(),
                    f"{split}_d_weight": d_weight.detach(),
                    f"{split}_disc_factor": torch.tensor(disc_factor),
                    f"{split}_g_loss": g_loss.detach(),
                    f"{split}_budget_loss": budget_loss.detach(),
                }
                return loss, log

            log = {
                f"{split}_total_loss": loss.detach().mean(),
                f"{split}_quant_loss": codebook_loss.detach().mean(),
                f"{split}_nll_loss": nll_loss.detach(),
                f"{split}_rec_loss": rec_loss.detach().mean(),
                f"{split}_p_loss": p_loss.detach(),
                f"{split}_d_weight": d_weight.detach(),
                f"{split}_disc_factor": torch.tensor(disc_factor),
                f"{split}_g_loss": g_loss.detach(),
            }
            return loss, log

        if optimizer_idx == 1:
            if self.disc_conditional:
                assert cond is not None, "Conditional discriminator requires 'cond'."
                logits_real = self.discriminator(
                    torch.cat((inputs.detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.detach(), cond), dim=1)
                )
            else:
                logits_real = self.discriminator(inputs.detach())
                logits_fake = self.discriminator(reconstructions.detach())

            disc_factor = (
                self.disc_factor
                if global_step >= self.discriminator_iter_start
                else 0.0
            )
            d_loss = disc_factor * self.disc_loss_fn(logits_real, logits_fake)

            log = {
                f"{split}_disc_loss": d_loss.detach().mean(),
                f"{split}_logits_real": logits_real.detach().mean(),
                f"{split}_logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, log
