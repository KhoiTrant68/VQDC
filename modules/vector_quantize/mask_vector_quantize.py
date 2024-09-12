import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from modules.vector_quantize import utils_quantize


class MaskVectorQuantize(nn.Module):
    def __init__(
        self,
        codebook_size,
        codebook_dim,
        kmeans_init=False,
        kmeans_iters=10,
        use_cosine_sim=False,
        channel_last=False,
        accept_image_fmap=True,
        commitment_beta=0.25,
        orthogonal_reg_weight=0.0,
        activate_mask_quantize=True,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
        self.use_cosine_sim = use_cosine_sim
        self.beta = commitment_beta

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        if not kmeans_init:
            self.embedding.weight.data.uniform_(
                -1.0 / self.codebook_size, 1.0 / self.codebook_size
            )
        else:
            self.embedding.weight.data.zero_()
        self.kmeans_iters = kmeans_iters
        self.register_buffer("initted", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(1, codebook_size))

        self.sample_fn = utils_quantize.batched_sample_vectors
        self.all_reduce_fn = utils_quantize.noop

        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.activate_mask_quantize = activate_mask_quantize

    def init_embed_(self, data):
        if self.initted.item():
            return

        data = rearrange(data, "... -> 1 (...) d").contiguous()
        embed, cluster_size = utils_quantize.kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.all_reduce_fn,
        )

        self.embedding.weight.data.copy_(embed.squeeze(0))
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.fill_(True)

    def forward(self, x, temp=0.0, codebook_mask=None):
        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            if codebook_mask is not None and self.activate_mask_quantize:
                codebook_mask = rearrange(
                    codebook_mask, "b c h w -> b (h w) c"
                ).contiguous()

        if not self.channel_last and not self.accept_image_fmap:
            x = rearrange(x, "b d n -> b n d").contiguous()

        shape, flatten = x.shape, rearrange(x, "h ... d -> h (...) d").contiguous()
        self.init_embed_(flatten)

        if self.use_cosine_sim:
            flatten_norm = F.normalize(flatten, p=2, dim=-1)
            weight_norm = F.normalize(self.embedding.weight, p=2, dim=-1).unsqueeze(0)
            dist = torch.einsum("h n d, h c d -> h n c", flatten_norm, weight_norm)
        else:
            flatten = flatten.view(-1, self.codebook_dim)
            dist = (
                -torch.sum(flatten**2, dim=1, keepdim=True)
                - torch.sum(self.embedding.weight**2, dim=1)
                + 2 * torch.einsum("bd,dn->bn", flatten, self.embedding.weight.t())
            )

        embed_ind = utils_quantize.gumbel_sample(dist, dim=-1, temperature=temp)
        embed_ind = embed_ind.view(*shape[:-1])
        x_q = self.embedding(embed_ind)

        if codebook_mask is not None and self.activate_mask_quantize:
            ratio = 1 / codebook_mask.mean()
            loss = (
                ratio * self.beta * ((x_q.detach() - x) ** 2 * codebook_mask).mean()
                + ratio * ((x_q - x.detach()) ** 2 * codebook_mask).mean()
            )
        else:
            loss = (
                self.beta * ((x_q.detach() - x) ** 2).mean()
                + ((x_q - x.detach()) ** 2).mean()
            )

        if self.orthogonal_reg_weight > 0.0:
            emb_weight_after_norm = F.normalize(self.embedding.weight, p=2, dim=-1)
            diff = torch.mm(
                emb_weight_after_norm, emb_weight_after_norm.t()
            ) - torch.eye(self.codebook_size, device=emb_weight_after_norm.device)
            ortho_reg_term = (
                self.orthogonal_reg_weight * (diff**2).sum() / (diff.size(0) ** 2)
            )
            loss += ortho_reg_term

        x_q = x + (x_q - x).detach()

        if not self.channel_last and not self.accept_image_fmap:
            x_q = rearrange(x_q, "b n d -> b d n").contiguous()

        if self.accept_image_fmap:
            x_q = rearrange(x_q, "b (h w) c -> b c h w", h=height, w=width).contiguous()
            embed_ind = rearrange(
                embed_ind, "b (h w) ... -> b h w ...", h=height, w=width
            ).contiguous()

        return x_q, loss, (None, None, embed_ind)

    def get_codebook_entry(self, indices, shape, *kwargs):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()
        return z_q

    @torch.no_grad()
    def embed_code_with_depth(self, code, to_latent_shape=False):
        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)
        embeds = [self.embedding(code_slice) for code_slice in code_slices]
        if to_latent_shape:
            embeds = [
                self.to_latent_shape(embed.squeeze(-2)).unsqueeze(-2)
                for embed in embeds
            ]
        return torch.cat(embeds, dim=-2), None
