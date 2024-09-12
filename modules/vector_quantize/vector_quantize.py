import torch
import torch.distributed as dist
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class VQEmbedding(nn.Embedding):
    """VQ embedding module with EMA update and random restart."""

    def __init__(
        self,
        n_embed,
        embed_dim,
        ema=True,
        decay=0.99,
        restart_unused_codes=True,
        eps=1e-5,
    ):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            with torch.no_grad():
                self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
                self.register_buffer("embed_ema", self.weight[:-1, :].clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        return torch.cdist(
            inputs.reshape(-1, self.weight.shape[-1]), self.weight[:-1, :]
        ).reshape(*inputs.shape[:-1], -1)

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)
        return distances.argmin(dim=-1)

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        # Use torch.sqrt instead of np.sqrt
        std = 0.01 / torch.sqrt(torch.tensor(embed_dim, device=x.device))
        x = x.repeat_interleave(n_repeats, dim=0) + torch.randn_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)

        cluster_size = torch.zeros(
            n_embed, dtype=torch.long, device=vectors.device
        ).scatter_add_(0, idxs, torch.ones_like(idxs, dtype=torch.long))
        vectors_sum_per_cluster = torch.zeros(
            n_embed, embed_dim, device=vectors.device
        ).scatter_add_(0, idxs.unsqueeze(1).expand(-1, embed_dim), vectors)

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(
            vectors_sum_per_cluster, alpha=1 - self.decay
        )

        if self.restart_unused_codes:
            unused_indices = torch.where(cluster_size == 0)[0]
            num_unused = unused_indices.shape[0]

            if num_unused > 0:
                if vectors.shape[0] < num_unused:
                    vectors = self._tile_with_noise(vectors, num_unused)

                random_indices = torch.randperm(
                    vectors.shape[0], device=vectors.device
                )[:num_unused]
                _vectors_random = vectors[random_indices]

                if dist.is_initialized():
                    dist.broadcast(_vectors_random, 0)

                self.embed_ema[unused_indices] = _vectors_random
                self.cluster_size_ema[unused_indices] = 1

    @torch.no_grad()
    def _update_embedding(self):
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :].copy_(
            self.embed_ema / normalized_cluster_size.reshape(-1, 1)
        )

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training and self.ema:
            self._update_buffers(inputs, embed_idxs)

        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        return super().forward(idxs)


class VectorQuantize2(nn.Module):
    def __init__(
        self,
        codebook_size,
        codebook_dim=None,
        accept_image_fmap=True,
        commitment_beta=0.25,
        decay=0.99,
        restart_unused_codes=True,
        channel_last=False,
    ):
        super().__init__()
        self.accept_image_fmap = accept_image_fmap
        self.beta = commitment_beta
        self.channel_last = channel_last
        self.codebook = VQEmbedding(
            codebook_size,
            codebook_dim,
            decay=decay,
            restart_unused_codes=restart_unused_codes,
        )
        nn.init.kaiming_uniform_(
            self.codebook.weight, a=0, mode="fan_in", nonlinearity="linear"
        )

    def forward(self, x, codebook_mask=None, *args, **ignorekwargs):
        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, "b c h w -> b (h w) c")

        if need_transpose:
            x = rearrange(x, "b d n -> b n d")

        flatten = rearrange(x, "h ... d -> h (...) d")

        x_q, x_code = self.codebook(flatten)

        if codebook_mask is not None:
            if codebook_mask.dim() == 4:
                codebook_mask = rearrange(codebook_mask, "b c h w -> b (h w) c")
                loss = self.beta * torch.mean(
                    (x_q.detach() - x) ** 2 * codebook_mask
                ) + torch.mean((x_q - x.detach()) ** 2 * codebook_mask)
            else:
                loss = self.beta * torch.mean(
                    (x_q.detach() - x) ** 2 * codebook_mask.unsqueeze(-1)
                ) + torch.mean((x_q - x.detach()) ** 2 * codebook_mask.unsqueeze(-1))
        else:
            loss = self.beta * torch.mean((x_q.detach() - x) ** 2) + torch.mean(
                (x_q - x.detach()) ** 2
            )

        x_q = x + (x_q - x).detach()

        if need_transpose:
            x_q = rearrange(x_q, "b n d -> b d n")

        if self.accept_image_fmap:
            x_q = rearrange(x_q, "b (h w) c -> b c h w", h=height, w=width)
            x_code = rearrange(x_code, "b (h w) ... -> b h w ...", h=height, w=width)
        return x_q, loss, (None, None, x_code)

    @torch.no_grad()
    def get_soft_codes(self, x, temp=1.0, stochastic=False):
        distances = self.codebook.compute_distances(x)
        soft_code = F.softmax(-distances / temp, dim=-1)

        if stochastic:
            code = torch.multinomial(soft_code.flatten(0, -2), 1).reshape(
                *soft_code.shape[:-1]
            )
        else:
            code = distances.argmin(dim=-1)

        return soft_code, code

    def get_codebook_entry(self, indices, *kwargs):
        z_q = self.codebook.embed(indices)
        return z_q
