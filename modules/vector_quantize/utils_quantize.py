import torch
from einops import rearrange
from torch.nn import functional as F


def noop(*args, **kwargs):
    """
    This function does nothing. It accepts any arguments and returns None.
    """


import torch


def gumbel_sample(t, temperature=1.0, dim=-1):
    """
    Draws a sample from a categorical distribution according to the Gumbel-Softmax trick.

    Args:
        t (torch.Tensor): Unnormalized log probabilities (logits).
        temperature (float, optional): Temperature parameter. Defaults to 1.0.
        dim (int, optional): Dimension to perform the operation. Defaults to -1.

    Returns:
        torch.Tensor: Indices of the sampled elements.
    """
    if temperature == 0:
        return t.argmax(dim=dim)

    # More efficient Gumbel noise generation:
    noise = torch.rand_like(t)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)

    return ((t / temperature) + gumbel_noise).argmax(dim=dim)


def batched_sample_vectors(samples, num):
    """
    Efficiently samples a fixed number of vectors from a batch of sample tensors.

    Handles cases where `num` is smaller or larger than the number of
    samples in each batch element.

    Args:
        samples (torch.Tensor): Batch of tensors to sample from,
                                 shape (batch_size, num_samples, dim).
        num (int): Number of vectors to sample from each tensor in the batch.

    Returns:
        torch.Tensor: Sampled vectors, shape (batch_size, num, dim).
    """
    batch_size, num_samples, dim = samples.shape
    device = samples.device

    if num_samples >= num:
        # Efficiently sample without replacement if enough samples
        indices = torch.randperm(num_samples, device=device)[:num]
        return samples[:, indices, :]  # Vectorized slicing
    else:
        # Sample with replacement if not enough samples
        # (This part remains similar to your original logic)
        indices = torch.randint(0, num_samples, (batch_size, num), device=device)
        return torch.gather(samples, 1, indices.unsqueeze(-1).expand(-1, -1, dim))


def batched_bincount(x, *, minlength):
    """
    Computes a batched version of torch.bincount.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, ...).
        minlength (int): Minimum length of the output bincount vectors.

    Returns:
        torch.Tensor: Tensor of shape (batch, minlength) containing the counts.
    """
    batch, device = x.shape[0], x.device
    return torch.stack(
        [torch.bincount(sample, minlength=minlength) for sample in x], dim=0
    ).to(device)


def kmeans(
    samples,
    num_clusters,
    num_iters=10,
    use_cosine_sim=False,
    sample_fn=None,  # No need to pass sample_fn if it's unused
    all_reduce_fn=None,  # No need to pass all_reduce_fn if it's unused
):
    """
    Performs k-means clustering on a batch of samples.

    Args:
        samples (torch.Tensor): Input samples of shape (batch, num_samples, dim).
        num_clusters (int): Number of clusters.
        num_iters (int, optional): Number of iterations. Defaults to 10.
        use_cosine_sim (bool, optional): Use cosine similarity instead of Euclidean distance.
                                          Defaults to False.
        sample_fn: This argument is not used and can be removed.
        all_reduce_fn: This argument is not used and can be removed.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The cluster means (batch, num_clusters, dim)
                                         and the cluster counts (batch, num_clusters).
    """
    num_codebooks, dim, dtype, device = (
        samples.shape[0],
        samples.shape[-1],
        samples.dtype,
        samples.device,
    )

    # Improved Initialization: Sample without replacement for better initial means
    means_indices = torch.stack(
        [torch.randperm(samples.shape[1])[:num_clusters] for _ in range(num_codebooks)],
        dim=0,
    ).to(device)
    means = torch.gather(samples, 1, means_indices.unsqueeze(-1).expand(-1, -1, dim))

    for _ in range(num_iters):
        if use_cosine_sim:
            # More efficient cosine similarity calculation
            samples_norm = F.normalize(samples, dim=-1)
            means_norm = F.normalize(means, dim=-1)
            dists = torch.einsum("b n d, b k d -> b n k", samples_norm, means_norm)
        else:
            dists = -torch.cdist(samples, means, p=2)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)

        # Optimized Cluster Mean Update
        new_means = torch.zeros(
            num_codebooks, num_clusters, dim, dtype=dtype, device=device
        )
        new_means.scatter_add_(1, buckets.unsqueeze(-1).expand(-1, -1, dim), samples)

        # Avoid division by zero
        bins_masked = torch.where(bins > 0, bins, torch.ones_like(bins))
        new_means = new_means / bins_masked.unsqueeze(-1)

        # Update means, handling empty clusters
        means = torch.where(rearrange(bins == 0, "... -> ... 1"), means, new_means)
        if use_cosine_sim:
            means = F.normalize(means, dim=-1)

    return means, bins
