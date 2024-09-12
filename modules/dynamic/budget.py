import torch
import torch.nn as nn


class BaseBudgetConstraint(nn.Module):
    """
    Base class for budget constraint modules. Provides common initialization and
    helper methods.
    """

    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.loss = nn.MSELoss()

    def calculate_ratio(self, gate, weights, target_area, base_area):
        """
        Calculates the budget ratio for a given grain size.

        Args:
            gate (torch.Tensor): Gate tensor (batch, num_grains, H, W).
            weights (torch.Tensor): Weights corresponding to each grain size.
            target_area (int): Total area of the target grain size.
            base_area (int): Base area for normalization.

        Returns:
            torch.Tensor: Budget ratio for the target grain size.
        """
        weights = weights.to("cuda")
        beta = (gate * weights[:, None, None]).sum(dim=1).sum() / gate.size(
            0
        ) - base_area
        return beta / target_area


class BudgetConstraint_RatioMSE_DualGrain(BaseBudgetConstraint):
    def __init__(
        self,
        target_ratio=0.0,
        gamma=1.0,
        min_grain_size=8,
        max_grain_size=16,
        calculate_all=True,
    ):
        super().__init__(gamma)
        self.target_ratio = target_ratio  # e.g., 0.8 means 80% are fine-grained
        self.calculate_all = calculate_all  # calculate all grains

        self.const = min_grain_size * min_grain_size
        self.max_const = max_grain_size * max_grain_size - self.const
        self.weights = torch.tensor(
            [1.0, 4.0], dtype=torch.float32
        )  # Weights for coarse and fine grains

    def forward(self, gate):
        # 0 for coarse-grained and 1 for fine-grained
        # gate: (batch, 2, min_grain_size, min_grain_size)
        budget_ratio = self.calculate_ratio(
            gate, self.weights, self.max_const, self.const
        )
        target_ratio = torch.full_like(budget_ratio, self.target_ratio)
        loss_budget = self.gamma * self.loss(budget_ratio, target_ratio)

        if self.calculate_all:
            loss_budget_last = self.gamma * self.loss(
                1 - budget_ratio, 1 - target_ratio
            )
            return loss_budget + loss_budget_last

        return loss_budget


class BudgetConstraint_NormedSeperateRatioMSE_TripleGrain(BaseBudgetConstraint):
    def __init__(
        self,
        target_fine_ratio=0.0,
        target_median_ratio=0.0,
        gamma=1.0,
        min_grain_size=8,
        median_grain_size=16,
        max_grain_size=32,
    ):
        super().__init__(gamma)
        assert target_fine_ratio + target_median_ratio <= 1.0
        self.target_fine_ratio = (
            target_fine_ratio  # e.g., 0.8 means 80% are fine-grained
        )
        self.target_median_ratio = target_median_ratio

        self.min_const = min_grain_size * min_grain_size
        self.median_const = median_grain_size * median_grain_size - self.min_const
        self.max_const = max_grain_size * max_grain_size - self.min_const
        self.weights_median = torch.tensor(
            [1.0, 4.0, 1.0], dtype=torch.float32
        )  # Weights for coarse, median, and fine grains (compensated)
        self.weights_fine = torch.tensor(
            [1.0, 1.0, 16.0], dtype=torch.float32
        )  # Weights for coarse, median, and fine grains (compensated)

    def forward(self, gate):
        # 0 for coarse-grained, 1 for median-grained, 2 for fine-grained
        # gate: (batch, 3, min_grain_size, min_grain_size)
        budget_ratio_median = self.calculate_ratio(
            gate, self.weights_median, self.median_const, self.min_const
        )
        target_ratio_median = torch.full_like(
            budget_ratio_median, self.target_median_ratio
        )
        loss_budget_median = self.loss(budget_ratio_median, target_ratio_median)

        budget_ratio_fine = self.calculate_ratio(
            gate, self.weights_fine, self.max_const, self.min_const
        )
        target_ratio_fine = torch.full_like(budget_ratio_fine, self.target_fine_ratio)
        loss_budget_fine = self.gamma * self.loss(budget_ratio_fine, target_ratio_fine)

        return loss_budget_fine + loss_budget_median
