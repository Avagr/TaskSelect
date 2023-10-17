import torch
from torch import nn


class BaseEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x) -> (torch.Tensor, dict[str, torch.Tensor]):
        raise NotImplementedError


class LinearEmbedding(BaseEmbedding):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x), {}


class TrickEmbedding(BaseEmbedding):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.mean_linear = nn.Linear(in_dim, out_dim)
        self.std_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        mean = self.mean_linear(x)
        log_var = self.std_linear(x)
        rand_var = torch.empty_like(mean).normal_()
        return mean + torch.exp(log_var * 0.5) * rand_var, {"mean": mean, "log_var": log_var}


class MLPEmbedding(BaseEmbedding):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__(in_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x), {}
