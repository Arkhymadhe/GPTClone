import torch
import numpy as np

from torch import nn


class Mask(nn.Module):
    def __init__(self, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.mask = torch.full(size=(1, seq_len, seq_len), fill_value=-torch.inf)
        self.mask = torch.triu(self.mask, diagonal=1)

    @torch.no_grad()
    def forward(self, x):
        return x + self.mask


class ALIBI(nn.Module):
    def __init__(self, mask=None, num_heads=8, seq_len=100):
        super(ALIBI, self).__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len

        self.slopes = np.geomspace(
            start=2 ** (-8 / num_heads), stop=2 ** (-8), num=num_heads, endpoint=True
        )

        if mask is not None:
            self.mask = torch.zeros_like(mask)
        else:
            self.mask = torch.zeros(size=(num_heads, self.seq_len, self.seq_len))

        for slope, index in zip(self.slopes, range(self.seq_len)):
            if index < 2:
                continue
            sub_mask = torch.arange(index) - (index - 1)

            self.mask[:, index, :index] = slope * sub_mask

    @torch.no_grad()
    def forward(self, x):
        masked_x = x + self.mask.to(x.device)
        return masked_x
