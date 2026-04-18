import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.linear3 = nn.Linear(hid_dim, out_dim)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """"
        (batch_size, in_dim) -> (batch_size, out_dim)
        """
        data = self.linear1(data)
        data = F.relu(data)
        data = self.linear2(data)
        data = F.relu(data)
        data = self.linear3(data)
        return data