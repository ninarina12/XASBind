import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, List


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=2, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return self.dropout(out)
    

class InOutKernel(NamedTuple):
    in_channels: int
    out_channels: int
    kernel_size: int
        
        
class SpectrumEncoder(nn.Module):
    def __init__(self, layer_params: List[InOutKernel], embed_dim=None):
        super().__init__()
        self.blocks = nn.Sequential()
        
        for layer in layer_params:
            block = ResBlock(
                in_ch=layer.in_channels,
                out_ch=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=2
            )
            self.blocks.append(block)

        last_out_ch = layer_params[-1].out_channels

        if embed_dim is not None:
            self.head = nn.Linear(last_out_ch, embed_dim)
        else:
            self.head = nn.Identity()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.blocks(x)
        x = x.mean(dim=2)
        x = self.head(x)
        x_normed = F.normalize(x, dim=1)
        return x_normed