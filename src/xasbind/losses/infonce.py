import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, tau=0.07):
        super().__init__()
        self.learnable_temp = nn.Parameter(torch.tensor(tau).log())

    def _infonce_pair(self, first_tensor: torch.Tensor, second_tensor: torch.Tensor):
        logits = (first_tensor @ second_tensor.T) / self.learnable_temp.exp()
        target = torch.arange(len(logits), device=logits.device)
        loss_first_second = F.cross_entropy(logits, target)
        loss_second_first = F.cross_entropy(logits.T, target)
        loss = (loss_first_second + loss_second_first) / 2
        return loss

    def forward(self, struct_embs: torch.Tensor, xanes_embs: torch.Tensor, exafs_embs: torch.Tensor):
        struct_xanes_loss = self._infonce_pair(struct_embs, xanes_embs)
        struct_exafs_loss = self._infonce_pair(struct_embs, exafs_embs)
        return struct_xanes_loss + struct_exafs_loss, struct_xanes_loss, struct_exafs_loss






