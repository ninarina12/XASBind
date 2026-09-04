import torch.nn as nn

class XASBind(nn.Module):
    def __init__(self, xanes_enc, exafs_enc, struct_enc, loss_fn):
        super().__init__()
        self.xanes_enc = xanes_enc
        self.exafs_enc = exafs_enc
        self.struct_enc = struct_enc
        self.loss_fn = loss_fn

    def forward(self, xanes, exafs, graph_batch):
        xanes_embs = self.xanes_enc(xanes)
        exafs_embs = self.exafs_enc(exafs)
        structs_embs = self.struct_enc(graph_batch)
        loss, sx_loss, se_loss = self.loss_fn(structs_embs, xanes_embs, exafs_embs)
        return loss, sx_loss, se_loss

    def encode(self, xanes, exafs, graph_batch):
        xanes_embs = self.xanes_enc(xanes)
        exafs_embs = self.exafs_enc(exafs)
        structs_embs = self.struct_enc(graph_batch)
        return structs_embs, xanes_embs, exafs_embs