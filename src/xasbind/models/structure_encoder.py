from e3nn import o3
from e3nn.nn import Gate
from e3nn.math import soft_one_hot_linspace
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, List


# ==================================================================================
# MULTI-LAYER + INVARIANT-SUMMARY MODIFICATIONS BEGIN
# ==================================================================================
def _extract_invariants(features: torch.Tensor, irreps: "o3.Irreps") -> torch.Tensor:
    """Given a flat tensor of shape (N, irreps.dim) with fields ordered as in `irreps`,
    return a flat tensor of shape (N, num_invariants) containing:
      - all scalar (l=0) values unchanged
      - the L2 norm of each non-scalar irrep (one scalar per irrep multiplicity)

    Example: irreps = "32x0e + 32x1o + 16x2e" -> 32 scalars + 32 vector norms + 16 tensor norms = 80.
    """
    outs = []
    offset = 0
    for mul, ir in irreps:
        d = ir.dim  # 2l+1
        block = features[:, offset:offset + mul * d]  # (N, mul*d)
        offset += mul * d
        if ir.l == 0:
            outs.append(block)
        else:
            # reshape to (N, mul, d) then take norm over d
            block = block.view(block.size(0), mul, d)
            outs.append(block.norm(dim=-1))
    return torch.cat(outs, dim=1)
# ==================================================================================
# MULTI-LAYER + INVARIANT-SUMMARY MODIFICATIONS END
# ==================================================================================

class RadialBasis(nn.Module):
    def __init__(self, r_max: int, num_basis: int):
        super().__init__()
        self.r_max = r_max
        self.num_basis = num_basis

    def forward(self, edge_length):
        radial_basis = soft_one_hot_linspace(
            edge_length,
            start=0.0, end=self.r_max, number=self.num_basis,
            basis="cosine", cutoff=True
        )
        return radial_basis
    

class E3EquivariantBlock(nn.Module):
    def __init__(self, in_irreps, out_irreps, sh_irreps, num_radial_basis, z_irreps, radial_hidden_dim=100, dropout=0.1):
        super().__init__()

        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)
        self.z_irreps = o3.Irreps(z_irreps)

        self.gate = self._build_gate(self.out_irreps)
        gate_in = self.gate.irreps_in

        self.sc = o3.FullyConnectedTensorProduct(in_irreps, z_irreps, gate_in, shared_weights=True)
        self.lin1 = o3.FullyConnectedTensorProduct(in_irreps, z_irreps, in_irreps, shared_weights=True)
        self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, in_irreps, shared_weights=False) # in_irreps if lin1/lin2 used
        # self.lin2 = o3.FullyConnectedTensorProduct(in_irreps, z_irreps, gate_in)
        self.post_linear = o3.Linear(in_irreps, gate_in)

        self.radial_net = nn.Sequential(
            nn.Linear(num_radial_basis, radial_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(radial_hidden_dim, self.tp.weight_numel)
        )

    def _build_gate(self, out_irreps):
        irreps = o3.Irreps(out_irreps)
        scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0])
        non_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l > 0])
        num_gates = sum(mul for mul, _ in non_scalars)
        gates = o3.Irreps(f"{num_gates}x0e") if num_gates > 0 else o3.Irreps("")

        return Gate(
            irreps_scalars = scalars, 
            act_scalars = [F.silu] * len(scalars),
            irreps_gates = gates, 
            act_gates = [torch.sigmoid] * len(gates),
            irreps_gated = non_scalars
        )

    def forward(self, node_features, edge_index, edge_sh, edge_radial, z_emb):
        src, dst = edge_index

        self_connection = self.sc(node_features, z_emb)

        # projected_in = self.lin1(node_features, z_emb)[src]
        radial_weights = self.radial_net(edge_radial)
        messages = self.tp(self.lin1(node_features, z_emb)[src], edge_sh, weight=radial_weights)

        num_nodes = node_features.size(0)
        aggregated = torch.zeros(num_nodes, messages.size(1), device=messages.device, dtype=messages.dtype)
        aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)

        # projected_out = self.lin2(aggregated, z_emb)

        return self.gate(self_connection + self.post_linear(aggregated))


class InOut(NamedTuple):
    in_irreps: str
    out_irreps: str


class StructureEncoder(nn.Module):
    def __init__(self, layer_params: List[InOut], num_radial_basis: int = 8, 
                 node_dim=94, lmax=2, r_max=5.0, embed_dim=None):
        super().__init__()

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.radial_basis = RadialBasis(r_max=r_max, num_basis=num_radial_basis)

        first_in_irreps = o3.Irreps(layer_params[0].in_irreps)
        last_out_irreps = o3.Irreps(layer_params[-1].out_irreps)
        if first_in_irreps.lmax != 0:
            raise ValueError("First input irreps must only be scalars!")
        if last_out_irreps.lmax != 0:
            raise ValueError("Final output irreps must only be scalars!")

        self.emb_x = nn.Sequential(nn.Linear(node_dim, first_in_irreps.dim), nn.ReLU())
        self.emb_z = nn.Sequential(nn.Linear(node_dim, first_in_irreps.dim), nn.Tanh())

        self.blocks = nn.ModuleList()

        z_irreps = str(first_in_irreps)

        for layer in layer_params:
            self.blocks.append(E3EquivariantBlock(layer.in_irreps, layer.out_irreps, str(self.sh_irreps), num_radial_basis, z_irreps))

        """
        last_out_dim = last_out_irreps.dim
        if embed_dim is not None and last_out_dim != embed_dim:
            self.head = nn.Linear(last_out_dim, embed_dim) 
        else:
            self.head = nn.Identity()
        """

        # ==================================================================================
        # MULTI-LAYER + INVARIANT-SUMMARY MODIFICATIONS BEGIN
        # ==================================================================================
        # Store per-block irreps for invariant extraction during forward
        self.block_irreps = [o3.Irreps(layer.out_irreps) for layer in layer_params]

        # Compute total invariant feature dimension across all blocks
        # Per block: sum of (mul for l=0 irreps) + (mul for l>0 irreps, one scalar per irrep)
        def _count_invariants(irreps):
            total = 0
            for mul, ir in irreps:
                total += mul  # scalars pass through, non-scalars contribute one norm each
            return total
        self.invariant_dims = [_count_invariants(irr) for irr in self.block_irreps]
        total_invariant_dim = sum(self.invariant_dims)

        # Per-block LayerNorms (so all blocks contribute on equal scale after concatenation)
        self.block_norms = nn.ModuleList([nn.LayerNorm(d) for d in self.invariant_dims])

        # Projection head from concatenated invariants -> embedding
        target_dim = embed_dim if embed_dim is not None else last_out_irreps.dim
        hidden_dim = max(512, target_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(total_invariant_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, target_dim),
        )
        # ==================================================================================
        # MULTI-LAYER + INVARIANT-SUMMARY MODIFICATIONS END
        # ==================================================================================


    def forward(self, batch):
        x = self.emb_x(batch.x_in.float())
        z = self.emb_z(batch.z_in.float())
        node_features = x * z

        pos = batch.pos.float()
        edge_shift = batch.edge_shift.float()
        lattice = batch.lattice.float()

        src, dst = batch.edge_index
        graph_edge = batch.batch[src]
        lattice_edge = lattice[graph_edge]

        vecs = pos[dst] - pos[src] + torch.einsum("ei, eij -> ej", edge_shift, lattice_edge)
        dists = torch.linalg.norm(vecs, dim=1)
        unit_vecs = torch.div(vecs, dists.unsqueeze(-1))
        edge_radial = self.radial_basis(dists)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, unit_vecs, normalize=True, normalization="component")

        """
        for block in self.blocks:
            node_features = block(node_features, batch.edge_index, edge_sh, edge_radial, z)

        z_nodes = torch.argmax(batch.z_in, dim=1) + 1
        z_absorbers = batch.absorber_Z[batch.batch]
        mask = (z_nodes == z_absorbers)
        masked_node_features = node_features[mask]
        masked_batch_index = batch.batch[mask]

        pooled = torch.zeros(batch.num_graphs, node_features.size(1),
                             device=node_features.device, dtype=node_features.dtype)
        index_expanded = masked_batch_index.unsqueeze(1).expand_as(masked_node_features)
        pooled.scatter_add_(0, index_expanded, masked_node_features)
        counts = torch.zeros(batch.num_graphs, 
                             device=node_features.device, dtype=node_features.dtype)
        counts.scatter_add_(0, masked_batch_index, torch.ones_like(masked_batch_index, dtype=node_features.dtype))
        pooled_mean = pooled / counts.clamp(min=1.0).unsqueeze(1)

        out = self.head(pooled_mean)
        out_normed = F.normalize(out, dim=1)
        return out_normed
        """

        # ==================================================================================
        # MULTI-LAYER + INVARIANT-SUMMARY MODIFICATIONS BEGIN
        # ==================================================================================
        # Collect each block's output BEFORE it goes into the next block, extract invariants
        z_nodes = torch.argmax(batch.z_in, dim=1) + 1
        z_absorbers = batch.absorber_Z[batch.batch]
        mask = (z_nodes == z_absorbers)
        masked_batch_index = batch.batch[mask]

        per_block_pooled = []
        for i, block in enumerate(self.blocks):
            node_features = block(node_features, batch.edge_index, edge_sh, edge_radial, z)

            # Extract invariants for this block's output
            invariants = _extract_invariants(node_features, self.block_irreps[i])  # (num_nodes, inv_dim_i)

            # Absorber-selective mean pooling on the invariants
            masked_inv = invariants[mask]  # (num_absorber_nodes, inv_dim_i)
            pooled = torch.zeros(batch.num_graphs, invariants.size(1),
                                 device=invariants.device, dtype=invariants.dtype)
            idx_exp = masked_batch_index.unsqueeze(1).expand_as(masked_inv)
            pooled.scatter_add_(0, idx_exp, masked_inv)
            counts = torch.zeros(batch.num_graphs,
                                 device=invariants.device, dtype=invariants.dtype)
            counts.scatter_add_(0, masked_batch_index,
                                torch.ones_like(masked_batch_index, dtype=invariants.dtype))
            pooled_mean = pooled / counts.clamp(min=1.0).unsqueeze(1)

            # Per-block LayerNorm so no block dominates due to scale
            pooled_mean = self.block_norms[i](pooled_mean)
            per_block_pooled.append(pooled_mean)

        # Concatenate invariants from all blocks -> project to embedding
        all_invariants = torch.cat(per_block_pooled, dim=1)  # (num_graphs, total_invariant_dim)
        out = self.head(all_invariants)
        out_normed = F.normalize(out, dim=1)
        return out_normed
        # ==================================================================================
        # MULTI-LAYER + INVARIANT-SUMMARY MODIFICATIONS END
        # ==================================================================================






