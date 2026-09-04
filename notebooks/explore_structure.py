# %%
# %%
import pandas as pd
import numpy as np
from pathlib import Path
import h5py
import json

MATS_DIR = Path("data/xas/matproj")
DATA_DIR = Path(MATS_DIR / "xas.h5")
meta = pd.read_csv(MATS_DIR / "filtered_metadata.csv")
print(meta.head())
# %%
with h5py.File(DATA_DIR, "r") as f:
    keys = list(f.keys())
    first_key = keys[0]
    first_group = f[first_key]
    raw_json = first_group.attrs["structure_json"]
    print(isinstance(raw_json, bytes))
    if isinstance(raw_json, bytes):
        raw_json = raw_json.decode('utf-8')
    parsed_json = json.loads(raw_json)
    print(f"First group name: {first_key}")
    print(json.dumps(parsed_json, indent=4))

# %%
from pymatgen.core import Structure
import json, h5py
with h5py.File("data/xas/matproj/xas.h5", "r") as f:
    grp = f["struct-0000488545b5_Ca_K_EXAFS"]
    s = Structure.from_dict(json.loads(grp.attrs["structure_json"]))

s.to(filename="structure.cif")
# %%
print("num_sites:", len(s))
print("species:", [str(site.specie) for site in s])
print("cart_coords shape:", s.cart_coords.shape)
print(s.cart_coords[:5])

print("lattice matrix")
print(s.lattice.matrix)

print("atomic numbers:", [site.specie.Z for site in s])


# %%
r_max = 5.0
center_indices, neighbor_indices, image_vectors, distances = s.get_neighbor_list(r_max)

print("num_edges:", len(center_indices))
print("distance min:", distances.min())
print("distance max:", distances.max())
print("distance mean:", distances.mean())

for j in range(min(5, len(center_indices))):
    src = center_indices[j]
    dst = neighbor_indices[j]
    img = image_vectors[j]
    dist = distances[j]
    print(f"  atom {src} ({s[src].specie}) -> atom {dst} ({s[dst].specie}), "
          f"image={img}, dist={dist:.3f} Å")
# %%
import torch
import ase
Z_max = 94
n_atoms = len(s)
atomic_numbers = [site.specie.Z for site in s]

# z_in: plain one-hot by atomic number
z_in = torch.zeros(n_atoms, Z_max, dtype=torch.float64)
for atom_idx, Z in enumerate(atomic_numbers):
    z_in[atom_idx, Z - 1] = 1.0    # Z is 1-indexed, tensor is 0-indexed

# x_in: mass-weighted one-hot
from ase.data import atomic_masses
x_in = torch.zeros(n_atoms, Z_max, dtype=torch.float64)
for atom_idx, Z in enumerate(atomic_numbers):
    x_in[atom_idx, Z - 1] = atomic_masses[Z]

# Verify: for atom 0 (which is Ca, Z=20), x_in should have mass 40.078 at index 19
print("atom 0 species:", s[0].specie, "Z:", s[0].specie.Z)
print("x_in[0] nonzero index:", torch.nonzero(x_in[0]).item(), "value:", x_in[0, s[0].specie.Z - 1].item())
print("z_in[0] nonzero index:", torch.nonzero(z_in[0]).item(), "value:", z_in[0, s[0].specie.Z - 1].item())

# %%
from torch_geometric.data import Data

data = Data(
    pos        = torch.tensor(s.cart_coords, dtype=torch.float64),
    x_in       = x_in,
    z_in       = z_in,
    edge_index = torch.tensor([center_indices, neighbor_indices], dtype=torch.long),
    edge_shift = torch.tensor(image_vectors, dtype=torch.float64),
    lattice    = torch.tensor(s.lattice.matrix, dtype=torch.float64).unsqueeze(0),
    symbol     = [str(site.specie) for site in s],
    absorber_Z = torch.tensor(20, dtype=torch.long),  # Ca is the absorber for this sample
)

# %%
print(data)
print("num_nodes:", data.num_nodes)
print("num_edges:", data.edge_index.shape[1])
print("symbols:", data.symbol)
print("absorber_Z:", data.absorber_Z.item())

src, dst = data.edge_index
edge_vec = (data.pos[dst] - data.pos[src]
            + torch.einsum('ni,ij->nj', data.edge_shift, data.lattice.squeeze(0)))
edge_dist = edge_vec.norm(dim=1)
print("edge distance min:", edge_dist.min().item())
print("edge distance max:", edge_dist.max().item())
print("edges with dist < 0.1:", (edge_dist < 0.1).sum().item())  # ideally 0
# %%
