import h5py
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from ase.data import atomic_masses
from pymatgen.core import Structure, Element

MATS_DIR = Path("data/xas/matproj")
FMETA_DIR = Path(MATS_DIR / "filtered_metadata.csv")
DATA_DIR = Path(MATS_DIR / "xas.h5")

def prep_spectra(meta_dir: Path, data_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    meta = pd.read_csv(meta_dir)

    n = len(meta)
    all_xanes = np.empty((n, 650))
    all_exafs = np.empty((n, 2950))

    with h5py.File(data_dir, "r") as f:
        for i, row in enumerate(tqdm(meta.itertuples(), total=len(meta))):
            xafs_grp = f[row.spectrum_id]
            x_xafs = xafs_grp["x"][:]
            y_xafs = xafs_grp["y"][:]
            e0 = float(xafs_grp.attrs["e0"])
            xanes_grid = np.linspace(e0 - 15, e0 + 50, 650)
            y_xanes = np.interp(xanes_grid, x_xafs, y_xafs)
            exafs_grid = np.linspace(e0 + 50, e0 + 1525, 2950)
            y_exafs = np.interp(exafs_grid, x_xafs, y_xafs)
            all_xanes[i] = y_xanes
            all_exafs[i] = y_exafs

    return all_xanes, all_exafs

def prep_structure(meta_dir: Path, data_dir: Path, r_max: float = 5.0):
    meta = pd.read_csv(meta_dir)
    Z_max = 94

    all_graphs = []

    with h5py.File(data_dir, "r") as f:
        for _, row in enumerate(tqdm(meta.itertuples(), total=len(meta))):
            xafs_grp = f[row.spectrum_id]
            raw_json = xafs_grp.attrs["structure_json"]
            s = Structure.from_dict(json.loads(raw_json))

            center_indices, neighbor_indices, image_vectors, _ = s.get_neighbor_list(r_max)

            n_atoms = len(s)
            atomic_numbers = [site.specie.Z for site in s]

            z_in = torch.zeros(n_atoms, Z_max, dtype=torch.float64)
            x_in = torch.zeros(n_atoms, Z_max, dtype=torch.float64)
            for atom_idx, Z in enumerate(atomic_numbers):
                z_in[atom_idx, Z - 1] = 1.0
                x_in[atom_idx, Z - 1] = atomic_masses[Z]

            absorber_Z = Element(row.absorbing_element).Z

            data = Data(
                pos=torch.tensor(s.cart_coords, dtype=torch.float64),
                x_in=x_in,
                z_in=z_in,
                edge_index=torch.tensor(np.stack([center_indices, neighbor_indices]), dtype=torch.long),
                edge_shift=torch.tensor(np.array(image_vectors), dtype=torch.float64),
                lattice=torch.tensor(s.lattice.matrix, dtype=torch.float64).unsqueeze(0),
                absorber_Z=torch.tensor(absorber_Z, dtype=torch.long),
            )

            all_graphs.append(data)

    return all_graphs


def main() -> None:
    print("Starting spectrum preprocessing...")
    xanes, exafs = prep_spectra(FMETA_DIR, DATA_DIR)
    struct_absorber = prep_structure(FMETA_DIR, DATA_DIR)
    np.save(MATS_DIR / "processed_xanes.npy", xanes)
    np.save(MATS_DIR / "processed_exafs.npy", exafs)
    torch.save(struct_absorber, MATS_DIR / "processed_structs.pt")
    print(f"Preprocessed spectra and structure files saved to {MATS_DIR}")

if __name__ == "__main__":
    main()