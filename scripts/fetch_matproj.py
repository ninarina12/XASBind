#!/usr/bin/env python
"""Fetch all XAS spectra from the Materials Project into a single HDF5 file.

Each (material_id, absorbing_element, edge, spectrum_type) becomes one HDF5
group holding the spectrum (x, y), metadata attributes, and the serialized
pymatgen Structure. A companion metadata.csv indexes every group for fast
downstream pairing/splitting.

Provide your Materials Project API key via the MP_API_KEY environment variable
(recommended) or --api-key. Do not commit your key.

Example:
    export MP_API_KEY=...        # or pass --api-key
    python scripts/fetch_matproj.py --output-dir data/xas/matproj
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import pandas as pd
from mp_api.client import MPRester
import hashlib
from tqdm import tqdm


def fetch(api_key: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_file = output_dir / "xas.h5"
    metadata_file = output_dir / "metadata.csv"

    with MPRester(api_key=api_key) as mpr:
        # xas_docs = mpr.materials.xas.search(fields=["material_id", "spectrum"])
        xas_docs = mpr.materials.xas.search(fields=["spectrum"])

    saved = skipped = 0
    metadata_list: list[dict] = []

    with h5py.File(h5_file, "w") as f:
        for xas_doc in tqdm(xas_docs, total=len(xas_docs), desc="Writing spectra"):
            try:
                spectrum = xas_doc.spectrum
                # mp_id = str(xas_doc.material_id)
                struct_dict = json.dumps(spectrum.structure.as_dict(), sort_keys=True)
                mp_id = "struct-" + hashlib.md5(struct_dict.encode()).hexdigest()[:12]
                absorbing_element = str(spectrum.absorbing_element)
                edge = str(spectrum.edge)
                spectrum_type = spectrum.spectrum_type
                e0 = spectrum.e0.item()
                structure = spectrum.structure

                group_name = f"{mp_id}_{absorbing_element}_{edge}_{spectrum_type}"
                grp = f.create_group(group_name)

                grp.create_dataset("x", data=spectrum.x, compression="gzip")
                grp.create_dataset("y", data=spectrum.y, compression="gzip")

                grp.attrs["material_id"] = mp_id
                grp.attrs["absorbing_element"] = absorbing_element
                grp.attrs["edge"] = edge
                grp.attrs["spectrum_type"] = spectrum_type
                grp.attrs["e0"] = e0
                grp.attrs["structure_json"] = json.dumps(structure.as_dict())

                metadata_list.append({
                    "spectrum_id": group_name,
                    "material_id": mp_id,
                    "absorbing_element": absorbing_element,
                    "edge": edge,
                    "spectrum_type": spectrum_type,
                    "e0": e0,
                    "x_length": len(spectrum.x),
                    "x_min": spectrum.x.min().item(),
                    "x_max": spectrum.x.max().item(),
                    "y_min": spectrum.y.min().item(),
                    "y_max": spectrum.y.max().item(),
                    "formula": structure.composition.reduced_formula,
                    "num_sites": len(structure),
                    "nelements": len(structure.composition.elements),
                })
                saved += 1

            except Exception as e:  # noqa: BLE001 - keep going, report first few
                skipped += 1
                if skipped <= 10:
                    print(f"  Error: {e}")

    pd.DataFrame(metadata_list).to_csv(metadata_file, index=False)
    print(f"  Saved:    {saved:,} spectra -> {h5_file}")
    print(f"  Skipped:  {skipped:,}")
    print(f"  Metadata: {metadata_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data/xas/matproj"),
                        help="Directory for xas.h5 and metadata.csv.")
    parser.add_argument("--api-key", default=os.environ.get("MP_API_KEY"),
                        help="Materials Project API key (default: $MP_API_KEY).")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("No API key found. Set MP_API_KEY or pass --api-key.")

    fetch(args.api_key, args.output_dir)


if __name__ == "__main__":
    main()
