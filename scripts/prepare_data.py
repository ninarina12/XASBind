#!/usr/bin/env python
"""Build the paired XANES/EXAFS dataset and grouped train/valid/test splits.

Reads metadata.csv (from fetch_matproj.py), pairs XANES with EXAFS per
(material_id, absorbing_element, edge), filters rare absorbing elements, then
writes paired_metadata.csv and {train,valid,test}_indices.npy.

The split groups on material_id (no structure leaks across splits) and
approximately stratifies by absorbing_element.

Example:
    python scripts/prepare_data.py --data-dir data/xas/matproj \
        --valid-size 0.2 --test-size 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from xasbind.data.splits import grouped_stratified_split, report_split


def build_pairs(metadata: pd.DataFrame) -> pd.DataFrame:
    """Pair XANES with EXAFS per (material_id, absorbing_element, edge).

    Stored *_idx values are positional indices into `metadata`, which the
    dataloader maps to HDF5 group names via metadata['spectrum_id'].
    """
    paired: list[dict] = []
    grouped = metadata.groupby(["material_id", "absorbing_element", "edge"])
    for (mat_id, elem, edge), group in tqdm(grouped, total=len(grouped), desc="Pairing"):
        types = group["spectrum_type"].values
        if "XANES" in types and "EXAFS" in types:
            paired.append({
                "material_id": mat_id,
                "absorbing_element": elem,
                "edge": edge,
                "xanes_idx": group[group["spectrum_type"] == "XANES"].index[0],
                "exafs_idx": group[group["spectrum_type"] == "EXAFS"].index[0],
                "source": "native_pair",
            })
        elif "XAFS" in types:
            # Could be split into XANES/EXAFS by an energy threshold later.
            paired.append({
                "material_id": mat_id,
                "absorbing_element": elem,
                "edge": edge,
                "xafs_idx": group[group["spectrum_type"] == "XAFS"].index[0],
                "source": "split_xafs",
            })
        # else: only XANES or only EXAFS -> skip for the paired dataset
    return pd.DataFrame(paired)


def filter_rare_elements(paired_df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    counts = paired_df["absorbing_element"].value_counts()
    valid = counts[counts >= min_samples].index
    removed = counts[counts < min_samples]
    if len(removed):
        print(f"Removing {len(removed)} elements with < {min_samples} samples:")
        for elem, c in removed.items():
            print(f"  {elem}: {c} samples")
        out = paired_df[paired_df["absorbing_element"].isin(valid)].copy()
        print(f"\nBefore: {len(paired_df)} pairs, {len(counts)} elements")
        print(f"After:  {len(out)} pairs, {len(valid)} elements")
        return out
    print(f"All elements have >= {min_samples} samples")
    return paired_df.copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/xas/matproj"))
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--n-splits", type=int, default=None,
                        help="Override the auto-chosen fold count.")
    parser.add_argument("--min-samples-per-element", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=12)
    args = parser.parse_args()

    metadata = pd.read_csv(args.data_dir / "metadata.csv")
    print(f"Total spectra: {len(metadata)}")

    paired_df = build_pairs(metadata)
    print("\nPaired dataset statistics:")
    print(f"  Native pairs (XANES & EXAFS): {(paired_df['source'] == 'native_pair').sum()}")
    print(f"  From XAFS (can be split):     {(paired_df['source'] == 'split_xafs').sum()}")
    print(f"  Total paired samples:         {len(paired_df)}\n")

    paired_df = filter_rare_elements(paired_df, args.min_samples_per_element)

    train_idx, valid_idx, test_idx, n_splits = grouped_stratified_split(
        paired_df,
        valid_size=args.valid_size,
        test_size=args.test_size,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    # Reset the same way the split did, so indices line up with the saved rows.
    paired_df = paired_df.reset_index(drop=True)

    print()
    report_split(paired_df, train_idx, valid_idx, test_idx,
                 n_splits, args.valid_size, args.test_size)

    paired_df.to_csv(args.data_dir / "paired_metadata.csv", index=True)  # index matters
    np.save(args.data_dir / "train_indices.npy", train_idx)
    np.save(args.data_dir / "valid_indices.npy", valid_idx)
    np.save(args.data_dir / "test_indices.npy", test_idx)
    print(f"\nWrote paired_metadata.csv and train/valid/test_indices.npy to {args.data_dir}")


if __name__ == "__main__":
    main()
