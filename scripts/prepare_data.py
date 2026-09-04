import h5py 
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from xasbind.data.splits import grouped_stratified_split, report_split


MATS_DIR = Path("data/xas/matproj")
META_DIR = Path(MATS_DIR / "metadata.csv")
DATA_DIR = Path(MATS_DIR / "xas.h5")
VALID_SIZE = 0.20
TEST_SIZE = 0.10
N_SPLITS = None
RANDOM_STATE = 20
MIN_SAMPLES_PER_ELEMENT = 50


def filter_data(meta_dir: Path, data_dir: Path) -> pd.DataFrame:
    log: list[dict] = []
    meta = pd.read_csv(meta_dir)

    meta_orig = meta.copy()
    before = len(meta)
    meta = meta[meta["spectrum_type"] == "XAFS"].copy()
    after = len(meta)
    log.append({"step" : "use only XAFS", 
                "rows_before": before, 
                "rows_after": after, 
                "removed": before - after})
    
    before = len(meta)
    meta = meta[meta["edge"] == "K"].copy()
    after = len(meta)
    log.append({"step" : "use only K-edge", 
                "rows_before": before, 
                "rows_after": after, 
                "removed": before - after})
    
    before = len(meta)
    meta = meta[(meta["e0"] >= 10) & (meta["e0"] <= 125000)].copy()
    after = len(meta)
    log.append({"step" : "e0 in [10, 125000] eV",
                "rows_before": before,
                "rows_after": after,
                "removed": before - after})
    
    before = len(meta)
    meta = meta[meta["y_min"] >= 0].copy()
    after = len(meta)
    log.append({"step" : "y_min >= 0",
                "rows_before": before,
                "rows_after": after,
                "removed": before - after})
    
    xanes_lookup = (
        meta_orig[(meta_orig["spectrum_type"] == "XANES") & (meta_orig["edge"] == "K")]
        .set_index(["material_id", "absorbing_element"])["spectrum_id"]
        .to_dict()
    )
    disagreements = np.full(len(meta), np.nan)
    ymax_offsets = np.full(len(meta), np.nan)
    with h5py.File(data_dir, "r") as f:
        for i, (_, row) in enumerate(tqdm(meta.iterrows(), total=len(meta))):
            xafs_grp = f[row["spectrum_id"]]
            x_xafs = xafs_grp["x"][:]
            y_xafs = xafs_grp["y"][:]
            e0 = float(xafs_grp.attrs["e0"])
            ymax_offsets[i] = float(x_xafs[np.argmax(y_xafs)]) - e0
            key = (row["material_id"], row["absorbing_element"])
            if key in xanes_lookup:
                xanes_grp = f[xanes_lookup[key]]
                x_xan = xanes_grp["x"][:]
                y_xan = xanes_grp["y"][:]
                y_xafs_on_xan = np.interp(x_xan, x_xafs, y_xafs)
                rng = y_xan.max() - y_xan.min()
                if rng > 0:
                    disagreements[i] = np.mean(np.abs(y_xafs_on_xan - y_xan)) / rng
    current = meta.copy()
    current["disagreement"] = disagreements
    current["ymax_offset"] = ymax_offsets
    before = len(current)
    meta = current[(current["disagreement"].isna()) | (current["disagreement"] <= 0.15)].copy()
    after = len(meta)
    log.append({"step" : "XAFS-XANES disagreement <= 0.15",
                "rows_before": before,
                "rows_after": after,
                "removed": before - after})
    
    before = len(meta)
    meta = meta[meta["ymax_offset"] <= 50.0].copy()
    after = len(meta)
    log.append({"step" : "y_max within 50 eV of e0",
                "rows_before": before,
                "rows_after": after,
                "removed": before - after})
    
    before = len(meta)
    meta = meta[meta["y_max"] < 10].copy()
    after = len(meta)
    log.append({"step" : "y_max < 10", 
                "rows_before": before, 
                "rows_after": after, 
                "removed": before - after})
    
    counts = meta["absorbing_element"].value_counts()
    keep_elements = counts[counts >= MIN_SAMPLES_PER_ELEMENT].index
    before = len(meta)
    meta = meta[meta["absorbing_element"].isin(keep_elements)].copy()
    after = len(meta)
    log.append({"step" : f"absorbing_element >= {MIN_SAMPLES_PER_ELEMENT} samples",
                "rows_before": before,
                "rows_after": after,
                "removed": before - after})
    
    
    element_counts = meta.groupby("material_id")["absorbing_element"].transform('nunique')
    meta["all_elements_present"] = meta["nelements"] == element_counts


    log = pd.DataFrame(log).to_string()
    return meta, log


def main() -> None:
    print("Starting data filtration...")
    filtered_meta, log = filter_data(META_DIR, DATA_DIR)
    print("Data filtration complete.")
    print(log)
    filtered_meta = filtered_meta.reset_index(drop=True)
    train_inds, valid_inds, test_inds, _ = grouped_stratified_split(filtered_meta, VALID_SIZE, TEST_SIZE, N_SPLITS, RANDOM_STATE)
    print("Dataset grouped into splits.")
    counts_check, dupes_check, strats_check = report_split(filtered_meta, train_inds, valid_inds, test_inds, VALID_SIZE, TEST_SIZE)
    print("Split report:")
    print(counts_check)
    print(dupes_check)
    print(strats_check)
    filtered_meta.to_csv(MATS_DIR / "filtered_metadata.csv")
    np.save(MATS_DIR / "train_inds.npy", train_inds)
    np.save(MATS_DIR / "valid_inds.npy", valid_inds)
    np.save(MATS_DIR / "test_inds.npy", test_inds)
    print(f"Dataset index files saved to {MATS_DIR}.")


if __name__ == "__main__":
    main()