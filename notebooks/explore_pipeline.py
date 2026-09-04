# %%

# %%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
# %%
DATA_DIR = Path("data/xas/matproj")
meta = pd.read_csv(DATA_DIR / "metadata.csv")
log: list[dict] = []

# %%
# initial diagnostics
before = len(meta)
print(meta.shape)
print(meta.head())
print(meta["spectrum_type"].value_counts())
print(meta["edge"].value_counts())
after = len(meta)
log.append({"step" : "initial diagnostics", 
           "rows_before": before, 
           "rows_after": after, 
           "removed": before - after})

# %%
# use only XAFS
before = len(meta)
meta_xafs = meta[meta["spectrum_type"] == "XAFS"].copy()
after = len(meta_xafs)
log.append({"step" : "use only XAFS", 
           "rows_before": before, 
           "rows_after": after, 
           "removed": before - after})


# %%
# use only K-edge
before = len(meta_xafs)
meta_xafs_k = meta_xafs[meta_xafs["edge"] == "K"].copy()
after = len(meta_xafs_k)
log.append({"step" : "use only K-edge", 
           "rows_before": before, 
           "rows_after": after, 
           "removed": before - after})


# %%
# e0 safety guard
before = len(meta_xafs_k)
meta_xafs_k_e0 = meta_xafs_k[(meta_xafs_k["e0"] >= 10) & (meta_xafs_k["e0"] <= 125000)].copy()
after = len(meta_xafs_k_e0)
log.append({"step" : "e0 in [10, 125000] eV",
           "rows_before": before,
           "rows_after": after,
           "removed": before - after})

# %%
# drop y_min < 0
before = len(meta_xafs_k_e0)
meta_xafs_k_e0_ypos = meta_xafs_k_e0[meta_xafs_k_e0["y_min"] >= 0].copy()
after = len(meta_xafs_k_e0_ypos)
log.append({"step" : "y_min >= 0",
           "rows_before": before,
           "rows_after": after,
           "removed": before - after})

# %%
# drop Krypton (mp-636056)
before = len(meta_xafs_k_e0_ypos)
meta_xafs_k_e0_ypos_nokr = meta_xafs_k_e0_ypos[meta_xafs_k_e0_ypos["material_id"] != "mp-636056"].copy()
after = len(meta_xafs_k_e0_ypos_nokr)
log.append({"step" : "drop mp-636056 (Kr)",
           "rows_before": before,
           "rows_after": after,
           "removed": before - after})

# %%
# compute XAFS-XANES disagreement and y_max offset from e0 (single HDF5 pass)
current = meta_xafs_k_e0_ypos_nokr
xanes_lookup = (
    meta[(meta["spectrum_type"] == "XANES") & (meta["edge"] == "K")]
    .set_index(["material_id", "absorbing_element"])["spectrum_id"]
    .to_dict()
)
disagreements = np.full(len(current), np.nan)
ymax_offsets = np.full(len(current), np.nan)
with h5py.File(DATA_DIR / "xas.h5", "r") as f:
    for i, (_, row) in enumerate(tqdm(current.iterrows(), total=len(current))):
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
current = current.copy()
current["disagreement"] = disagreements
current["ymax_offset"] = ymax_offsets

# %%
# disagreement <= 0.15 (keep NaN = no XANES counterpart to score against)
before = len(current)
meta_q = current[(current["disagreement"].isna()) | (current["disagreement"] <= 0.15)].copy()
after = len(meta_q)
log.append({"step" : "XAFS-XANES disagreement <= 0.15",
           "rows_before": before,
           "rows_after": after,
           "removed": before - after})

# %%
# y_max within 50 eV of e0
before = len(meta_q)
meta_q_ymax = meta_q[meta_q["ymax_offset"] <= 50.0].copy()
after = len(meta_q_ymax)
log.append({"step" : "y_max within 50 eV of e0",
           "rows_before": before,
           "rows_after": after,
           "removed": before - after})

# %%
# drop rare absorbing elements (< 50 samples)
MIN_SAMPLES_PER_ELEMENT = 50
counts = meta_q_ymax["absorbing_element"].value_counts()
keep_elements = counts[counts >= MIN_SAMPLES_PER_ELEMENT].index
before = len(meta_q_ymax)
meta_final = meta_q_ymax[meta_q_ymax["absorbing_element"].isin(keep_elements)].copy()
after = len(meta_final)
log.append({"step" : f"absorbing_element >= {MIN_SAMPLES_PER_ELEMENT} samples",
           "rows_before": before,
           "rows_after": after,
           "removed": before - after})

# %%
# filtration summary
print(pd.DataFrame(log).to_string(index=False))

# %%
# write filtered metadata
meta_final.reset_index(drop=True).to_csv(DATA_DIR / "paired_metadata.csv", index=True)

# %%
print(*log, sep='\n')
# TODO: train/valid/test split via xasbind.data.splits.grouped_stratified_split


# %%
# %%
# material attrition: how many materials lost absorber sites during filtration?

# original (pre-filter) absorber sets, restricted to the same scope we were filtering
orig_scope = meta[(meta["spectrum_type"] == "XAFS") & (meta["edge"] == "K")]
orig_sets = orig_scope.groupby("material_id")["absorbing_element"].agg(set)
final_sets = meta_final.groupby("material_id")["absorbing_element"].agg(set)

# align: for every material that survived at all, compare its surviving set to its original
common = final_sets.index
lost_counts = (orig_sets.loc[common].apply(len) - final_sets.apply(len))

print(f"materials in original K-edge XAFS scope:  {len(orig_sets)}")
print(f"materials surviving (>=1 site) in meta_final: {len(final_sets)}")
print(f"materials completely wiped:               {len(orig_sets) - len(final_sets)}")
print()
print("of the surviving materials, sites lost:")
print(lost_counts.value_counts().sort_index().to_string())

# Result:
"""
materials in original K-edge XAFS scope:  44744
materials surviving (>=1 site) in meta_final: 42022
materials completely wiped:               2722

of the surviving materials, sites lost:
absorbing_element
0    21373
1    15622
2     4796
3      228
4        3
"""

# %%
orig_size_in_surviving = orig_sets.loc[common].apply(len)
print()
print("among surviving materials, original site count distribution:")
print(orig_size_in_surviving.value_counts().sort_index().to_string())
print()
multi = orig_size_in_surviving > 1
print(f"of {multi.sum()} surviving multi-site materials, "
      f"{(lost_counts[multi] == 0).sum()} are fully intact "
      f"({(lost_counts[multi] == 0).mean():.1%})")
"""
among surviving materials, original site count distribution:
absorbing_element
1     2270
2     7286
3    19352
4    10241
5     2538
6      315
7       20

of 39752 surviving multi-site materials, 19103 are fully intact (48.1%)
"""
# %%
