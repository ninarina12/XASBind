# %%

# %%
# 0 - Imports, Load
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/xas/matproj")
meta = pd.read_csv(DATA_DIR / "metadata.csv")
meta.head()

# %%
# 1 - Unique material_ids
mat_ids = meta['material_id']
print(f"total material_ids:  {len(mat_ids)}")
mat_ids_set = set(mat_ids)
print(f" unique material_ids: {len(mat_ids_set)}")
print(f"{((len(mat_ids) - len(mat_ids_set)) / len(mat_ids)) * 100} % of the material_ids are repeats.")

# %%
# 2 - Distribution of spectrum_type
import matplotlib.pyplot as plt
spec_type = meta['spectrum_type'].value_counts()
plt.figure(figsize=(8, 5))
spec_type.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("spectrum_type frequency dist")
plt.xlabel("spectrum_type")
plt.ylabel("count")
plt.show()

# %%
# 3 - Distribution of most comment absorbing_element
abs_element = meta['absorbing_element'].value_counts().head(20)
plt.figure(figsize=(8, 5))
abs_element.plot(kind='bar', color='red', edgecolor='black')
plt.title('absorbing_element freq dist')
plt.xlabel("absorbing_element")
plt.ylabel("count")
plt.show() 

# %%
# 4 - Spectrum length distribution
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 15))
meta["x_length"].hist(bins=50, ax = axes[0])
axes[0].set_title("x_length")
meta["x_min"].hist(bins=50, ax = axes[1])
axes[1].set_title("x_min")
meta["x_max"].hist(bins=50, ax=axes[2])
axes[2].set_title("x_max")
meta.groupby("spectrum_type")[["x_min", "x_max", "x_length"]].describe().xs('mean', axis=1, level=1)

# %%
# 5 - Distribution of energy range relative to e0
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12,15))
(meta["x_min"] - meta["e0"]).hist(bins=50, ax=axes[0])
axes[0].set_title("offset of x_min from e0")
(meta["x_max"] - meta["e0"]).hist(bins=50, ax=axes[1])
axes[1].set_title("offset of x_max from e0")

# %%
# 6 - Per-type spacing
meta["spacing"] = (meta["x_max"] - meta["x_min"]) / (meta["x_length"] - 1)
print(meta.groupby("spectrum_type")["spacing"].describe())
meta.hist(column="x_length", by="spectrum_type", bins=50, figsize=(12, 4), layout=(1, 3))


# %%
# 7 - Investigate XANES Outlier
outliers = meta[(meta.spectrum_type=="XANES") & (meta.x_length==500)]
outliers_L_edge = meta[(meta.spectrum_type=="XANES") & (meta.x_length==500) & (meta.edge=="L2,3")]
print(outliers)
print(len(outliers))
print(len(outliers_L_edge))

# %%
# 8 - Examine HDF5 file
import h5py
with h5py.File(DATA_DIR / "xas.h5", "r") as f:
    print(len(f))
    keys = list(f.keys())
    print(keys[:5])

    chosen_idx = 0
    meta_row = meta.iloc[chosen_idx]
    grp = f[meta_row["spectrum_id"]]
    print(dict(grp.attrs))
    print(list(grp.keys()))
    print(grp["x"].shape, grp["x"].dtype)
    print(grp["y"].shape, grp["y"].dtype)
    print(meta_row)



# %%
# 9 - Plot a single spectrum
row = meta[(meta.absorbing_element == "Mn") & (meta.edge == "K") & (meta.spectrum_type == "XANES")].iloc[0]
sample = row["material_id"]
row2 = meta[(meta.absorbing_element == "Mn") & (meta.edge == "K") & (meta.spectrum_type == "EXAFS") & (meta.material_id == sample)].iloc[0]
row3 = meta[(meta.absorbing_element == "Mn") & (meta.edge == "K") & (meta.spectrum_type == "XAFS") & (meta.material_id == sample)].iloc[0]



with h5py.File(DATA_DIR / "xas.h5", "r") as f:
    grp = f[row["spectrum_id"]]
    x = grp["x"][:]
    y = grp["y"][:]
    e0 = grp.attrs["e0"]

    grp2 = f[row2["spectrum_id"]]
    x2 = grp2["x"][:]
    y2 = grp2["y"][:]
    e02 = grp2.attrs["e0"]

    grp3 = f[row3["spectrum_id"]]
    x3 = grp3["x"][:]
    y3 = grp3["y"][:]
    e03 = grp3.attrs["e0"]

plt.plot(x, y, label="XANES")
plt.plot(x2, y2, label="EXAFS")
plt.plot(x3, y3, label="XAFS")
plt.axvline(e0, ls="--", color="black", label=f"e0 from XANES = {e0:.1f} eV")
plt.axvline(e02, ls="--", color="green", label=f"e0 from EXAFS = {e02:.1f} eV")
plt.axvline(e03, ls="--", color="orange", label=f"e0 from XAFS = {e03:.1f} eV")
plt.xlim(e0 - 20, e0 + 60)
plt.legend()

plt.savefig("notebooks/cell9_fe_Mn.png", dpi=150, bbox_inches="tight")

# %%
# 10 - Check for universal overlap of XAFS and XANES
import numpy as np
from tqdm import tqdm

triple_cols = ["material_id", "absorbing_element", "edge"]

diffs = []
skipped = 0

with h5py.File(DATA_DIR / "xas.h5", "r") as f:
    for key, sub in tqdm(meta.groupby(triple_cols)):
        xanes_rows = sub[sub["spectrum_type"] == "XANES"]
        xafs_rows = sub[sub["spectrum_type"] == "XAFS"]

        if xanes_rows.empty or xafs_rows.empty:
            skipped +=1
            continue

        xanes_id = xanes_rows.iloc[0]["spectrum_id"]
        xafs_id = xafs_rows.iloc[0]["spectrum_id"]

        x_xanes = f[xanes_id]["x"][:]
        y_xanes = f[xanes_id]["y"][:]
        x_xafs = f[xafs_id]["x"][:]
        y_xafs = f[xafs_id]["y"][:]

        mask = (x_xafs >= x_xanes.min()) & (x_xafs <= x_xanes.max())
        if mask.sum() < 5:
            skipped += 1
            continue

        y_xanes_on_xafs = np.interp(x_xafs[mask], x_xanes, y_xanes)

        amp = np.ptp(y_xanes)
        if amp == 0:
            skipped += 1
            continue
        err = np.mean(np.abs(y_xafs[mask] - y_xanes_on_xafs)) / amp
        diffs.append(err)

print(f"compared: {len(diffs)}   skipped: {skipped}")
diffs = np.array(diffs)
print("median:", np.median(diffs), "  mean:", diffs.mean(), "  p95:", np.quantile(diffs, 0.95))

plt.hist(diffs, bins=50)
plt.xlabel("mean |XAFS - XANES| / range(XANES)")
plt.ylabel("count")
plt.title(f"XAFS vs XANES agreement (n={len(diffs)})")
        

# %%
# 11 - e0 consistency across spectrum_types
e0_wide = meta.pivot_table(
    index=triple_cols,
    columns="spectrum_type",
    values="e0",
).dropna()
print("XAFS  - XANES:"); print((e0_wide["XAFS"]  - e0_wide["XANES"]).describe())
print("EXAFS - XANES:"); print((e0_wide["EXAFS"] - e0_wide["XANES"]).describe())
# %%
# %%
# 10b - Zoom into the tail of the disagreement distribution
import numpy as np

print(f"% below 0.01 (1%):  {(diffs < 0.01).mean() * 100:.2f}")
print(f"% below 0.05 (5%):  {(diffs < 0.05).mean() * 100:.2f}")
print(f"% below 0.10 (10%): {(diffs < 0.10).mean() * 100:.2f}")
print(f"% below 0.15 (15%): {(diffs < 0.15).mean() * 100:.2f}")
print(f"% below 0.20 (20%): {(diffs < 0.20).mean() * 100:.2f}")
print(f"max: {diffs.max():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left: log-scale, full range — shows overall shape including tail
axes[0].hist(diffs, bins=np.logspace(-4, 1, 60))
axes[0].set_xscale("log")
axes[0].set_xlabel("disagreement (log)")
axes[0].set_ylabel("count")
axes[0].set_title("Full distribution, log x")
axes[0].axvline(0.05, color="red", ls="--", label="5%")
axes[0].legend()

# Right: zoom on the 1–15% tail you care about for thresholding
axes[1].hist(diffs[(diffs >= 0.01) & (diffs <= 0.15)], bins=50)
axes[1].set_xlabel("disagreement")
axes[1].set_ylabel("count")
axes[1].set_title("Zoom: 1% – 15% tail")
axes[1].axvline(0.05, color="red", ls="--", label="5%")
axes[1].legend()

plt.tight_layout()

# %%
# how many materials have insane e0?
print("XANES e0 outside [0, 50000] eV:", ((meta["e0"] < 0) | (meta["e0"] > 50000)).sum())
print("XANES e0 outside [0, 50000] by type:")
print(meta[(meta["e0"] < 0) | (meta["e0"] > 50000)]["spectrum_type"].value_counts())

# look at typical |XAFS - XANES| e0 difference, ignoring extreme outliers
diff_e0 = (e0_wide["XAFS"] - e0_wide["XANES"]).abs()
print("\n|XAFS - XANES| e0:")
print(diff_e0.describe(percentiles=[0.5, 0.9, 0.99, 0.999]))

# same for EXAFS
diff_e0_exafs = (e0_wide["EXAFS"] - e0_wide["XANES"]).abs()
print("\n|EXAFS - XANES| e0:")
print(diff_e0_exafs.describe(percentiles=[0.5, 0.9, 0.99, 0.999]))

# %%
