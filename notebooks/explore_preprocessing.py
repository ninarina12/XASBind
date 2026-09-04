# %%

# %%
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %%
FMETA_DIR = Path("data/xas/matproj")
fmeta = pd.read_csv(FMETA_DIR / "filtered_metadata.csv")
print(fmeta.shape)

met = fmeta.copy()

met["x_relmax"] = met["x_max"] - met["e0"]
counts = pd.cut(met["x_relmax"], bins=5).value_counts().sort_index()
print(counts)

# %%
import h5py
print(met.groupby("spectrum_type")[["x_length", "x_min", "x_max"]].mean())


# %%
fmeta = pd.read_csv("data/xas/matproj/filtered_metadata.csv")
row = fmeta.iloc[0]
sid = row["spectrum_id"]

with h5py.File("data/xas/matproj/xas.h5", "r") as f:
    x = f[sid]["x"][:]
    y = f[sid]["y"][:]

diffs = np.diff(x)
print(f"spectrum_id: {sid}")
print(f"e0: {row['e0']:.2f}, x_min: {x[0]:.2f}, x_max: {x[-1]:.2f}")
print(f"num points: {len(x)}")
print(f"energy steps — min: {diffs.min():.4f}, max: {diffs.max():.4f}, mean: {diffs.mean():.4f}, std: {diffs.std():.4f}")
print(f"first 10 diffs: {np.round(diffs[:10], 4)}")
print(f"last 10 diffs:  {np.round(diffs[-10:], 4)}")

# how many raw points fall in the XANES region (e0 to e0+50)?
e0 = row["e0"]
xanes_mask = (x >= e0 - 9) & (x <= e0 + 50)
print(f"\npoints in XANES region [e0, e0+50]: {xanes_mask.sum()}")
print(f"diffs in XANES region: {np.round(np.diff(x[xanes_mask]), 4)}")
# %%
e0 = row["e0"]
exafs_mask = (x >= e0 + 50)
print(f"\npoints in EXAFS region [e0+50): {exafs_mask.sum()}")
print(f"diffs in EXAFS region: {np.round(np.diff(x[exafs_mask]), 4)}")

# %%
# average xafs spectrum width
met["x_range"] = met["x_max"] - met["x_min"]
met["x_range"].plot(kind='hist', bins=100, edgecolor='black')

# %%
print(met["nelements"].head())

# %%
duplicate_count = met.duplicated(subset=["material_id", "absorbing_element"]).sum()
print(duplicate_count)

# %%
import h5py 
import numpy as np
import random

fmeta = pd.read_csv("data/xas/matproj/filtered_metadata.csv")
row_num = random.randint(0, len(fmeta))
row = fmeta.iloc[row_num]
spec_id = row["spectrum_id"]


with h5py.File("data/xas/matproj/xas.h5", "r") as f:
    x = f[spec_id]["x"][:]
    y = f[spec_id]["y"][:]

fig, ax = plt.subplots(figsize=(100, 5))
ax.plot(x, y, '-', linewidth=0.8, color='steelblue', label='Curve')
ax.plot(x, y, 'o', markersize=2, color='red', label=f'Sample points ({len(x)})')
ax.axvline(row["e0"], color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {row["e0"]:.1f}')
ax.axvline((row["e0"] + 50), color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {row["e0"]:.1f}')
ax.set_title(f'{spec_id}')
ax.set_xlabel('Energy (eV)')
# ax.set_xlim(1000, 1200)
ax.set_ylabel('mu(E)')
ax.legend()
plt.tight_layout()
plt.show()






# %%
# XANES region: everything up to e0 + 75
e0 = row["e0"]
mask = x <= e0 + 75
x_xanes, y_xanes = x[mask], y[mask]

# Resample at 0.1 eV
x_resampled = np.arange(x_xanes[0], x_xanes[-1], 0.1)
y_resampled = np.interp(x_resampled, x_xanes, y_xanes)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top: raw points
axes[0].plot(x_xanes, y_xanes, '-', linewidth=0.8, color='steelblue', label='Curve')
axes[0].plot(x_xanes, y_xanes, 'o', markersize=3, color='red', label=f'Raw points ({len(x_xanes)})')
axes[0].axvline(e0, color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {e0:.1f}')
axes[0].axvline((e0 + 50), color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {e0 + 50:.1f}')
axes[0].set_title(f'{spec_id} — XANES raw')
axes[0].set_xlabel('Energy (eV)')
axes[0].set_ylabel('μ(E)')
axes[0].legend()

# Bottom: resampled at 0.1 eV
axes[1].plot(x_resampled, y_resampled, '-', linewidth=0.8, color='steelblue', label='Curve')
axes[1].plot(x_resampled, y_resampled, 'o', markersize=2, color='red', label=f'0.1 eV points ({len(x_resampled)})')
axes[1].axvline(e0, color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {e0:.1f}')
axes[1].axvline((e0 + 50), color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {e0 + 50:.1f}')
axes[1].set_title(f'{spec_id} — XANES resampled 0.1 eV')
axes[1].set_xlabel('Energy (eV)')
axes[1].set_ylabel('μ(E)')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Raw: {len(x_xanes)} points, Resampled: {len(x_resampled)} points")

# %%
# %%
met["x_relmin"] = met["x_min"] - met["e0"]
counts = pd.cut(met["x_relmin"], bins=10).value_counts().sort_index()
print(counts)
bin_categories = pd.cut(met["x_relmin"], bins=10)
# %%
from tqdm import tqdm

x_relmins = []
y_firsts = []

with h5py.File("data/xas/matproj/xas.h5", "r") as f:
    for _, row in tqdm(met.iterrows(), total=len(met)):
        grp = f[row["spectrum_id"]]
        x = grp["x"][:]
        y = grp["y"][:]
        x_relmins.append(x[0] - row["e0"])
        y_firsts.append(y[0])

x_relmins = np.array(x_relmins)
y_firsts = np.array(y_firsts)

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(x_relmins, y_firsts, s=1, alpha=0.3)
ax.set_xlabel('x_min - e0 (eV)')
ax.set_ylabel('y at leftmost point')
ax.set_title('Absorption at left edge of each spectrum')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()

# %%
met["bin_range"] = pd.cut(met["x_relmin"], bins=10)

met["y_firsts"] = y_firsts  
summary = met.groupby("bin_range", observed=False).agg(
    count=('x_relmin', 'count'),
    avg_x_relmin=('x_relmin', 'mean'),
    avg_y_absorption=('y_firsts', 'mean')
)

print(summary.to_string())

# %%
# %%

# %%
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %%
FMETA_DIR = Path("data/xas/matproj")
fmeta = pd.read_csv(FMETA_DIR / "filtered_metadata.csv")
print(fmeta.shape)

met = fmeta.copy()

met["x_relmax"] = met["x_max"] - met["e0"]
counts = pd.cut(met["x_relmax"], bins=5).value_counts().sort_index()
print(counts)

# %%
import h5py
print(met.groupby("spectrum_type")[["x_length", "x_min", "x_max"]].mean())


# %%
fmeta = pd.read_csv("data/xas/matproj/filtered_metadata.csv")
row = fmeta.iloc[0]
sid = row["spectrum_id"]

with h5py.File("data/xas/matproj/xas.h5", "r") as f:
    x = f[sid]["x"][:]
    y = f[sid]["y"][:]

diffs = np.diff(x)
print(f"spectrum_id: {sid}")
print(f"e0: {row['e0']:.2f}, x_min: {x[0]:.2f}, x_max: {x[-1]:.2f}")
print(f"num points: {len(x)}")
print(f"energy steps — min: {diffs.min():.4f}, max: {diffs.max():.4f}, mean: {diffs.mean():.4f}, std: {diffs.std():.4f}")
print(f"first 10 diffs: {np.round(diffs[:10], 4)}")
print(f"last 10 diffs:  {np.round(diffs[-10:], 4)}")

# how many raw points fall in the XANES region (e0 to e0+50)?
e0 = row["e0"]
xanes_mask = (x >= e0 - 9) & (x <= e0 + 50)
print(f"\npoints in XANES region [e0, e0+50]: {xanes_mask.sum()}")
print(f"diffs in XANES region: {np.round(np.diff(x[xanes_mask]), 4)}")
# %%
e0 = row["e0"]
exafs_mask = (x >= e0 + 50)
print(f"\npoints in EXAFS region [e0+50): {exafs_mask.sum()}")
print(f"diffs in EXAFS region: {np.round(np.diff(x[exafs_mask]), 4)}")

# %%
# average xafs spectrum width
met["x_range"] = met["x_max"] - met["x_min"]
met["x_range"].plot(kind='hist', bins=100, edgecolor='black')

# %%
print(met["nelements"].head())

# %%
duplicate_count = met.duplicated(subset=["material_id", "absorbing_element"]).sum()
print(duplicate_count)

# %%
import h5py 
import numpy as np
import random

fmeta = pd.read_csv("data/xas/matproj/filtered_metadata.csv")
row_num = random.randint(0, len(fmeta))
row = fmeta.iloc[row_num]
spec_id = row["spectrum_id"]


with h5py.File("data/xas/matproj/xas.h5", "r") as f:
    x = f[spec_id]["x"][:]
    y = f[spec_id]["y"][:]

fig, ax = plt.subplots(figsize=(100, 5))
ax.plot(x, y, '-', linewidth=0.8, color='steelblue', label='Curve')
ax.plot(x, y, 'o', markersize=2, color='red', label=f'Sample points ({len(x)})')
ax.axvline(row["e0"], color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {row["e0"]:.1f}')
ax.axvline((row["e0"] + 50), color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {row["e0"]:.1f}')
ax.set_title(f'{spec_id}')
ax.set_xlabel('Energy (eV)')
# ax.set_xlim(1000, 1200)
ax.set_ylabel('mu(E)')
ax.legend()
plt.tight_layout()
plt.show()






# %%
# XANES region: everything up to e0 + 75
e0 = row["e0"]
mask = x <= e0 + 75
x_xanes, y_xanes = x[mask], y[mask]

# Resample at 0.1 eV
x_resampled = np.arange(x_xanes[0], x_xanes[-1], 0.1)
y_resampled = np.interp(x_resampled, x_xanes, y_xanes)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top: raw points
axes[0].plot(x_xanes, y_xanes, '-', linewidth=0.8, color='steelblue', label='Curve')
axes[0].plot(x_xanes, y_xanes, 'o', markersize=3, color='red', label=f'Raw points ({len(x_xanes)})')
axes[0].axvline(e0, color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {e0:.1f}')
axes[0].axvline((e0 + 50), color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {e0 + 50:.1f}')
axes[0].set_title(f'{spec_id} — XANES raw')
axes[0].set_xlabel('Energy (eV)')
axes[0].set_ylabel('μ(E)')
axes[0].legend()

# Bottom: resampled at 0.1 eV
axes[1].plot(x_resampled, y_resampled, '-', linewidth=0.8, color='steelblue', label='Curve')
axes[1].plot(x_resampled, y_resampled, 'o', markersize=2, color='red', label=f'0.1 eV points ({len(x_resampled)})')
axes[1].axvline(e0, color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {e0:.1f}')
axes[1].axvline((e0 + 50), color='gray', linestyle='--', linewidth=0.8, label=f'e0 = {e0 + 50:.1f}')
axes[1].set_title(f'{spec_id} — XANES resampled 0.1 eV')
axes[1].set_xlabel('Energy (eV)')
axes[1].set_ylabel('μ(E)')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Raw: {len(x_xanes)} points, Resampled: {len(x_resampled)} points")

# %%
# %%
met["x_relmax"] = met["x_max"] - met["e0"]
counts = pd.cut(met["x_relmax"], bins=10).value_counts().sort_index()
print(counts)
bin_categories = pd.cut(met["x_relmax"], bins=10)
# %%
from tqdm import tqdm

x_relmaxs = []
y_lasts = []

with h5py.File("data/xas/matproj/xas.h5", "r") as f:
    for _, row in tqdm(met.iterrows(), total=len(met)):
        grp = f[row["spectrum_id"]]
        x = grp["x"][:]
        y = grp["y"][:]
        x_relmaxs.append(x[-1] - row["e0"])
        y_lasts.append(y[-1])

x_relmaxs = np.array(x_relmaxs)
y_lasts = np.array(y_lasts)

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(x_relmaxs, y_lasts, s=1, alpha=0.3)
ax.set_xlabel('x_max - e0 (eV)')
ax.set_ylabel('y at rightmost point')
ax.set_title('Absorption at right edge of each spectrum')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()

# %%
met["bin_range"] = pd.cut(met["x_relmax"], bins=10)

met["y_lasts"] = y_lasts  
summary = met.groupby("bin_range", observed=False).agg(
    count=('x_relmax', 'count'),
    avg_x_relmin=('x_relmax', 'mean'),
    avg_y_absorption=('y_lasts', 'mean')
)

print(summary.to_string())

# %%
# %%
from scipy.stats import wasserstein_distance
from tqdm import tqdm

emd_scores = []
pre_edge_y_vals = []

with h5py.File("data/xas/matproj/xas.h5", "r") as f:
    for _, row in tqdm(fmeta.iterrows(), total=len(fmeta)):
        grp = f[row["spectrum_id"]]
        x = grp["x"][:]
        y = grp["y"][:]
        e0 = row["e0"]

        # --- Task 1: EMD for XANES resampling loss ---
        xanes_mask = (x >= e0 - 15) & (x <= e0 + 50)
        x_raw = x[xanes_mask]
        y_raw = y[xanes_mask]

        if len(x_raw) >= 2:
            # Step 1: resample to 0.1 eV (what we'd save)
            x_coarse = np.arange(e0 - 15, e0 + 50, 0.1)
            y_coarse = np.interp(x_coarse, x_raw, y_raw)
            # Step 2: reconstruct at original raw positions from coarse grid
            y_reconstructed = np.interp(x_raw, x_coarse, y_coarse)
            # Step 3: EMD between original and reconstructed
            emd_scores.append(wasserstein_distance(y_raw, y_reconstructed))

        # --- Task 2: all y-values in pre-edge [e0-15, e0] ---
        pre_edge_mask = (x >= e0 - 15) & (x < e0)
        if pre_edge_mask.sum() > 0:
            pre_edge_y_vals.extend(y[pre_edge_mask].tolist())

emd_scores = np.array(emd_scores)
pre_edge_y_vals = np.array(pre_edge_y_vals)

# %%
# Task 1 results
print(f"EMD scores — mean: {emd_scores.mean():.6f}, median: {np.median(emd_scores):.6f}, "
      f"max: {emd_scores.max():.6f}, 95th pctl: {np.percentile(emd_scores, 95):.6f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(emd_scores, bins=100, edgecolor='black')
ax.set_xlabel('EMD (raw vs 0.1 eV resampled)')
ax.set_ylabel('Count')
ax.set_title('XANES resampling loss (Earth Mover\'s Distance)')
plt.tight_layout()
plt.show()

# %%
# Task 2 results
print(f"Pre-edge y-values — mean: {pre_edge_y_vals.mean():.4f}, median: {np.median(pre_edge_y_vals):.4f}, "
      f"max: {pre_edge_y_vals.max():.4f}, 95th pctl: {np.percentile(pre_edge_y_vals, 95):.4f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(pre_edge_y_vals, bins=100, edgecolor='black')
ax.set_xlabel('μ(E) in pre-edge region [e0-15, e0]')
ax.set_ylabel('Count')
ax.set_title('Distribution of absorption values in pre-edge region')
plt.tight_layout()
plt.show()

# %%
from tqdm import tqdm

left_edge_y = []

with h5py.File("data/xas/matproj/xas.h5", "r") as f:
    for _, row in tqdm(fmeta.iterrows(), total=len(fmeta)):
        grp = f[row["spectrum_id"]]
        y = grp["y"][:]
        left_edge_y.append(y[0])

left_edge_y = np.array(left_edge_y)

print(f"Leftmost y-value per spectrum:")
print(f"  mean:      {left_edge_y.mean():.4f}")
print(f"  median:    {np.median(left_edge_y):.4f}")
print(f"  std:       {left_edge_y.std():.4f}")
print(f"  max:       {left_edge_y.max():.4f}")
print(f"  95th pctl: {np.percentile(left_edge_y, 95):.4f}")
print(f"  99th pctl: {np.percentile(left_edge_y, 99):.4f}")
print(f"  < 0.1:     {(left_edge_y < 0.1).sum()} ({100*(left_edge_y < 0.1).mean():.1f}%)")
print(f"  < 0.5:     {(left_edge_y < 0.5).sum()} ({100*(left_edge_y < 0.5).mean():.1f}%)")
print(f"  > 1.0:     {(left_edge_y > 1.0).sum()} ({100*(left_edge_y > 1.0).mean():.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(left_edge_y, bins=100, edgecolor='black')
axes[0].set_xlabel('y[0] (leftmost absorption value)')
axes[0].set_ylabel('Count')
axes[0].set_title('All spectra — leftmost y-value')

# Zoomed in to see the bulk near 0
axes[1].hist(left_edge_y[left_edge_y < 0.5], bins=100, edgecolor='black')
axes[1].set_xlabel('y[0] (leftmost absorption value)')
axes[1].set_ylabel('Count')
axes[1].set_title('Zoomed: spectra with y[0] < 0.5')

plt.tight_layout()
plt.show()

# %%
