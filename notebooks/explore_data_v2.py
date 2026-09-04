
# %%
# %%
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/xas/matproj")
meta = pd.read_csv(DATA_DIR / "metadata.csv")
print(meta.shape)
print(meta.dtypes)
print(meta.head())
print((meta["absorbing_element"] == "Kr").sum())

# %%
print((meta["absorbing_element"] == "Kr").sum())

# %%
print(meta.iloc[0]["x_min"])
print(meta["spectrum_type"].value_counts())
# %%
print(meta.head())
print(meta["edge"].value_counts())


# %%
print(meta.groupby("spectrum_type")[["x_length", "x_min", "x_max"]].mean())

# %%
meta["x_relmin"] = meta["x_min"] - meta["e0"]
meta["x_relmax"] = meta["x_max"] - meta["e0"]
print(meta.groupby("spectrum_type")["x_relmin"].mean())
print(meta.groupby("spectrum_type")["x_relmax"].mean())
# %%
meta["spacing"] = (meta["x_max"] - meta["x_min"]) / (meta["x_length"] - 1)
print(meta.groupby("spectrum_type")["spacing"].mean())
# %%
sets = meta.groupby(["material_id", "absorbing_element", "edge"])["spectrum_type"].apply(set)

# %%
has_all = sets[sets == {"XANES", "EXAFS", "XAFS"}]
chosen_material = has_all.index[0][0]
sub = meta[(meta["material_id"] == chosen_material) & (meta["edge"] == "K")]
print(sub)

# %%
import h5py
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

spec_types = ["XANES", "EXAFS", "XAFS"]

with h5py.File(DATA_DIR / "xas.h5", "r") as f:
    for spec_type in spec_types:

        row = sub[sub["spectrum_type"] == spec_type].iloc[0]
        spec_id = row["spectrum_id"]
        grp = f[spec_id]
        x = grp["x"][:]
        y = grp["y"][:]
        e0 = grp.attrs["e0"]

        ax.plot(x, y, linewidth=(3 - spec_types.index(spec_type)), label=spec_type)
        plt.axvline(x=e0, color='red', linestyle='--', label=f"e0 from {spec_type}: {e0}")

e0_ref = sub[sub["spectrum_type"] == "XAFS"].iloc[0]["e0"]
ax.set_xlim(e0_ref - 10, e0_ref + 80)
plt.legend()
plt.show()
        

# %%
print(sets.apply(frozenset).value_counts())
# %%
# Cell 9 - XAFS vs XANES agreement across full K-edge dataset
import numpy as np
from tqdm import tqdm

diffs = []
skipped = 0

grouped = meta[meta["edge"] == "K"].groupby(["material_id", "absorbing_element", "edge"])

with h5py.File(DATA_DIR / "xas.h5", "r") as f:
    for key, group in tqdm(grouped, total=len(grouped)):
        types = set(group["spectrum_type"])
        if "XANES" not in types or "XAFS" not in types:
            skipped += 1
            continue

        xanes_id = group[group["spectrum_type"] == "XANES"].iloc[0]["spectrum_id"]
        xafs_id  = group[group["spectrum_type"] == "XAFS"].iloc[0]["spectrum_id"]

        x_xanes = f[xanes_id]["x"][:]
        y_xanes = f[xanes_id]["y"][:]
        x_xafs  = f[xafs_id]["x"][:]
        y_xafs  = f[xafs_id]["y"][:]

        # interpolate XAFS onto the XANES x-grid (XANES energy window only)
        y_xafs_on_xanes = np.interp(x_xanes, x_xafs, y_xafs)

        amp = np.ptp(y_xanes)  # peak-to-peak range of XANES
        if amp == 0:
            skipped += 1
            continue

        err = np.mean(np.abs(y_xafs_on_xanes - y_xanes)) / amp
        diffs.append(err)

diffs = np.array(diffs)
print(f"compared: {len(diffs)}   skipped: {skipped}")
print(f"median: {np.median(diffs):.4f}   p95: {np.quantile(diffs, 0.95):.4f}   max: {diffs.max():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(diffs, bins=np.logspace(-4, 1, 60))
axes[0].set_xscale("log")
axes[0].set_xlabel("disagreement (log scale)")
axes[0].set_ylabel("count")
axes[0].set_title("Full distribution")

axes[1].hist(diffs[(diffs >= 0.01) & (diffs <= 0.15)], bins=50)
axes[1].set_xlabel("disagreement")
axes[1].set_ylabel("count")
axes[1].set_title("Zoom: 1%-15% tail")

plt.tight_layout()
plt.show()
# %%
for p in [50, 90, 95, 99, 99.9, 99.99]:
    print(f"p{p}: {np.quantile(diffs, p/100):.4f}")
print(f"max: {diffs.max():.2f}")
print(f"n above 1.0 (100%): {(diffs > 1.0).sum()}")
print(f"n above 0.5  (50%): {(diffs > 0.5).sum()}")
print(f"n above 0.15 (15%): {(diffs > 0.15).sum()}")
# %%
print(meta["absorbing_element"].unique())
# %%
# Cell 11 - per-element breakdown of bad subpopulation (disagreement > 0.15)
# Re-run the same loop, this time collecting (element, key) alongside diffs
THRESHOLD = 0.15

keys_compared = []
diffs_xanes = []
skipped = 0

grouped = meta[meta["edge"] == "K"].groupby(["material_id", "absorbing_element", "edge"])

with h5py.File(DATA_DIR / "xas.h5", "r") as f:
    for key, group in tqdm(grouped, total=len(grouped)):
        types = set(group["spectrum_type"])
        if "XANES" not in types or "XAFS" not in types:
            skipped += 1
            continue

        xanes_id = group[group["spectrum_type"] == "XANES"].iloc[0]["spectrum_id"]
        xafs_id  = group[group["spectrum_type"] == "XAFS"].iloc[0]["spectrum_id"]

        x_xanes = f[xanes_id]["x"][:]
        y_xanes = f[xanes_id]["y"][:]
        x_xafs  = f[xafs_id]["x"][:]
        y_xafs  = f[xafs_id]["y"][:]

        y_xafs_on_xanes = np.interp(x_xanes, x_xafs, y_xafs)
        amp = np.ptp(y_xanes)
        if amp == 0:
            skipped += 1
            continue

        err = np.mean(np.abs(y_xafs_on_xanes - y_xanes)) / amp
        keys_compared.append(key)  # (material_id, absorbing_element, edge)
        diffs_xanes.append(err)

diffs_xanes = np.array(diffs_xanes)
keys_compared = np.array(keys_compared)  # shape (n, 3)

bad_mask = diffs_xanes > THRESHOLD
bad_keys = keys_compared[bad_mask]

bad_elements = pd.Series(bad_keys[:, 1], name="absorbing_element")
total_by_elem = pd.Series(keys_compared[:, 1], name="absorbing_element").value_counts()
bad_by_elem   = bad_elements.value_counts()
bad_rate = (bad_by_elem / total_by_elem).dropna().sort_values(ascending=False)

print(f"Total bad (>{THRESHOLD}): {bad_mask.sum()} / {len(diffs_xanes)}")
print("\nBad count by element (top 20):")
print(bad_by_elem.head(20))
print("\nBad rate by element (top 20):")
print(bad_rate.head(20))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
bad_by_elem.head(20).plot(kind="bar", ax=axes[0], color="salmon", edgecolor="black")
axes[0].set_title(f"Bad sample count by element (disagreement > {THRESHOLD})")
axes[0].set_ylabel("count")

bad_rate.head(20).plot(kind="bar", ax=axes[1], color="steelblue", edgecolor="black")
axes[1].set_title(f"Bad rate by element (disagreement > {THRESHOLD})")
axes[1].set_ylabel("fraction bad")

plt.tight_layout()
plt.show()

# %%
print(meta["y_min"].head())
print("space")
print(meta["y_max"].head())
print("space")
print(meta["y_min"].describe())
print("space")
print(meta["y_max"].describe())
target_percentiles = [0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]

print(meta["y_max"].quantile(target_percentiles))
threshold = 1.0E03
above_thresh_elements = meta.loc[meta["y_max"] > threshold, ["y_max", "absorbing_element", "spectrum_id"]]
print(above_thresh_elements)

# %%
fig, ax = plt.subplots()

with h5py.File(DATA_DIR / "xas.h5", "r") as f:

    grp = f["mp-632296_H_K_XAFS"]
    x = grp["x"][:]
    y = grp["y"][:]
    e0 = grp.attrs["e0"]


    grp2 = f["mp-632296_H_K_XANES"]
    x2 = grp2["x"][:]
    y2 = grp2["y"][:]
    e02 = grp2.attrs["e0"]


    ax.plot(x, y, label='high XAFS')
    ax.plot(x2, y2, label='high XANES')
    plt.axvline(x=e0, color='red', linestyle='--', label=f"e0 from K XAFS: {e0}")
    plt.axvline(x=e02, color='blue', linestyle='--', label=f"e0 from K XANES: {e02}")

ax.set_xlim(e0 - 10, e0 + 10)
plt.legend()
plt.show()

# %%
print(meta.loc[meta["spectrum_id"] == "mp-632296_H_K_XAFS"]["y_max"])
with h5py.File(DATA_DIR / "xas.h5", "r") as f:
    grp = f["mp-632296_H_K_XAFS"]
    x = grp["x"][:]
    y = grp["y"][:]
    e0 = grp.attrs["e0"]
    print(x[20:21])
    print(e0)
    print(y[20])
    print(y[:60])


# %%
print((meta["absorbing_element"] == "Kr").sum())




# %%
# Cell 12 - Summary: filter rules for prepare_data.py
print("""
FILTER RULES FOR prepare_data.py
=================================

Data source:
  - Use ONLY spectrum_type == "XAFS" as the source spectrum.
  - Ignore standalone XANES and EXAFS entirely.
  - This covers 135,922 K-edge triples (every XAFS comes with the full triple).

Modality construction:
  - XANES modality: XAFS[:e0 + 50 eV]  (the near-edge region)
  - EXAFS modality: XAFS[e0 + 50 eV:]  (the post-edge region)
  - Split point e0 + 50 eV is consistent with the observed XANES x_relmax ~ 50 eV.

Edge filter:
  - Keep only edge == "K".
  - L-edges (L2, L3, L2,3) are ~90k rows; excluded from this project.

Quality filter:
  - Compute mean |XAFS_interp - XANES| / range(XANES) over the XANES energy window.
  - Discard triples where this disagreement > 0.15 (15%).
  - This removes ~2% of triples (2,901 / 135,922).
  - Applied globally: bad rate per element peaks at ~9% (Tl, H). No element
    is systematically broken, so we don't need to exclude an entire element.

e0:
  - No corrupt e0 values found in K-edge XAFS rows (min 13.6 eV, max 122,360 eV).
  - No e0 filter needed, but can add e0 < 10 or e0 > 125000 as a safety guard.
      
Y-value filters:
  - Remove y_min < 0 (none found currently, but jic).
  - Remove material_id == "mp-636056" (Krypton, two spectra, unphysical shape).
  - Remove any XAFS spectrum where the x-position of y_max is more than ((50)) eV
    above e0 (peak outside the XANES window indicates corrupted/unphysical data;
    catches cases like H mp-632296 where y_max is 20x the edge-jump value and
    located well into the EXAFS region).

Rare element filter (already in prepare_data.py):
  - Drop absorbing_element with < 50 samples after the above filters.
""")

# %%

# %%
