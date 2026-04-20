import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
import glob
import os
import calendar
plt.style.use('../plotstyling.mplstyle')

files = sorted(glob.glob("../../../DataStorage/AWS/l2_cloud_signal/*.nc"))

x_all             = []
y_all             = []
lat_all           = []
fov_idx_all       = []
month_all         = []

for f in files:
    ds = xr.open_dataset(f)

    month = int(os.path.basename(f).split("_")[3][4:6])

    mean_flat = ds.Ta_CloudSignal_AWS33_mean.values.ravel()
    q_flat    = (ds.Ta_CloudSignal_AWS33_quantiles.values[:,:,3] -
                 ds.Ta_CloudSignal_AWS33_quantiles.values[:,:,1]).ravel()
    lat_flat  = ds.latitude.values.ravel()

    fov_idx = np.broadcast_to(
        ds.l1b_index_fovs.values[np.newaxis, :],
        ds.Ta_CloudSignal_AWS33_mean.shape
    ).ravel().copy()

    nan_mask = np.isnan(mean_flat)
    bad_mask = ds.flag_bad_data.values.ravel() != 0
    valid    = ~nan_mask & ~bad_mask

    x_all.append(mean_flat[valid])
    y_all.append(q_flat[valid])
    lat_all.append(lat_flat[valid])
    fov_idx_all.append(fov_idx[valid])
    month_all.append(np.full(valid.sum(), month, dtype=np.int8))

    ds.close()

x       = np.concatenate(x_all)
y       = np.concatenate(y_all)
lat     = np.concatenate(lat_all)
fov_idx = np.concatenate(fov_idx_all)
month   = np.concatenate(month_all)

lat_mask = lat < -60
mask_a = lat_mask & (x > 10) & (y > 27)
mask_b = lat_mask & (x > 20) & (y > 0.65 * x) & (y < 27)
mask_c = lat_mask & (x > 30) & (y < 8)
region_colors  = ["gold", "dodgerblue", "crimson"]
region_labels  = [
    r"CS $>$ 30 K, IQR $<$ 8 K",
    r"CS $>$ 20 K, IQR $>$ 0.65·CS",
    r"CS $>$ 10 K, IQR $>$ 25 K",
]
region_masks   = [mask_c, mask_b, mask_a]
region_zorders = [3, 4, 5]
xlim = np.linspace(0, 100, 200)

months_present = sorted(np.unique(month))

for m in months_present:
    month_mask = month == m
    valid_plot = lat_mask & month_mask & ~np.isnan(y) & ~np.isnan(x)
    xm, ym = x[valid_plot], y[valid_plot]

    if len(xm) == 0:
        continue

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(f"{calendar.month_name[m]}, Antarctica")

    # --- subplot 1: cluster colours ---
    ax = axes[0]
    ax.scatter(xm, ym, c="gray", s=5, alpha=0.1, ec="none", zorder=1, label="All cases")
    for mask, color, label, zo in zip(region_masks, region_colors, region_labels, region_zorders):
        ax.scatter(x[mask & valid_plot], y[mask & valid_plot],
                   c=color, s=10, ec="none", zorder=zo, alpha=0.6, label=label)
    ax.axhline(27,           c="crimson",    ls="--", lw=1, alpha=0.7)
    ax.plot(xlim, 0.65*xlim, c="dodgerblue", ls="--", lw=1, alpha=0.7)
    ax.axhline(8,            c="gold",       ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Cloud Signal AWS33 (mean) [K]")
    ax.set_ylabel("IQR (0.84 - 0.16) [K]")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.set_title("CS vs IQR — clusters")
    #ax.legend(markerscale=3)

    # --- subplot 2: coloured by fov index ---
    ax = axes[1]
    sc = ax.scatter(xm, ym, c=fov_idx[valid_plot], s=5, alpha=0.7, ec="none",
                    cmap="twilight_r", zorder=1)
    fig.colorbar(sc, ax=ax, label="FOV index")
    ax.set_xlabel("Cloud Signal AWS33 (mean) [K]")
    ax.set_ylabel("IQR (0.84 - 0.16) [K]")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.set_title("CS vs IQR — FOV index")

    # --- subplot 3: coloured by day of month ---
    day = np.array([int(os.path.basename(f).split("_")[3][6:8])
                    for f in files
                    for _ in range(1)], dtype=np.int8)  # placeholder, see note
    ax = axes[2]
    ax.scatter(xm, ym, c="gray", s=5, alpha=0.1, ec="none", zorder=1)
    ax.set_xlabel("Cloud Signal AWS33 (mean) [K]")
    ax.set_ylabel("IQR (0.84 - 0.16) [K]")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.set_title(f"N = {len(xm):,}")

    plt.tight_layout()
    plt.savefig(
        f"../figures/retrievals_on_obs/cs_iqr_antarctica_{m:02d}_{calendar.month_abbr[m]}.png",
        dpi=200, bbox_inches="tight", facecolor="white"
    )