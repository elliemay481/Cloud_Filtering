import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.cm import get_cmap
from scipy.stats import binned_statistic
import glob
plt.style.use('../plotstyling.mplstyle')

files = sorted(glob.glob("../../../DataStorage/AWS/l2_cloud_signal/*.nc"))

x_all             = []
y_all             = []
lat_all           = []
fov_idx_all       = []
remap_dist_all    = []
model_variant_all = []
month_all         = []

for f in files:
    ds = xr.open_dataset(f)

    # extract month from filename e.g. "20250605023209_20250605041037"
    month = int(os.path.basename(f).split("_")[3][4:6])

    mean_flat = ds.Ta_CloudSignal_AWS33_mean.values.ravel()
    q_flat    = (ds.Ta_CloudSignal_AWS33_quantiles.values[:,:,3] -
                 ds.Ta_CloudSignal_AWS33_quantiles.values[:,:,1]).ravel()
    lat_flat  = ds.latitude.values.ravel()

    fov_idx = np.broadcast_to(
        ds.l1b_index_fovs.values[np.newaxis, :],
        ds.Ta_CloudSignal_AWS33_mean.shape
    ).ravel().copy()

    remap_dist    = ds.remap_distance.values.ravel()
    model_variant = ds.model_variant.values.ravel()

    nan_mask = np.isnan(mean_flat)
    bad_mask = ds.flag_bad_data.values.ravel() != 0
    valid    = ~nan_mask & ~bad_mask

    x_all.append(mean_flat[valid])
    y_all.append(q_flat[valid])
    lat_all.append(lat_flat[valid])
    fov_idx_all.append(fov_idx[valid])
    remap_dist_all.append(remap_dist[valid])
    model_variant_all.append(model_variant[valid])
    month_all.append(np.full(valid.sum(), month, dtype=np.int8))

    ds.close()

x             = np.concatenate(x_all)
y             = np.concatenate(y_all)
lat           = np.concatenate(lat_all)
fov_idx       = np.concatenate(fov_idx_all)
remap_dist    = np.concatenate(remap_dist_all)
model_variant = np.concatenate(model_variant_all)
month         = np.concatenate(month_all)

variant_classes = sorted(np.unique(model_variant))
variant_int     = np.array([variant_classes.index(v) for v in model_variant])

lat_mask = np.abs(lat) > 60
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

valid_plot = lat_mask & ~np.isnan(y) & ~np.isnan(x)
xm, ym = x[valid_plot], y[valid_plot]

fig, axes = plt.subplots(1, 3, figsize=(36, 10))

# --- subplot 1: cluster colours ---
ax = axes[0]
ax.scatter(xm, ym, c="gray", s=5, alpha=0.1, ec="none", zorder=1, label="All cases")
for mask, color, label, zo in zip(region_masks, region_colors, region_labels, region_zorders):
    ax.scatter(x[mask & valid_plot], y[mask & valid_plot],
               c=color, s=10, ec="none", zorder=zo, alpha=0.6, label=label)
xlim = np.linspace(0, 100, 200)
ax.axhline(27,           c="crimson",    ls="--", lw=1, alpha=0.7)
ax.plot(xlim, 0.65*xlim, c="dodgerblue", ls="--", lw=1, alpha=0.7)
ax.axhline(8,            c="gold",       ls="--", lw=1, alpha=0.7)
ax.set_xlabel("Cloud Signal AWS33 (mean) [K]")
ax.set_ylabel("IQR (0.84 - 0.16) [K]")
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.set_title("CS vs IQR — clusters")
ax.legend(markerscale=3)

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

# --- subplot 3: coloured by month ---
ax = axes[2]
months_present = sorted(np.unique(month[valid_plot]))
n_months       = len(months_present)
cmap_month     = get_cmap("tab10", n_months)
norm_month     = BoundaryNorm(np.arange(-0.5, n_months), n_months)
month_int      = np.array([months_present.index(m) for m in month[valid_plot]])
sc = ax.scatter(xm, ym, c=month_int, s=5, alpha=0.5, ec="none",
                cmap=cmap_month, norm=norm_month, zorder=1)
import calendar
cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(n_months))
cbar.ax.set_yticklabels([calendar.month_abbr[m] for m in months_present])
cbar.set_label("Month")
ax.set_xlabel("Cloud Signal AWS33 (mean) [K]")
ax.set_ylabel("IQR (0.84 - 0.16) [K]")
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.set_title("CS vs IQR — month")

plt.tight_layout()
plt.savefig("../figures/retrievals_on_obs/cs_iqr_coloured_by_variable_highlats.png",
            dpi=200, bbox_inches="tight", facecolor="white")