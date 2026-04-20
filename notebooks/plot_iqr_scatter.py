import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cmc
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
import glob
import os
plt.style.use('../plotstyling.mplstyle')

# --- Load single-file special points ---
datetime_str = "20250605023209_20250605041037"
cs_ds = xr.open_dataset(f"../../../DataStorage/AWS/l2_cloud_signal/l2_cloud_signal_{datetime_str}.nc")

start, end = 2500, 3000
mean_flat_t = cs_ds.Ta_CloudSignal_AWS33_mean.values[start:end].ravel()
q_flat_t    = (cs_ds.Ta_CloudSignal_AWS33_quantiles.values[start:end,:,3] -
               cs_ds.Ta_CloudSignal_AWS33_quantiles.values[start:end,:,1]).ravel()
lat_flat_t  = cs_ds.latitude.values[start:end].ravel()
nan_mask_t  = np.isnan(mean_flat_t)
bad_mask_t  = cs_ds.flag_bad_data.values[start:end].ravel() != 0
valid_t     = ~nan_mask_t & ~bad_mask_t
x_t         = mean_flat_t[valid_t]
y_t         = q_flat_t[valid_t]
special_mask     = x_t > 50
x_special        = x_t[special_mask]
y_special        = y_t[special_mask]
print(f"Found {special_mask.sum()} 'bad' points")
cs_ds.close()

# --- Load all files ---
files = sorted(glob.glob("../../../DataStorage/AWS/l2_cloud_signal/*.nc"))

x_all        = []
y_all        = []
lat_all      = []
lon_all      = []
most_prob_all = []
tb33_all     = []
tb44_all     = []

for f in files:
    ds = xr.open_dataset(f)

    mean_flat      = ds.Ta_CloudSignal_AWS33_mean.values.ravel()
    q_flat         = (ds.Ta_CloudSignal_AWS33_quantiles.values[:,:,3] -
                      ds.Ta_CloudSignal_AWS33_quantiles.values[:,:,1]).ravel()
    lat_flat       = ds.latitude.values.ravel()
    lon_flat       = ds.longitude.values.ravel()
    most_prob_flat = ds.Ta_CloudSignal_AWS33_most_prob.values.ravel()

    ch_idx   = list(ds.channel.values).index("AWS33")
    tb33_flat = ds.tb.values[:, :, ch_idx].ravel()
    ch_idx   = list(ds.channel.values).index("AWS44")
    tb44_flat = ds.tb.values[:, :, ch_idx].ravel()

    nan_mask = np.isnan(mean_flat)
    bad_mask = ds.flag_bad_data.values.ravel() != 0
    valid    = ~nan_mask & ~bad_mask

    x_all.append(mean_flat[valid])
    y_all.append(q_flat[valid])
    lat_all.append(lat_flat[valid])
    lon_all.append(lon_flat[valid])
    most_prob_all.append(most_prob_flat[valid])
    tb33_all.append(tb33_flat[valid])
    tb44_all.append(tb44_flat[valid])

    ds.close()

x         = np.concatenate(x_all)
y         = np.concatenate(y_all)
lat       = np.concatenate(lat_all)
lon       = np.concatenate(lon_all)
most_prob = np.concatenate(most_prob_all)
tb33      = np.concatenate(tb33_all)
tb44      = np.concatenate(tb44_all)

# --- define regions ---
#lat_mask = (np.abs(lat) > 30) & (np.abs(lat) < 60)
lat_mask = (np.abs(lat) < 30)
"""
mask_a = lat_mask & (x > 10) & (y > 35)
mask_b = lat_mask & (x > 20) & (y > 0.65 * x) & (y < 27)
mask_c = lat_mask & (x > 30) & (y < 8)
region_colors = ["gold", "dodgerblue", "crimson"]
region_labels = [
    r"CS $>$ 30 K, IQR $<$ 8 K",
    r"CS $>$ 20 K, IQR $>$ 0.65·CS",
    r"CS $>$ 10 K, IQR $>$ 35 K",
]
region_masks = [mask_c, mask_b, mask_a]
region_zorders = [3, 4, 5]   # ← gold=3, blue=4, crimson=5; blue beats gold always
"""

# for tropics:
mask_c = lat_mask & (x > 25) & (y > 1.1*x)

region_colors = ["crimson"]
region_labels = [
    r"CS $>$ 25 K, IQR $>$ CS",
]
region_masks = [mask_c]
region_zorders = [5]   # ← gold=3, blue=4, crimson=5; blue beats gold always


with np.errstate(invalid='ignore', divide='ignore'):
    iqr_norm = np.where(x > 0, y / x, np.nan)

fig = plt.figure(figsize=(60, 10))
gs  = fig.add_gridspec(2, 5, width_ratios=[2, 1, 1, 1, 1], hspace=0.05, wspace=0.1)

ax_north = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_south = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax_sc    = fig.add_subplot(gs[:, 1])
ax_sc2   = fig.add_subplot(gs[:, 2])
ax_sc3   = fig.add_subplot(gs[:, 3])
ax_sc4   = fig.add_subplot(gs[:, 4])

# --- maps ---
for ax, extent, title in zip(
    [ax_north, ax_south],
    [[-180, 180, 0, 30], [-180, 180, -30, 0]],
    ["(lat > 60°)", "(lat < -60°)"],
):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=3)
    ax.add_feature(cfeature.LAND,      facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="white",     zorder=0)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    ax.set_extent(extent)
    for mask, color, label, zo in zip(region_masks, region_colors, region_labels, region_zorders):
        ax.scatter(lon[mask], lat[mask], c=color, s=2, ec="none",
                   zorder=zo, alpha=0.5, label=label)
ax_north.legend(markerscale=6)

# --- CS vs IQR ---
#valid_plot = (np.abs(lat) > 30) & (np.abs(lat) < 60) & ~np.isnan(y) & ~np.isnan(x)
valid_plot = (np.abs(lat) < 30) & ~np.isnan(y) & ~np.isnan(x)

ax_sc.scatter(x[valid_plot], y[valid_plot],
              c="gray", s=20, alpha=0.2, ec="none", zorder=1, label="All cases")
for mask, color, label in zip(region_masks, region_colors, region_labels):
    ax_sc.scatter(x[mask & valid_plot], y[mask & valid_plot],
                  c=color, s=20, ec="none", zorder=3, alpha=0.6, label=label)
xlim = np.linspace(0, 100, 200)
ax_sc.plot(xlim, 1*xlim, c="crimson", ls="--", lw=1, alpha=0.7)
ax_sc.set_xlabel("Cloud Signal AWS33 (mean) [K]")
ax_sc.set_ylabel("IQR (0.84 - 0.16) [K]")
ax_sc.set_xlim(0, 100)
ax_sc.set_ylim(0, 50)
ax_sc.set_title("CS vs IQR")
ax_sc.legend(markerscale=2)
"""
# --- CS vs IQR/mean ---
valid_plot2 = valid_plot & ~np.isnan(iqr_norm)
ax_sc2.scatter(x[valid_plot2], iqr_norm[valid_plot2],
               c="gray", s=20, alpha=0.2, ec="none", zorder=1, label="All cases")
for mask, color, label in zip(region_masks, region_colors, region_labels):
    ax_sc2.scatter(x[mask & valid_plot2], iqr_norm[mask & valid_plot2],
                   c=color, s=20, ec="none", zorder=3, alpha=0.6, label=label)
ax_sc2.set_xlabel("Cloud Signal AWS33 (mean) [K]")
ax_sc2.set_ylabel("IQR / mean")
ax_sc2.set_xlim(0, 100)
ax_sc2.set_ylim(0, 10)
ax_sc2.set_title("CS vs IQR/mean")
ax_sc2.legend(markerscale=2)

# --- CS mean vs most probable ---
valid_plot3 = valid_plot & ~np.isnan(most_prob)
ax_sc3.scatter(x[valid_plot3], most_prob[valid_plot3],
               c="gray", s=20, alpha=0.2, ec="none", zorder=1, label="All cases")
for mask, color, label in zip(region_masks, region_colors, region_labels):
    ax_sc3.scatter(x[mask & valid_plot3], most_prob[mask & valid_plot3],
                   c=color, s=20, ec="none", zorder=3, alpha=0.6, label=label)
ax_sc3.plot([0, 100], [0, 100], c="k", ls="--", lw=1, alpha=0.5, label="1:1")
ax_sc3.set_xlabel("CS mean [K]")
ax_sc3.set_ylabel("CS most probable [K]")
ax_sc3.set_xlim(0, 100)
ax_sc3.set_ylim(0, 100)
ax_sc3.set_title("Mean vs most probable")
ax_sc3.legend(markerscale=2)
"""
# --- CS mean vs TB AWS33 ---
valid_plot4 = valid_plot & ~np.isnan(tb33)
ax_sc4.scatter(tb33[valid_plot4], tb44[valid_plot4],
               c="gray", s=20, alpha=0.2, ec="none", zorder=0, label="All cases")
for mask, color, label, zo in zip(region_masks, region_colors, region_labels, region_zorders):
    ax_sc4.scatter(tb33[mask & valid_plot4], tb44[mask & valid_plot4],  # ← ax_sc4, tb33/tb44
                   c=color, s=20, ec="none", zorder=zo, alpha=0.6, label=label)
ax_sc4.plot([80,320], [80,320], ls="--", c="k")
ax_sc4.set_xlabel("Ta AWS33 [K]")
ax_sc4.set_ylabel("Ta AWS44 [K]")
ax_sc4.set_xlim(80, 300)
ax_sc4.set_ylim(80, 300)
ax_sc4.legend(markerscale=2)

plt.savefig("../figures/retrievals_on_obs/world_map_and_scatter_regions_tropics.png",
            dpi=200, bbox_inches="tight", facecolor="white")