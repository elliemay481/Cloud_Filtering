import xarray as xr
import os
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import re
from matplotlib.colors import ListedColormap
from pyproj import Transformer

plt.style.use('../plotstyling.mplstyle')

AWS_CHANNELS = [31, 32, 33, 34, 35, 36]

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_sensing_start(folder_name):
    match = re.search(r'OPE_(\d{14})_\d{14}', folder_name)
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S") if match else None

def get_closest_MTG(dt, max_delta_minutes=30):
    deltas = np.abs([(t - dt).total_seconds() for t in mtg_times_arr])
    idx = np.argmin(deltas)
    if deltas[idx] > max_delta_minutes * 60:
        return None, None
    return mtg_files[mtg_times_arr[idx]], mtg_times_arr[idx]

def setup_mtg(ds_MTG):
    proj_attrs = ds_MTG["mtg_geos_projection"].attrs
    geo_proj = ccrs.Geostationary(
        satellite_height=proj_attrs["perspective_point_height"],
        central_longitude=proj_attrs["longitude_of_projection_origin"],
        sweep_axis=proj_attrs["sweep_angle_axis"],
        globe=ccrs.Globe(
            semimajor_axis=proj_attrs["semi_major_axis"],
            semiminor_axis=proj_attrs["semi_minor_axis"]
        )
    )
    x = ds_MTG["x"].values[::-1]
    y = ds_MTG["y"].values
    cloud_phase = ds_MTG["retrieved_cloud_phase"].values[::-1, :]
    masked_phase = np.ma.masked_where(cloud_phase != 2, cloud_phase)
    h = proj_attrs["perspective_point_height"]
    extent = [np.tan(x[0])*h, np.tan(x[-1])*h, np.tan(y[0])*h, np.tan(y[-1])*h]
    return masked_phase, extent, geo_proj

def create_plot(cloud_masks, surface_masks, cs_ds, ds_MTG, bad_mask, i, step):
    """
    cloud_masks : dict  {channel_number: 2-D bool array}
                  keys are integers from AWS_CHANNELS (31–36)
    """
    n_panels = len(AWS_CHANNELS) + 1          # 6 AWS + 1 MTG
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(8 * n_panels, 12),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    lons = cs_ds.longitude[i:i+step].values.copy()
    lats = cs_ds.latitude[i:i+step].values.copy()
    lons = np.where(np.isfinite(lons), lons, 0.0)
    lats = np.where(np.isfinite(lats), lats, 0.0)

    extent_kw = dict(
        crs=ccrs.PlateCarree(),
        extents=[
            np.nanmin(cs_ds.longitude[i:i+step].values),
            np.nanmax(cs_ds.longitude[i:i+step].values),
            np.nanmin(cs_ds.latitude[i:i+step].values),
            np.nanmax(cs_ds.latitude[i:i+step].values),
        ]
    )

    for ax in axes:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, zorder=3)
        ax.add_feature(cfeature.LAND,  facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="white",     zorder=0)
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray',
                          alpha=0.5, linestyle='--')
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {'size': 14}
        ax.set_extent(extent_kw["extents"])

    # ── shared overlay masks ──────────────────────────────────────────────────
    bad_data   = np.where(bad_mask, 0.0, np.nan)
    bad_masked = np.ma.masked_invalid(bad_data)

    def plot_swath_panel(ax, data_arr, cloud_mask, surface_mask_ch):
        ax.pcolormesh(lons, lats, np.ma.masked_where(~cloud_mask, data_arr),
                      transform=ccrs.PlateCarree(), cmap=ListedColormap(["steelblue"]))
        if surface_mask_ch.any():
            surface_data_ch   = np.where(surface_mask_ch, 0.0, np.nan)
            surface_masked_ch = np.ma.masked_invalid(surface_data_ch)
            ax.pcolormesh(lons, lats, surface_masked_ch,
                          transform=ccrs.PlateCarree(), cmap=ListedColormap(["indianred"]))
        ax.pcolormesh(lons, lats, bad_masked,
                      transform=ccrs.PlateCarree(), cmap=ListedColormap(["grey"]))

    # ── AWS panels (indices 0–5) ──────────────────────────────────────────────
    for panel_idx, ch in enumerate(AWS_CHANNELS):
        var_name = f"Ta_CloudSignal_AWS{ch}_mean"
        data_arr = cs_ds[var_name][i:i+step].values
        plot_swath_panel(axes[panel_idx], data_arr, cloud_masks[ch], surface_masks[ch])
        axes[panel_idx].set_title(f"AWS{ch} cloud signal $>$ 1 K", fontsize=16)

    # ── MTG panel (last, index 6) ─────────────────────────────────────────────
    axes[-1].set_title("MTG ice cloud mask", fontsize=16)
    masked_phase, extent, geo_proj = setup_mtg(ds_MTG)
    axes[-1].imshow(masked_phase, origin="upper", extent=extent,
                    transform=geo_proj, cmap="Set3", interpolation="none")

    # ── swath outline on all panels ───────────────────────────────────────────
    raw_lons = cs_ds.longitude[i:i+step].values
    raw_lats = cs_ds.latitude[i:i+step].values
    valid = np.isfinite(raw_lons) & np.isfinite(raw_lats)

    left_valid  = valid[:, 0]
    right_valid = valid[:, -1]
    bot_valid   = valid[-1, :]
    top_valid   = valid[0, ::-1]

    swath_lon = np.concatenate([
        raw_lons[:, 0][left_valid],
        raw_lons[-1, :][bot_valid],
        raw_lons[:, -1][right_valid][::-1],
        raw_lons[0, ::-1][top_valid],
    ])
    swath_lat = np.concatenate([
        raw_lats[:, 0][left_valid],
        raw_lats[-1, :][bot_valid],
        raw_lats[:, -1][right_valid][::-1],
        raw_lats[0, ::-1][top_valid],
    ])

    for ax in axes:
        ax.plot(swath_lon, swath_lat, transform=ccrs.PlateCarree(),
                color="k", linewidth=1.5, zorder=5)

    return fig


# ── pre-compute MTG lat/lon bounding boxes cheaply ───────────────────────────

def get_mtg_lonlat_bounds(nc_file):
    ds = xr.open_dataset(nc_file)
    proj_attrs = ds["mtg_geos_projection"].attrs
    h     = proj_attrs["perspective_point_height"]
    lon_0 = proj_attrs["longitude_of_projection_origin"]
    a     = proj_attrs["semi_major_axis"]
    b     = proj_attrs["semi_minor_axis"]
    x = ds["x"].values
    y = ds["y"].values
    ds.close()

    x_sample = x[np.linspace(0, len(x)-1, 50, dtype=int)]
    y_sample = y[np.linspace(0, len(y)-1, 50, dtype=int)]
    xs, ys = np.meshgrid(np.tan(x_sample) * h, np.tan(y_sample) * h)

    proj_str = (f"+proj=geos +lon_0={lon_0} +h={h} "
                f"+a={a} +b={b} +sweep=y +units=m")
    transformer = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)

    lons, lats = transformer.transform(xs.ravel(), ys.ravel())
    valid = np.isfinite(lons) & np.isfinite(lats)

    if not valid.any():
        return None

    return (lons[valid].min(), lons[valid].max(),
            lats[valid].min(), lats[valid].max())


# ── build MTG file index ──────────────────────────────────────────────────────

MTG_path = "../../../DataStorage/MTG/"

mtg_files  = {}
mtg_bounds = {}

for nc_file in glob.glob(MTG_path + "**/*.nc", recursive=True):
    t = parse_sensing_start(os.path.dirname(nc_file))
    if t is None:
        continue
    mtg_files[t] = nc_file
    bounds = get_mtg_lonlat_bounds(nc_file)
    if bounds:
        mtg_bounds[t] = bounds
    print(f"Indexed MTG: {t} | bounds: {bounds}")

mtg_times     = sorted(mtg_files.keys())
mtg_times_arr = np.array(mtg_times)


# ── loop over all retrieval files ─────────────────────────────────────────────

retrieval_files = sorted(glob.glob(
    "../../../DataStorage/AWS/l2_cloud_signal/l2_cloud_signal_*.nc"
))

step = 300
current_mtg_file = None
ds_MTG_cached    = None

for retrieval_filename in retrieval_files:
    print(f"\n{'='*60}\nProcessing: {retrieval_filename}")
    retrieval_ds = xr.open_dataset(retrieval_filename)
    datetime_str = retrieval_filename[-32:-3]
    scans        = retrieval_ds.scan.values

    print(datetime_str)
    print(scans[0])

    for i in range(0, len(scans), step):
        chunk  = scans[i:i+step]
        mid_dt = pd.Timestamp(chunk[len(chunk) // 2]).to_pydatetime()
        print(mid_dt)

        mtg_file, mtg_t = get_closest_MTG(mid_dt)
        if mtg_file is None:
            print(f"  Chunk {i}: no MTG file within 30 min of {mid_dt}, skipping")
            continue
        print(mtg_t)

        # ── 1. skip if chunk has no overlap with MTG coverage ────────────────
        chunk_lons = retrieval_ds.longitude[i:i+step].values
        chunk_lats = retrieval_ds.latitude[i:i+step].values
        bounds = mtg_bounds.get(mtg_t)
        if bounds:
            lon_min, lon_max, lat_min, lat_max = bounds
            in_view = (
                (chunk_lons >= lon_min) & (chunk_lons <= lon_max) &
                (chunk_lats >= lat_min) & (chunk_lats <= lat_max)
            )
            if not in_view.any():
                print(f"  Chunk {i}: outside MTG coverage, skipping")
                continue

        # ── 2. only reload MTG if it changed ─────────────────────────────────
        if mtg_file != current_mtg_file:
            if ds_MTG_cached is not None:
                ds_MTG_cached.close()
            print(f"  Loading MTG: {mtg_t}")
            ds_MTG_cached    = xr.open_dataset(mtg_file)
            current_mtg_file = mtg_file

        # ── 3. build masks ────────────────────────────────────────────────────
        bad_mask = retrieval_ds.flag_bad_data[i:i+step].values != 0

        cloud_masks   = {}
        surface_masks = {}
        for ch in AWS_CHANNELS:
            var_name = f"Ta_CloudSignal_AWS{ch}_mean"
            nan_mask_ch      = np.isnan(retrieval_ds[var_name][i:i+step].values)
            surface_masks[ch] = nan_mask_ch & ~bad_mask
            cloud_masks[ch]   = ~surface_masks[ch] & (
                retrieval_ds[var_name][i:i+step].values > 1
            )

        # ── 4. plot & save ────────────────────────────────────────────────────
        fig = create_plot(cloud_masks, surface_masks, retrieval_ds, ds_MTG_cached,
                          bad_mask, i, step)

        plt.savefig(
            f"../figures/retrievals_on_obs/MTG_comparison/"
            f"cloud_filtered_and_MTG_{datetime_str}_{i}_{i+step}.png",
            dpi=200, bbox_inches="tight", facecolor="white"
        )
        plt.close(fig)

    retrieval_ds.close()

if ds_MTG_cached is not None:
    ds_MTG_cached.close()