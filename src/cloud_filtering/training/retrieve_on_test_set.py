from pathlib import Path
import torch
from quantnn.mrnn import MRNN
from toolbox.models.ICINN import AWS_CF_Model
import pickle
import xarray as xr
import numpy as np
from datetime import datetime
from quantnn.mrnn import Quantiles
import hashlib

import config as config
import retrieval_preprocessing

def compute_md5(path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def verify_md5(path: Path, expected_md5: str, name: str):
    actual_md5 = compute_md5(path)

    if actual_md5 != expected_md5:
        raise ValueError(
            f"MD5 mismatch for {name}!\n"
            f"Expected: {expected_md5}\n"
            f"Got:      {actual_md5}\n"
            f"File:     {path}"
        )

def load_model(tag):

    input_variables = config.get_input_variables(tag)
    output_variables = config.get_output_variables(tag)

    n_quantile_outputs = {name: config.N_QUANTS for name in output_variables}
    losses = {name: Quantiles(config.QUANTILES) for name in output_variables}

    model = AWS_CF_Model(
        n_inputs=len(input_variables),
        n_outputs=n_quantile_outputs,
        **config.MODEL
    )

    model_path = config.MODELS[tag]["path_model"]
    scaler_path = config.MODELS[tag]["path_scalers"]

    verify_md5(model_path, config.MODELS[tag]["md5_model"], "Model")
    verify_md5(scaler_path, config.MODELS[tag]["md5_scalers"], "Scalers")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    mrnn = MRNN(losses=losses, model=model)

    # load scalers
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    return mrnn, scalers


def create_dataset_of_retrievals(filename, tag, y_quantiles, y_mean, surface_mask):

    group3_channels = [31, 32, 33, 34, 35, 36]

    ds = xr.open_dataset(filename)

    ds_out = xr.Dataset(
        coords={
            "number": np.arange(N_samples),
            "quantile": config.QUANTILES,
            "masked_channel": [f"AWS{ch}" for ch in group3_channels],
        }
    )

    ds_out["Latitude"] = xr.DataArray(ds["Latitude"], dims=("number",))
    ds_out["Longitude"] = xr.DataArray(ds["Longitude"], dims=("number",))
    ds_out["CloudSat_Datetime"] = xr.DataArray(ds["CloudSat_Datetime"], dims=("number",))
    ds_out["Fwp"] = xr.DataArray(ds["Fwp"], dims=("number",))

    for var in input_variables:
        ds_out[var] = xr.DataArray(ds[var], dims=("number",))

    for var in output_variables:
        channel_id = var[-2:]
        cloud_signal = ds[f"Ta_Clearsky_AWS{channel_id}"] - ds[f"Ta_Allsky_AWS{channel_id}"]
        ds_out[f"{var}_true"] = xr.DataArray(cloud_signal, dims=("number",))

        # (Small fix: these lines should be inside the loop if y_quantiles/y_mean are per-var)
        ds_out[f"{var}_quantiles"] = xr.DataArray(y_quantiles[var], dims=("number", "quantile"))
        ds_out[f"{var}_mean"] = xr.DataArray(y_mean[var], dims=("number",))


    surface_mask_2d = np.stack([ surface_mask[ch] for ch in group3_channels], axis=1)  # (N, 5)

    ds_out["surface_mask"] = xr.DataArray(
        surface_mask_2d.astype(bool),
        dims=("number", "surface_channel"),
        coords={
            "number": ds_out["number"],
            "surface_channel": ds_out["surface_channel"],
        },
    )

    ds_out.attrs["model_type"] = tag

    ds.close()
    return ds_out




def predict_quantiles(x, quantnn, quantiles):
    
    '''
    x : scaled input data for retrievals, e.g. Ta
    quantnn : xarray dataset, simulated data
    quantiles : list of quantiles to predict
    '''

    y_pred = quantnn.predict(x)

    y_quantiles = quantnn.posterior_quantiles(y_pred=y_pred, quantiles=quantiles)
    
    return y_quantiles


def run_batch(x, start, end, output_variables, y_quantiles, y_mean):
    y_pred_quantiles = predict_quantiles(x[start:end], mrnn, config.QUANTILES)

    for var in output_variables:

        y_quantiles[var][start:end, :] = y_pred_quantiles[var].astype(np.float32)

        y_mean = mrnn.posterior_mean(y_pred={var: y_quantiles[var]})

        y_mean[var][start:end, :] = y_mean.astype(np.float32)

    return y_quantiles, y_mean


if __name__ == "__main__":

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    surface_masks_test = retrieval_preprocessing.surface_filtering(filename=config.AWS_TEST_SET)

    for tag in config.VARIANTS.keys():

        input_variables = config.get_input_variables(tag)
        output_variables = config.get_output_variables(tag)

        mrnn, scalers = load_model(tag)

        x_test, y_test, _, surface_mask = retrieval_preprocessing.prepare_data(tag, config.AWS_TRAINING_SET, input_variables, surface_masks_test, train=False, scalers=scalers)

        # add noise to Ta
        for i, ch in enumerate(range(len(input_variables))):
            x_test[:,i] += np.random.normal(0, config.AWS_CHANNEL_NOISE[ch], len(x_test[:,i]))

        ds = xr.open_dataset(filename=config.AWS_TEST_SET)
        N_samples = len(ds["Latitude"])
        ds.close()

        # preallocate prediction arrays
        y_quantiles = {var: np.empty((N_samples, config.N_QUANTS), dtype=np.float32) for var in output_variables}
        y_mean = {var: np.empty((N_samples, config.N_QUANTS), dtype=np.float32) for var in output_variables}

        batch_size = 1e5
        for start in range(0, N_samples, batch_size):
            end = min(start + batch_size, N_samples)

            y_quantiles, y_mean = run_batch(x_test, start, end, output_variables, y_quantiles, y_mean)

        ds_retrievals = create_dataset_of_retrievals(config.AWS_TEST_SET, tag, y_quantiles, y_mean, surface_mask)

        today = datetime.today().strftime("%Y%m%d")
        out_path = config.DATA_DIR / f"cloud_signal_test_set_retrievals_{tag}_{today}.nc"
        ds_retrievals.to_netcdf(out_path)
        print(f"Saved: {out_path}")