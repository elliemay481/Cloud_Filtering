from pathlib import Path
import torch
from quantnn.mrnn import MRNN
from QRNN_model import CloudSignalModel_MultiOutput
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

    model = CloudSignalModel_MultiOutput(
        n_inputs=len(input_variables),
        n_outputs=n_quantile_outputs,
        **config.MODEL
    )

    model_path = config.MODELS[tag]["path_model"]
    scaler_path = config.MODELS[tag]["path_scalers"]

    #verify_md5(model_path, config.MODELS[tag]["md5_model"], "Model")
    #verify_md5(scaler_path, config.MODELS[tag]["md5_scalers"], "Scalers")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    mrnn = MRNN(losses=losses, model=model)

    # load scalers
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    return mrnn, scalers


def create_dataset_of_retrievals(filename, input_variables, output_variables, tag, y_quantiles, y_mean, y_test, x_test, N_samples, mask):

    group3_channels = [31, 32, 33, 34, 35, 36]

    ds = xr.open_dataset(filename)

    ds_out = xr.Dataset(
        coords={
            "number": np.arange(N_samples),
            "quantile": config.QUANTILES,
            "number_test_set": np.arange(ds["Latitude"].shape[0]),
        }
    )

    ds_out["Latitude"] = xr.DataArray(ds["Latitude"][mask], dims=("number",))
    ds_out["Longitude"] = xr.DataArray(ds["Longitude"][mask], dims=("number",))
    ds_out["CloudSat_Datetime"] = xr.DataArray(ds["CloudSat_Datetime"][mask], dims=("number",))
    ds_out["Fwp"] = xr.DataArray(ds["Fwp"][mask], dims=("number",))

    for var in input_variables:
        ds_out[var] = xr.DataArray(x_test[var], dims=("number",))
    print(ds_out[var].shape)

    for var in output_variables:
        channel_id = var[-2:]
        #cloud_signal = ds[f"Ta_Clearsky_AWS{channel_id}"] - ds[f"Ta_Allsky_AWS{channel_id}"]
        #ds_out[f"{var}_true"] = xr.DataArray(cloud_signal, dims=("number",))
        ds_out[f"{var}_true"] = xr.DataArray(y_test[var], dims=("number",))

        ds_out[f"{var}_quantiles"] = xr.DataArray(y_quantiles[var], dims=("number", "quantile"))
        ds_out[f"{var}_mean"] = xr.DataArray(y_mean[var], dims=("number",))


    #surface_mask_2d = np.stack([mask[ch] for ch in group3_channels], axis=1)  # (N, 5)

    ds_out["surface_mask"] = xr.DataArray(
        mask.astype(bool),
        dims=("number_test_set"),
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
        y_quantiles[var][start:end, :] = y_pred_quantiles[var].detach().numpy().astype(np.float32)

        y_pred_mean = mrnn.posterior_mean(y_pred={var: y_quantiles[var][start:end, :]})

        y_mean[var][start:end] = y_pred_mean[var].astype(np.float32)

    return y_quantiles, y_mean


if __name__ == "__main__":

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    surface_masks_test = retrieval_preprocessing.surface_filtering(filename=config.AWS_TEST_SET)

    for tag in config.VARIANTS.keys():
        print("-------")
        print(tag)
        input_variables = config.get_input_variables(tag)
        output_variables = config.get_output_variables(tag)
        print(output_variables)
        mrnn, scalers = load_model(tag)

        x_test, y_test, _, surface_mask = retrieval_preprocessing.prepare_data(tag, config.AWS_TEST_SET, input_variables, output_variables, surface_masks_test, train=False, scalers=scalers)
        x_test_np = np.vstack([x_test[v] for v in input_variables]).T

        # add noise to Ta
        for i, ch in enumerate(input_variables[:-1]): # all except mirror angle
            x_test_np[:,i] += np.random.normal(0, config.AWS_CHANNEL_NOISE[ch], len(x_test_np[:,i]))

        #ds = xr.open_dataset(config.AWS_TEST_SET)
        #N_samples = len(ds["Latitude"])
        #ds.close()
        N_samples = x_test_np.shape[0]
        print(x_test_np.shape)


        #print(N_samples)
        # preallocate prediction arrays
        y_quantiles = {var: np.empty((N_samples, config.N_QUANTS), dtype=np.float32) for var in output_variables}
        y_mean = {var: np.empty(N_samples, dtype=np.float32) for var in output_variables}

        batch_size = 1e5
        for start in range(0, int(N_samples), int(batch_size)):
            end = int(min(start + batch_size, N_samples))

            y_quantiles, y_mean = run_batch(x_test_np, start, end, output_variables, y_quantiles, y_mean)

        ds_retrievals = create_dataset_of_retrievals(config.AWS_TEST_SET, input_variables, output_variables, tag, y_quantiles, y_mean, y_test, x_test, N_samples, surface_mask)

        today = datetime.today().strftime("%Y%m%d")
        out_path = config.DATA_DIR / f"cloud_signal_test_set_retrievals_{tag}_{today}.nc"
        ds_retrievals.to_netcdf(out_path)
        print(f"Saved: {out_path}")

