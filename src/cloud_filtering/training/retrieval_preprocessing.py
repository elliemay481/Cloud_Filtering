# preprocessing_cloudsignal.py
import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch

import config as config

from surface_algorithm import surface_mask_simulations


class MyDataset(Dataset):
    def __init__(self, x_data, y_data, y_labels):
        self.x_data = x_data
        self.y_data = y_data
        self.y_labels = y_labels

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = tuple(self.y_data[var][idx] for var in self.y_labels)
        return x, y
    

def collate(batch, y_labels):
    x = torch.stack([item[0] for item in batch])
    y = {
        var: torch.tensor(np.array([item[1][i] for item in batch]))
        for i, var in enumerate(y_labels)
    }
    return x, y


def create_dataloader(x_data, y_data, y_labels, batch_size, shuffle):
    dataset = MyDataset(x_data, y_data, y_labels)

    # to generate batches of data from the dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collate(x, y_labels),
    )

    return dataloader


def surface_filtering(filename) -> xr.Dataset:
    """
    Keep only samples where all required channels are NOT surface impacted.

    dropped_channel_ids: channel (ids) that must be "clean" (mask False) for this model.
    """
    #if not dropped_channel_ids:
    #    return ds
    ds = xr.open_dataset(filename)

    surface_masks = surface_mask_simulations(ds)

    m31 = surface_masks["31"]
    m32 = surface_masks["32"]
    m33 = surface_masks["33"]
    m34 = surface_masks["34"]
    m35 = surface_masks["35"]
    m36 = surface_masks["36"]

    model_mask_variants = {
        "aws31_36": (~m32),
        "aws32_36": (~m33),
        "aws33_36": ( m31) & (~m34),
        "aws34_36": ( m32) & (~m35),
        "aws35_36": ( m33) & (~m36),
    }

    return model_mask_variants

def scale_x(x, train=True, scalers=None):

    variables_to_scale = ["MirrorAngle"]

    if train:
        scalers = {}
    
    for var in variables_to_scale:
        data = x[var].reshape(-1, 1)

        # want to scale both training and validation set according to training data statistics
        if train: 
            data_scaler = MinMaxScaler(feature_range=(80, 320))
            data_scaler.fit(data)
            scalers[f"{var}_scaler"] = data_scaler
        else:
            data_scaler = scalers[f"{var}_scaler"]

        data_scaled = data_scaler.transform(data)
        x[var] = data_scaled.flatten()

    return x, scalers

def prepare_data(tag, filename, input_variables, output_variables, surface_masks, train=True):

    ds = xr.open_dataset(filename)

    # select relevant surface mask
    mask = surface_masks[tag]

    x = {}
    y = {}

    for var in output_variables:
        channel_id = var[-2:]
        cloud_signal = ds[f"Ta_Clearsky_AWS{channel_id}"] - ds[f"Ta_Allsky_AWS{channel_id}"]
        y[var] = cloud_signal.values[train]

    for var in input_variables:
        x[var] = ds[var].values[mask]

    x, scalers = scale_x(x, train=train, scalers=scalers)

    ds.close()

    return x, y, scalers, mask

def prep_for_training(tag, input_variables, output_variables, surface_masks_training, surface_masks_validation):

    x_train, y_train, scalers, _ = prepare_data(tag, config.AWS_TRAINING_SET, input_variables, output_variables, surface_masks_training, train=True, scalers=None)
    x_val, y_val, _ = prepare_data(tag, config.AWS_VALIDATION_SET, input_variables, output_variables, surface_masks_validation, train=False, scalers=scalers)

    # prepare for pytorch
    x_train = np.vstack(list(x_train.values())).T
    x_val = np.vstack(list(x_val.values())).T

    x_train = torch.as_tensor(x_train, dtype=torch.float32)
    x_val = torch.as_tensor(x_val, dtype=torch.float32)

    # create data loader
    training_loader = create_dataloader(
        x_train,
        y_train,
        y_labels=list(y_train.keys()),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = create_dataloader(
        x_val, y_val, y_labels=list(y_val.keys()), batch_size=config.BATCH_SIZE, shuffle=True
    )

    return training_loader, validation_loader, scalers
