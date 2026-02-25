# train_cloudsignal.py
import pickle
import numpy as np
import torch
from pathlib import Path

from quantnn.mrnn import MRNN
from toolbox.models.ICINN import AWS_CF_Model
from quantnn.mrnn import Quantiles

import config as config
import retrieval_preprocessing


def save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def train_variant(tag, surface_masks_training, surface_masks_validation):

    print(f"\n===== Training {tag} =====")

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    input_variables = config.get_input_variables(tag)
    output_variables = config.get_output_variables(tag)
    dropped_variables = config.VARIANTS[tag]

    # dropped variables -> channel ids
    dropped_channel_ids = []
    for ch in dropped_variables:
        dropped_channel_ids.append(config.CHANNEL_IDS[ch])

    # create dataset with relevant channels, scale variables, and prep for pytorch
    training_loader, validation_loader, scalers = retrieval_preprocessing.prepare_data(tag, input_variables, output_variables, 
        surface_masks_training, surface_masks_validation
    )

    n_quantile_outputs = {name: config.N_QUANTS for name in output_variables}
    losses = {name: Quantiles(config.QUANTILES) for name in output_variables}


    
    model = AWS_CF_Model(
        n_inputs=len(input_variables),
        n_outputs=n_quantile_outputs,
        **config.MODEL
    )

    mrnn = MRNN(losses=losses, model=model)

    for stage in config.TRAINING_STAGES:
        optimizer = torch.optim.SGD(
            mrnn.model.parameters(),
            lr=stage["lr"],
            momentum=0.5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, stage["epochs"]
        )

        # select noise of the channels we use
        sigma_noise = np.array([
            config.AWS_CHANNEL_NOISE[v]
            for v in input_variables
            if v in config.AWS_CHANNEL_NOISE
        ])
        
        mrnn.train(
            training_loader,
            validation_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            adversarial_training=config.ADVERSARIAL_TRAINING,
            n_epochs=stage["epochs"],
            device=config.DEVICE,
            sigma_noise=sigma_noise,
        )

        today = datetime.today().strftime("%Y%m%d")

    model_path  = Path(str(config.MODEL_TEMPLATE).format(tag=tag))
    scaler_path = Path(str(config.SCALER_TEMPLATE).format(tag=tag))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    save_pickle(scalers, scaler_path)

    print(f"Saved model  -> {model_path}")
    print(f"Saved scalers-> {scaler_path}")
    print(f"n_inputs={len(input_variables)} | dropped={dropped_variables}")
    

if __name__ == "__main__":

    surface_masks_training = retrieval_preprocessing.surface_filtering(filename=config.AWS_TRAINING_SET)
    surface_masks_validation = retrieval_preprocessing.surface_filtering(filename=config.AWS_VALIDATION_SET)

    for tag in config.VARIANTS.keys():
        train_variant(tag, surface_masks_training, surface_masks_validation)
