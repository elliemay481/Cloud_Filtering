import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

import quantnn
from quantnn import models
from quantnn import QRNN
from quantnn.mrnn import Quantiles, Density, Mean, Classification, MRNN
import os

from toolbox.models.ICINN import ICIModel, ICIModel_MultiOutput

from toolbox.retrieval_helper_functions import log_linear_transformation
import toolbox.retrieval_functions_aws as retrieval_functions
import toolbox.data_functions as data_functions
from toolbox import data_helper
from toolbox import utils

np.random.seed(0)
AWS_channel_noise = np.array([0.6, 0.7, 0.7, 1, 1, 1.3, 1.7, 1.4, 1.2, 1])
#AWS_channel_noise = np.zeros(10)

n_channels = 11

# load data used to train NN
data_directory = "/cephyr/users/maye/Vera/data/AWS/"
train_data_path = data_directory + "aws_train_dataset_new_120825.pkl"
val_data_path = data_directory + "aws_validate_dataset_new_120825.pkl"
#training_data = utils.load_processed_data(train_data_path)
#validation_data = utils.load_processed_data(val_data_path)
with open(train_data_path, "rb") as f:
    training_data = pickle.load(f)
with open(val_data_path, "rb") as f:
    validation_data = pickle.load(f)


# define input and output variables
n_inputs = 11

input_variables = [
    "Ta_Allsky_AWS31",
    "Ta_Allsky_AWS32",
    "Ta_Allsky_AWS33",
    "Ta_Allsky_AWS34",
    "Ta_Allsky_AWS35",
    "Ta_Allsky_AWS36",
    "Ta_Allsky_AWS41",
    "Ta_Allsky_AWS42",
    "Ta_Allsky_AWS43",
    "Ta_Allsky_AWS44",
    "MirrorAngle",
]
output_variables = [
    "CloudSignal_AWS31",
    "CloudSignal_AWS32",
    "CloudSignal_AWS33",
    "CloudSignal_AWS34",
    "CloudSignal_AWS35",
    "CloudSignal_AWS36",
    "CloudSignal_AWS41",
    "CloudSignal_AWS42",
    "CloudSignal_AWS43",
    "CloudSignal_AWS44",
]

n_quants = 17
n_quantile_outputs = {
    "CloudSignal_AWS31": n_quants,
    "CloudSignal_AWS32": n_quants,
    "CloudSignal_AWS33": n_quants,
    "CloudSignal_AWS34": n_quants,
    "CloudSignal_AWS35": n_quants,
    "CloudSignal_AWS36": n_quants,
    "CloudSignal_AWS41": n_quants,
    "CloudSignal_AWS42": n_quants,
    "CloudSignal_AWS43": n_quants,
    "CloudSignal_AWS44": n_quants}

losses = {
    "CloudSignal_AWS31": Quantiles(np.linspace(0.01, 0.99, n_quants)),
    "CloudSignal_AWS32": Quantiles(np.linspace(0.01, 0.99, n_quants)),
    "CloudSignal_AWS33": Quantiles(np.linspace(0.01, 0.99, n_quants)),
    "CloudSignal_AWS34": Quantiles(np.linspace(0.01, 0.99, n_quants)),
    "CloudSignal_AWS35": Quantiles(np.linspace(0.01, 0.99, n_quants)),
    "CloudSignal_AWS36": Quantiles(np.linspace(0.01, 0.99, n_quants)),
    "CloudSignal_AWS41": Quantiles(np.linspace(0.01, 0.99, n_quants)),
    "CloudSignal_AWS42": Quantiles(np.linspace(0.01, 0.99, n_quants)),
    "CloudSignal_AWS43": Quantiles(np.linspace(0.01, 0.99, n_quants)),
    "CloudSignal_AWS44": Quantiles(np.linspace(0.01, 0.99, n_quants)),
}

# setup training and validation data
training_loader, validation_loader, scalers = retrieval_functions.prepare_training_data(
    training_data,
    validation_data,
    batch_size=512,
    output_variables=output_variables,
    input_variables=input_variables,
    cloud_filter=True,
    cloud_signal=True,
)

model = ICIModel_MultiOutput(
    n_inputs=n_inputs,
    n_outputs=n_quantile_outputs,
#    n_layers=12,
#    width=512,
    n_layers=2,
    width=8,
    batch_norm=True,
)
mrnn = MRNN(losses=losses, model=model)


# train the model
n_epochs = 10
"""
optimizer = torch.optim.SGD(mrnn.model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
mrnn.train(
    training_loader,
    validation_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    adversarial_training=0.05,
    n_epochs=10,
    device="cpu",
    sigma_noise=AWS_channel_noise,
)
"""
optimizer = torch.optim.SGD(mrnn.model.parameters(), lr=0.01, momentum=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
mrnn.train(
    training_loader,
    validation_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    adversarial_training=0.05,
    n_epochs=n_epochs,
    device="cpu",
    sigma_noise=AWS_channel_noise,
)


optimizer = torch.optim.SGD(mrnn.model.parameters(), lr=0.001, momentum=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
mrnn.train(
    training_loader,
    validation_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    adversarial_training=0.05,
    n_epochs=n_epochs,
    device="cpu",
    sigma_noise=AWS_channel_noise,
)
"""
optimizer = torch.optim.SGD(mrnn.model.parameters(), lr=0.0001, momentum=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
mrnn.train(
    training_loader,
    validation_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    adversarial_training=0.05,
    n_epochs=n_epochs,
    device="cpu",
    sigma_noise=AWS_channel_noise,
)

optimizer = torch.optim.SGD(mrnn.model.parameters(), lr=0.00001, momentum=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
mrnn.train(
    training_loader,
    validation_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    adversarial_training=0.05,
    n_epochs=n_epochs,
    device="cpu",
    sigma_noise=AWS_channel_noise,
)
"""
model_path = "/cephyr/users/maye/Vera/data/models/MRNN_AWS_cloudfiltermodel_cloudsignal_smallmodel.pt"

torch.save(mrnn.model.state_dict(), model_path)
model.load_state_dict(torch.load(model_path))
model.eval()

# save the scalers for predictions
with open("/cephyr/users/maye/Vera/data/models/MRNN_AWS_cloudfilterscalers_cloudsignal_smallmodel.pkl", "wb") as f:
    pickle.dump(scalers, f)
