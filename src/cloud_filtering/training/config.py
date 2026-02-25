# config_cloudsignal.py
import numpy as np
from pathlib import Path
from datetime import datetime

SEED = 0

#DATA_DIR = Path("/cephyr/users/maye/Vera/data/AWS/")
DATA_DIR = Path("/home/eleanor/Documents/Research/PHD/DataStorage/AWS/")

# Where trained models/scalers go
MODEL_DIR = Path("/cephyr/users/maye/Vera/data/models/cloud_signal/")

today = datetime.today().strftime("%Y%m%d")
MODEL_TEMPLATE   = MODEL_DIR / "MRNN_AWS_cloudsignal_{tag}_{today}.pt"
SCALER_TEMPLATE  = MODEL_DIR / "MRNN_AWS_cloudsignal_scalers_{tag}_{today}.pkl"

# pre-divided training set? probably yes
AWS_TRAINING_SET = DATA_DIR / "aws_database_2025-11-10_training.nc"
AWS_VALIDATION_SET = DATA_DIR / "aws_database_2025-11-10_validation.nc"
AWS_TEST_SET = DATA_DIR / "aws_database_2025-11-10_test.nc"



# -------------------------------------------------
# Inputs / outputs
# -------------------------------------------------

BASE_INPUT_VARIABLES = [
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

BASE_OUTPUT_VARIABLES = [
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

# -------------------------------------------------
# 5 models
# progressively remove channels based on surface filtering
# note: never train if only AWS36/AWS41 remains
# -------------------------------------------------

VARIANTS = {
    "aws31_36": [],
    "aws32_36": ["Ta_Allsky_AWS31"],
    "aws33_36": ["Ta_Allsky_AWS31", "Ta_Allsky_AWS32"],
    "aws34_36": ["Ta_Allsky_AWS31", "Ta_Allsky_AWS32", "Ta_Allsky_AWS33", "Ta_Allsky_AWS44"],
    "aws35_36": ["Ta_Allsky_AWS31", "Ta_Allsky_AWS32", "Ta_Allsky_AWS33", "Ta_Allsky_AWS44", "Ta_Allsky_AWS34", "Ta_Allsky_AWS43"],
}


def get_input_variables(tag):
    """Return input list for a given variant."""
    drop = set(VARIANTS[tag])
    inputs = [v for v in BASE_INPUT_VARIABLES if v not in drop]

    # never train if only AWS36 remains from AWS3x
    #aws3x = [v for v in inputs if v.startswith("Ta_Allsky_AWS3")]
    #if aws3x == ["Ta_Allsky_AWS36"]:
    #    raise ValueError("Invalid variant: only AWS36 remaining.")

    return inputs

def get_output_variables(tag):
    """Return input list for a given variant."""
    drop = set(VARIANTS[tag])
    outputs = []
    for var in BASE_OUTPUT_VARIABLES:
        var_allsky = f"CloudSignal_AWS{var[-2:]}"
        if var_allsky not in drop:
            outputs.append(var)

    return outputs

CHANNEL_IDS = {
    "Ta_Allsky_AWS31": 31,
    "Ta_Allsky_AWS32": 32,
    "Ta_Allsky_AWS33": 33,
    "Ta_Allsky_AWS34": 34,
    "Ta_Allsky_AWS35": 35,
    "Ta_Allsky_AWS36": 36,
    "Ta_Allsky_AWS41": 41,
    "Ta_Allsky_AWS42": 42,
    "Ta_Allsky_AWS43": 43,
    "Ta_Allsky_AWS44": 44,
}


# -------------------------------------------------
# Quantiles / losses
# -------------------------------------------------

N_QUANTS = 99
QUANTILES = np.linspace(0.01, 0.99, N_QUANTS)

# -------------------------------------------------
# Model parameters
# -------------------------------------------------

MODEL = {
    "n_layers": 2,
    "width": 8,
    "batch_norm": True,
}

# -------------------------------------------------
# Training parameters
# -------------------------------------------------

TRAINING_STAGES = [
    {"lr": 1e-2, "epochs": 10},
    {"lr": 1e-3, "epochs": 10},
]

BATCH_SIZE = 512
DEVICE = "cpu"
ADVERSARIAL_TRAINING = 0.05

AWS_CHANNEL_NOISE = {
    "Ta_Allsky_AWS31": 0.6,
    "Ta_Allsky_AWS32": 0.7,
    "Ta_Allsky_AWS33": 0.7,
    "Ta_Allsky_AWS34": 1.0,
    "Ta_Allsky_AWS35": 1.0,
    "Ta_Allsky_AWS36": 1.3,
    "Ta_Allsky_AWS41": 1.7,
    "Ta_Allsky_AWS42": 1.4,
    "Ta_Allsky_AWS43": 1.2,
    "Ta_Allsky_AWS44": 1.0,
}



MODELS = {
    "aws31_36": {
        "path_model":   MODEL_DIR / "MRNN_AWS_model_aws31_36_2025-11-26.pt",
        "md5_model":    "8f206334d81f959226035bf9dfd8743d",
        "path_scalers": MODEL_DIR / "MRNN_AWS_scalers_aws31_36_2025-11-26.pkl",
        "md5_scalers":  "5e9d7f6a69815554c4dadfdf6465ec62",
    },
    "aws32_36": {
        "path_model":   MODEL_DIR / "MRNN_AWS_model_aws32_36_2025-11-26.pt",
        "md5_model":    "....",
        "path_scalers": MODEL_DIR / "MRNN_AWS_scalers_aws32_36_2025-11-26.pkl",
        "md5_scalers":  "....",
    },
    "aws33_36": {
        "path_model":   MODEL_DIR / "MRNN_AWS_model_aws33_36_2025-11-26.pt",
        "md5_model":    "....",
        "path_scalers": MODEL_DIR / "MRNN_AWS_scalers_aws33_36_2025-11-26.pkl",
        "md5_scalers":  "....",
    },
    "aws34_36": {
        "path_model":   MODEL_DIR / "MRNN_AWS_model_aws34_36_2025-11-26.pt",
        "md5_model":    "....",
        "path_scalers": MODEL_DIR / "MRNN_AWS_scalers_aws34_36_2025-11-26.pkl",
        "md5_scalers":  "....",
    },
    "aws35_36": {
        "path_model":   MODEL_DIR / "MRNN_AWS_model_aws35_36_2025-11-26.pt",
        "md5_model":    "....",
        "path_scalers": MODEL_DIR / "MRNN_AWS_scalers_aws35_36_2025-11-26.pkl",
        "md5_scalers":  "....",
    },
}