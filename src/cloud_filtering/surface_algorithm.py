import numpy as np


CHANNELS = [
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
]

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

slope_threshold = {
    "AWS33": 0.99,
    "AWS34": 0.98,
    "AWS35": 0.97,
}

group3_channels = [31, 32, 33, 34, 35, 36]
channel_pairs = [(33, 44), (34, 43), (35, 42), (36, 41)]
pair_map = dict(channel_pairs)

def surface_mask(ds, channel, thr_g3=-1, thr_g4=-1, thr_pair=5):

    Tb = ds.aws_toa_brightness_temperature
    N = ds.number.shape[0]
    mask_all = {channel: np.zeros(N, dtype=bool) for channel in group3_channels}

    if channel in [31, 32]:

        # --- condition 1 (diff between neighbouring 3X channel) ---
        dTa_g3 = Tb.sel(n_channels=f"AWS{channel}") - Tb.sel(
            n_channels=f"AWS{channel+1}"
        )
        cond1 = dTa_g3 < 0

        # not really sure here - testing
        # --- condition 2 (cfiltering out of water vapour saturation cases) ---
        dTa_33_44 = Tb.sel(n_channels=f"AWS33") - Tb.sel(n_channels=f"AWS44")
        cond2 = dTa_33_44 < 15

        numer = Tb.sel(n_channels=f"AWS44")*Tb.sel(n_channels=f"AWS34")
        denom = Tb.sel(n_channels=f"AWS33")*Tb.sel(n_channels=f"AWS43")
        cond3 = (numer / denom) > slope_threshold["AWS33"]

        base = cond1 & cond2 & cond3
        use_cond4 = np.abs(latitude) < 60

        mask = base & (~use_cond4 | cond4)

        # surface impact when all are true
        #mask = cond1 & cond2 & cond3

        mask_all[channel] = mask

    elif channel in [33, 34, 35]:

        # --- condition 1 (group 3 diff) ---
        dTa_g3 = Tb.sel(n_channels=f"AWS{channel}") - Tb.sel(
            n_channels=f"AWS{channel+1}"
        )
        cond1 = dTa_g3 < thr_g3

        # --- condition 2 (group 4 diff) ---
        g4_1 = pair_map[channel]
        g4_2 = pair_map[channel + 1]
        dTa_g4 = Tb.sel(n_channels=f"AWS{g4_1}") - Tb.sel(n_channels=f"AWS{g4_2}")
        cond2 = dTa_g4 < thr_g4

        # --- condition 3 (cross-pair) ---
        dTa_pair = Tb.sel(n_channels=f"AWS{channel}") - Tb.sel(n_channels=f"AWS{g4_1}")
        cond3 = dTa_pair < thr_pair

        numer = Tb.sel(n_channels=f"AWS{g4_1}")*Tb.sel(n_channels=f"AWS{channel+1}")
        denom = Tb.sel(n_channels=f"AWS{channel}")*Tb.sel(n_channels=f"AWS{g4_2}")
        cond4 = (numer / denom) > slope_threshold[channel]

        # surface impact when all are true

        base = cond1 & cond2 & cond3
        use_cond4 = np.abs(latitude) < 60

        mask = base & (~use_cond4 | cond4)
        
        #mask =  cond1 & cond2 & cond3 & cond4
        mask_all[channel] = mask

        if channel == 35:
                mask_all[36] = mask

    for i in range(len(group3_channels) - 2, -1, -1):  # walk backwards, start at 35

        mask_all[group3_channels[i]] = mask_all[group3_channels[i]] | mask_all[group3_channels[i + 1]] # check right
        # A channelis True if itself OR a previous channel is True:
        #   (False | False) -> False
        #   (True  | False) -> True
        #   (False | True)  -> True
        #   (True  | True)  -> True

    return mask_all


def surface_mask_simulations(ds):

    N = ds.number.shape[0]
    mask_all = {channel: np.zeros(N, dtype=bool) for channel in group3_channels} # one column for each model variant
    #cond4_all = {channel: np.zeros(N, dtype=bool) for channel in group3_channels} # for testing
    #cond5_all = {channel: np.zeros(N, dtype=bool) for channel in group3_channels} # for testing

    latitude = ds["Latitude"].values
    # add noise to simulations
    #for ch in CHANNELS:
    #    noisy_tb = ds[ch].values + np.random.normal(0, AWS_CHANNEL_NOISE[ch], N)  # or len(ds[ch])
    #    ds[f"{ch}_noisy"] = (ds[ch].dims, noisy_tb)
    
    for channel in group3_channels:

        # condition 0 (remove tropical DC from classifying as surface impact)
        #cond0 = (ds["Ta_Allsky_AWS33"].values < 150) & (np.abs(ds["Latitude"].values) < 30)

        if channel in [31, 32]:

            # --- condition 1 (group 3 diff) ---
            dTa_g3 = ds[f"Ta_Allsky_AWS{channel}"].values - ds[f"Ta_Allsky_AWS{channel+1}"].values
            cond1 = dTa_g3 < 0

            # --- condition 2 (filtering out of water vapour saturation cases) ---
            dTa_pair = ds[f"Ta_Allsky_AWS33"].values - ds[f"Ta_Allsky_AWS44"].values
            cond2 = dTa_pair < 15

            # --- condition 3 (removal of some cloudy cases):
            numer = ds[f"Ta_Allsky_AWS44"].values*ds[f"Ta_Allsky_AWS34"].values
            denom = ds[f"Ta_Allsky_AWS33"].values*ds[f"Ta_Allsky_AWS43"].values
            #slope = (numer / (denom))
            #cond4 = slope <= 1
            cond3 = (numer / denom) > slope_threshold["AWS33"]

            # surface impact when all are true
            base = cond1 & cond2 & cond3
            use_cond3 = np.abs(latitude) < 60

            mask = base & (~use_cond3 | cond3)

            mask_all[channel] = mask
            #cond4_all[channel] = cond3


        elif channel in [33, 34, 35]:

            # --- condition 1 (surface impact check - lower altitude channel sees colder) ---
            dTa_g3 = ds[f"Ta_Allsky_AWS{channel}"].values - ds[f"Ta_Allsky_AWS{channel+1}"].values
            cond1 = dTa_g3 < 0

            # --- condition 2 (surface impact check - lower altitude channel sees colder) ---
            g4_1 = pair_map[channel]
            g4_2 = pair_map[channel + 1]

            dTa_g4 = ds[f"Ta_Allsky_AWS{g4_1}"].values - ds[f"Ta_Allsky_AWS{g4_2}"].values
            cond2 = dTa_g4 < 0

            # --- condition 3 (filtering out of water vapour saturation cases) ---
            dTa_pair = ds[f"Ta_Allsky_AWS{channel}"].values - ds[f"Ta_Allsky_AWS{g4_1}"].values
            cond3 = dTa_pair < 15

            # --- condition 4 (separating scattering dTb from surface emissivity dTb):
            numer = ds[f"Ta_Allsky_AWS{channel}"].values - ds[f"Ta_Allsky_AWS{g4_1}"].values
            denom = ds[f"Ta_Allsky_AWS{channel}"].values - ds[f"Ta_Allsky_AWS{g4_2}"].values
            #cond4 = (numer/(denom)) < 1

            #Tb33 = ds[f"Ta_Allsky_AWS{channel}"].values
            #Tb34 = ds[f"Ta_Allsky_AWS{channel+1}"].values
            #Tb43 = ds[f"Ta_Allsky_AWS{g4_1}"].values
            #Tb44 = ds[f"Ta_Allsky_AWS{g4_2}"].values

            #cond4 = (Tb33 < Tb43) & (Tb43 < Tb44)

            numer = ds[f"Ta_Allsky_AWS{g4_1}"].values*ds[f"Ta_Allsky_AWS{channel+1}"].values
            denom = ds[f"Ta_Allsky_AWS{channel}"].values*ds[f"Ta_Allsky_AWS{g4_2}"].values
            #slope = (numer / (denom))
            #cond4 = slope <= 1
            cond4 = (numer / denom) > slope_threshold[f"AWS{channel}"]
            #cond5 = dTa_pair < 10
            #cond4 = (Tb44/Tb43) > (Tb33/Tb34)

            # surface impact when all are true
            base = cond1 & cond2 & cond3
            use_cond4 = np.abs(latitude) < 60

            mask = base & (~use_cond4 | cond4)
            #mask_2 = base & (~use_cond4 | cond5)

            mask_all[channel] = mask
            #cond4_all[channel] = cond4
            #cond5_all[channel] = cond5

            if channel == 35:
                mask_all[36] = mask



    for i in range(len(group3_channels) - 2, -1, -1):  # walk backwards, start at 35

        mask_all[group3_channels[i]] = mask_all[group3_channels[i]] | mask_all[group3_channels[i + 1]] # check right
        # A channelis True if itself OR a previous channel is True:
        #   (False | False) -> False
        #   (True  | False) -> True
        #   (False | True)  -> True
        #   (True  | True)  -> True

    return mask_all
