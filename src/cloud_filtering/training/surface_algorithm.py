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
    "AWS33": 1.00,
    "AWS34": 1.00,
    "AWS35": 1.00,
}

group3_channels = [31, 32, 33, 34, 35, 36]
channel_pairs = [(33, 44), (34, 43), (35, 42), (36, 41)]
pair_map = dict(channel_pairs)


def deep_convection_mask(lat, Ta):
    """
    Returns array that is True where deep convection is detected.
    Cases where this is True should NOT be flagged as surface-impacted.

    Parameters
    ----------
    lat
    Ta : dict 
    """

    eps = 0.5  # K

    tropics_mask = (lat > -60) & (lat < 60)

    Ta_3533 = Ta[35] - Ta[33]
    Ta_3534 = Ta[35] - Ta[34]
    Ta_3433 = Ta[34] - Ta[33]

    condition_dc = (
        tropics_mask
        & (Ta[32] < 240)
        & (Ta[33] < 240)
        & (Ta[34] < 240)
        & (Ta[35] < 240)
        & (Ta[36] < 240)
        & (Ta[41] < 220)
        & (Ta[42] < 220)
        & (Ta[43] < 220)
        & (Ta[44] < 220)
        & (Ta_3533 >= Ta_3534 - eps)
        & (Ta_3534 >= Ta_3433 - eps)
        & (Ta_3433 >= -eps)
        & ((np.abs(Ta[33] - Ta[34]) - np.abs(Ta[44] - Ta[43])) > -eps)
        #& ((Ta[33] - Ta[34]) < 0)
        & ((Ta[44] - Ta[43]) < 0)
    )

    return condition_dc


def surface_mask(ds):

    Tb = ds.aws_toa_brightness_temperature
    N = ds.number.shape[0]
    mask_all = {channel: np.zeros(N, dtype=bool) for channel in group3_channels}

    lat = ds.aws_lat.values

    Ta = {ch: Tb.sel(n_channels=f"AWS{ch}").values for ch in [31, 32, 33, 34, 35, 36, 41, 42, 43, 44]}
    cond_dc = deep_convection_mask(lat, Ta)

    for channel in group3_channels:

        if channel in [31, 32]:

            # --- condition 1 (diff between neighbouring 3X channel) ---
            dTa_group3 = Tb.sel(n_channels=f"AWS{channel}") - Tb.sel(
                n_channels=f"AWS{channel+1}"
            )
            cond1 = dTa_group3 < 0

            # not really sure here - testing
            # --- condition 2 (filtering out of water vapour saturation cases) ---
            dTa_33_44 = Tb.sel(n_channels=f"AWS33") - Tb.sel(n_channels=f"AWS44")
            cond2 = dTa_33_44 < 15

            numer = Tb.sel(n_channels=f"AWS44") * Tb.sel(n_channels=f"AWS34")
            denom = Tb.sel(n_channels=f"AWS33") * Tb.sel(n_channels=f"AWS43")
            cond3 = (numer / denom) > slope_threshold["AWS33"]

            base = cond1 & cond2
            use_cond3 = (
                np.abs(ds.aws_lat) < 60
            )  # only apply ratio threshold for mid and low lats

            mask = base & (~use_cond3 | cond3)

            mask_all[channel] = mask

        elif channel in [33, 34, 35]:

            # --- condition 1 (difference between group 3 neighbours) ---
            dTa_group3 = Tb.sel(n_channels=f"AWS{channel}") - Tb.sel(
                n_channels=f"AWS{channel+1}"
            )
            cond1 = dTa_group3 < 0

            # --- condition 2 (difference between group 4 neighbours) ---
            group4_1 = pair_map[channel]  # i.e. the partner of our group 3 channel
            group4_2 = pair_map[
                channel + 1
            ]  # i.e. the partner of our group 3 channel's neighbour
            dTa_group4 = Tb.sel(n_channels=f"AWS{group4_1}") - Tb.sel(
                n_channels=f"AWS{group4_2}"
            )
            cond2 = dTa_group4 < 0

            # --- condition 3 (difference between channel pairs) ---
            dTa_pair = Tb.sel(n_channels=f"AWS{channel}") - Tb.sel(
                n_channels=f"AWS{group4_1}"
            )
            cond3 = dTa_pair < 15

            numer = Tb.sel(n_channels=f"AWS{group4_1}") * Tb.sel(
                n_channels=f"AWS{channel+1}"
            )
            denom = Tb.sel(n_channels=f"AWS{channel}") * Tb.sel(n_channels=f"AWS{group4_2}")
            cond4 = (numer / denom) > slope_threshold[f"AWS{channel}"]

            # surface impact when all are true
            base = cond1 & cond2 & cond3
            use_cond4 = np.abs(ds.aws_lat) < 60

            mask = base & (~use_cond4 | cond4)

            # mask =  cond1 & cond2 & cond3 & cond4
            mask_all[channel] = mask

            if channel == 35:
                mask_all[36] = mask

    for i in range(len(group3_channels) - 2, -1, -1):  # walk backwards, start at 35

        mask_all[group3_channels[i]] = (
            mask_all[group3_channels[i]] | mask_all[group3_channels[i + 1]]
        )  # check right
        # A channelis True if itself OR a previous channel is True:
        #   (False | False) -> False
        #   (True  | False) -> True
        #   (False | True)  -> True
            #   (True  | True)  -> True


    # exclude any DC cases
    for ch in group3_channels:
        mask_all[ch] = mask_all[ch] & ~cond_dc

    return mask_all


def surface_mask_tmp(ds):

    Tb = ds.tb
    N = ds.scan.shape[0]
    mask_all = {channel: np.zeros(N, dtype=bool) for channel in group3_channels}

    lat = ds.latitude.values

    Ta = {ch: Tb.sel(channel=f"AWS{ch}").values for ch in [31, 32, 33, 34, 35, 36, 41, 42, 43, 44]}
    cond_dc = deep_convection_mask(lat, Ta)

    for channel in group3_channels:

        if channel in [31, 32]:

            # --- condition 1 (diff between neighbouring 3X channel) ---
            dTa_group3 = Tb.sel(channel=f"AWS{channel}") - Tb.sel(
                channel=f"AWS{channel+1}"
            )
            cond1 = dTa_group3 < 0

            # not really sure here - testing
            # --- condition 2 (filtering out of water vapour saturation cases) ---
            dTa_33_44 = Tb.sel(channel=f"AWS33") - Tb.sel(channel=f"AWS44")
            cond2 = dTa_33_44 < 15

            numer = Tb.sel(channel=f"AWS44") * Tb.sel(channel=f"AWS34")
            denom = Tb.sel(channel=f"AWS33") * Tb.sel(channel=f"AWS43")
            cond3 = (numer / denom) > slope_threshold["AWS33"]

            base = cond1 & cond2
            use_cond3 = (
                np.abs(lat) < 60
            )  # only apply ratio threshold for mid and low lats

            mask = base & (~use_cond3 | cond3)

            mask_all[channel] = mask

        elif channel in [33, 34, 35]:

            # --- condition 1 (difference between group 3 neighbours) ---
            dTa_group3 = Tb.sel(channel=f"AWS{channel}") - Tb.sel(
                channel=f"AWS{channel+1}"
            )
            cond1 = dTa_group3 < 0

            # --- condition 2 (difference between group 4 neighbours) ---
            group4_1 = pair_map[channel]  # i.e. the partner of our group 3 channel
            group4_2 = pair_map[
                channel + 1
            ]  # i.e. the partner of our group 3 channel's neighbour
            dTa_group4 = Tb.sel(channel=f"AWS{group4_1}") - Tb.sel(
                channel=f"AWS{group4_2}"
            )
            cond2 = dTa_group4 < 0

            # --- condition 3 (difference between channel pairs) ---
            dTa_pair = Tb.sel(channel=f"AWS{channel}") - Tb.sel(
                channel=f"AWS{group4_1}"
            )
            cond3 = dTa_pair < 15

            numer = Tb.sel(channel=f"AWS{group4_1}") * Tb.sel(
                channel=f"AWS{channel+1}"
            )
            denom = Tb.sel(channel=f"AWS{channel}") * Tb.sel(channel=f"AWS{group4_2}")
            cond4 = (numer / denom) >= slope_threshold[f"AWS{channel}"]

            # surface impact when all are true
            base = cond1 & cond2 & cond3
            use_cond4 = np.abs(lat) < 60

            mask = base & (~use_cond4 | cond4)

            # mask =  cond1 & cond2 & cond3 & cond4
            mask_all[channel] = mask

            if channel == 35:
                mask_all[36] = mask

    for i in range(len(group3_channels) - 2, -1, -1):  # walk backwards, start at 35

        mask_all[group3_channels[i]] = (
            mask_all[group3_channels[i]] | mask_all[group3_channels[i + 1]]
        )  # check right
        # A channelis True if itself OR a previous channel is True:
        #   (False | False) -> False
        #   (True  | False) -> True
        #   (False | True)  -> True
            #   (True  | True)  -> True


        # exclude any DC cases
        for ch in group3_channels:
            mask_all[ch] = mask_all[ch]

        return mask_all


def surface_mask_simulations(ds):

    N = ds.number.shape[0]
    mask_all = {
        channel: np.zeros(N, dtype=bool) for channel in group3_channels
    }  # one column for each model variant
    # cond4_all = {channel: np.zeros(N, dtype=bool) for channel in group3_channels} # for testing
    # cond5_all = {channel: np.zeros(N, dtype=bool) for channel in group3_channels} # for testing

    latitude = ds["Latitude"].values
    # add noise to simulations
    # for ch in CHANNELS:
    #    noisy_tb = ds[ch].values + np.random.normal(0, AWS_CHANNEL_NOISE[ch], N)  # or len(ds[ch])
    #    ds[f"{ch}_noisy"] = (ds[ch].dims, noisy_tb)

    Ta = {ch: ds[f"Ta_Allsky_AWS{ch}"].values for ch in [31, 32, 33, 34, 35, 36, 41, 42, 43, 44]}
    cond_dc = deep_convection_mask(latitude, Ta)

    for channel in group3_channels:

        # condition 0 (remove tropical DC from classifying as surface impact)
        # cond0 = (ds["Ta_Allsky_AWS33"].values < 150) & (np.abs(ds["Latitude"].values) < 30)

        if channel in [31, 32]:

            # --- condition 1 (group 3 diff) ---
            dTa_g3 = (
                ds[f"Ta_Allsky_AWS{channel}"].values
                - ds[f"Ta_Allsky_AWS{channel+1}"].values
            )
            cond1 = dTa_g3 < 0

            # --- condition 2 (filtering out of water vapour saturation cases) ---
            dTa_pair = ds[f"Ta_Allsky_AWS33"].values - ds[f"Ta_Allsky_AWS44"].values
            cond2 = dTa_pair < 15

            # --- condition 3 (removal of some cloudy cases):
            numer = ds[f"Ta_Allsky_AWS44"].values * ds[f"Ta_Allsky_AWS34"].values
            denom = ds[f"Ta_Allsky_AWS33"].values * ds[f"Ta_Allsky_AWS43"].values
            cond3 = (numer / denom) > slope_threshold["AWS33"]

            # surface impact when all are true
            base = cond1 & cond2
            use_cond3 = np.abs(latitude) < 60

            mask = base & (~use_cond3 | cond3)

            mask_all[channel] = mask
            # cond4_all[channel] = cond3

        elif channel in [33, 34, 35]:

            # --- condition 1 (surface impact check - lower altitude channel sees colder) ---
            dTa_g3 = (
                ds[f"Ta_Allsky_AWS{channel}"].values
                - ds[f"Ta_Allsky_AWS{channel+1}"].values
            )
            cond1 = dTa_g3 < 0

            # --- condition 2 (surface impact check - lower altitude channel sees colder) ---
            g4_1 = pair_map[channel]
            g4_2 = pair_map[channel + 1]

            dTa_g4 = (
                ds[f"Ta_Allsky_AWS{g4_1}"].values - ds[f"Ta_Allsky_AWS{g4_2}"].values
            )
            cond2 = dTa_g4 < 0

            # --- condition 3 (filtering out of water vapour saturation cases) ---
            dTa_pair = (
                ds[f"Ta_Allsky_AWS{channel}"].values - ds[f"Ta_Allsky_AWS{g4_1}"].values
            )
            cond3 = dTa_pair < 15

            # --- condition 4 (separating scattering dTb from surface emissivity dTb):
            numer = (
                ds[f"Ta_Allsky_AWS{channel}"].values - ds[f"Ta_Allsky_AWS{g4_1}"].values
            )
            denom = (
                ds[f"Ta_Allsky_AWS{channel}"].values - ds[f"Ta_Allsky_AWS{g4_2}"].values
            )

            numer = (
                ds[f"Ta_Allsky_AWS{g4_1}"].values
                * ds[f"Ta_Allsky_AWS{channel+1}"].values
            )
            denom = (
                ds[f"Ta_Allsky_AWS{channel}"].values * ds[f"Ta_Allsky_AWS{g4_2}"].values
            )
            cond4 = (numer / denom) > slope_threshold[f"AWS{channel}"]

            # surface impact when all are true
            base = cond1 & cond2 & cond3
            use_cond4 = np.abs(latitude) < 60

            mask = base & (~use_cond4 | cond4)

            mask_all[channel] = mask

            if channel == 35:
                mask_all[36] = mask

    for i in range(len(group3_channels) - 2, -1, -1):  # walk backwards, start at 35

        mask_all[group3_channels[i]] = (
            mask_all[group3_channels[i]] | mask_all[group3_channels[i + 1]]
        )  # check right
        # A channelis True if itself OR a previous channel is True:
        #   (False | False) -> False
        #   (True  | False) -> True
        #   (False | True)  -> True
        #   (True  | True)  -> True

    for ch in group3_channels:
        mask_all[ch] = mask_all[ch] & ~cond_dc

    return mask_all
