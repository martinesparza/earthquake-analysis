from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaldata as pyal
from sklearn.decomposition import PCA


def reshape_to_trials(signal_1d, trial_length_samples):
    total_samples = len(signal_1d)
    if total_samples % trial_length_samples != 0:
        raise ValueError("Total number of samples is not divisible by trial length.")

    n_trials = total_samples // trial_length_samples
    return signal_1d.reshape(n_trials, trial_length_samples)


def get_trial_x_time_per_neuron(df, area, neuron_id, trial_length=200):
    df_trials = pyal.select_trials(df, df.trial_name == "trial")
    trials_arr = pyal.concat_trials(df_trials[:-1], f"{area}_spikes")[:, neuron_id]

    return reshape_to_trials(trials_arr, trial_length)


def get_data_array(
    data_list: list[pd.DataFrame],
    trial_cat="values_Sol_direction",
    epoch: Callable = None,
    area: str = "M1",
    units: Callable = None,
    model: Callable = "pca",
    n_components: int = 10,
) -> np.ndarray:
    """
    Applies the `model` to the `data_list` and return a data matrix of the shape: sessions x targets x trials x time x modes
    with the minimum number of trials and timepoints shared across all the datasets/targets.

    Parameters
    ----------
    `data_list`: list of pd.dataFrame datasets from pyalData (could also be a single dataset)
    `epoch`: an epoch function of the type `pyal.generate_epoch_fun()`
    `area`: area, either: 'PFC', or 'PFC_removed_var', ...
    `model`: a model that implements `.fit()`, `.transform()` and `n_components`. By default: `PCA(10)`. If it's an integer: `PCA(integer)`.
    `n_components`: use `model`, this is for backward compatibility
    'trial_cat': str representing category by which trials are grouped: eg 'cue_id','target_id'
    Returns
    -------
    `AllData`: np.ndarray

    Signature
    -------
    AllData = get_data_array(data_list, delay_epoch, area='PFC', model=10)
    all_data = np.reshape(AllData, (-1,10))
    """

    if isinstance(data_list, pd.DataFrame):
        data_list = [data_list]
    if model is None:
        model = PCA(n_components=n_components, svd_solver="full")
        pca_field = "_pca"
    elif isinstance(model, int):
        model = PCA(n_components=model, svd_solver="full")
        pca_field = "_pca"
    elif model == "pca":
        model = PCA(n_components=n_components, svd_solver="full")
        pca_field = "_pca"
    else:
        raise ValueError(
            "Invalid model specified. Choose 'isomap', 'pca', or specify number of components for PCA."
        )

    field = f"{area}_rates"

    n_shared_trial = np.inf
    target_ids = np.unique(data_list[0][trial_cat])

    for df in data_list:
        for target in target_ids:
            df_ = pyal.select_trials(df, df[trial_cat] == target)
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))
    n_shared_trial = int(n_shared_trial)

    # finding the number of timepoints
    # print(len(data_list))
    if epoch is not None:
        # print(data_list[0]["MOp_rates"][0].shape)
        df_ = pyal.restrict_to_interval(data_list[0], epoch_fun=epoch)
    n_timepoints = int(df_[field][0].shape[0])

    # pre-allocating the data matrix
    AllData = np.empty(
        (len(data_list), len(target_ids), n_shared_trial, n_timepoints, model.n_components)
    )

    rng = np.random.default_rng(12345)
    for session, df in enumerate(data_list):
        df_ = pyal.restrict_to_interval(df, epoch_fun=epoch) if epoch is not None else df
        if f"{area}_pca" not in df_.columns:
            rates = np.concatenate(df_[field].values, axis=0)
            if units is not None:
                rates = rates[:, units[0] : units[1]]
            rates_model = model.fit(rates)
            df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, pca_field)
        else:
            pca_field = f"{area}_pca"
        for targetIdx, target in enumerate(target_ids):
            df__ = pyal.select_trials(df_, df_[trial_cat] == target)
            all_id = df__.trial_id.to_numpy()
            # to guarantee shuffled ids
            while ((all_id_sh := rng.permutation(all_id)) == all_id).all():
                continue
            all_id = all_id_sh
            df__ = pyal.select_trials(
                df__, lambda trial: trial.trial_id in all_id[:n_shared_trial]
            )
            for trial, trial_rates in enumerate(df__[pca_field]):
                AllData[session, targetIdx, trial, :, :] = trial_rates

    return AllData


rng = np.random.default_rng(12345)


def add_bhv(trial_data, bhv_fields=["all"]):
    if bhv_fields[0] == "all":
        bhv_fields = [
            "calibrated_eyes_pos",
            "left_hand_pos",
            "tail_pos",
            "eye_acceleration",
            "eye_azimuth",
            "eye_direction",
            "eye_distance_traveled",
            "eye_eccentricity",
            "eye_grid_ID",
            "eye_velocity",
            "eyebrow_pos",
            "head_pos",
            "eyes_pos",
            "head_acceleration",
            "head_direction",
            "head_distance_traveled",
            "head_grid_ID",
            "head_velocity",
            "head_tilt",
            "nose_pos",
            "pupil_size",
            "right_hand_pos",
            "Motion_energy_arm",
            "Motion_energy_head",
        ]
    print(trial_data.session[0])
    bhv_list = []
    for trial in range(len(trial_data)):
        design_matrix = np.empty((trial_data["right_knee"][trial].shape[0], 0))
        for bhv in bhv_fields:
            design_matrix = np.column_stack((design_matrix, trial_data[bhv][trial]))
        bhv_list.append(design_matrix)
    trial_data["bhv"] = bhv_list
    return trial_data


def add_history(data: np.ndarray, n_hist: int) -> np.ndarray:
    """
    Adds history to the columns of `data`, by stacking `n_hist` previous time bins

    Parameters
    ----------
    `data`: the data matrix, T x n with _T_ time points and _n_ neurons/components/features.

    `n_hist` : number of time rows to be added.

    Returns
    -------
    An array of _T_  x _(n x n_hist+1)_

    """
    out = np.hstack([np.roll(data, shift, axis=0) for shift in range(n_hist + 1)])
    out[:n_hist, data.shape[1] :] = 0
    return out


def add_history_to_data_array(allData, n_hist):
    """
    applies `add_history` to each trial

    Parameters
    ----------
    `allData`: the data matrix coming from `dt.add_history`

    `n_hist` : number of time rows to be added.

    Returns
    -------
    Similar to the output of `dt.get_data_array`, with extra PC columns.
    """
    assert allData.ndim == 5, "Wrong input size"
    newShape = list(allData.shape)
    newShape[-1] *= n_hist + 1

    out = np.empty(newShape)
    for session, sessionData in enumerate(allData):
        for target, targetData in enumerate(sessionData):
            for trial, trialData in enumerate(targetData):
                out[session, target, trial, :, :] = add_history(trialData, n_hist)
    return out


def interpolate_nans(matrix):
    """
    interpolate NaN sequence of maximum 5 consecutive bins (150ms)
    """
    if matrix.ndim == 1:
        # print(matrix)
        interpolated_series = pd.Series(matrix).interpolate(limit_area="inside", limit=5)
        # print(interpolated_series.values)
        return interpolated_series.values
    else:
        interpolated_matrix = np.empty_like(matrix)
        for i in range(matrix.shape[1]):
            column = matrix[:, i]
            interpolated_series = pd.Series(column).interpolate(limit_area="inside", limit=5)
            interpolated_matrix[:, i] = interpolated_series.values
        # print(interpolated_matrix)
        return interpolated_matrix


def get_data_array_and_pos(
    data_list: list[pd.DataFrame],
    trial_cat,
    epoch,
    area: str = "PFC",
    n_components: int = 10,
    normalize_pos=False,
    model=None,
    n_neighbors=10,
    pca_bhv=False,
    bhv=["all"],
) -> np.ndarray:
    """
    Applies PCA to the data and return a data matrix of the shape: sessions x targets x  trials x time x PCs
    with the minimum number of trials and timepoints shared across all the datasets/targets.

    Parameters
    ----------
    `data_list`: list of pd.dataFrame datasets from pyal-data
    `epoch`: an epoch function of the type `pyal.generate_epoch_fun`
    `area`: area, either: 'M1', or 'S1', or 'PMd'

    Returns
    -------
    `AllData`: np.array

    Signature
    -------
    AllData = get_data_array(data_list, execution_epoch, area='M1', n_components=10)
    all_data = np.reshape(AllData, (-1,10))
    """
    if isinstance(data_list, pd.DataFrame):
        data_list = [data_list]
    if model is None:
        model = PCA(n_components=n_components, svd_solver="full")
        field_name = "_pca"
    elif isinstance(model, int):
        model = PCA(n_components=model, svd_solver="full")
        field_name = "_pca"
    elif model == "pca":
        model = PCA(n_components=n_components, svd_solver="full")
        field_name = "_pca"

    else:
        raise ValueError(
            "Invalid model specified. Choose 'isomap' or specify number of components for PCA."
        )

    def normal_mov(df: pd.DataFrame, field: str = "hTrjB") -> pd.DataFrame:
        """
        normalises based on 99th percentile for the magnitude of the movement
        """
        df = df.copy()
        magnitude = np.percentile(np.abs(np.concatenate(df[field]).flatten()), 99)
        df[field] = [pos / magnitude for pos in df[field]]
        return df

    # if trial_cat == "Target_id":
    #     target_ids = [1,2]
    # elif trial_cat == "Cue_id":
    #     target_ids = [1,2,3,4]
    # elif trial_cat == "Position_id":
    #     target_ids = [1,2,3,4,5,6,7,8]

    target_ids = np.unique(data_list[0][trial_cat])
    target_ids = target_ids[target_ids != 0]
    field = f"{area}_rates"
    n_shared_trial = np.inf
    n_targets = len(target_ids)
    pos_field = "bhv"
    for i, df in enumerate(data_list):
        df = add_bhv(df, bhv)
        df = pyal.restrict_to_interval(df, epoch_fun=epoch)
        for target in target_ids:
            df_ = pyal.select_trials(df, df[trial_cat] == target)
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))

    n_shared_trial = int(n_shared_trial)
    # print(f"n_shared_trial: {n_shared_trial}")

    # finding the number of timepoints
    df_ = pyal.restrict_to_interval(data_list[0], epoch_fun=epoch)
    n_timepoints = int(df_[field][0].shape[0])
    # if pca_bhv:
    #     n_outputs = 10
    # else:
    # n_outputs = df[bhv][0].shape[-1]
    n_outputs = df["bhv"][0].shape[-1]
    # if pca_bhv:
    #     pos_field_to_keep = "bhv_pca"
    # else:
    #     pos_field_to_keep = pos_field
    pos_field_to_keep = pos_field
    # n_shared_trial will change
    # pre-allocating the data matrix
    AllData = np.empty(
        (len(data_list), n_targets, n_shared_trial, n_timepoints, n_components)
    )
    AllVel = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, n_outputs))

    for target in target_ids:
        df_ = pyal.select_trials(df, df[trial_cat] == target)
        n_shared_trial = np.min((df_.shape[0], n_shared_trial))

    n_shared_trial = int(n_shared_trial)
    # print(f"n_shared_trial: {n_shared_trial}")

    # finding the number of timepoints
    df_ = pyal.restrict_to_interval(data_list[0], epoch_fun=epoch)
    n_timepoints = int(df_[field][0].shape[0])
    # if pca_bhv:
    #     n_outputs = 10
    # else:
    # n_outputs = df[bhv][0].shape[-1]
    n_outputs = df["bhv"][0].shape[-1]
    # if pca_bhv:
    #     pos_field_to_keep = "bhv_pca"
    # else:
    #     pos_field_to_keep = pos_field
    pos_field_to_keep = pos_field
    # n_shared_trial will change
    # pre-allocating the data matrix
    AllData = np.empty(
        (len(data_list), n_targets, n_shared_trial, n_timepoints, n_components)
    )
    AllVel = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, n_outputs))
    for session, df in enumerate(data_list):
        df_ = pyal.restrict_to_interval(df, epoch_fun=epoch)
        df_ = add_bhv(df_, bhv)

        for trial in range(len(df_)):
            df_[pos_field][trial] = interpolate_nans(df_[pos_field][trial])
        # df_= df_[~df_[pos_field].apply(contains_nan_in_matrix)]

        # df_ = df_.reset_index(drop=True)

        pos_mean = np.nanmean(pyal.concat_trials(df_, pos_field), axis=0)
        df_[pos_field] = [pos - pos_mean for pos in df_[pos_field]]
        # if normalize_pos:
        #     df_ = normal_mov(df_,pos_field)

        rates = np.concatenate(df_[field].values, axis=0)

        rates_model = model.fit(rates)
        df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, field_name)

        # if pca_bhv:
        #     bhv = np.concatenate(df_[pos_field].values, axis=0)
        #     model = PCA(n_components=10, svd_solver='full')
        #     bhv_model = model.fit(bhv)
        #     df_ = pyal.apply_dim_reduce_model(df_, bhv_model, pos_field,pos_field_to_keep )

        for targetIdx, target in enumerate(target_ids):
            df__ = pyal.select_trials(df_, df_[trial_cat] == target)
            all_id = df__.trial_id.to_numpy()
            rng.shuffle(all_id)
            # select the right number of trials to each target
            df__ = pyal.select_trials(
                df__, lambda trial: trial.trial_id in all_id[:n_shared_trial]
            )
            for trial, (trial_rates, trial_vel) in enumerate(
                zip(df__[field_name], df__[pos_field_to_keep])
            ):
                AllData[session, targetIdx, trial, :, :] = trial_rates
                AllVel[session, targetIdx, trial, :, :] = trial_vel

    return AllData, AllVel
