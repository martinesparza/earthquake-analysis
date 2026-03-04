from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaldata as pyal
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

from tools.dimensionality.pca import compute_pca


def shift_time_no_wrap(arr, shift, fill_value=np.nan):
    """
    Shift along time axis=1 without wrap-around.
    Positive shift moves data to later times (right).
    Negative shift moves data to earlier times (left).
    """
    arr = np.asarray(arr)
    out = np.full_like(
        arr,
        fill_value=fill_value,
        dtype=float if np.isnan(fill_value) else arr.dtype,
    )

    if shift == 0:
        return arr.copy()

    T = arr.shape[1]

    if shift > 0:
        # t -> t+shift
        out[:, shift:T, :] = arr[:, : T - shift, :]
    else:
        s = -shift
        # t -> t-s
        out[:, : T - s, :] = arr[:, s:T, :]

    return out


def add_concat_trial_start(td: pd.DataFrame):
    td = td.copy()
    td["concat_trial_start"] = pd.Series([None] * len(td), dtype="object")

    for idx, row in td.iterrows():
        if row.trial_name != "trial":
            continue
        td.at[idx, "concat_trial_start"] = td.iloc[idx - 1].trial_length
    return td


def add_concat_perturb_time(td: pd.DataFrame):
    td = td.copy()
    td["concat_perturb_time"] = pd.Series([None] * len(td), dtype="object")

    for idx, row in td.iterrows():
        if row.trial_name != "trial":
            continue
        td.at[idx, "concat_perturb_time"] = td.iloc[idx - 1].trial_length + row.idx_sol_on
    return td


def concat_previous_intertrial_signal(td, signal, features: np.ndarray | None = None):
    if features is None:
        features = np.arange(td[signal].values[0].shape[-1])

    td = td.copy()
    new_signal = signal + "_concat"
    td[new_signal] = pd.Series([None] * len(td), dtype="object")

    for pos in range(len(td)):
        row = td.iloc[pos]
        if row.trial_name != "trial":
            continue
        prev_tf = td.iloc[pos - 1][signal][:, features]  # (T, n_features)
        curr_tf = td.iloc[pos][signal][:, features]  # (T, n_features)

        td.at[td.index[pos], new_signal] = np.concatenate([prev_tf, curr_tf], axis=0)

    return td


def _get_min_number_shared_trials(df, label_field):
    return df[label_field].value_counts().min()


def balance_classes(df, label_field, random_state=42):
    min_trials = _get_min_number_shared_trials(df, label_field)
    # print(f"Min g. trials per condition: {min_trials}")
    labels = np.unique(df[label_field].values)
    subsets = []
    for label in labels:
        subset = df[df[label_field] == label].sample(n=min_trials, random_state=random_state)
        subsets.append(subset)

    # Concatenate and shuffle the result
    balanced_df = (
        pd.concat(subsets).sample(frac=1, random_state=random_state).reset_index(drop=True)
    )
    return balanced_df


def remove_trials_wo_motion_before_event(df, motion_field, event_field, verbose=True):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if motion_field not in df.columns or event_field not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{motion_field}' and '{event_field}' columns."
        )
    event_onset = df[event_field].iloc[0]
    initial_count = len(df)

    filtered_df = df[df[motion_field].apply(lambda x: np.any(np.array(x) < event_onset))]

    dropped_count = initial_count - len(filtered_df)
    if verbose:
        print(
            f"Dropped {dropped_count} of {initial_count} rows ({dropped_count/initial_count:.2%})."
        )
    return filtered_df.reset_index(drop=True)


def _smooth_data(arr, sigma=1):
    norm_arr = gaussian_filter1d(arr, sigma=sigma, axis=0)
    return norm_arr


def _find_number_timepoints(df, field, epoch):
    if epoch is not None:
        df = pyal.restrict_to_interval(df, epoch_fun=epoch)
    n_timepoints = int(df[field][0].shape[0])
    return n_timepoints


def _find_number_shared_trial(data_list, target_ids, trial_cat, epoch):
    n_shared_trial = np.inf
    for df in data_list:
        if epoch is not None:
            df = pyal.restrict_to_interval(df, epoch_fun=epoch)
        for target in target_ids:
            df_ = pyal.select_trials(df, df[trial_cat] == target)
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))
    n_shared_trial = int(n_shared_trial)
    return n_shared_trial


def _get_bhv_dims(df: pd.DataFrame, bhv: list):
    """Get the number of behavioural dimensions in a dataframe

    Args:
        df (pd.DataFrame): data
        bhv (list): List of keypoints / angles

    Returns:
        int: number of dimensions
    """
    return np.column_stack(df[bhv].values[0]).shape[1]


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
    area: str = "MOp",
    units: Callable = None,
    model: Callable = "pca",
    n_components: int = 10,
    bhv: None | list = None,
    norm_bhv: bool = True,
    sigma: float = 1.0,
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

    # Definitions
    field = f"{area}_rates"
    target_ids = np.unique(data_list[0][trial_cat])
    n_shared_trial = _find_number_shared_trial(data_list, target_ids, trial_cat, epoch)
    n_timepoints = _find_number_timepoints(data_list[0], epoch=epoch, field=field)

    # pre-allocating the data matrix
    AllData = np.empty(
        (len(data_list), len(target_ids), n_shared_trial, n_timepoints, model.n_components)
    )
    if bhv is not None:
        bhv_dims = _get_bhv_dims(data_list[0], bhv)
        AllBhv = np.empty(
            (len(data_list), len(target_ids), n_shared_trial, n_timepoints, bhv_dims)
        )

    # Begin processing sessions
    rng = np.random.default_rng(12345)
    for session, df in enumerate(data_list):

        if bhv is not None:
            # Add behaviour
            df = add_bhv(df, bhv)

            # Interpolate nans
            # for trial in range(len(df)):
            #     df["bhv"][trial] = interpolate_nans(df["bhv"][trial])

            # Normalise behaviour
            if norm_bhv:
                df["bhv"] = [_smooth_data(bhv_arr, sigma=sigma) for bhv_arr in df["bhv"]]

        # Restrict to interval
        df_ = pyal.restrict_to_interval(df, epoch_fun=epoch) if epoch is not None else df

        # Apply dim reduction
        if f"{area}_pca" not in df_.columns:
            rates = np.concatenate(df_[field].values, axis=0)
            if units is not None:
                rates = rates[:, units[0] : units[1]]
            rates_model = model.fit(rates)
            df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, pca_field)
        else:
            pca_field = f"{area}_pca"

        # Populate general array
        for targetIdx, target in enumerate(target_ids):
            df__ = pyal.select_trials(df_, df_[trial_cat] == target)
            all_id = df__.trial_id.to_numpy()
            rng.shuffle(all_id)  # shuffle ids

            df__ = pyal.select_trials(
                df__, lambda trial: trial.trial_id in all_id[:n_shared_trial]
            )

            # Convert lists to NumPy arrays for vectorised assignment
            trial_rates_array = np.stack(df__[pca_field].to_list(), axis=0)
            AllData[session, targetIdx, : len(trial_rates_array), :, :] = trial_rates_array

            if bhv is not None:
                trial_bhv_array = np.stack(df__["bhv"].to_list(), axis=0)
                AllBhv[session, targetIdx, : len(trial_bhv_array), :, :] = trial_bhv_array

    return AllData if bhv is None else (AllData, AllBhv)


# rng = np.random.default_rng(12345)


def add_pca_field(trial_data, signal, n_components):
    pca_model = compute_pca(np.concatenate(trial_data[signal].values), n_components)
    trial_data = pyal.apply_dim_reduce_model(trial_data, pca_model, signal, f"{signal}_pca")

    return trial_data


def add_pca_df(
    trial_data: pd.DataFrame,
    pca_fields: list | str = ["all"],
    n_components: int | None = None,
):

    if not isinstance(pca_fields, list):
        pca_fields = [pca_fields]
    if pca_fields == ["all"]:
        pca_fields = [col for col in trial_data.columns if col.endswith("_rates")]
        try:
            pca_fields.remove("all_rates")
        except:
            print("No <all> field")

    for pca_field in pca_fields:
        trial_data = add_pca_field(trial_data, pca_field, n_components)

    return trial_data


def add_bhv(trial_data, bhv_fields=["all"]):
    if bhv_fields[0] == "all":
        bhv_fields = [
            "left_ankle",
            "left_ankle_angle",
            "left_elbow",
            "left_elbow_angle",
            "left_foot",
            "left_knee",
            "left_knee_angle",
            "left_paw",
            "left_shoulder",
            "left_wrist",
            "left_wrist_angle",
            "right_ankle",
            "right_ankle_angle",
            "right_elbow",
            "right_elbow_angle",
            "right_foot",
            "right_knee",
            "right_knee_angle",
            "right_paw",
            "right_shoulder",
            "right_wrist",
            "right_wrist_angle",
            "shoulder_center",
            "tail_base",
            "tail_middle",
            "tail_tip",
        ]
    trial_data = trial_data.copy()
    bhv_list = []
    for trial in range(len(trial_data)):
        design_matrix = np.empty((trial_data["right_knee"].values[trial].shape[0], 0))
        for bhv in bhv_fields:
            design_matrix = np.column_stack((design_matrix, trial_data[bhv].values[trial]))
        bhv_list.append(design_matrix)
    trial_data.loc[:, "bhv"] = bhv_list
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
    epoch=None,
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
        if epoch is not None:
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
    rng = np.random.default_rng(12345)

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
