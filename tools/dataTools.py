
import os
import sys
sys.path.append("../")
from typing import Callable
from tools.dsp.preprocessing import preprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaldata as pyal
from sklearn.decomposition import PCA
from tools.params import Params
from tools.params import paths
from typing import List, Tuple, Optional
def get_n_time(df, trial_name, field = "MOp_rates"):
    return np.concatenate(df[df['trial_name'] == trial_name][field].values, axis=0).shape[0]

def load_sessions(session_names, data_root = paths.data_root, prep = True, only_trials = False):
    if isinstance(session_names, pd.DataFrame):
        session_names = [session_names]
    prep_dfs = []
    for session in session_names:
        print(session)
        animal = session.split('_')[0]
        data_dir = f"{data_root}/{animal}/{session}"
        for file in range(4):
            fname = os.path.join(data_dir, f"{session}_pyaldata_{file}.mat")
            if os.path.exists(fname):
                df = pyal.mat2dataframe(fname, shift_idx_fields=False)
                # concatenate the different parts of the session into one dataframe
                if file == 0:
                    full_df = df
                else:
                    full_df = pd.concat([full_df, df], ignore_index=True)
    
        prep_df = preprocess(full_df,only_trials=only_trials, repair_time_varying_fields=['MotSen1_X', 'MotSen1_Y']) if prep else full_df
        prep_dfs.append(prep_df)
    return prep_dfs

def add_no_mua_field(df, areas = ["MOp","SSp","CP","VAL"]):
    for area in areas:
        rates_field = f"{area}_rates"
        label_field = f"{area}_kslabel" if f"{area}_kslabel" in df.columns else f"{area}_KSLabel"
        out_field = f"{area}_rates_good"

        labels0 = np.asarray(df.iloc[0][label_field])
        keep_good = (labels0 == "good")
        df[out_field] = df[rates_field].apply(lambda r: r[:, keep_good])
    return df

def split_into_windows(
    trial_duration_s: float = 7 * 60,      
    window_lengths_s: Tuple[float, ...] = (1.0, 3.0, 5.0),
    gap_s: float = 6.0,
    seed: Optional[int] = None,
    start_offset_s: float = 0.0,          # optional random offset before first window
) -> List[Tuple[float, float]]:
    """
    Create a sequential schedule of windows inside a continuous free period:
      [window] then [gap] then [window] then [gap] ...
    Window lengths are randomly drawn from window_lengths_s.

    Returns:
      List of (start_time_s, end_time_s) for each window (in seconds, relative to free start).
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    if start_offset_s > 0:
        max_offset = min(start_offset_s, max(0.0, trial_duration_s - min(window_lengths_s)))
        t = float(rng.uniform(0.0, max_offset))
    windows: List[Tuple[float, float]] = []

    while True:
        L = float(rng.choice(window_lengths_s))
        if t + L > trial_duration_s:
            break
        windows.append((t, t + L))
        t = t + L + gap_s  

        if t >= trial_duration_s:
            break
    return windows
#############################
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
    if n_components is None:
        n_components = df_[field][0].shape[-1]
    AllData = np.empty(
        (len(data_list), len(target_ids), n_shared_trial, n_timepoints, n_components)
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
keypoints = [
    "shoulder_center",
    "left_shoulder",
    "left_paw",
    "right_shoulder",
    "right_elbow",
    "right_paw",
    "hip_center",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_knee",
    "right_ankle",
    "right_foot",
    "tail_base",
    "tail_middle",
    "tail_tip",
    "left_elbow",
    "left_wrist",
    "right_wrist"
]
angles = [
    "right_knee_angle",
    "left_knee_angle",
    "right_ankle_angle",
    "left_ankle_angle",
    "right_elbow_angle",
    "left_elbow_angle"
]
# 1) compute velocities for any list of (T×D) fields in your df
def add_velocity_fields(df, fields = None, dt=Params.BIN_SIZE):
    """
    For each field in `fields`, compute its frame‐to‐frame velocity and add a new column
    `<field>_vel` to the DataFrame.

    - If df[f] is (T,3): velocity[t] = sqrt(sum((pos[t] - pos[t-1])^2)) / dt
    - If df[f] is (T,)  : velocity[t] = abs(angle[t] - angle[t-1]) / dt
    The first velocity is set equal to the second, so length remains T.
    """
    if fields is None:
        fields = keypoints + angles
    df = df.copy()
    for f in fields:
        vel_list = []
        for arr in df[f].values:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] == 3:
            
                dpos = np.diff(arr, axis=0) / dt             
                mag = np.sqrt((dpos**2).sum(axis=1))           
            else:
                
                flat = arr.ravel()
                mag = np.abs(np.diff(flat)) / dt            

        
            if mag.size > 0:
                first = mag[0]
                vel = np.concatenate([[first], mag])         
            else:
                # degenerate case (T<2) just zeros
                vel = np.zeros_like(arr.ravel())

            vel_list.append(vel)
        df[f + "_vel"] = vel_list
    return df

def add_bhv(trial_data, bhv_fields = ["all"], remove_baseline = False):
    if isinstance(bhv_fields, str):
        bhv_fields = [bhv_fields]
    if bhv_fields[0] == "all_angles":
        bhv_fields = [
                    "right_knee_angle",
                    "left_knee_angle",
                    "right_ankle_angle",
                    "left_ankle_angle",
                    "right_elbow_angle",
                    "left_elbow_angle"]
                    
    elif bhv_fields[0] == "all_keypoints":
        bhv_fields = [
                 "shoulder_center",
                "left_shoulder",
                "left_paw",
                "right_shoulder",
                "right_elbow",
                "right_paw",
                "hip_center",
                "left_knee",
                "left_ankle",
                "left_foot",
                "right_knee",
                "right_ankle",
                "right_foot",
                "tail_base",
                "tail_middle",
                "tail_tip",
                "left_elbow",
                "left_wrist",
                "right_wrist",
                    ]
    elif bhv_fields[0] == "all_keypoints_vel":
        bhv_fields = [
                 "shoulder_center_vel",
                "left_shoulder_vel",
                "left_paw_vel",
                "right_shoulder_vel",
                "right_elbow_vel",
                "right_paw_vel",
                "hip_center_vel",
                "left_knee_vel",
                "left_ankle_vel",
                "left_foot_vel",
                "right_knee_vel",
                "right_ankle_vel",
                "right_foot_vel",
                "tail_base_vel",
                "tail_middle_vel",
                "tail_tip_vel",
                "left_elbow_vel",
                "left_wrist_vel",
                "right_wrist_vel",
                    ]
    elif bhv_fields[0] == "all_angles_vel":
        bhv_fields = [
                    "right_knee_angle_vel",
                    "left_knee_angle_vel",
                    "right_ankle_angle_vel",
                    "left_ankle_angle_vel",
                    "right_elbow_angle_vel",
                    "left_elbow_angle_vel"]
    elif bhv_fields[0] == "all":
        bhv_fields = [
                    "shoulder_center",
                    "left_shoulder",
                    "left_paw",
                    "right_shoulder",
                    "right_elbow",
                    "right_paw",
                    "hip_center",
                    "left_knee",
                    "left_ankle",
                    "left_foot",
                    "right_knee",
                    "right_ankle",
                    "right_foot",
                    "tail_base",
                    "tail_middle",
                    "tail_tip",
                    "left_elbow",
                    "left_wrist",
                    "right_wrist",
                    "right_knee_angle",
                    "left_knee_angle",
                    "right_ankle_angle",
                    "left_ankle_angle",
                    "right_elbow_angle",
                    "left_elbow_angle",
                     "shoulder_center_vel",
                "left_shoulder_vel",
                "left_paw_vel",
                "right_shoulder_vel",
                "right_elbow_vel",
                "right_paw_vel",
                "hip_center_vel",
                "left_knee_vel",
                "left_ankle_vel",
                "left_foot_vel",
                "right_knee_vel",
                "right_ankle_vel",
                "right_foot_vel",
                "tail_base_vel",
                "tail_middle_vel",
                "tail_tip_vel",
                "left_elbow_vel",
                "left_wrist_vel",
                "right_wrist_vel",
                "right_knee_angle_vel",
                    "left_knee_angle_vel",
                    "right_ankle_angle_vel",
                    "left_ankle_angle_vel",
                    "right_elbow_angle_vel",
                    "left_elbow_angle_vel"
                ]
    elif bhv_fields[0] == "all_no_vel":
        bhv_fields = [
                    "shoulder_center",
                    "left_shoulder",
                    "left_paw",
                    "right_shoulder",
                    "right_elbow",
                    "right_paw",
                    "hip_center",
                    "left_knee",
                    "left_ankle",
                    "left_foot",
                    "right_knee",
                    "right_ankle",
                    "right_foot",
                    "tail_base",
                    "tail_middle",
                    "tail_tip",
                    "left_elbow",
                    "left_wrist",
                    "right_wrist",
                    "right_knee_angle",
                    "left_knee_angle",
                    "right_ankle_angle",
                    "left_ankle_angle",
                    "right_elbow_angle",
                    "left_elbow_angle",
                #      "shoulder_center_vel",
                # "left_shoulder_vel",
                # "left_paw_vel",
                # "right_shoulder_vel",
                # "right_elbow_vel",
                # "right_paw_vel",
                # "hip_center_vel",
                # "left_knee_vel",
                # "left_ankle_vel",
                # "left_foot_vel",
                # "right_knee_vel",
                # "right_ankle_vel",
                # "right_foot_vel",
                # "tail_base_vel",
                # "tail_middle_vel",
                # "tail_tip_vel",
                # "left_elbow_vel",
                # "left_wrist_vel",
                # "right_wrist_vel",
                # "right_knee_angle_vel",
                #     "left_knee_angle_vel",
                #     "right_ankle_angle_vel",
                #     "left_ankle_angle_vel",
                #     "right_elbow_angle_vel",
                #     "left_elbow_angle_vel"
                ]
    bhv_list = []
    for trial in range(len(trial_data)):
        design_matrix = np.empty((trial_data["right_knee"][trial].shape[0],0))
        # also return bhv names, and add x,y,x depending on how many columns each field has
        for bhv in bhv_fields:
            if bhv not in trial_data.columns and "vel" in bhv:
               trial_data = add_velocity_fields(trial_data,fields = [bhv.replace("_vel","")]) 
            bhv_trial = trial_data[bhv][trial]
            if remove_baseline and "vel" not in bhv and "angle" not in bhv:
                baseline = np.mean(bhv_trial[trial_data["idx_sol_on"][trial]-int(1/Params.BIN_SIZE):trial_data["idx_sol_on"][trial],:],axis = 0)
                # print(baseline)
                bhv_trial = bhv_trial-baseline
            design_matrix = np.column_stack((design_matrix,bhv_trial))
        bhv_list.append(design_matrix)
    trial_data["bhv"] = bhv_list
    # keep bhv names but if it is in keypoints add x, y z
    bhv_names = []
    for bhv in bhv_fields:
        if bhv in keypoints:
            bhv_names.extend([f"{bhv}_x",f"{bhv}_y",f"{bhv}_z"])
        else:
            bhv_names.append(bhv)

    return trial_data, bhv_names

def add_history(data:np.ndarray, n_hist:int) -> np.ndarray:
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
    out = np.hstack([np.roll(data, shift, axis=0) for shift in range(n_hist+1)])
    out[:n_hist,data.shape[1]:] = 0
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
    assert allData.ndim == 5, 'Wrong input size'
    newShape = list(allData.shape)
    newShape[-1] *= (n_hist+1)
    
    out = np.empty(newShape)
    for session,sessionData in enumerate(allData):
        for target,targetData in enumerate(sessionData):
            for trial,trialData in enumerate(targetData):
                out[session,target,trial,:,:] = add_history(trialData, n_hist)
    return out

def interpolate_nans(matrix):
    """
    interpolate NaN sequence of maximum 5 consecutive bins (150ms)
    """
    if matrix.ndim == 1:
        # print(matrix)
        interpolated_series = pd.Series(matrix).interpolate(limit_area='inside', limit=5)
        # print(interpolated_series.values)
        return interpolated_series.values
    else:
        interpolated_matrix = np.empty_like(matrix)
        for i in range(matrix.shape[1]):
            column = matrix[:, i]
            interpolated_series = pd.Series(column).interpolate(limit_area='inside', limit=5)
            interpolated_matrix[:, i] = interpolated_series.values
        # print(interpolated_matrix)
        return interpolated_matrix
    
def normal_mov(df: pd.DataFrame, field:str ='hTrjB') -> pd.DataFrame:
        """
        normalises based on 99th percentile for the magnitude of the movement
        """
        df = df.copy()
        magnitude = np.percentile(np.abs(np.concatenate(df[field]).flatten()), 99)
        df[field] = [pos/magnitude for pos in df[field]]
        return df

def get_data_array_and_pos_old(data_list: list[pd.DataFrame], trial_cat,epoch , area: str ='PFC', n_components: int =10, normalize_pos = False,model = None, n_neighbors = 10,pca_bhv = False, bhv = ["all"]) -> np.ndarray:
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
        model = PCA(n_components=n_components, svd_solver='full')
        field_name = "_pca"
    elif isinstance(model, int):
        model = PCA(n_components=model, svd_solver='full')
        field_name = "_pca"
    elif model == 'pca':
        model = PCA(n_components=n_components,svd_solver='full')
        field_name = "_pca"

    else:
        raise ValueError("Invalid model specified. Choose 'isomap' or specify number of components for PCA.")
    
    
    
    # if trial_cat == "Target_id":
    #     target_ids = [1,2]
    # elif trial_cat == "Cue_id":
    #     target_ids = [1,2,3,4]
    # elif trial_cat == "Position_id":
    #     target_ids = [1,2,3,4,5,6,7,8]
    # elif trial_cat == "Sol_direction":


    target_ids = np.unique(data_list[0][trial_cat])
    target_ids = target_ids[target_ids!=0]
    field = f'{area}_rates'
    n_shared_trial = np.inf
    n_targets = len(target_ids)
    pos_field = "bhv"
    for i,df in enumerate(data_list):
        df = add_bhv(df,bhv)
        df = pyal.restrict_to_interval(df,epoch_fun=epoch)
        for target in target_ids:
            df_ = pyal.select_trials(df, df[trial_cat] == target)
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))

    n_shared_trial = int(n_shared_trial)
    # print(f"n_shared_trial: {n_shared_trial}")

    # finding the number of timepoints
    df_ = pyal.restrict_to_interval(data_list[0],epoch_fun=epoch)
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
    AllData = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, n_components))
    AllVel  = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, n_outputs))

    for target in target_ids:
            df_ = pyal.select_trials(df, df[trial_cat] == target)
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))

    n_shared_trial = int(n_shared_trial)
    # print(f"n_shared_trial: {n_shared_trial}")

    # finding the number of timepoints
    df_ = pyal.restrict_to_interval(data_list[0],epoch_fun=epoch)
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
    AllData = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, n_components))
    AllVel  = np.empty((len(data_list), n_targets, n_shared_trial, n_timepoints, n_outputs))
    for session, df in enumerate(data_list):
        df_ = pyal.restrict_to_interval(df, epoch_fun=epoch)
        df_ = add_bhv(df_,bhv)

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
            
        for targetIdx,target in enumerate(target_ids):
            df__ = pyal.select_trials(df_, df_[trial_cat]==target)
            all_id = df__.trial_id.to_numpy()
            rng.shuffle(all_id)
            # select the right number of trials to each target
            df__ = pyal.select_trials(df__, lambda trial: trial.trial_id in all_id[:n_shared_trial])
            for trial, (trial_rates,trial_vel) in enumerate(zip(df__[field_name], df__[pos_field_to_keep])):
                AllData[session,targetIdx,trial, :, :] = trial_rates
                AllVel[session,targetIdx,trial, :, :] = trial_vel

    return AllData, AllVel


def normal_mov_per_channel(df: pd.DataFrame, field: str = 'bhv') -> pd.DataFrame:
    """
    For each column in the T×D arrays of df[field], 
    divides that column by its own 99th percentile magnitude.
    """
    df = df.copy()
    # stack into (total_time × D)
    all_data = np.concatenate(df[field].values, axis=0)  # shape (N, D)
    # compute per-column 99th percentile of abs values
    p99 = np.percentile(np.abs(all_data), 99, axis=0)    # shape (D,)
    # avoid divide-by-zero
    p99[p99 == 0] = 1.0
    # then for each trial, divide each column by its p99
    df[field] = [trial / p99 for trial in df[field]]
    return df
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pyaldata as pyal




def get_data_array_and_pos_(data_list: list[pd.DataFrame], trial_cat,epoch , area: str ='MOp', n_components: int =10, normalize_pos = False,model = None, n_neighbors = 10,pca_bhv = False, bhv = ["all"],bhv_components = 1, shuffle_id = True) -> np.ndarray:
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
    for df in data_list:
        if n_components>df[f"{area}_rates"][0].shape[-1]:
            n_components = None
            print("No. PCs larger than number of no. neurons.")
            break
    if isinstance(data_list, pd.DataFrame):
        data_list = [data_list]
    if model is None:
        model = PCA(n_components=n_components, svd_solver='full')
        field_name = "_pca"
    elif isinstance(model, int):
        model = PCA(n_components=model, svd_solver='full')
        field_name = "_pca"
    elif model == 'pca':
        model = PCA(n_components=n_components,svd_solver='full')
        field_name = "_pca"

    else:
        raise ValueError("Invalid model specified. Choose 'isomap' or specify number of components for PCA.")
    

    trials = data_list[0]["trial_name"][0] =="trial"
    if trials:
        target_ids = np.unique(data_list[0][trial_cat])
    else:
        epoch = None
        target_ids = [0]
    field = f'{area}_rates'
    n_shared_trial = np.inf
    n_shared_timepoints = np.inf
    n_targets = len(target_ids)
    pos_field = "bhv"
    for i,df in enumerate(data_list):
        df = add_bhv(df,bhv)
        df = pyal.restrict_to_interval(df,epoch_fun=epoch) if epoch is not None else df
        for target in target_ids:
            if trials:
                df_ = pyal.select_trials(df, df[trial_cat] == target)
            else:
                df_ = df
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))
            n_shared_timepoints= np.min((df_[field][0].shape[0],n_shared_timepoints))

    n_shared_trial = int(n_shared_trial)
    n_shared_timepoints = int(n_shared_timepoints)
    # print(f"n_shared_trial: {n_shared_trial}")

 
    if pca_bhv:
        n_outputs = bhv_components
        pos_field_to_keep = "bhv_pca"
    else:
        n_outputs = df["bhv"][0].shape[-1]
        pos_field_to_keep = pos_field
    # n_shared_trial will change
    # pre-allocating the data matrix
    if n_components is None:
        n_components = df_[field][0].shape[-1]
    AllData = np.empty((len(data_list), n_targets, n_shared_trial,n_shared_timepoints, n_components))
    AllVel  = np.empty((len(data_list), n_targets, n_shared_trial, n_shared_timepoints, n_outputs))


    for session, df in enumerate(data_list):
        df_ = pyal.restrict_to_interval(df, epoch_fun=epoch) if epoch is not None else df
        
        df_ = add_bhv(df_,bhv)
        ########
        # for trial in range(len(df_)):
        #     df_[pos_field][trial] = interpolate_nans(df_[pos_field][trial])  
        #########
        # df_= df_[~df_[pos_field].apply(contains_nan_in_matrix)]
        
        # df_ = df_.reset_index(drop=True)
        # if "vel" not in bhv[0]:
        #     pos_mean = np.nanmean(pyal.concat_trials(df_, pos_field), axis=0)
        #     df_[pos_field] = [pos - pos_mean for pos in df_[pos_field]] 
        # if normalize_pos:
        #     df_ = normal_mov(df_,pos_field)
        # print("Check update")
        rates = np.concatenate(df_[field].values, axis=0)
        
        rates_model = model.fit(rates)
        df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, field_name)
       
        if pca_bhv:
            bhv = np.concatenate(df_[pos_field].values, axis=0)
            model = PCA(n_components=bhv_components, svd_solver='full')
            bhv_model = model.fit(bhv)
            df_ = pyal.apply_dim_reduce_model(df_, bhv_model, pos_field,pos_field_to_keep )
            
        for targetIdx,target in enumerate(target_ids):
            if trials:
                df__ = pyal.select_trials(df_, df_[trial_cat]==target)
            else:
                df__ = df_
            all_id = df__.trial_id.to_numpy()
            if shuffle_id:
                rng.shuffle(all_id)
            # select the right number of trials to each target
            df__ = pyal.select_trials(df__, lambda trial: trial.trial_id in all_id[:n_shared_trial])
            for trial, (trial_rates,trial_vel) in enumerate(zip(df__[field_name], df__[pos_field_to_keep])):
                AllData[session,targetIdx,trial, :, :] = trial_rates[:n_shared_timepoints]
                AllVel[session,targetIdx,trial, :, :] = trial_vel[:n_shared_timepoints]

    return AllData, AllVel

def get_data_array_and_pos(data_list: list[pd.DataFrame], trial_cat,epoch , area: str ='MOp', n_components: int =10, normalize_pos = False,model = None, n_neighbors = 10,pca_bhv = False, bhv = ["all"],bhv_components = 1, shuffle_id = True, remove_baseline = False, trial_type = "trial") -> np.ndarray:
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
    for df in data_list:
        if n_components>df[f"{area}_rates"][0].shape[-1]:
            n_components = None
            print("No. PCs larger than number of no. neurons.")
            break
    if isinstance(data_list, pd.DataFrame):
        data_list = [data_list]
    # if model is None:
    #     model = PCA(n_components=n_components, svd_solver='full')
    #     field_name = "_pca"
    elif isinstance(model, int):
        model = PCA(n_components=model, svd_solver='full')
        field_name = "_pca"
    elif model == 'pca':
        model = PCA(n_components=n_components,svd_solver='full')
        field_name = "_pca"

    # else:
    #     raise ValueError("Invalid model specified. Choose 'isomap' or specify number of components for PCA.")
    

    if model is None:
        n_components = data_list[0][f"{area}_rates"][0].shape[-1]

    if trial_type != "trial":
        epoch = None
        target_ids = [0]
    field = f'{area}_rates'
    n_shared_trial = np.inf
    n_max_timepoints = 0 
    # n_min_timepoints = np.inf
    
    pos_field = "bhv"
    for i,df in enumerate(data_list):
        df = pyal.select_trials(df, df.trial_name == trial_type)
        target_ids = np.unique(df[trial_cat]) if trial_type == "trial" else [0]
        df = add_bhv(df,bhv)
        df = pyal.restrict_to_interval(df,epoch_fun=epoch) if epoch is not None else df
        for target in target_ids:
            if trial_type == "trial":
                df_ = pyal.select_trials(df, df[trial_cat] == target)
                n_max_timepoints= np.max((df_[field][0].shape[0],n_max_timepoints))
                # n_min_timepoints= np.min((df_[field][0].shape[0],n_min_timepoints))
            else:
                df_ = df
                for trial in range(len(df_)):
                    n_max_timepoints= np.max((df_[field][trial].shape[0],n_max_timepoints))
                    # n_min_timepoints= np.min((df_[field][0].shape[0],n_min_timepoints))
            n_shared_trial = np.min((df_.shape[0], n_shared_trial))
            

    n_shared_trial = int(n_shared_trial)
    n_max_timepoints = int(n_max_timepoints)
    # print(f"n_shared_trial: {n_shared_trial}")
    # print(n_max_timepoints)

 
    if pca_bhv:
        n_outputs = bhv_components
        pos_field_to_keep = "bhv_pca"
    else:
        n_outputs = df["bhv"][0].shape[-1]
        pos_field_to_keep = pos_field
    # n_shared_trial will change
    # pre-allocating the data matrix
    if n_components is None:
        n_components = df_[field][0].shape[-1]
    n_targets = len(target_ids)
    AllData = np.empty((len(data_list), n_targets, n_shared_trial,n_max_timepoints, n_components))
    AllVel  = np.empty((len(data_list), n_targets, n_shared_trial, n_max_timepoints, n_outputs))
    AllData[:] = np.nan
    AllVel[:] = np.nan
    print(AllData.shape)


    for session, df in enumerate(data_list):
        df_ = pyal.select_trials(df, df.trial_name == trial_type)
        target_ids = np.unique(df_[trial_cat]) if trial_type == "trial" else [0]
        df_ = pyal.restrict_to_interval(df_, epoch_fun=epoch) if epoch is not None else df_
        
        df_ = add_bhv(df_,bhv, remove_baseline = remove_baseline)
        ########
        # for trial in range(len(df_)):
        #     df_[pos_field][trial] = interpolate_nans(df_[pos_field][trial])  
        #########
        # df_= df_[~df_[pos_field].apply(contains_nan_in_matrix)]
        
        # df_ = df_.reset_index(drop=True)
        # if "vel" not in bhv[0]:
        #     pos_mean = np.nanmean(pyal.concat_trials(df_, pos_field), axis=0)
        #     df_[pos_field] = [pos - pos_mean for pos in df_[pos_field]] 
        # if normalize_pos:
        #     df_ = normal_mov(df_,pos_field)
        # print("Check update")
        if model is not None:
            rates = np.concatenate(df_[field].values, axis=0)
            
            rates_model = model.fit(rates)
            df_ = pyal.apply_dim_reduce_model(df_, rates_model, field, field_name) 
        else:
            field_name = field
        if pca_bhv:
            bhv = np.concatenate(df_[pos_field].values, axis=0)
            model = PCA(n_components=bhv_components, svd_solver='full')
            bhv_model = model.fit(bhv)
            df_ = pyal.apply_dim_reduce_model(df_, bhv_model, pos_field,pos_field_to_keep )
        
        for targetIdx,target in enumerate(target_ids):
            print(target)
            if trial_type == "trial":
                df__ = pyal.select_trials(df_, df_[trial_cat]==target)
            else:
                df__ = df_
            all_id = df__.trial_id.to_numpy()
            if shuffle_id:
                rng.shuffle(all_id)
            # select the right number of trials to each target
            df__ = pyal.select_trials(df__, lambda trial: trial.trial_id in all_id[:n_shared_trial])
            for trial, (trial_rates,trial_vel) in enumerate(zip(df__[field_name], df__[pos_field_to_keep])):
                # print(trial_rates.shape,AllData[session,targetIdx,trial, :trial_rates.shape[0], :].shape)
                AllData[session,targetIdx,trial, :trial_rates.shape[0], :] = trial_rates
                AllVel[session,targetIdx,trial, :trial_vel.shape[0], :] = trial_vel

    return AllData, AllVel


import numpy as np
import pandas as pd
import torch
from typing import Any, Callable, Optional, Sequence, Tuple


def _as_time_by_features_np(arr: Any) -> Optional[np.ndarray]:
    if arr is None:
        return None
    A = np.asarray(arr)
    if A.size == 0:
        return None
    if A.ndim == 1:
        return A.astype(float)[:, None]
    if A.ndim == 2:
        return A.astype(float)
    return A.reshape(A.shape[0], -1).astype(float)


def add_trial_metric_field(
    prep_df: pd.DataFrame,
    input_field: str,
    metric_fn: Callable[[np.ndarray], Any],
    new_field: str,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Compute one metric per row/trial from input_field and add it as new_field.

    metric_fn receives a (T, F) numpy array and should return scalar/array-like.
    """
    out_df = prep_df if inplace else prep_df.copy()
    values = []

    for _, row in out_df.iterrows():
        A = _as_time_by_features_np(row.get(input_field, None))
        if A is None:
            values.append(None)
            continue
        v = metric_fn(A)
        if v is None:
            values.append(None)
            continue
        v_arr = np.asarray(v)
        if v_arr.size == 0:
            values.append(None)
        elif v_arr.ndim == 0:
            values.append(float(v_arr))
        else:
            values.append(v_arr.reshape(-1).astype(float))

    out_df[new_field] = values
    return out_df


def build_window_tensor_sliding_session(
    prep_df: pd.DataFrame,
    window_size_s: float,
    field: str,
    step_size_s: Optional[float] = None,
    overlap_pct: Optional[float] = None,
    bin_size_col: str = "bin_size",
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    Concatenate all rows in prep_df in row order and build overlapping/sliding windows.

    step_size_s and overlap_pct are both supported. If both are provided, step_size_s is used.
    overlap_pct is a fraction in [0, 1), where 0.5 means 50% overlap.

    Returns:
      meta_df: one row per window (session-level sample indices)
      X: (n_windows, w, F)
    """
    if len(prep_df) == 0:
        return pd.DataFrame(), torch.empty((0, 0, 0), dtype=torch.float32)

    dt0 = float(prep_df[bin_size_col].iloc[0])

    w = int(round(window_size_s / dt0))
    if w <= 0:
        raise ValueError("window_size_s is too small relative to bin_size.")

    if step_size_s is not None:
        step = int(round(step_size_s / dt0))
        overlap_pct_used = 1.0 - (step / w)
    else:
        if overlap_pct is None:
            raise ValueError("Provide either step_size_s or overlap_pct.")
        if not (0 <= overlap_pct < 1):
            raise ValueError("overlap_pct must be in [0, 1).")
        step = int(round(w * (1.0 - overlap_pct)))
        overlap_pct_used = overlap_pct

    if step <= 0:
        raise ValueError("Derived step size is <= 0. Use lower overlap_pct or larger step_size_s.")

    As = []
    F_ref = None
    for _, row in prep_df.iterrows():
        A = _as_time_by_features_np(row.get(field, None))
        if A is None:
            continue
        if F_ref is None:
            F_ref = A.shape[1]
        elif A.shape[1] != F_ref:
            raise ValueError(f"Feature dim mismatch: expected {F_ref}, got {A.shape[1]}.")
        As.append(A)

    if not As:
        return pd.DataFrame(), torch.empty((0, w, 0), dtype=torch.float32)

    A_cat = np.concatenate(As, axis=0)
    T, F = A_cat.shape
    if T < w:
        return pd.DataFrame(), torch.empty((0, w, F), dtype=torch.float32)

    starts = np.arange(0, T - w + 1, step, dtype=int)
    Xw = np.stack([A_cat[s:s + w, :] for s in starts], axis=0)

    meta_df = pd.DataFrame({
        "window_index": np.arange(len(starts), dtype=int),
        "start_frame": starts,
        "end_frame": starts + w - 1,
        "bin_size": dt0,
        "window_size_samples": w,
        "step_size_samples": step,
        "window_size_s": w * dt0,
        "step_size_s": step * dt0,
        "overlap_pct": overlap_pct_used,
    })
    X = torch.from_numpy(Xw).to(dtype=torch.float32)
    return meta_df, X




def build_window_tensor_concat_by_trial_type(
    prep_df: pd.DataFrame,
    window_size_s: float,
    field: str,
    trial_type_col: str = "trial_name",
    bin_size_col: str = "bin_size",
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    Concatenate all rows within each trial type, window within that type only (no crossing types).

    Returns:
      meta_df: one row per window, only column is 'trial_type'
      X: (nW_total, w, F)
    """
    if len(prep_df) == 0:
        return pd.DataFrame(columns=["trial_type"]), torch.empty((0, 0, 0), dtype=torch.float32)

    dt0 = float(prep_df[bin_size_col].iloc[0])
    if not np.allclose(prep_df[bin_size_col].astype(float).to_numpy(), dt0):
        raise ValueError("bin_size varies across rows; enforce constant bin_size or adapt w per row.")

    w = int(round(window_size_s / dt0))
    if w <= 0:
        raise ValueError("window_size_s is too small relative to bin_size.")

    # Preserve order of first appearance of types
    types_in_order = list(pd.unique(prep_df[trial_type_col]))

    meta_parts = []
    X_parts = []
    F_global = None

    for t in types_in_order:
        sub = prep_df[prep_df[trial_type_col] == t]

        As = []
        F_ref = None
        for _, row in sub.iterrows():
            A = _as_time_by_features_np(row.get(field, None))
            if A is None:
                continue
            if F_ref is None:
                F_ref = A.shape[1]
            elif A.shape[1] != F_ref:
                raise ValueError(f"Feature dim mismatch within type '{t}': expected {F_ref}, got {A.shape[1]}.")
            As.append(A)

        if not As:
            continue

        A_cat = np.concatenate(As, axis=0)  # (Tcat, F)
        Tcat, F = A_cat.shape
        nW = Tcat // w
        if nW <= 0:
            continue

        A_cat = A_cat[: nW * w, :]
        Xw = A_cat.reshape(nW, w, F)
        Xt = torch.from_numpy(Xw).to(dtype=torch.float32)

        if F_global is None:
            F_global = F
        elif F != F_global:
            raise ValueError(f"Feature dim mismatch across types: expected {F_global}, got {F} for type '{t}'.")

        X_parts.append(Xt)
        meta_parts.append(pd.DataFrame({"trial_type": [t] * nW}))

    meta_df = pd.concat(meta_parts, ignore_index=True) if meta_parts else pd.DataFrame(columns=["trial_type"])
    if X_parts:
        X = torch.cat(X_parts, dim=0)
    else:
        # If nothing yielded windows, keep shape consistent with requested w
        X = torch.empty((0, w, 0), dtype=torch.float32)

    return meta_df, X

import numpy as np
import pandas as pd
from typing import Sequence, Optional, List
def trial_change_boundaries(meta_df: pd.DataFrame, col: str = "trial_name") -> List[int]:
    """
    Returns boundary indices k such that a change occurs between k-1 and k.
    These are suitable for drawing a line at x = k-0.5 on an imshow heatmap.
    """
    
    names = meta_df[col].astype(str).to_numpy()
    if names.size == 0:
        return []
    change = names[1:] != names[:-1]
    # boundary index is the index of the first element of the new block
    return (np.where(change)[0] + 1).astype(int).tolist()