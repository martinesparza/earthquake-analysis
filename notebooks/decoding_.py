import os
import sys
sys.path.append("../")

import pyaldata as pyal
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tools.params import Params
from tools import dataTools as dt

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
# test 
from sklearn.metrics import accuracy_score, make_scorer, r2_score
from tools.decoding import regression as reg
# for each behavioural widnow:
##### one input matrix, one decoder, one output matrix
def build_input_matrix(data_all_areas_session,labels, delays, start_t = None, end_t = None, WINDOW_perturb = None):
    # add delayed versions of the features to the matrix
    n_targets, n_trials, n_time, total_pcs = data_all_areas_session.shape
    # print(data_all_areas_session.shape)
    start_idx = int((start_t-WINDOW_perturb[0])/Params.BIN_SIZE) if start_t is not None else 0
    end_idx = int((end_t-WINDOW_perturb[0])/Params.BIN_SIZE) if end_t is not None else n_time
    if not (0 <= start_idx < n_time) or not (0 < end_idx <= n_time) or not (start_idx < end_idx):
        raise ValueError("Requested time window is outside the epoch or invalid.")
    # n_time = int((WINDOW_perturb[1]-WINDOW_perturb[0])/Params.BIN_SIZE) +1
    indices = np.arange(start_idx, end_idx, dtype=int)
    
    valid = np.ones(indices.shape, dtype=bool)
    for d in delays:
        valid &= (indices + d >= 0) & (indices + d < n_time)
    t_idx = indices[valid]
    if t_idx.size == 0:
        raise ValueError("No valid time bins given the delays and window.")
    T = t_idx.size
    n_delays = len(delays)
    n_features = total_pcs * n_delays

    X = np.empty((n_targets, n_trials, T, n_features), dtype=data_all_areas_session.dtype)
    feature_names = []

    for g, (area, pc_area) in enumerate(labels):
        feature_data = data_all_areas_session[:, :, :, g]  # [n_targets, n_trials, n_time]
        for j, d in enumerate(delays):
            idx_del = t_idx + d
            col = feature_data[:, :, idx_del]               # [n_targets, n_trials, T]
            feat_idx = g * n_delays + j
            X[:, :, :, feat_idx] = col
            feature_names.append({
                "area": area,
                "pc": int(pc_area),
                "lag": float(d * Params.BIN_SIZE),        
                "name": f"{area}|pc{int(pc_area)}|lag{d}",
            })
    
    return X, feature_names
def get_input_output_matrix(data_all_areas_session, bhv_session, labels, delays, start_t = None, end_t = None, WINDOW_perturb = None):
    # combine the 2 functions so the invalid points are removed from both input and output matrices
    # add delayed versions of the features to the matrix
    n_targets, n_trials, n_time, total_pcs = data_all_areas_session.shape
    # print(data_all_areas_session.shape)
    start_idx = int((start_t-WINDOW_perturb[0])/Params.BIN_SIZE) if start_t is not None else 0
    end_idx = int((end_t-WINDOW_perturb[0])/Params.BIN_SIZE) if end_t is not None else n_time
    if not (0 <= start_idx < n_time) or not (0 < end_idx <= n_time) or not (start_idx < end_idx):
        raise ValueError("Requested time window is outside the epoch or invalid.")
    # n_time = int((WINDOW_perturb[1]-WINDOW_perturb[0])/Params.BIN_SIZE) +1
    indices = np.arange(start_idx, end_idx, dtype=int)
    
    valid = np.ones(indices.shape, dtype=bool)
    for d in delays:
        valid &= (indices + d >= 0) & (indices + d < n_time)
    t_idx = indices[valid]
    if t_idx.size == 0:
        raise ValueError("No valid time bins given the delays and window.")
    T = t_idx.size
    n_delays = len(delays)
    n_features = total_pcs * n_delays

    X = np.empty((n_targets, n_trials, T, n_features), dtype=data_all_areas_session.dtype)
    y = bhv_session[:,:,t_idx,:]  # [n_targets, n_trials, T, bhv_features]
    feature_names = []

    for g, (area, pc_area) in enumerate(labels):
        feature_data = data_all_areas_session[:, :, :, g]  # [n_targets, n_trials, n_time]
        for j, d in enumerate(delays):
            idx_del = t_idx + d
            col = feature_data[:, :, idx_del]               # [n_targets, n_trials, T]
            feat_idx = g * n_delays + j
            X[:, :, :, feat_idx] = col
            feature_names.append({
                "area": area,
                "pc": int(pc_area),
                "lag": float(d * Params.BIN_SIZE),        
                "name": f"{area}|pc{int(pc_area)}|lag{d}",
            })
    
    return X,y, feature_names
def get_output_matrix(bhv_session,start_t = None,end_t = None, WINDOW_perturb = None):
    output_feature_names = None
    n_targets, n_trials, n_time, bhv_features = bhv_session.shape
    # print(bhv_session.shape)
    start_idx = int((start_t-WINDOW_perturb[0])/Params.BIN_SIZE) if start_t is not None else 0
    end_idx = int((end_t-WINDOW_perturb[0])/Params.BIN_SIZE) if end_t is not None else n_time
    if not (0 <= start_idx < n_time) or not (0 < end_idx <= n_time) or not (start_idx < end_idx):
        raise ValueError("Requested time window to predict is outside the epoch or invalid.")
    bhv_trimmed = bhv_session[:,:,start_idx:end_idx,:]
    n_targets, n_trials, n_time, bhv_features = bhv_trimmed.shape
    print(f"Output matrix shape: {bhv_trimmed.shape}")
    return bhv_trimmed,output_feature_names
# def remove_nans(matrix,rates):
#     """
#     Remove NaN values from bhv data and the corresponding timepoints from the neural data
#     """
#     [~np.isnan(matrix)]  
#     nan_rows = np.isnan(matrix) if matrix.ndim ==1 else np.isnan(matrix).any(axis=1) 
#     # print(nan_rows)
#     cleaned_matrix = matrix[~nan_rows]  
#     cleaned_rates = rates[~nan_rows]  
#     return cleaned_matrix, cleaned_rates
# def score_metric(y_true,y_pred):
#     score = r2_score(y_true, y_pred, multioutput="variance_weighted")
#     return score
def shuffle_features(input_matrix, feature_names, features_to_shuffle):
    """
    Shuffle the features in the input matrix based on the provided feature names.
    """
    shuffled_matrix = input_matrix.copy()
    for feature in features_to_shuffle:
        if feature in feature_names:
            idx = feature_names.index(feature)
            np.random.shuffle(shuffled_matrix[:, idx])
    return shuffled_matrix

def test_feature_lb(input_matrix, feature_names, feature_to_test, output_matrix):
    test_matrix = shuffle_features(input_matrix, feature_names, [feature_to_test])
    _,scores = decode(test_matrix,output_matrix)
    return scores

def test_feature_hb(input_matrix, feature_names, feature_to_test,output_matrix):
    test_matrix = shuffle_features(input_matrix, feature_names, feature_names.remove(feature_to_test))
    _,scores = decode(test_matrix,output_matrix)
    return scores
def remove_mean(y):
    y_mean = np.nanmean(y, axis=0)
    y_centered = y - y_mean
    return y_centered

from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# ----------------------------
# Metrics (R2 + RMSE)
# ----------------------------
def score_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns:
      r2   : variance-weighted multioutput R^2 (as you used before)
      rmse : scalar RMSE over all samples and outputs
    """
    r2 = r2_score(y_true, y_pred, multioutput="variance_weighted")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": float(r2), "rmse": rmse}


# ----------------------------
# NaN handling (unchanged logic, but made explicit)
# ----------------------------
def remove_nans(X: np.ndarray, y: np.ndarray):
    """
    Remove rows where either X has any NaN OR y has any NaN.
    Assumes X is 2D [n_samples, n_features], y is 2D [n_samples, n_outputs] or 1D.
    """
    if y.ndim == 1:
        y_nan = np.isnan(y)
    else:
        y_nan = np.isnan(y).any(axis=1)

    X_nan = np.isnan(X).any(axis=1)
    keep = ~(X_nan | y_nan)
    return X[keep], y[keep]


@dataclass
class PCARidgeDecoder:
    pca: PCA
    reg: object           # whatever reg.fit_semedo_ridge returns (sklearn-like)
    y_mean: np.ndarray    # [n_outputs,]


def fit_decoder_pca_ridge(
    X_train_2d: np.ndarray,
    y_train_2d: np.ndarray,
    n_components: int,
    reg_module,
    svd_solver: str = "full",
):
    """
    Fits PCA on X_train and ridge on PCA(X_train).
    Centers y by training mean (and stores it).
    """
    y_mean = np.nanmean(y_train_2d, axis=0)
    y_train_c = y_train_2d - y_mean

    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    Xtr_p = pca.fit_transform(X_train_2d)

    reg_model = reg_module.fit_semedo_ridge(Xtr_p, y_train_c)
    return PCARidgeDecoder(pca=pca, reg=reg_model, y_mean=y_mean)


def predict_decoder(dec: PCARidgeDecoder, X_2d: np.ndarray) -> np.ndarray:
    Xp = dec.pca.transform(X_2d)
    y_pred_c = dec.reg.predict(Xp)
    return y_pred_c + dec.y_mean


def evaluate_decoder(dec: PCARidgeDecoder, X_test_2d: np.ndarray, y_test_2d: np.ndarray) -> dict:
    """
    Applies the decoder and returns both R2 and RMSE.
    IMPORTANT: y_test is centred using TRAIN y_mean implicitly via predict (we compare in original space).
    """
    y_pred = predict_decoder(dec, X_test_2d)
    return score_metrics(y_test_2d, y_pred)


# ----------------------------
# CV decode (returns BOTH metrics + fitted "full" decoder)
# ----------------------------
def decode(
    X, y,
    n_components,
    reg_module,
    n_splits=10,
    random_state=None,
    shuffle=True,
    window_size=None,
    max_train=None,
    max_test=None,
):
    """
    Returns:
      fold_metrics: list of dicts [{"r2":..., "rmse":...}, ...]
      full_decoder: PCARidgeDecoder fit on all available (NaN-removed) samples
      last_test_idx: indices from the last fold (for plotting/debug)
    """
    fold_metrics = []
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    n_targets, n_trials_x, n_time_x, n_features = X.shape
    _, n_trials_y, n_time_y, bhv_features = y.shape
    assert n_trials_x == n_trials_y and n_time_x == n_time_y, "X-y shape mismatch"

    # helper to flatten
    def flatten_Xy(X_blk, y_blk):
        X2 = X_blk.reshape(-1, n_features)
        y2 = y_blk.reshape(-1, bhv_features)
        X2, y2 = remove_nans(X2, y2)
        if max_train is not None:  # only applied by caller to train blocks
            pass
        return X2, y2

    # Decide splitting strategy
    if window_size is not None:
        bins_per_window = int(window_size / Params.BIN_SIZE)
        n_total_tp = n_trials_x * n_time_x
        n_windows = n_total_tp // bins_per_window
        n_use = n_windows * bins_per_window

        # reshape into windows along the (trial,time) axis
        X_lin = X.reshape(n_targets, n_trials_x * n_time_x, n_features)[:, :n_use, :]
        y_lin = y.reshape(n_targets, n_trials_x * n_time_x, bhv_features)[:, :n_use, :]

        Xw = X_lin.reshape(n_targets, n_windows, bins_per_window, n_features)
        yw = y_lin.reshape(n_targets, n_windows, bins_per_window, bhv_features)
        yw = yw-yw.mean(axis=2, keepdims=True)  # center within window

        window_indices = np.arange(n_windows)

        for train_idx, test_idx in kf.split(window_indices):
            X_train = Xw[:, train_idx, :, :].reshape(-1, n_features)
            y_train = yw[:, train_idx, :, :].reshape(-1, bhv_features)
            X_test  = Xw[:, test_idx,  :, :].reshape(-1, n_features)
            y_test  = yw[:, test_idx,  :, :].reshape(-1, bhv_features)

            X_train, y_train = remove_nans(X_train, y_train)
            X_test,  y_test  = remove_nans(X_test,  y_test)

            if max_train is not None:
                X_train = X_train[:max_train, :]
                y_train = y_train[:max_train, :]
            if max_test is not None:
                X_test = X_test[:max_test, :]
                y_test = y_test[:max_test, :]

            dec = fit_decoder_pca_ridge(X_train, y_train, n_components, reg_module)
            fold_metrics.append(evaluate_decoder(dec, X_test, y_test))

    elif n_trials_x >= n_splits:
        trial_indices = np.arange(n_trials_x)
        for train_idx, test_idx in kf.split(trial_indices):
            X_train = X[:, train_idx, :, :].reshape(-1, n_features)
            y_train = y[:, train_idx, :, :].reshape(-1, bhv_features)
            X_test  = X[:, test_idx,  :, :].reshape(-1, n_features)
            y_test  = y[:, test_idx,  :, :].reshape(-1, bhv_features)

            X_train, y_train = remove_nans(X_train, y_train)
            X_test,  y_test  = remove_nans(X_test,  y_test)

            if max_train is not None:
                X_train = X_train[:max_train, :]
                y_train = y_train[:max_train, :]
            if max_test is not None:
                X_test = X_test[:max_test, :]
                y_test = y_test[:max_test, :]

            dec = fit_decoder_pca_ridge(X_train, y_train, n_components, reg_module)
            fold_metrics.append(evaluate_decoder(dec, X_test, y_test))

    else:
        time_indices = np.arange(n_time_x)
        for train_idx, test_idx in kf.split(time_indices):
            X_train = X[:, :, train_idx, :].reshape(-1, n_features)
            y_train = y[:, :, train_idx, :].reshape(-1, bhv_features)
            X_test  = X[:, :, test_idx,  :].reshape(-1, n_features)
            y_test  = y[:, :, test_idx,  :].reshape(-1, bhv_features)

            X_train, y_train = remove_nans(X_train, y_train)
            X_test,  y_test  = remove_nans(X_test,  y_test)

            if max_train is not None:
                X_train = X_train[:max_train, :]
                y_train = y_train[:max_train, :]
            if max_test is not None:
                X_test = X_test[:max_test, :]
                y_test = y_test[:max_test, :]

            dec = fit_decoder_pca_ridge(X_train, y_train, n_components, reg_module)
            fold_metrics.append(evaluate_decoder(dec, X_test, y_test))

    # Fit "full" decoder on all samples (use the same max_train/max_test logic? Typically no.)
    X_all = X.reshape(-1, n_features)
    y_all = y.reshape(-1, bhv_features)
    X_all, y_all = remove_nans(X_all, y_all)
    full_decoder = fit_decoder_pca_ridge(X_all, y_all, n_components, reg_module)

    return fold_metrics, full_decoder, test_idx


# ----------------------------
# Cross-condition tests requested:
#   - train inter PCA+reg on pre
#   - train pre PCA+reg on inter
# ----------------------------
def cross_condition_test(
    X_train_cond, y_train_cond,
    X_test_cond, y_test_cond,
    n_components,
    reg_module,
    max_train=None,
    max_test=None,
):
    Xtr = X_train_cond.reshape(-1, X_train_cond.shape[-1])
    ytr = y_train_cond.reshape(-1, y_train_cond.shape[-1])
    Xte = X_test_cond.reshape(-1, X_test_cond.shape[-1])
    yte = y_test_cond.reshape(-1, y_test_cond.shape[-1])

    Xtr, ytr = remove_nans(Xtr, ytr)
    Xte, yte = remove_nans(Xte, yte)

    if max_train is not None:
        Xtr = Xtr[:max_train, :]
        ytr = ytr[:max_train, :]
    if max_test is not None:
        Xte = Xte[:max_test, :]
        yte = yte[:max_test, :]

    dec = fit_decoder_pca_ridge(Xtr, ytr, n_components, reg_module)
    metrics = evaluate_decoder(dec, Xte, yte)
    return metrics, dec

# def get_data_array_and_pos_all_areas(df_list, trial_cat,epoch,areas,components,pca_bhv = False, bhv = ["all"], bhv_components = 1):
#     all_neural_data = np.empty((0,)) 
#     for area in areas:
#         neural_data_area, bhv_data = dt.get_data_array_and_pos(df_list, trial_cat,epoch,area,n_components = components[area], pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components)
#         # concatenate along the components axis
#         if all_neural_data.shape[0] == 0:
#             all_neural_data = neural_data_area
#         else:
#             all_neural_data = np.concatenate((all_neural_data, neural_data_area), axis=-1)
        

#     print(f"Neural data shape: {all_neural_data.shape}, Behavioral data shape: {bhv_data.shape}")
#     return all_neural_data, bhv_data
def get_data_array_and_pos_all_areas(df_list, trial_cat,epoch,areas,components,pca_bhv = False, bhv = ["all"], bhv_components = 1, random_state = None, trial_type = "trial",model = "pca"):
    all_neural_data = np.empty((0,)) 
    for area in areas:
        neural_data_area, bhv_data = dt.get_data_array_and_pos(df_list, trial_cat,epoch,area,n_components = components[area], pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components, shuffle_id = False, trial_type = trial_type, model = model)
        # concatenate along the components axis
        if all_neural_data.shape[0] == 0:
            all_neural_data = neural_data_area
        else:
            all_neural_data = np.concatenate((all_neural_data, neural_data_area), axis=-1)
    n_sessions, n_targets, n_trials, _, _ = all_neural_data.shape
    rng_root = np.random.SeedSequence(random_state)
    child_seeds = rng_root.spawn(n_sessions * n_targets)
    k = 0
    for s in range(n_sessions):
        for t in range(n_targets):
            rng_st = np.random.default_rng(child_seeds[k]); k += 1
            perm = rng_st.permutation(n_trials)
            all_neural_data[s, t] = all_neural_data[s, t, perm, :, :]
            bhv_data[s,        t] = bhv_data[s,        t, perm, :, :]  

    print(f"Neural data shape: {all_neural_data.shape}, Behavioral data shape: {bhv_data.shape}, time samples: {all_neural_data.shape[1]*all_neural_data.shape[2]*all_neural_data.shape[3]}")
    return all_neural_data, bhv_data

sessions = [
    'M062_2025_03_21_14_00',
    # 'M062_2025_03_20_14_00',
    # 'M062_2025_03_19_14_00',
    # 'M061_2025_03_06_14_00',
    # 'M061_2025_03_05_14_00',
    # "M061_2025_03_04_10_00",
    # "M063_2025_03_12_14_00",  
    # "M063_2025_03_13_14_00",  
    # "M063_2025_03_14_15_30",
    # "M078_2025_08_06_15_00"


]
all_dfs = []
for session in sessions:
    print(session)
    animal = session.split('_')[0]
    data_dir = f"/data/bnd-data/raw/{animal}/{session}"
    for file in range(4):
        fname = os.path.join(data_dir, f"{session}_pyaldata_{file}.mat")
        print(fname)
        if os.path.exists(fname):
            print(fname)
            df = pyal.mat2dataframe(fname, shift_idx_fields=False)
            # concatenate the different parts of the session into one dataframe
            if file == 0:
                full_df = df
            else:
                full_df = pd.concat([full_df, df], ignore_index=True)
    all_dfs.append(full_df)
    
# from tools.dsp.preprocessing import preprocess
# all_dfs_trials = []
# for dataframe in all_dfs:
#     all_dfs_trials.append(preprocess(dataframe, only_trials=True))

from tools.dsp.preprocessing import preprocess
all_dfs_all = []
for dataframe in all_dfs:
    all_dfs_all.append(preprocess(dataframe, only_trials=False))

## predictors
df_0 = all_dfs_all[0]
# df_0["trial_name"][0] = "free0"
# df_0["trial_name"][len(df_0)-1] = "free1"
bhv_fields = [ 
                        # "bhv"
                #     "shoulder_center",
                #     "left_shoulder",
                    "left_paw",
                #     "right_shoulder",
                #     "right_elbow",
                    "left_foot",
                #     "right_paw",
                #     # "hip_center",
                #     # "left_knee",
                    "left_ankle",
                    
                # #     "right_knee",
                # #     "right_ankle",
                #     "right_foot",
                # #     "tail_base",
                # #     "tail_middle",
                # #     "tail_tip",
                # #     "left_elbow",
                # #     "left_wrist",
                # #     "right_wrist",
                # #     "right_knee_angle",
                # #     "left_knee_angle",
                # #     "right_ankle_angle",
                # #     "left_ankle_angle",
                # #     "right_elbow_angle",
                # #     "left_elbow_angle",
                # #      "shoulder_center_vel",
                # # "left_shoulder_vel",
                # "left_paw_vel",
                # # "right_shoulder_vel",
                # # "right_paw_vel",
                # # "hip_center_vel",
                # # "left_knee_vel",
                # # "left_knee_angle_vel",
                # # "left_ankle_vel",
                # #  "left_ankle_angle_vel",
                # "left_foot_vel",
                # # "right_knee_vel",
                # #  "right_knee_angle_vel",
                # # "right_ankle_vel",
                # #  "right_ankle_angle_vel",
                # # "right_foot_vel",
                # # "tail_base_vel",
                # # "tail_middle_vel",
                # # "tail_tip_vel",
                # # "left_elbow_vel",
                # #  "left_elbow_angle_vel",
                # #  "right_elbow_vel",
                # # "right_elbow_angle_vel",
                # # "left_wrist_vel",
                # # "right_wrist_vel",
               
            
                   
                   
                  
                   
                ]
# areas 
areas_list = [["MOp"],["CP"],["SSp"],["VAL"]]
n_components = 30


# components = {"MOp": 50, "CP": 50, "SSp": 50,"VAL": 50}

# delays
max_delay = 0
min_delays = [0,-0.06]
delays_step = 0.02


## behavioral variables to decode
pca_bhv = False
bhv_components = 1


## for trial data:
trial_cat = "values_Sol_direction" # the trial data willl be balanced based on this trial category
# restrict trial data to window
WINDOW_perturb = (-1,3) # time window (in seconds) relative to the perturbation onset
epoch = pyal.generate_epoch_fun(
        start_point_name="idx_sol_on",
        rel_start=int(WINDOW_perturb[0] / Params.BIN_SIZE),
        rel_end=int(WINDOW_perturb[1] / Params.BIN_SIZE),
    )
for bhv in bhv_fields:
    results_df = pd.DataFrame()
    if bhv not in df_0.columns and "vel" in bhv:
        df_0 = dt.add_velocity_fields(df_0,fields = [bhv.replace("_vel","")]) 
    
    for min_delay in min_delays:
        assert delays_step >= Params.BIN_SIZE, f"delays_step should be >= bin size ({Params.BIN_SIZE})"
        delays = list(range(int(min_delay/Params.BIN_SIZE), int(max_delay/Params.BIN_SIZE)+1, int(delays_step/Params.BIN_SIZE)))
        for areas in areas_list:
            components = {}
            for area in areas:
                components[area] = n_components
            # data_all_areas_perturb, bhv_perturb  = get_data_array_and_pos_all_areas([df_0], trial_cat, epoch, areas,components, pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components, random_state = 0, trial_type = "trial", model = None)
            # ammend get_data_array_and_pos_all_areas to get data for intertrial, pretrial and posttrial periods
            data_all_areas_inter, bhv_inter  = get_data_array_and_pos_all_areas([df_0], trial_cat, epoch, areas,components, pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components, random_state = 0, trial_type = "intertrial", model = None)
            data_all_areas_pre, bhv_pre  = get_data_array_and_pos_all_areas([df_0], trial_cat, epoch, areas,components, pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components, random_state = 0, trial_type = "free0", model = None)
            # data_all_areas_post, bhv_post  = get_data_array_and_pos_all_areas([df_0], trial_cat, epoch, areas,components, pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components, random_state = 0, trial_type = "free1", model = None)
            # keep the minimum number of (n_trials * n_targets) and n_time in perturb and intertrial
            labels = []
            for a in areas:
                for pc in range(components[a]):
                    labels.append((a, pc))
            
            ## build decoder
            # for trial data
            start_t, end_t = 0, 0.5 # wrt t_0 (perturbation onset)
            #  check that the time window is within the epoch
            assert start_t >= WINDOW_perturb[0] and end_t <= WINDOW_perturb[1], "The time window is outside the epoch"
            n_splits = 5
            # do only one cross-validation fold per session for testing
            for session in range(data_all_areas_inter.shape[0]):
                print(f"Session {session}")
                # perturbation data
                # data_all_areas_session = data_all_areas_perturb[session]
                # bhv_session = bhv_perturb[session]

                # input_matrix_perturb, output_matrix_perturb,feature_names = get_input_output_matrix(data_all_areas_session,bhv_session,labels, delays, start_t, end_t, WINDOW_perturb)
                # output_matrix_perturb, output_feature_names = get_output_matrix(bhv_session,start_t, end_t, WINDOW_perturb)
                # intertrial data
                data_all_areas_session = data_all_areas_inter[session]
                bhv_session = bhv_inter[session]  
                input_matrix_inter,output_matrix_inter, feature_names = get_input_output_matrix(data_all_areas_session,bhv_session,labels, delays, start_t = None, end_t = None, WINDOW_perturb = None)
                # output_matrix_inter, output_feature_names = get_output_matrix(bhv_session,start_t =None, end_t = None, WINDOW_perturb = None)
                print(f"Input matrix inter shape: {input_matrix_inter.shape}, output matrix inter shape: {output_matrix_inter.shape}")
                # pretrial data
                data_all_areas_session = data_all_areas_pre[session]
                bhv_session = bhv_pre[session]  
                input_matrix_pre,output_matrix_pre, feature_names = get_input_output_matrix(data_all_areas_session,bhv_session,labels, delays, start_t = None, end_t = None, WINDOW_perturb = None)
                # output_matrix_pre, output_feature_names= get_output_matrix(bhv_session,start_t =None, end_t = None, WINDOW_perturb = None)
                # data_all_areas_session = data_all_areas_post[session]
                # bhv_session = bhv_post[session]  
                # input_matrix_post, output_matrix_post,feature_names = get_input_output_matrix(data_all_areas_session,bhv_session,labels, delays, start_t = None, end_t = None, WINDOW_perturb = None)
                # output_matrix_post, output_feature_names = get_output_matrix(bhv_session,start_t =None, end_t = None, WINDOW_perturb = None)
                # max_test = int(input_matrix_post.shape[0]*input_matrix_post.shape[1]*input_matrix_post.shape[2]/n_splits)
                # max_train = int(input_matrix_post.shape[0]*input_matrix_post.shape[1]*input_matrix_post.shape[2]-max_test)
                max_test = int(input_matrix_pre.shape[0]*input_matrix_pre.shape[1]*input_matrix_pre.shape[2]/n_splits)
                max_train = int(input_matrix_pre.shape[0]*input_matrix_pre.shape[1]*input_matrix_pre.shape[2]-max_test)
                print(f"Max train: {max_train}, max test: {max_test}")
                print(f"Max train: {max_train}, max test: {max_test}")
                # decode perturbation data
                # print("base score perturb")
                # base_score_perturb, decoder_perturb, test_idx_perturb = decode(input_matrix_perturb, output_matrix_perturb, n_splits=n_splits,max_train = max_train, max_test = max_test, shuffle = False)
                # results_df = pd.concat([results_df, pd.DataFrame({
                #     "session": all_dfs_all[session].session[0],
                #     "train_cond": "perturb",
                #     "test_cond": "perturb",
                #     "score": [base_score_perturb],
                #     "areas":  " ".join(areas),
                #     "n_components": n_components,
                #     "delays": [delays],
                    

                #     })])
                # results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test3_{bhv}.pkl")
                
                # decode intertrial data
                print("base score inter")
                # Example: base scores (within-condition CV)
                metrics_inter_folds, full_dec_inter, _ = decode(
                    input_matrix_inter, output_matrix_inter,
                    n_components=n_components, reg_module=reg,
                    n_splits=n_splits, max_train=max_train, max_test=max_test,
                    shuffle=False
                )
                metrics_pre_folds, full_dec_pre, _ = decode(
                    input_matrix_pre, output_matrix_pre,
                    n_components=n_components, reg_module=reg,
                    n_splits=n_splits, max_train=max_train, max_test=max_test,
                    shuffle=True, window_size=1, random_state=42
                )

                # Summaries if you want scalar entries for your dataframe:
                inter_r2_mean  = float(np.mean([m["r2"]   for m in metrics_inter_folds]))
                inter_rmse_mean= float(np.mean([m["rmse"] for m in metrics_inter_folds]))
                pre_r2_mean    = float(np.mean([m["r2"]   for m in metrics_pre_folds]))
                pre_rmse_mean  = float(np.mean([m["rmse"] for m in metrics_pre_folds]))

                                # Train on inter, test on pre
                metrics_inter_to_pre, dec_inter_to_pre = cross_condition_test(
                    input_matrix_inter, output_matrix_inter,
                    input_matrix_pre,   output_matrix_pre,
                    n_components=n_components, reg_module=reg,
                    max_train=max_train, max_test=max_test
                )

                # Train on pre, test on inter
                metrics_pre_to_inter, dec_pre_to_inter = cross_condition_test(
                    input_matrix_pre,   output_matrix_pre,
                    input_matrix_inter, output_matrix_inter,
                    n_components=n_components, reg_module=reg,
                    max_train=max_train, max_test=max_test
                )

                results_df = pd.concat([results_df, pd.DataFrame([{
                    "session": all_dfs_all[session].session[0],
                    "train_cond": "inter",
                    "test_cond": "inter",
                    "r2": inter_r2_mean,
                    "rmse": inter_rmse_mean,
                    "areas": " ".join(areas),
                    "n_components": n_components,
                    "delays": [delays],
                }])])

                results_df = pd.concat([results_df, pd.DataFrame([{
                    "session": all_dfs_all[session].session[0],
                    "train_cond": "pre",
                    "test_cond": "pre",
                    "r2": pre_r2_mean,
                    "rmse": pre_rmse_mean,
                    "areas": " ".join(areas),
                    "n_components": n_components,
                    "delays": [delays],
                }])])

                results_df = pd.concat([results_df, pd.DataFrame([{
                    "session": all_dfs_all[session].session[0],
                    "train_cond": "inter",
                    "test_cond": "pre",
                    "r2": metrics_inter_to_pre["r2"],
                    "rmse": metrics_inter_to_pre["rmse"],
                    "areas": " ".join(areas),
                    "n_components": n_components,
                    "delays": [delays],
                }])])

                results_df = pd.concat([results_df, pd.DataFrame([{
                    "session": all_dfs_all[session].session[0],
                    "train_cond": "pre",
                    "test_cond": "inter",
                    "r2": metrics_pre_to_inter["r2"],
                    "rmse": metrics_pre_to_inter["rmse"],
                    "areas": " ".join(areas),
                    "n_components": n_components,
                    "delays": [delays],
                }])])

                results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test5_{bhv}.pkl")


