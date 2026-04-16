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
def remove_nans(matrix,rates):
    """
    Remove NaN values from bhv data and the corresponding timepoints from the neural data
    """
    [~np.isnan(matrix)]  
    nan_rows = np.isnan(matrix) if matrix.ndim ==1 else np.isnan(matrix).any(axis=1) 
    # print(nan_rows)
    cleaned_matrix = matrix[~nan_rows]  
    cleaned_rates = rates[~nan_rows]  
    return cleaned_matrix, cleaned_rates
def score_metric(y_true,y_pred):
    score = r2_score(y_true, y_pred, multioutput="variance_weighted")
    return score
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

def decode(X,y,n_splits=10,random_state = None,shuffle = True, window_size = None,max_train = None, max_test = None):
    scores = []
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    n_targets, n_trials_x, n_time_x, n_features = X.shape
    _, n_trials_y, n_time_y, bhv_features = y.shape
    assert n_trials_x == n_trials_y and n_time_x == n_time_y, "X-y shape mismatch"
    #  the order of trials for each condition is already shuffled 
    
    
    if window_size is not None:
        # reshape timepoints into trials of size window_size and split based on that
        n_windows = n_time_x*n_trials_x // int(window_size/Params.BIN_SIZE)
        # make sure that you can reshape. if not, remove some timepoints from the end
        n_timepoints_to_use = n_windows * int(window_size/Params.BIN_SIZE)
        X = X[:, : , :n_timepoints_to_use, :].reshape(n_targets, n_trials_x, n_timepoints_to_use, n_features)
        y = y[:, : , :n_timepoints_to_use, :].reshape(n_targets, n_trials_x, n_timepoints_to_use, bhv_features)
        time_indices = np.arange(n_windows)
        X_reshaped = X.reshape(n_targets, n_windows, int(window_size/Params.BIN_SIZE), n_features)
        y_reshaped = y.reshape(n_targets, n_windows, int(window_size/Params.BIN_SIZE), bhv_features)
        for train_idx, test_idx in kf.split(time_indices):
            X_train = X_reshaped[ :, train_idx, :, :].reshape(-1, n_features)
            y_train = y_reshaped[ :, train_idx, :, :].reshape(-1,bhv_features)
            X_test = X_reshaped[ :, test_idx, :, :].reshape(-1, n_features)
            y_test = y_reshaped[ :, test_idx, :, :].reshape(-1, bhv_features)
            # remove timepoints that have NaNs in either X or y
            X_train, y_train = remove_nans(X_train, y_train)
            X_test, y_test = remove_nans(X_test, y_test)
            if max_train is not None:
                X_train = X_train[:max_train,:]
                y_train = y_train[:max_train,:]
            if max_test is not None:
                X_test = X_test[:max_test,:]
                y_test = y_test[:max_test,:]
            
            y_train = remove_mean(y_train)
            y_test = remove_mean(y_test)

            # fit pca model, then linear regression; at the end fit model to all data and return it; keep test indices to test the models from the different blocks
            model = PCA(n_components, svd_solver = "full")
            X_train = model.fit_transform(X_train)
            X_test = model.transform(X_test)
            model_fold = reg.fit_semedo_ridge(X_train, y_train)
            y_pred = model_fold.predict(X_test)
            scores.append(score_metric(y_test, y_pred))
            print(scores[-1])   
            pca_full_model = model.fit_transform(X_reshaped.reshape(-1, n_features))
            # reg_full_model = reg.fit_semedo_ridge(X_reshaped.reshape(-1, n_features),y_reshaped.reshape(-1, bhv_features))
            reg_full_model = reg.fit_semedo_ridge(X_reshaped.reshape(-1, n_features),y_reshaped.reshape(-1, bhv_features))


            # break # start with one fold
    elif n_trials_x >= n_splits: 
        trial_indices = np.arange(n_trials_x)
        for train_idx, test_idx in kf.split(trial_indices):
            X_train = X[:, train_idx, :, :].reshape(-1, n_features)
            y_train = y[:, train_idx, :, :].reshape(-1,bhv_features)
            X_test = X[:, test_idx, :, :].reshape(-1, n_features)
            y_test = y[:, test_idx, :, :].reshape(-1, bhv_features)
            X_train, y_train = remove_nans(X_train, y_train)
            X_test, y_test = remove_nans(X_test, y_test)
            if max_train is not None:
                X_train = X_train[:max_train,:]
                y_train = y_train[:max_train,:]
            if max_test is not None:
                X_test = X_test[:max_test,:]
                y_test = y_test[:max_test,:]

            y_train = remove_mean(y_train)
            y_test = remove_mean(y_test)
            model = PCA(n_components, svd_solver = "full")
            X_train = model.fit_transform(X_train)
            X_test = model.transform(X_test)
            model_fold = reg.fit_semedo_ridge(X_train, y_train)
            y_pred = model_fold.predict(X_test)
            scores.append(score_metric(y_test, y_pred))
            print(scores[-1])   
            pca_full_model = model.fit_transform(X_reshaped.reshape(-1, n_features))
            reg_full_model = reg.fit_semedo_ridge(X_reshaped.reshape(-1, n_features),y_reshaped.reshape(-1, bhv_features))

            # break # start with one fold
    else: # what this does is to split based on timepoints if there are not enough trials
        time_indices = np.arange(n_time_x)
        for train_idx, test_idx in kf.split(time_indices):
            X_train = X[:, :, train_idx, :].reshape(-1, n_features)
            y_train = y[:, :, train_idx, :].reshape(-1,bhv_features)
            X_test = X[:, :, test_idx, :].reshape(-1, n_features)
            y_test = y[:, :, test_idx, :].reshape(-1, bhv_features)
            X_train, y_train = remove_nans(X_train, y_train)
            X_test, y_test = remove_nans(X_test, y_test)
            if max_train is not None:
                X_train = X_train[:max_train,:]
                y_train = y_train[:max_train,:]
            if max_test is not None:
                X_test = X_test[:max_test,:]
                y_test = y_test[:max_test,:]
            y_train = remove_mean(y_train)
            y_test = remove_mean(y_test)
            model = PCA(n_components, svd_solver = "full")
            X_train = model.fit_transform(X_train)
            X_test = model.transform(X_test)
            model_fold = reg.fit_semedo_ridge(X_train, y_train)
            y_pred = model_fold.predict(X_test)
            scores.append(score_metric(y_test, y_pred))
            print(scores[-1])   
            pca_full_model = model.fit_transform(X_reshaped.reshape(-1, n_features))
            reg_full_model = reg.fit_semedo_ridge(X_reshaped.reshape(-1, n_features),y_reshaped.reshape(-1, bhv_features))  
            # break # start with one fold

    print(f"After removing NaNs, train shape: {X_train.shape}, test shape: {X_test.shape}")
    return scores, pca_full_model, reg_full_model, test_idx
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
            data_all_areas_perturb, bhv_perturb  = get_data_array_and_pos_all_areas([df_0], trial_cat, epoch, areas,components, pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components, random_state = 0, trial_type = "trial", model = None)
            # ammend get_data_array_and_pos_all_areas to get data for intertrial, pretrial and posttrial periods
            data_all_areas_inter, bhv_inter  = get_data_array_and_pos_all_areas([df_0], trial_cat, epoch, areas,components, pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components, random_state = 0, trial_type = "intertrial", model = None)
            data_all_areas_pre, bhv_pre  = get_data_array_and_pos_all_areas([df_0], trial_cat, epoch, areas,components, pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components, random_state = 0, trial_type = "free0", model = None)
            data_all_areas_post, bhv_post  = get_data_array_and_pos_all_areas([df_0], trial_cat, epoch, areas,components, pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components, random_state = 0, trial_type = "free1", model = None)
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
            for session in range(data_all_areas_perturb.shape[0]):
                print(f"Session {session}")
                # perturbation data
                data_all_areas_session = data_all_areas_perturb[session]
                bhv_session = bhv_perturb[session]

                input_matrix_perturb, output_matrix_perturb,feature_names = get_input_output_matrix(data_all_areas_session,bhv_session,labels, delays, start_t, end_t, WINDOW_perturb)
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
                data_all_areas_session = data_all_areas_post[session]
                bhv_session = bhv_post[session]  
                input_matrix_post, output_matrix_post,feature_names = get_input_output_matrix(data_all_areas_session,bhv_session,labels, delays, start_t = None, end_t = None, WINDOW_perturb = None)
                # output_matrix_post, output_feature_names = get_output_matrix(bhv_session,start_t =None, end_t = None, WINDOW_perturb = None)
                max_test = int(input_matrix_post.shape[0]*input_matrix_post.shape[1]*input_matrix_post.shape[2]/n_splits)
                max_train = int(input_matrix_post.shape[0]*input_matrix_post.shape[1]*input_matrix_post.shape[2]-max_test)
                print(f"Max train: {max_train}, max test: {max_test}")
                # decode perturbation data
                print("base score perturb")
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
                base_score_inter, decoder_inter, test_idx_inter = decode(input_matrix_inter, output_matrix_inter, n_splits=n_splits,max_train = max_train, max_test = max_test, shuffle = False)
                # decode pretrial data
                results_df = pd.concat([results_df, pd.DataFrame({
                    "session": all_dfs_all[session].session[0],
                    "train_cond": "inter",
                    "test_cond": "inter",
                    "score": [base_score_inter],
                    "areas":  " ".join(areas),
                    "n_components": n_components,
                    "delays": [delays],})])
                results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test4_{bhv}.pkl")

                print("base score pre")
                base_score_pre, decoder_pre, test_idx_pre = decode(input_matrix_pre, output_matrix_pre, n_splits=n_splits,max_train = max_train, max_test = max_test, shuffle = True, window_size = 1, random_state=42)
                results_df = pd.concat([results_df, pd.DataFrame({
                    "session": all_dfs_all[session].session[0],
                    "train_cond": "pre",
                    "test_cond": "pre",
                    "score": [base_score_pre],
                    "areas": " ".join(areas),
                    "n_components": n_components,
                    "delays": [delays],
                    })])
                results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test4_{bhv}.pkl")

                # # decode posttrial data
                # print("base score post")
                # base_score_post, decoder_post, test_idx_post = decode(input_matrix_post, output_matrix_post, n_splits=n_splits,max_train = max_train, max_test = max_test, shuffle = True, window_size = 1, random_state=42)    
                # results_df = pd.concat([results_df, pd.DataFrame({
                #     "session": all_dfs_all[session].session[0],
                #     "train_cond": "post",
                #     "test_cond": "post",
                #     "score": [base_score_post],
                #     "areas": " ".join(areas),
                #     "n_components": n_components,
                #     "delays": [delays],
                #     })])
                #     # break
                # results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test3_{bhv}.pkl")

                # test each decoder on the other data: for one of the test idx of a certain period plot the predictions obtained using each decoder
            results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test4_{bhv}.pkl")

# import os
# import sys
# sys.path.append("../")

# import pyaldata as pyal
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')

# from tools.params import Params
# from tools import dataTools as dt
# from tools.decoding import regression as reg

# from sklearn.model_selection import KFold
# from sklearn.metrics import r2_score, mean_squared_error


# # =========================
# # Matrix builders
# # =========================
# def build_input_matrix(data_all_areas_session, labels, delays, start_t=None, end_t=None, WINDOW_perturb=None):
#     n_targets, n_trials, n_time, total_pcs = data_all_areas_session.shape

#     start_idx = int((start_t - WINDOW_perturb[0]) / Params.BIN_SIZE) if start_t is not None else 0
#     end_idx   = int((end_t   - WINDOW_perturb[0]) / Params.BIN_SIZE) if end_t   is not None else n_time
#     if not (0 <= start_idx < n_time) or not (0 < end_idx <= n_time) or not (start_idx < end_idx):
#         raise ValueError("Requested time window is outside the epoch or invalid.")

#     indices = np.arange(start_idx, end_idx, dtype=int)
#     valid = np.ones(indices.shape, dtype=bool)
#     for d in delays:
#         valid &= (indices + d >= 0) & (indices + d < n_time)

#     t_idx = indices[valid]
#     if t_idx.size == 0:
#         raise ValueError("No valid time bins given the delays and window.")

#     T = t_idx.size
#     n_delays = len(delays)
#     n_features = total_pcs * n_delays

#     X = np.empty((n_targets, n_trials, T, n_features), dtype=data_all_areas_session.dtype)
#     feature_names = []

#     for g, (area, pc_area) in enumerate(labels):
#         feature_data = data_all_areas_session[:, :, :, g]  # [n_targets, n_trials, n_time]
#         for j, d in enumerate(delays):
#             idx_del = t_idx + d
#             col = feature_data[:, :, idx_del]               # [n_targets, n_trials, T]
#             feat_idx = g * n_delays + j
#             X[:, :, :, feat_idx] = col
#             feature_names.append({
#                 "area": area,
#                 "pc": int(pc_area),
#                 "lag": float(d * Params.BIN_SIZE),
#                 "name": f"{area}|pc{int(pc_area)}|lag{d}",
#             })

#     return X, feature_names


# def get_input_output_matrix(data_all_areas_session, bhv_session, labels, delays,
#                             start_t=None, end_t=None, WINDOW_perturb=None):
#     n_targets, n_trials, n_time, total_pcs = data_all_areas_session.shape

#     if WINDOW_perturb is None:
#         # use full window
#         start_idx = 0
#         end_idx = n_time
#     else:
#         start_idx = int((start_t - WINDOW_perturb[0]) / Params.BIN_SIZE) if start_t is not None else 0
#         end_idx   = int((end_t   - WINDOW_perturb[0]) / Params.BIN_SIZE) if end_t   is not None else n_time

#     if not (0 <= start_idx < n_time) or not (0 < end_idx <= n_time) or not (start_idx < end_idx):
#         raise ValueError("Requested time window is outside the epoch or invalid.")

#     indices = np.arange(start_idx, end_idx, dtype=int)

#     valid = np.ones(indices.shape, dtype=bool)
#     for d in delays:
#         valid &= (indices + d >= 0) & (indices + d < n_time)

#     t_idx = indices[valid]
#     if t_idx.size == 0:
#         raise ValueError("No valid time bins given the delays and window.")

#     T = t_idx.size
#     n_delays = len(delays)
#     n_features = total_pcs * n_delays

#     X = np.empty((n_targets, n_trials, T, n_features), dtype=data_all_areas_session.dtype)
#     y = bhv_session[:, :, t_idx, :]  # [n_targets, n_trials, T, bhv_features]

#     feature_names = []
#     for g, (area, pc_area) in enumerate(labels):
#         feature_data = data_all_areas_session[:, :, :, g]  # [n_targets, n_trials, n_time]
#         for j, d in enumerate(delays):
#             idx_del = t_idx + d
#             col = feature_data[:, :, idx_del]
#             feat_idx = g * n_delays + j
#             X[:, :, :, feat_idx] = col
#             feature_names.append({
#                 "area": area,
#                 "pc": int(pc_area),
#                 "lag": float(d * Params.BIN_SIZE),
#                 "name": f"{area}|pc{int(pc_area)}|lag{d}",
#             })

#     return X, y, feature_names


# # =========================
# # Cleaning + metrics
# # =========================
# def remove_nans_xy(X, y):
#     """
#     Remove rows where X or y contain NaNs.
#     X: [N, n_features]
#     y: [N, n_outputs]
#     """
#     if X.ndim != 2 or y.ndim != 2:
#         raise ValueError("remove_nans_xy expects 2D arrays")

#     bad_x = np.isnan(X).any(axis=1)
#     bad_y = np.isnan(y).any(axis=1)
#     keep = ~(bad_x | bad_y)
#     return X[keep], y[keep]


# def score_r2(y_true, y_pred):
#     return r2_score(y_true, y_pred, multioutput="variance_weighted")


# def score_rmse(y_true, y_pred):
#     # average over outputs (uniform), then take sqrt
#     return float(np.sqrt(mean_squared_error(y_true, y_pred, multioutput="uniform_average")))


# # =========================
# # Decoders
# # =========================
# def decode(X, y, n_splits=10, random_state=None, shuffle=True,
#            window_size=None, max_train=None, max_test=None):
#     """
#     Returns:
#       r2_scores:   list[float]  (per fold)
#       rmse_scores: list[float]  (per fold)
#       model_fold: fitted model from last fold
#       test_idx:   indices from last fold (trial/time window indices depending on mode)
#     """
#     r2_scores = []
#     rmse_scores = []
#     kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

#     n_targets, n_trials_x, n_time_x, n_features = X.shape
#     _, n_trials_y, n_time_y, bhv_features = y.shape
#     assert n_trials_x == n_trials_y and n_time_x == n_time_y, "X-y shape mismatch"

#     model_fold = None
#     test_idx = None

#     if window_size is not None:
#         bins_per_win = int(window_size / Params.BIN_SIZE)
#         if bins_per_win <= 0:
#             raise ValueError("window_size must be >= BIN_SIZE")

#         n_windows = (n_time_x * n_trials_x) // bins_per_win
#         n_timepoints_to_use = n_windows * bins_per_win

#         X_trim = X[:, :, :n_timepoints_to_use, :].reshape(n_targets, n_trials_x, n_timepoints_to_use, n_features)
#         y_trim = y[:, :, :n_timepoints_to_use, :].reshape(n_targets, n_trials_x, n_timepoints_to_use, bhv_features)

#         X_reshaped = X_trim.reshape(n_targets, n_windows, bins_per_win, n_features)
#         y_reshaped = y_trim.reshape(n_targets, n_windows, bins_per_win, bhv_features)

#         time_indices = np.arange(n_windows)
#         for train_idx, test_idx in kf.split(time_indices):
#             X_train = X_reshaped[:, train_idx, :, :].reshape(-1, n_features)
#             y_train = y_reshaped[:, train_idx, :, :].reshape(-1, bhv_features)
#             X_test  = X_reshaped[:, test_idx,  :, :].reshape(-1, n_features)
#             y_test  = y_reshaped[:, test_idx,  :, :].reshape(-1, bhv_features)

#             X_train, y_train = remove_nans_xy(X_train, y_train)
#             X_test,  y_test  = remove_nans_xy(X_test,  y_test)

#             if max_train is not None:
#                 X_train = X_train[:max_train, :]
#                 y_train = y_train[:max_train, :]
#             if max_test is not None:
#                 X_test = X_test[:max_test, :]
#                 y_test = y_test[:max_test, :]

#             model_fold = reg.fit_semedo_ridge(X_train, y_train)
#             y_pred = model_fold.predict(X_test)

#             r2 = score_r2(y_test, y_pred)
#             rmse = score_rmse(y_test, y_pred)

#             r2_scores.append(r2)
#             rmse_scores.append(rmse)

#             print(f"fold r2={r2:.4f} rmse={rmse:.4f}")

#     elif n_trials_x >= n_splits:
#         trial_indices = np.arange(n_trials_x)
#         for train_idx, test_idx in kf.split(trial_indices):
#             X_train = X[:, train_idx, :, :].reshape(-1, n_features)
#             y_train = y[:, train_idx, :, :].reshape(-1, bhv_features)
#             X_test  = X[:, test_idx,  :, :].reshape(-1, n_features)
#             y_test  = y[:, test_idx,  :, :].reshape(-1, bhv_features)

#             X_train, y_train = remove_nans_xy(X_train, y_train)
#             X_test,  y_test  = remove_nans_xy(X_test,  y_test)

#             if max_train is not None:
#                 X_train = X_train[:max_train, :]
#                 y_train = y_train[:max_train, :]
#             if max_test is not None:
#                 X_test = X_test[:max_test, :]
#                 y_test = y_test[:max_test, :]

#             model_fold = reg.fit_semedo_ridge(X_train, y_train)
#             y_pred = model_fold.predict(X_test)

#             r2 = score_r2(y_test, y_pred)
#             rmse = score_rmse(y_test, y_pred)

#             r2_scores.append(r2)
#             rmse_scores.append(rmse)

#             print(f"fold r2={r2:.4f} rmse={rmse:.4f}")

#     else:
#         time_indices = np.arange(n_time_x)
#         for train_idx, test_idx in kf.split(time_indices):
#             X_train = X[:, :, train_idx, :].reshape(-1, n_features)
#             y_train = y[:, :, train_idx, :].reshape(-1, bhv_features)
#             X_test  = X[:, :, test_idx,  :].reshape(-1, n_features)
#             y_test  = y[:, :, test_idx,  :].reshape(-1, bhv_features)

#             X_train, y_train = remove_nans_xy(X_train, y_train)
#             X_test,  y_test  = remove_nans_xy(X_test,  y_test)

#             if max_train is not None:
#                 X_train = X_train[:max_train, :]
#                 y_train = y_train[:max_train, :]
#             if max_test is not None:
#                 X_test = X_test[:max_test, :]
#                 y_test = y_test[:max_test, :]

#             model_fold = reg.fit_semedo_ridge(X_train, y_train)
#             y_pred = model_fold.predict(X_test)

#             r2 = score_r2(y_test, y_pred)
#             rmse = score_rmse(y_test, y_pred)

#             r2_scores.append(r2)
#             rmse_scores.append(rmse)

#             print(f"fold r2={r2:.4f} rmse={rmse:.4f}")

#     print(f"After removing NaNs, last fold train shape: {X_train.shape}, test shape: {X_test.shape}")
#     return r2_scores, rmse_scores, model_fold, test_idx


# def train_test_cross(X_train4d, y_train4d, X_test4d, y_test4d, max_train=None, max_test=None):
#     """
#     Train on all samples from train condition, test on all samples from test condition.
#     Returns: (r2, rmse, fitted_model)
#     """
#     n_targets, n_trials, n_time, n_features = X_train4d.shape
#     _, _, _, bhv_features = y_train4d.shape

#     Xtr = X_train4d.reshape(-1, n_features)
#     ytr = y_train4d.reshape(-1, bhv_features)
#     Xte = X_test4d.reshape(-1, n_features)
#     yte = y_test4d.reshape(-1, bhv_features)

#     Xtr, ytr = remove_nans_xy(Xtr, ytr)
#     Xte, yte = remove_nans_xy(Xte, yte)

#     if max_train is not None:
#         Xtr = Xtr[:max_train, :]
#         ytr = ytr[:max_train, :]
#     if max_test is not None:
#         Xte = Xte[:max_test, :]
#         yte = yte[:max_test, :]

#     model = reg.fit_semedo_ridge(Xtr, ytr)
#     ypred = model.predict(Xte)

#     r2 = score_r2(yte, ypred)
#     rmse = score_rmse(yte, ypred)
#     return r2, rmse, model


# # =========================
# # Data extraction (all areas)
# # =========================
# def get_data_array_and_pos_all_areas(df_list, trial_cat, epoch, areas, components,
#                                     pca_bhv=False, bhv=["all"], bhv_components=1,
#                                     random_state=None, trial_type="trial"):
#     all_neural_data = np.empty((0,))
#     for area in areas:
#         neural_data_area, bhv_data = dt.get_data_array_and_pos(
#             df_list, trial_cat, epoch, area,
#             n_components=components[area],
#             pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components,
#             shuffle_id=False, trial_type=trial_type
#         )
#         if all_neural_data.shape[0] == 0:
#             all_neural_data = neural_data_area
#         else:
#             all_neural_data = np.concatenate((all_neural_data, neural_data_area), axis=-1)

#     n_sessions, n_targets, n_trials, _, _ = all_neural_data.shape
#     rng_root = np.random.SeedSequence(random_state)
#     child_seeds = rng_root.spawn(n_sessions * n_targets)

#     k = 0
#     for s in range(n_sessions):
#         for t in range(n_targets):
#             rng_st = np.random.default_rng(child_seeds[k]); k += 1
#             perm = rng_st.permutation(n_trials)
#             all_neural_data[s, t] = all_neural_data[s, t, perm, :, :]
#             bhv_data[s, t] = bhv_data[s, t, perm, :, :]

#     print(
#         f"Neural data shape: {all_neural_data.shape}, Behavioral data shape: {bhv_data.shape}, "
#         f"time samples: {all_neural_data.shape[1]*all_neural_data.shape[2]*all_neural_data.shape[3]}"
#     )
#     return all_neural_data, bhv_data


# =========================
# # Main
# # =========================
# sessions = [
#     'M061_2025_03_06_14_00',
# ]

# all_dfs = []
# for session in sessions:
#     print(session)
#     animal = session.split('_')[0]
#     data_dir = f"/data/bnd-data/raw/{animal}/{session}"
#     for file in range(4):
#         fname = os.path.join(data_dir, f"{session}_pyaldata_{file}.mat")
#         print(fname)
#         if os.path.exists(fname):
#             df = pyal.mat2dataframe(fname, shift_idx_fields=False)
#             if file == 0:
#                 full_df = df
#             else:
#                 full_df = pd.concat([full_df, df], ignore_index=True)
#     all_dfs.append(full_df)

# from tools.dsp.preprocessing import preprocess
# all_dfs_all = []
# for dataframe in all_dfs:
#     all_dfs_all.append(preprocess(dataframe, only_trials=False))

# df_0 = all_dfs_all[0]
# df_0["trial_name"][0] = "free0"
# df_0["trial_name"][len(df_0)-1] = "free1"

# bhv_fields = [
#     "left_paw",
#     "left_foot",
#     "right_paw",
#     "right_foot",
#     "left_paw_vel",
#     "left_foot_vel",
# ]

# areas_list = [["MOp"], ["CP"], ["SSp"], ["VAL"]]
# n_components = 30

# max_delay = 0
# min_delays = [0, -0.06]
# delays_step = 0.02

# pca_bhv = False
# bhv_components = 1

# trial_cat = "values_Sol_direction"
# WINDOW_perturb = (-1, 3)
# epoch = pyal.generate_epoch_fun(
#     start_point_name="idx_sol_on",
#     rel_start=int(WINDOW_perturb[0] / Params.BIN_SIZE),
#     rel_end=int(WINDOW_perturb[1] / Params.BIN_SIZE),
# )

# for bhv in bhv_fields:
#     results_df = pd.DataFrame()

#     if bhv not in df_0.columns and "vel" in bhv:
#         df_0 = dt.add_velocity_fields(df_0, fields=[bhv.replace("_vel", "")])

#     for min_delay in min_delays:
#         assert delays_step >= Params.BIN_SIZE, f"delays_step should be >= bin size ({Params.BIN_SIZE})"
#         delays = list(
#             range(
#                 int(min_delay / Params.BIN_SIZE),
#                 int(max_delay / Params.BIN_SIZE) + 1,
#                 int(delays_step / Params.BIN_SIZE),
#             )
#         )

#         for areas in areas_list:
#             components = {area: n_components for area in areas}

#             data_all_areas_perturb, bhv_perturb = get_data_array_and_pos_all_areas(
#                 [df_0], trial_cat, epoch, areas, components,
#                 pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components,
#                 random_state=0, trial_type="trial"
#             )

#             data_all_areas_inter, bhv_inter = get_data_array_and_pos_all_areas(
#                 [df_0], trial_cat, epoch, areas, components,
#                 pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components,
#                 random_state=0, trial_type="intertrial"
#             )

#             data_all_areas_free0, bhv_free0 = get_data_array_and_pos_all_areas(
#                 [df_0], trial_cat, epoch, areas, components,
#                 pca_bhv=pca_bhv, bhv=bhv, bhv_components=bhv_components,
#                 random_state=0, trial_type="free0"
#             )

#             # labels for delayed features
#             labels = []
#             for a in areas:
#                 for pc in range(components[a]):
#                     labels.append((a, pc))

#             # decoder window for perturbation (relative to sol_on)
#             start_t, end_t = 0, 0.5
#             assert start_t >= WINDOW_perturb[0] and end_t <= WINDOW_perturb[1], "Time window outside epoch"

#             n_splits = 5

#             for s in range(data_all_areas_perturb.shape[0]):
#                 print(f"Session idx {s}")

#                 # --- build matrices ---
#                 Xp, yp, _ = get_input_output_matrix(
#                     data_all_areas_perturb[s], bhv_perturb[s],
#                     labels, delays, start_t, end_t, WINDOW_perturb
#                 )

#                 Xi, yi, _ = get_input_output_matrix(
#                     data_all_areas_inter[s], bhv_inter[s],
#                     labels, delays, start_t=None, end_t=None, WINDOW_perturb=None
#                 )

#                 Xf0, yf0, _ = get_input_output_matrix(
#                     data_all_areas_free0[s], bhv_free0[s],
#                     labels, delays, start_t=None, end_t=None, WINDOW_perturb=None
#                 )

#                 # cap training/testing sizes consistently across conditions (optional)
#                 n_cap = min(int(np.prod(Xf0.shape[:3])), int(np.prod(Xi.shape[:3])))
#                 max_test = n_cap // n_splits
#                 max_train = n_cap - max_test
#                 print(f"Max train: {max_train}, max test: {max_test}")

                

#                 print("CV inter -> inter")
#                 r2s, rmses, model_i, _ = decode(
#                     Xi, yi, n_splits=n_splits, max_train=max_train, max_test=max_test, shuffle=False
#                 )
#                 results_df = pd.concat([results_df, pd.DataFrame({
#                     "session": [all_dfs_all[s].session[0]],
#                     "train_cond": ["inter"],
#                     "test_cond": ["inter"],
#                     "r2_folds": [r2s],
#                     "rmse_folds": [rmses],
#                     "r2_mean": [float(np.mean(r2s))],
#                     "rmse_mean": [float(np.mean(rmses))],
#                     "areas": [" ".join(areas)],
#                     "n_components": [n_components],
#                     "delays": [delays],
#                 })], ignore_index=True)
#                 results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test2_{bhv}.pkl")

#                 print("CV free0 -> free0")
#                 r2s, rmses, model_f0, _ = decode(
#                     Xf0, yf0, n_splits=n_splits, max_train=max_train, max_test=max_test,
#                     shuffle=True, window_size=1, random_state=42
#                 )
#                 results_df = pd.concat([results_df, pd.DataFrame({
#                     "session": [all_dfs_all[s].session[0]],
#                     "train_cond": ["free0"],
#                     "test_cond": ["free0"],
#                     "r2_folds": [r2s],
#                     "rmse_folds": [rmses],
#                     "r2_mean": [float(np.mean(r2s))],
#                     "rmse_mean": [float(np.mean(rmses))],
#                     "areas": [" ".join(areas)],
#                     "n_components": [n_components],
#                     "delays": [delays],
#                 })], ignore_index=True)
#                 results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test2_{bhv}.pkl")

#                 # --- within-condition CV ---
#                 print("CV perturb -> perturb")
#                 r2s, rmses, model_p, _ = decode(
#                     Xp, yp, n_splits=n_splits, max_train=max_train, max_test=max_test, shuffle=False
#                 )
#                 results_df = pd.concat([results_df, pd.DataFrame({
#                     "session": [all_dfs_all[s].session[0]],
#                     "train_cond": ["perturb"],
#                     "test_cond": ["perturb"],
#                     "r2_folds": [r2s],
#                     "rmse_folds": [rmses],
#                     "r2_mean": [float(np.mean(r2s))],
#                     "rmse_mean": [float(np.mean(rmses))],
#                     "areas": [" ".join(areas)],
#                     "n_components": [n_components],
#                     "delays": [delays],
#                 })], ignore_index=True)
#                 results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test2_{bhv}.pkl")

#                 # --- cross-condition generalisation (NEW) ---
#                 print("Cross train free0 -> test inter")
#                 r2, rmse, _ = train_test_cross(Xf0, yf0, Xi, yi)
#                 results_df = pd.concat([results_df, pd.DataFrame({
#                     "session": [all_dfs_all[s].session[0]],
#                     "train_cond": ["free0"],
#                     "test_cond": ["inter"],
#                     "r2_folds": [[r2]],     # keep schema consistent
#                     "rmse_folds": [[rmse]],
#                     "r2_mean": [float(r2)],
#                     "rmse_mean": [float(rmse)],
#                     "areas": [" ".join(areas)],
#                     "n_components": [n_components],
#                     "delays": [delays],
#                 })], ignore_index=True)
#                 results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test2_{bhv}.pkl")

#                 print("Cross train inter -> test free0")
#                 r2, rmse, _ = train_test_cross(Xi, yi, Xf0, yf0)
#                 results_df = pd.concat([results_df, pd.DataFrame({
#                     "session": [all_dfs_all[s].session[0]],
#                     "train_cond": ["inter"],
#                     "test_cond": ["free0"],
#                     "r2_folds": [[r2]],
#                     "rmse_folds": [[rmse]],
#                     "r2_mean": [float(r2)],
#                     "rmse_mean": [float(rmse)],
#                     "areas": [" ".join(areas)],
#                     "n_components": [n_components],
#                     "delays": [delays],
#                 })], ignore_index=True)
#                 results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test2_{bhv}.pkl")

#             results_df.to_pickle(f"/home/il620/earthquake-analysis/notebooks/results_test2_{bhv}.pkl")
