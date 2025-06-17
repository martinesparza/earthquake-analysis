# === Standard Library Imports ===
import os
import sys
import importlib
import time

# === Third-Party Imports ===
import numpy as np
import pandas as pd
from scipy.io import savemat

# === PyalData Import ===
sys.path.append("/home/zms24/Desktop")  # Adjust if needed
import PyalData.pyaldata as pyal  # type: ignore

# === Tools Package Import (Relative to Script Location) ===
# Dynamically determine the project root relative to this script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import custom tool modules
from tools.curbd import curbd
from tools.dsp.preprocessing import preprocess
from tools.rnn_and_curbd import rnn as rnnz
from tools.rnn_and_curbd import plotting as pltz
from tools.rnn_and_curbd import model_analysis as analyz
from tools.rnn_and_curbd import curbd as curbdz

# Reload modules (for development convenience)
importlib.reload(rnnz)
importlib.reload(pltz)
importlib.reload(analyz)
importlib.reload(curbdz)

# === Set Global Random Seed ===
np.random.seed(44)

# === Load Data ===
data_dir = "/data/raw/M062/M062_2025_03_19_14_00"
mat_file_0= "M062_2025_03_19_14_00_pyaldata_0.mat"
mat_file_1= "M062_2025_03_19_14_00_pyaldata_1.mat"
mat_file_2= "M062_2025_03_19_14_00_pyaldata_2.mat"

fname0 = os.path.join(data_dir, mat_file_0)
fname1 = os.path.join(data_dir, mat_file_1)
fname2 = os.path.join(data_dir, mat_file_2)

print(f"\nLoading data from: {fname0}, {fname1} and {fname2}")

df0 = pyal.mat2dataframe(fname0, shift_idx_fields=True)
df1 = pyal.mat2dataframe(fname1, shift_idx_fields=True)
df2 = pyal.mat2dataframe(fname2, shift_idx_fields=True)
df = pd.concat([df0, df1, df2], ignore_index=True)
df = df.drop(columns="all_spikes") # the content is incorrect

# === Preprocessing ===
print("Preprocessing data...")
df_ = preprocess(df, only_trials=True)
BIN_SIZE = df_['bin_size'][0]
# get 'all_rates' column
areas =[ "MOp_rates", "SSp_rates", "CP_rates", "VAL_rates"]
df_ = pyal.merge_signals(df_, areas, "all_rates")

# correct trial length - this is an error in pyaldata
df_['trial_length'] = (df_['trial_length'] / (BIN_SIZE * 100)).astype(int)
df_ = df_[df_['trial_length'] == 200]

# === Metadata ===
session_id = mat_file_0.replace("_pyaldata_0.mat", "")
mouse = session_id.split('_')[0]
perturb_time_idx = df_.idx_sol_on[0]
perturb_time_sec = perturb_time_idx * BIN_SIZE

sol_angles = sorted(df_.values_Sol_direction.unique())
trial_labels = [f"solenoid {angle}" for angle in sol_angles]
num_trials = len(df_)

print(f"Mouse: {mouse}")
print(f"Number of trials: {num_trials}")
print(f"Perturbation time (bins): {perturb_time_idx}, ({perturb_time_sec:.2f} sec)")

# === Single trial RNN setup ===
print("Trimming trials...")
start_bin = perturb_time_idx - (int(1 / BIN_SIZE)) # 1 second before perturbation
length_bin = (int(4 / BIN_SIZE)) # 4 second time window 

df_interval = rnnz.restrict_time_interval(df_, areas+["all_rates"], start_bin, length_bin) # its actually only the 'all_rates' column that will be useful
df_interval = df_interval[df_interval['trial_length'] == length_bin]
num_of_trials = np.arange(10, 101, 10) # number of trials to train the single-trial RNN on :) 

times = {} 

for num in num_of_trials:
    # RNN parameters
    dtFactor = 2
    nRunTrain = 500

    print(f"Computing single trial activity for {num} trials...")
    selected_data = df_interval[:num]['all_rates']
    concat_rates = np.concatenate(selected_data, axis=0)
    single_trial_activity = np.transpose(concat_rates)

    print(f"Single-trial activity shape: {single_trial_activity.shape}")
    reset_points = rnnz.get_reset_points(df_interval, single_trial_activity, areas, dtFactor)
    print(reset_points)
    regions_arr = rnnz.get_regions(df_interval, areas)

    print(f"Building RNN with {len(regions_arr)} region(s)")
    print("Regions:", [r[0] for r in regions_arr])
    start_time = time.time()  # Start timer

    print(f"\nRunning RNN training for {nRunTrain} runs with dtFactor={dtFactor} and {num} trial number...")
    rnn_model, rnn_accuracy_fig = rnnz.run_rnn(
        single_trial_activity,
        reset_points,
        regions_arr,
        df_,
        mouse,
        dtFactor=dtFactor,
        nRunTrain=nRunTrain
    )
    print("RNN training complete.")
    end_time = time.time()  # End timer
    elapsed = end_time - start_time
    times[num] = elapsed
    print(f"Time taken for {num} trials: {elapsed:.2f} seconds")

    # === Save Model ===
    save_dir = "/home/zms24/Desktop/rnn_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"single_{num}_rnn_model_{session_id}.mat")

    print(f"\nSaving RNN model to: {save_path}")
    try:
        rnn_to_save = rnn_model.copy()
        if isinstance(rnn_to_save['params'].get('nonLinearity'), np.ufunc):
            rnn_to_save['params']['nonLinearity'] = rnn_to_save['params']['nonLinearity'].__name__
        savemat(save_path, {"rnn_model": rnn_to_save})
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving RNN model: {e}")

print("\n=== RNN Training Time Summary ===")
for num, seconds in times.items():
    print(f"{num} trials: {seconds:.2f} seconds")

# === Save timing summary ===
summary_path = os.path.join(save_dir, f"rnn_training_time_summary_{session_id}.txt")

with open(summary_path, "w") as f:
    f.write("=== RNN Training Time Summary ===\n")
    for num, seconds in times.items():
        f.write(f"{num} trials: {seconds:.2f} seconds\n")

print("\n=== THE END! ===")
