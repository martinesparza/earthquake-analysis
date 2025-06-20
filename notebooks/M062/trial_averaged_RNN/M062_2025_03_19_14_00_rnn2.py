# === Standard Library Imports ===
import os
import sys
import importlib

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
sol_durations = sorted(df_.values_Sol_duration.unique())
trial_labels = [f"solenoid {angle}" for angle in sol_angles]
num_trials = len(df_)

print(f"Mouse: {mouse}")
print(f"Duration labels", sol_durations)
print(f"Number of trials: {num_trials}")
print(f"Perturbation time (bins): {perturb_time_idx}, ({perturb_time_sec:.2f} sec)")

# === RNN Setup ===
dtFactor = 2
print("Computing average activity by trial...")
trial_avg_rates, trial_avg_labels = rnnz.average_by_categories(df_, sol_angles, sol_durations)
concat_rates = np.concatenate(trial_avg_rates, axis=0)
trial_avg_activity = np.transpose(concat_rates)

print(f"Averaged activity shape: {trial_avg_activity.shape}")
reset_points = rnnz.get_reset_points(df_, trial_avg_activity, areas, dtFactor)
regions_arr = rnnz.get_regions(df_, areas)
print(reset_points)

print(f"Building RNN with {len(regions_arr)} region(s)")
print("Regions:", [r[0] for r in regions_arr])

# === Train RNN ===
nRunTrain = 500
print(f"\nRunning RNN training for {nRunTrain} runs with dtFactor={dtFactor}...")
rnn_model, rnn_accuracy_fig = rnnz.run_rnn(
    trial_avg_activity,
    reset_points,
    regions_arr,
    df_,
    mouse,
    dtFactor=dtFactor,
    nRunTrain=nRunTrain
)
print("RNN training complete.")

# === Save Model ===
save_dir = "/home/zms24/Desktop/rnn_models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"rnn2_model_{session_id}.mat")

print(f"\nSaving RNN model to: {save_path}")
try:
    rnn_to_save = rnn_model.copy()
    rnn_to_save['categories'] = trial_avg_labels
    if isinstance(rnn_to_save['params'].get('nonLinearity'), np.ufunc):
        rnn_to_save['params']['nonLinearity'] = rnn_to_save['params']['nonLinearity'].__name__
    savemat(save_path, {"rnn_model": rnn_to_save})
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving RNN model: {e}")