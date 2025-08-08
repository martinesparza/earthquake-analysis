# === Standard Library Imports ===
import os
import sys
import importlib

# === Third-Party Imports ===
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.signal import find_peaks

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
data_dir = "/data/raw/M062/M062_2025_03_21_14_00"
mat_file_0= "M062_2025_03_21_14_00_pyaldata_0.mat"
mat_file_1= "M062_2025_03_21_14_00_pyaldata_1.mat"
mat_file_2= "M062_2025_03_21_14_00_pyaldata_2.mat"

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
df_ = preprocess(df)
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

# === Detecting gait cycles === 
def detect_gait_cycles(angle_data, distance=4, peak_height=80, peak_threshold=0.3):
    """
    Detects gait cycles in a 1D joint angle signal based on peak detection.

    Parameters:
    - angle_data : list or np.ndarray
        1D sequence of joint angles.
    - distance : int
        Minimum distance between peaks (in frames).
    - peak_height : float
        Minimum height of detected peaks.
    - peak_threshold : float
        Threshold to use for peak detection.

    Returns:
    - gait_cycles : list of tuples
        List of (start_idx, end_idx) tuples representing gait cycles.
    """
    angle_seq = np.array(angle_data)

    # Detect peaks
    peaks, _ = find_peaks(angle_seq, distance=distance, height=peak_height, threshold=peak_threshold)

    # Define gait cycles between successive peaks
    gait_cycles = [(peaks[i], peaks[i + 1]) for i in range(len(peaks) - 1)]

    return gait_cycles

gait_rows = []
gait_id_counter = 0

for trial_idx, trial_row in df_.iterrows():
    angle_data = trial_row['right_ankle_angle']
    
    # Detect gait cycles
    gait_cycles = detect_gait_cycles(angle_data, distance=4, )
    
    sol_on_idx = trial_row['idx_sol_on']

    # Handle empty arrays or single-element arrays
    if isinstance(sol_on_idx, (list, np.ndarray)):
        if len(sol_on_idx) == 0 or np.array(sol_on_idx).size == 0:
            sol_on_idx = None
        elif np.array(sol_on_idx).size == 1:
            sol_on_idx = np.array(sol_on_idx).item()

    for start, end in gait_cycles:
        # Only skip cycles after perturbation if sol_on_idx is defined
        if sol_on_idx is not None and end >= sol_on_idx:
            continue

        # Get neural data slices during the gait cycle
        gait_VAL = trial_row['VAL_rates'][start:end]
        gait_SSp = trial_row['SSp_rates'][start:end]
        gait_CP = trial_row['CP_rates'][start:end]
        gait_MOp = trial_row['MOp_rates'][start:end]
        gait_all = trial_row['all_rates'][start:end]
        gait_len = end - start
        
        # Append new row
        gait_rows.append({
            'gait_id': gait_id_counter,
            'trial_id': trial_row['trial_id'],
            'trial_name': trial_row['trial_name'],
            'trial_length': trial_row['trial_length'],
            'gait_length': gait_len,
            'bin_size': trial_row['bin_size'],
            'values_Sol_direction': trial_row['values_Sol_direction'],
            'sol_level_id': trial_row['sol_level_id'],
            'VAL_rates': gait_VAL,
            'SSp_rates': gait_SSp,
            'CP_rates': gait_CP,
            'MOp_rates': gait_MOp,
            'all_rates': gait_all,
            'gait_start_idx': start,
            'gait_end_idx': end
        })
        
        gait_id_counter += 1

df_gait = pd.DataFrame(gait_rows)

# === Asign solenoind angles for averaging purposes === 

## for the trials the solenoid angles are 0 - 11
## lets make the free period solenoid angles 12 - 15
## lets make the intertrial solenoid angles 16 - 19
## and lastly the 2nd free running period 20 - 23

for idx, row in df_gait.iterrows():
    if row['trial_name'] == 'free' and row['trial_id'] == 0: # first free period
        df_gait.at[idx, 'values_Sol_direction'] = np.random.randint(12, 16)
    elif row['trial_name'] == 'intertrial': # intertrial period
        df_gait.at[idx, 'values_Sol_direction'] = np.random.randint(16, 20)
    elif row['trial_name'] == 'free' and row['trial_id'] != 0: # last free period
        df_gait.at[idx, 'values_Sol_direction'] = np.random.randint(20, 24)

# trim all _rates columns
brain_areas = areas.copy()
brain_areas.append("all_rates")

for idx in df_gait.index:
    for area in brain_areas:
        arr = df_gait.at[idx, area]
        if arr is not None and isinstance(arr, np.ndarray):
            df_gait.at[idx, area] = arr[:4]
        else:
            print(f"Wrong spike array format at index: {idx}!")

# change gait_length column accordingly
for idx in df_gait.index:
    df_gait.at[idx, 'gait_length'] = len(df_gait.at[idx, 'MOp_rates'])

sol_angles = sorted(df_gait.values_Sol_direction.unique())
trial_labels = [f"solenoid {angle}" for angle in sol_angles]
num_trials = len(df_gait)

print(f"Mouse: {mouse}")
print(f"Number of trials: {num_trials}")
print(f"Perturbation time (bins): {perturb_time_idx}, ({perturb_time_sec:.2f} sec)")

# === RNN Setup ===
dtFactor = 2
print("Computing average activity by trial...")
trial_avg_rates = rnnz.average_by_trial(df_, sol_angles)
concat_rates = np.concatenate(trial_avg_rates, axis=0)
trial_avg_activity = np.transpose(concat_rates)

print(f"Averaged activity shape: {trial_avg_activity.shape}")
reset_points = rnnz.get_reset_points(df_, trial_avg_activity, areas, dtFactor)
regions_arr = rnnz.get_regions(df_, areas)

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
save_path = os.path.join(save_dir, f"rnn_model_{session_id}_gait.mat")

print(f"\nSaving RNN model to: {save_path}")
try:
    rnn_to_save = rnn_model.copy()
    if isinstance(rnn_to_save['params'].get('nonLinearity'), np.ufunc):
        rnn_to_save['params']['nonLinearity'] = rnn_to_save['params']['nonLinearity'].__name__
    savemat(save_path, {"rnn_model": rnn_to_save})
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving RNN model: {e}")