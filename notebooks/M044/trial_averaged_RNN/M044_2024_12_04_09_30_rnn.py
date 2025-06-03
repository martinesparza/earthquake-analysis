# === Imports ===
import os
import sys
import numpy as np
import pandas as pd
from scipy.io import savemat

# Import pyaldata
sys.path.append("/home/zms24/Desktop")
import PyalData.pyaldata as pyal

# Import functions in tools
project_root = os.path.abspath("earthquake/earthquake-analysis")
sys.path.insert(0, project_root) if project_root not in sys.path else None
from tools.curbd import curbd
from tools.dsp.preprocessing import preprocess
from tools.rnn_and_curbd import rnn as rnnz
from tools.rnn_and_curbd import plotting as pltz
from tools.rnn_and_curbd import model_analysis as analyz
from tools.rnn_and_curbd import curbd as curbdz

# Reload in case of updates (safe for development)
import importlib
importlib.reload(rnnz)
importlib.reload(pltz)
importlib.reload(analyz)
importlib.reload(curbdz)

# Set random seed
np.random.seed(44)

# # === Load Data ===
# data_dir = "/data/bnd-data/raw/M044/M044_2024_12_04_09_30"
# mat_file = "M044_2024_12_04_09_30_pyaldata.mat"
# fname = os.path.join(data_dir, mat_file)

# print(f"\nLoading data from: {fname}")
# df = pyal.mat2dataframe(fname, shift_idx_fields=True)

# # === Preprocessing ===
# print("Preprocessing data...")
# df_ = preprocess(df, only_trials=True)
# df_["M1_rates"] = [trial[:, 300:] for trial in df_["all_rates"]]
# df_["Dls_rates"] = [trial[:, :300] for trial in df_["all_rates"]]
# areas = ["M1_rates", "Dls_rates"]

# # === Metadata ===
# session_id = mat_file.replace("_pyaldata.mat", "")
# mouse = session_id.split('_')[0]
# BIN_SIZE = df['bin_size'][0]
# perturb_time_idx = df_.idx_sol_on[0]
# perturb_time_sec = perturb_time_idx * BIN_SIZE

# sol_angles = sorted(df_.values_Sol_direction.unique())
# trial_labels = [f"solenoid {angle}" for angle in sol_angles]
# num_trials = len(df_)

# print(f"Mouse: {mouse}")
# print(f"Number of trials: {num_trials}")
# print(f"Perturbation time (bins): {perturb_time_idx}, ({perturb_time_sec:.2f} sec)")

# # === RNN Setup ===
# dtFactor = 2
# print("Computing average activity by trial...")
# trial_avg_rates = rnnz.average_by_trial(df_, sol_angles)
# concat_rates = np.concatenate(trial_avg_rates, axis=0)
# trial_avg_activity = np.transpose(concat_rates)

# print(f"Averaged activity shape: {trial_avg_activity.shape}")
# reset_points = rnnz.get_reset_points(df_, trial_avg_activity, areas, dtFactor)
# regions_arr = rnnz.get_regions(df_, areas)

# print(f"Building RNN with {len(regions_arr)} region(s)")
# print("Regions:", [r[0] for r in regions_arr])

# # === Train RNN ===
# nRunTrain = 500
# print(f"\nRunning RNN training for {nRunTrain} runs with dtFactor={dtFactor}...")
# rnn_model, rnn_accuracy_fig = rnnz.run_rnn(
#     trial_avg_activity,
#     reset_points,
#     regions_arr,
#     df_,
#     mouse,
#     dtFactor=dtFactor,
#     nRunTrain=nRunTrain
# )
# print("RNN training complete.")

# # === Save Model ===
# save_dir = "/home/zms24/Desktop/rnn_models"
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, f"rnn_model_{session_id}.mat")

# print(f"\nSaving RNN model to: {save_path}")
# try:
#     rnn_to_save = rnn_model.copy()
#     if isinstance(rnn_to_save['params'].get('nonLinearity'), np.ufunc):
#         rnn_to_save['params']['nonLinearity'] = rnn_to_save['params']['nonLinearity'].__name__
#     savemat(save_path, {"rnn_model": rnn_to_save})
#     print("Model saved successfully.")
# except Exception as e:
#     print(f"Error saving RNN model: {e}")


