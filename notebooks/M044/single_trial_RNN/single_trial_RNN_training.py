import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append("../../")

import pyaldata as pyal
from tools.dsp.preprocessing import preprocess
from tools.rnn_and_curbd.RNN_functions import get_reset_points, get_regions, RNN

def main(file_path, dtFactor, tauRNN, nRunTrain, ampInWN):
    np.random.seed(44)

    #### Load data ####
    mat_file = os.path.basename(file_path)
    data_dir = os.path.dirname(file_path)

    df = pyal.mat2dataframe(file_path, shift_idx_fields=True)
    mouse = mat_file.split('_')[0]

    #### Preprocess data ####
    df_ = preprocess(df, only_trials=True)
    df_ = pyal.select_trials(df_, "idx_trial_end > 30365")  # Remove first 5 minutes because the switch was off

    brain_areas = ["Dls_rates", "M1_rates"]
    df_["M1_rates"] = [df_["all_rates"][i][:, 300:] for i in range(len(df_))]
    df_["Dls_rates"] = [df_["all_rates"][i][:, 0:300] for i in range(len(df_))]

    df_['trial_length'] = df_['all_rates'].apply(lambda x: x.shape[0])

    #### Prepare data for RNN ####
    concat_trials = pyal.concat_trials(df_, signal="all_rates")
    reset_points, trial_len = get_reset_points(df_, concat_trials, brain_areas, dtFactor)
    activity = np.transpose(concat_trials)
    regions = get_regions(df_, brain_areas)

    #### RNN training ####
    rnn_model = RNN(
        activity, regions, df_, mouse, graph=False,
        resetPoints = reset_points,
        dtFactor=dtFactor,
        ampInWN=ampInWN,
        tauRNN=tauRNN,
        nRunTrain=nRunTrain
    )
    # Create filenames with mouse ID
    df_filename = f"df_preprocessed_{mouse}_single_trial.mat"
    rnn_filename = f"rnn_model_output_{mouse}_single_trial.mat"

    # Save df_
    df_dict = {col: df_[col].to_numpy() for col in df_.columns}
    savemat(df_filename, df_dict)

    # Save RNN model
    try:
        savemat(rnn_filename, rnn_model)
    except Exception as e:
        print(f"Error saving RNN model for mouse {mouse}:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN on neural activity data.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the .mat file")
    parser.add_argument("--dtFactor", type=int, default=4, help="Time downsampling factor")
    parser.add_argument("--tauRNN", type=float, default=0.2, help="RNN time constant")
    parser.add_argument("--nRunTrain", type=int, default=200, help="Number of training runs")
    parser.add_argument("--ampInWN", type=float, default=0.001, help="Amplitude of white noise input")

    args = parser.parse_args()
    main(
        file_path=args.file_path,
        dtFactor=args.dtFactor,
        tauRNN=args.tauRNN,
        nRunTrain=args.nRunTrain,
        ampInWN=args.ampInWN
    )