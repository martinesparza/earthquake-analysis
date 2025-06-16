"""
Module for running lstm experiments
"""

import pyaldata as pyal

import tools.decoding.lstm_tools as lstm
import tools.dsp as dsp


def load_data(cfg: dict):

    df = pyal.load_pyaldata(cfg["data_dir"] + cfg["session"][:4] + "/" + cfg["session"])
    df = dsp.preprocess(df, only_trials=False, combine_time_bins=cfg["combine_time_bins"])

    return df


def run_lstm_experiment(cfg: dict):

    # Load data
    df = load_data(cfg["data"])

    # Preprocess data
    data, labels = lstm.preprocess(df, cfg["preprocess"])

    # K-fold evaluation
    results = lstm.k_fold_eval(data, labels, cfg)

    # Save config

    return
