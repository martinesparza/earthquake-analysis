"""
Module for running lstm experiments
"""

import pickle

import pandas as pd
import pyaldata as pyal
import yaml

import tools.decoding.lstm_tools as lstm
import tools.dsp as dsp


def get_keypoints_angles(df: pd.DataFrame, keypoints_angles: list) -> list:
    """Returns keypoints and angles and check if they are present

    Args:
        df (pd.DataFrame): trial data
        keypoints_angles (list): list of keypoints

    Raises:
        ValueError: _description_

    Returns:
        list: keypoints and angles
    """

    keypoints_angles_ = []
    prefixes = ("right", "left", "shoulder", "hip", "tail")
    # Define prefixes

    if keypoints_angles == "all":
        # Filter column names
        keypoints_angles_ = [col for col in df.columns if col.startswith(prefixes)]

    elif keypoints_angles == "all_keypoints":

        # Filter column names
        keypoints_angles_ = [
            col
            for col in df.columns
            if col.startswith(prefixes) and not col.endswith("_angle")
        ]

    elif keypoints_angles == "all_angles":
        keypoints_angles_ = [col for col in df.columns if col.endswith("_angle")]

    else:
        for key_ang in keypoints_angles:
            if key_ang not in df.columns:
                raise ValueError(
                    f"The following column is missing from the DataFrame: {key_ang}"
                )
        keypoints_angles_ = keypoints_angles

    return keypoints_angles_


def load_data(cfg: dict) -> pd.DataFrame:
    """Load data from directory and run generic preprocessing

    Args:
        cfg (dict): config

    Returns:
        pd.DataFrame: pyaldata
    """

    df = pyal.load_pyaldata(cfg["data_dir"] + cfg["session"][:4] + "/" + cfg["session"])
    df = dsp.preprocess(df, only_trials=False, combine_time_bins=cfg["combine_time_bins"])

    return df


def run_lstm_experiment(cfg: dict):
    """Experimetn-specific running routing

    Args:
        cfg (dict): experiment config
    """

    # Initialize results
    results = {}

    # Load data
    df = load_data(cfg["data"])

    keypoints_angles = get_keypoints_angles(df, cfg["preprocess"]["bhv"])

    # Iterate keypoints
    for keypoint_angle in keypoints_angles:

        if not isinstance(keypoint_angle, list):
            keypoint_angle = [keypoint_angle]

        results["-".join(keypoint_angle)] = {}

        # Iterate area
        for area in cfg["preprocess"]["areas"]:

            # Preprocess data
            data, labels = lstm.preprocess(
                df,
                keypoint_angle,
                area,
                cfg["preprocess"],
            )

            # K-fold evaluation
            results["-".join(keypoint_angle)][area] = lstm.k_fold_eval(
                data=data, labels=labels, area=area, keypoint_angle=keypoint_angle, cfg=cfg
            )

    # Save config
    filename = f"{cfg['name']}_config.yaml"
    with open(cfg["results"]["results_dir"] + "/" + filename, "w") as file:
        yaml.dump(cfg, file)

    filename = f"{cfg['name']}_results.pkl"
    with open(cfg["results"]["results_dir"] + "/" + filename, "wb") as f:
        pickle.dump(results, f)

    return
