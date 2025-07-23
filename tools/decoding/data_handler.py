"""
Module to handle data for decoding
"""

import numpy as np
import pandas as pd
import pyaldata as pyal

import tools.dataTools as dt

from ..dimensionality import compute_pca
from ..dsp import preprocess


def _get_all_data_and_kinematics_from_df(
    df: pd.DataFrame,
    area: str,
    bhv: list,
    n_components: int,
) -> tuple:
    """Parses data from entire recording

    Args:
        df (pd.DataFrame): pyaldata
        area (str): brain area
        bhv (list): list of keypoints
        n_components (int): components for pca

    Returns:
        tuple: data, labels
    """

    # Data
    rates = np.concatenate(df[f"{area}_rates"].values, axis=0)
    pca = compute_pca(rates, n_components)
    data = pca.fit_transform(rates)

    # Labels
    labels = None
    if bhv is not None:
        df = dt.add_bhv(df, bhv)
        labels = np.concatenate(df["bhv"])

    return data, labels


def _get_trialdata_and_kinematics_from_df(
    df: pd.DataFrame,
    area: str,
    kinematics: list,
    epoch: tuple,
    n_components: int,
    trial_cat: str | None,
) -> tuple:
    """Parses data from perturbation trials

    Args:
        df (pd.DataFrame): pyaldata
        area (str): brain
        bhv (list): list of keypoints
        epoch (tuple): epoch to crop around
        n_components (int): components for pca

    Returns:
        tuple: data, labels
    """

    if epoch is not None:
        epoch = pyal.generate_epoch_fun(
            start_point_name="idx_sol_on",
            rel_start=int(epoch[0] / df.bin_size.values[0]),
            rel_end=int(epoch[1] / df.bin_size.values[0]),
        )

    arr_data_arr_kinematics = dt.get_data_array(
        data_list=pyal.select_trials(df, df.trial_name == "trial"),
        trial_cat="values_Sol_direction" if trial_cat is None else trial_cat,
        epoch=epoch,
        area=area,
        bhv=kinematics,
        n_components=n_components,
    )
    if isinstance(arr_data_arr_kinematics, tuple):  # Bhv present
        data, kinematics = arr_data_arr_kinematics

        _, n_targets, n_trials, n_time, n_comp = data.shape
        _, n_targets, n_trials, n_time, n_keypoints = kinematics.shape
        data = data.reshape((n_targets * n_trials, n_time, n_comp))
        kinematics = kinematics.reshape((n_targets * n_trials, n_time, n_keypoints))

    else:
        data = arr_data_arr_kinematics
        _, n_targets, n_trials, n_time, n_comp = data.shape
        data = data.reshape((n_targets * n_trials, n_time, n_comp))
        kinematics = None

    return data, kinematics


class DecodingDataHandler:
    """
    Data handling class
    """

    def __init__(self, data_dir: str, session: str, combine_time_bins: bool = False):
        self.data_dir = data_dir
        self.session = session
        self.df = self.load(combine_time_bins)

    def load(self, combine_time_bins):
        """Load data from pyaldata file"""

        df = pyal.load_pyaldata(self.data_dir + self.session[:4] + "/" + self.session)
        df = preprocess(df, only_trials=False, combine_time_bins=combine_time_bins)

        return df

    def clean(self):
        pass

    def pca_transform(
        self,
        area: str,
        condition: str,
        kinematics: list | None,
        epoch: tuple,
        n_components: int,
        trial_cat: str | None,
        query: str = None,
    ):
        """Apply pca transformation to data

        Args:
            area (str): Brain area
            condition (str): Trial, free, intertrial
            epoch (tuple): tuple of start and end to crop
            n_components (int): components to use in pca
        """

        if query is not None:
            df = pyal.select_trials(self.df, query)
        else:
            df = self.df

        if condition == "trial":
            data, kinematics = _get_trialdata_and_kinematics_from_df(
                df=df,
                area=area,
                kinematics=kinematics,
                epoch=epoch,
                n_components=n_components,
                trial_cat=trial_cat,
            )

        else:  # None condition -> all timepoints
            data, kinematics = _get_all_data_and_kinematics_from_df(
                df=self.df, area=area, bhv=kinematics, n_components=n_components
            )

        return data, kinematics

    def validate(self):
        pass

    def wrangle(
        self,
        area: str,
        condition: str,
        kinematics: list | None,
        epoch: tuple,
        n_components: int,
        trial_cat: str | None,
        query: str = None,
    ):
        """
        Main pipeline
        """

        self.clean()

        data, kinematics = self.pca_transform(
            area, condition, kinematics, epoch, n_components, trial_cat, query
        )

        self.validate()

        return data, kinematics

    def get_data_and_kinematics(
        self,
        area: str = "MOp",
        condition: str | None = None,
        kinematics: list | None = None,
        n_components: int = 20,
        epoch: tuple | None = (-1, 3),
        trial_cat: str = None,
        query: str = None,
    ) -> tuple:
        """Entry point function to generate data and labels

        Args:
            area (str, optional): Area to parse. Defaults to "MOp".
            condition (str | None, optional): Condition to parse. Defaults to None.
            bhv (list | None, optional): List of keypoints. Defaults to None.
            n_components (int, optional): components to use in pca. Defaults to 20.
            epoch (tuple, optional): Start and end of window. Defaults to (-1, 3).

        Returns:
            tuple: data, labels
        """
        data, kinematics = self.wrangle(
            area, condition, kinematics, epoch, n_components, trial_cat, query
        )

        return data if kinematics is None else (data, kinematics)

    def get_feature(self, condition: str, feature: str):
        df = pyal.select_trials(self.df, self.df.trial_name == condition)
        if isinstance(df.loc[0][feature], (list, tuple, np.ndarray)):
            feature_ = np.concatenate(df[feature].values)
        else:
            feature_ = df[feature].values

        return feature_
