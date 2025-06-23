"""
Module to handle data for decoding
"""

import numpy as np
import pandas as pd
import pyaldata as pyal

import tools.dataTools as dt

from ..dimensionality import compute_pca
from ..dsp import preprocess


def _get_all_data_and_labels_from_df(
    df: pd.DataFrame,
    area: str,
    bhv: list,
    n_components: int,
) -> tuple:

    # Data
    rates = np.concatenate(df[f"{area}_rates"].values, axis=0)
    pca = compute_pca(rates, n_components)
    data = pca.fit_transform(rates)

    # Labels
    labels = df[bhv].to_numpy()

    return data, labels


def _get_trialdata_and_labels_from_df(
    df: pd.DataFrame,
    area: str,
    bhv: list,
    epoch: tuple,
    n_components: int,
) -> tuple:

    if epoch is not None:
        epoch = pyal.generate_epoch_fun(
            start_point_name="idx_sol_on",
            rel_start=int(epoch[0] / df.bin_size.values[0]),
            rel_end=int(epoch[1] / df.bin_size.values[0]),
        )

    arr_data, arr_bhv = dt.get_data_array(
        data_list=df,
        trial_cat="values_Sol_direction",
        epoch=epoch,
        area=area,
        bhv=bhv,
        n_components=n_components,
    )
    _, n_targets, n_trials, n_time, n_comp = arr_data.shape
    _, n_targets, n_trials, n_time, n_keypoints = arr_bhv.shape

    data = arr_data.reshape((n_targets * n_trials, n_time, n_comp))
    labels = arr_bhv.reshape((n_targets * n_trials, n_time, n_keypoints))

    return data, labels


class DecodingDataHandler:
    def __init__(
        self,
        data_dir: str,
        session: str,
    ):
        self.data_dir = data_dir
        self.session = session

    def load(self):

        df = pyal.load_pyaldata(self.data_dir + self.session[:4] + "/" + self.session)
        df = preprocess(df, only_trials=False, combine_time_bins=True)

        self.df = df
        return

    def clean(self):
        pass

    def pca_transform(self, area, condition, bhv, epoch, n_components):

        if condition == "trial":
            data, labels = _get_trialdata_and_labels_from_df(
                self.df, area, bhv, epoch, n_components
            )

        else:  # None condition -> all timepoints
            data, labels = _get_all_data_and_labels_from_df(self.df, area, bhv, n_components)

        self.data, self.labels = data, labels

        return

    def integrate(self):
        pass

    def validate(self):
        pass

    def wrangle(self, area, condition, bhv, epoch, n_components):

        self.load()

        self.clean()

        self.pca_transform(area, condition, bhv, epoch, n_components)

        self.integrate()

        self.validate()

        return

    def get_data_and_labels(
        self,
        area: str = "MOp",
        condition: str | None = None,
        bhv: list | None = None,
        n_components: int = 20,
        epoch: tuple = (-1, 3),
    ):

        self.wrangle(area, condition, bhv, epoch, n_components)

        return self.data, self.labels
