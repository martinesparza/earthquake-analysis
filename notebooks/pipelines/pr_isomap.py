import os
import sys

sys.path.append("../../")
import pickle

import numpy as np
import pyaldata as pyal

from tools.dimensionality.participation import get_pr_for_subsets_of_neurons, isomap_pr
from tools.dsp.preprocessing import preprocess
from tools.params import Params

isomap_prs = {}
areas = ["MOp", "SSp", "CP", "VAL"]


data_dir = "/data/bnd-data/raw/"
results_path = "/home/me24/repos/earthquake-analysis/results/"


sessions = [
    # "M046_2024_12_18_16_00",
    # "M046_2024_12_19_13_30",
    # "M061_2025_03_05_14_00",
    # "M061_2025_03_06_14_00",
    # "M062_2025_03_19_14_00",
    "M062_2025_03_20_14_00",
    # "M062_2025_03_21_14_00",
    # "M063_2025_03_12_14_00",
    # "M063_2025_03_13_14_00",
    # "M063_2025_03_14_15_30",
]

step_size = 10
n_iter = 5

for session in sessions:

    isomap_prs[session] = {}
    df = pyal.load_pyaldata(data_dir + session[:4] + "/" + session)
    df = preprocess(df, only_trials=False)

    for area in areas:
        print(f"Processing area {area}")

        isomap_prs[session][area] = {}
        for condition in ["free", "intertrial", "trial"]:
            print(f"\tProcessing condition {condition}")

            tmp_df = pyal.select_trials(df, df.trial_name == condition)

            if condition == "free":

                for idx, label in enumerate(["free0", "free1"]):
                    isomap_prs[session][area][f"{label}"] = {}

                    rates = tmp_df[f"{area}_rates"].values[idx]

                    isomap_prs[session][area][f"{label}"]["neurons"] = np.arange(
                        5, rates.shape[1] + 1, step_size
                    )
                    isomap_prs[session][area][f"{label}"]["PR"] = (
                        get_pr_for_subsets_of_neurons(rates, niter=n_iter, linear=False)
                    )

            elif condition == "trial":
                isomap_prs[session][area][condition] = {}

                tmp_df = pyal.restrict_to_interval(
                    tmp_df, epoch_fun=Params.perturb_epoch_long
                )
                rates = pyal.concat_trials(tmp_df, f"{area}_rates")
                isomap_prs[session][area][condition]["neurons"] = np.arange(
                    5, rates.shape[1] + 1, step_size
                )
                isomap_prs[session][area][condition]["PR"] = get_pr_for_subsets_of_neurons(
                    rates, niter=n_iter, linear=False
                )

            elif condition == "intertrial":
                isomap_prs[session][area][condition] = {}

                rates = pyal.concat_trials(tmp_df, f"{area}_rates")
                isomap_prs[session][area][condition]["neurons"] = np.arange(
                    5, rates.shape[1] + 1, step_size
                )
                isomap_prs[session][area][condition]["PR"] = get_pr_for_subsets_of_neurons(
                    rates, niter=n_iter, linear=False
                )

del df
results_path = "/home/me24/repos/earthquake-analysis/results/isomap/"
with open(results_path + f"isomap_pr_long_perturb_{session}.pkl", "wb") as f:
    pickle.dump(isomap_prs, f)
