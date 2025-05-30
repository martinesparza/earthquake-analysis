import pickle
import sys

import numpy as np
import pyaldata as pyal

sys.path.append("../../")

import tools.dimensionality.participation as part
from tools.dsp.preprocessing import preprocess
from tools.params import Params

prs_with_variable_neurons = {}
areas = ["MOp", "SSp", "CP", "VAL"]

sessions = [
    # "M046_2024_12_18_16_00",
    # "M046_2024_12_19_13_30",
    # "M061_2025_03_05_14_00",
    "M061_2025_03_06_14_00",
    "M062_2025_03_19_14_00",
    "M062_2025_03_20_14_00",
    "M062_2025_03_21_14_00",
    "M063_2025_03_12_14_00",
    "M063_2025_03_13_14_00",
    # "M063_2025_03_14_15_30",
]

# ========================================== Linear ===========================================

data_dir = "/data/bnd-data/raw/"
step_size = 10
n_iter = 20

for session in sessions:
    prs_with_variable_neurons[session] = {}
    df = pyal.load_pyaldata(data_dir + session[:4] + "/" + session)
    df = preprocess(df, only_trials=False)

    for area in areas:
        print(f"Processing area {area}")

        prs_with_variable_neurons[session][area] = {}
        for condition in ["free", "intertrial", "trial"]:
            print(f"\tProcessing condition {condition}")
            tmp_df = pyal.select_trials(df, df.trial_name == condition)

            if condition == "free":

                for idx, label in enumerate(["free0", "free1"]):
                    prs_with_variable_neurons[session][area][f"{label}"] = {}

                    rates = tmp_df[f"{area}_rates"].values[idx]

                    prs_with_variable_neurons[session][area][f"{label}"]["neurons"] = (
                        np.arange(5, rates.shape[1] + 1, step_size)
                    )
                    prs_with_variable_neurons[session][area][f"{label}"]["PR"] = (
                        part.get_pr_for_subsets_of_neurons(rates, niter=n_iter)
                    )

            elif condition == "trial":
                prs_with_variable_neurons[session][area][condition] = {}

                tmp_df = pyal.restrict_to_interval(
                    tmp_df, epoch_fun=Params.perturb_epoch_long
                )
                rates = pyal.concat_trials(tmp_df, f"{area}_rates")
                prs_with_variable_neurons[session][area][condition]["neurons"] = np.arange(
                    5, rates.shape[1] + 1, step_size
                )
                prs_with_variable_neurons[session][area][condition]["PR"] = (
                    part.get_pr_for_subsets_of_neurons(rates, niter=n_iter)
                )

            elif condition == "intertrial":
                prs_with_variable_neurons[session][area][condition] = {}

                rates = pyal.concat_trials(tmp_df, f"{area}_rates")
                prs_with_variable_neurons[session][area][condition]["neurons"] = np.arange(
                    5, rates.shape[1] + 1, step_size
                )
                prs_with_variable_neurons[session][area][condition]["PR"] = (
                    part.get_pr_for_subsets_of_neurons(rates, niter=n_iter)
                )


del df
results_path = "/home/me24/repos/earthquake-analysis/results/"
with open(results_path + "pr_all_sess_diff_num_neurons_long_perturb.pkl", "wb") as f:
    pickle.dump(prs_with_variable_neurons, f)
