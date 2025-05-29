import pickle
import sys

import numpy as np
import pyaldata as pyal

sys.path.append("../../")

import tools.dimensionality.participation as part
from tools.dsp.preprocessing import preprocess
from tools.params import Params

pcs_vaf = {}
areas = ["MOp", "SSp", "CP", "VAL"]

sessions = [
    # "M046_2024_12_18_16_00",
    # "M046_2024_12_19_13_30",
    # "M061_2025_03_05_14_00",
    # "M061_2025_03_06_14_00",
    "M062_2025_03_19_14_00",
    # "M062_2025_03_20_14_00",
    # "M062_2025_03_21_14_00",
    # "M063_2025_03_12_14_00",
    # "M063_2025_03_13_14_00",
    # "M063_2025_03_14_15_30",
]

# ========================================== Linear ===========================================

data_dir = "/data/bnd-data/raw/"
step_size = 10
n_iter = 20

for session in sessions:
    pcs_vaf[session] = {}
    df = pyal.load_pyaldata(data_dir + session[:4] + "/" + session)
    df = preprocess(df, only_trials=False)

    for area in areas:
        print(f"Processing area {area}")

        pcs_vaf[session][area] = {}
        for condition in ["free", "intertrial", "trial"]:
            print(f"\tProcessing condition {condition}")
            tmp_df = pyal.select_trials(df, df.trial_name == condition)

            if condition == "free":

                for idx, label in enumerate(["free0", "free1"]):
                    pcs_vaf[session][area][f"{label}"] = {}

                    rates = tmp_df[f"{area}_rates"].values[idx]

                    pcs_vaf[session][area][f"{label}"]["neurons"] = np.arange(
                        5, rates.shape[1] + 1, step_size
                    )
                    pcs_vaf[session][area][f"{label}"]["dim"] = (
                        part.get_pcs_percentVAF_subsets_neurons(rates, niter=n_iter)
                    )

            elif condition == "trial":
                pcs_vaf[session][area][condition] = {}

                tmp_df = pyal.restrict_to_interval(
                    tmp_df, epoch_fun=Params.perturb_epoch_long
                )
                rates = pyal.concat_trials(tmp_df, f"{area}_rates")
                pcs_vaf[session][area][condition]["neurons"] = np.arange(
                    5, rates.shape[1] + 1, step_size
                )
                pcs_vaf[session][area][condition]["dim"] = (
                    part.get_pcs_percentVAF_subsets_neurons(rates, niter=n_iter)
                )

            elif condition == "intertrial":
                pcs_vaf[session][area][condition] = {}

                rates = pyal.concat_trials(tmp_df, f"{area}_rates")
                pcs_vaf[session][area][condition]["neurons"] = np.arange(
                    5, rates.shape[1] + 1, step_size
                )
                pcs_vaf[session][area][condition]["dim"] = (
                    part.get_pcs_percentVAF_subsets_neurons(rates, niter=n_iter)
                )


del df
results_path = "/home/me24/repos/earthquake-analysis/results/"
with open(results_path + "pcs_vaf.pkl", "wb") as f:
    pickle.dump(pcs_vaf, f)
