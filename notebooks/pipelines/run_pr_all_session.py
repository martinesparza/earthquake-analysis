import pickle
import sys

import numpy as np
import pyaldata as pyal

sys.path.append("../../")

import tools.dimensionality.participation as part
import tools.dsp as dsp
from tools.params import Params

prs_with_variable_neurons = {}
areas = ["MOp", "SSp", "CP", "VAL"]

SESSIONS_ALL = [
    "M061_2025_03_04_10_00",
    "M061_2025_03_05_14_00",
    # "M061_2025_03_06_14_00",
    "M062_2025_03_20_14_00",
    "M062_2025_03_21_14_00",
    "M063_2025_03_13_14_00",
    "M063_2025_03_14_15_30",
    "M078_2025_08_06_15_00",
    "M086_2025_12_10_15_00",
]

RESULTS_DIR = "/data/equake_results/pr_all_sessions/"

# ========================================== Linear ===========================================

data_dir = "/data/bnd-data/raw/"
step_size = 10
n_iter = 20


for session in SESSIONS_ALL:
    print(f"Session: {session}")
    session_results = {}
    df_tr, _ = dsp.load_and_process_session(session, data_dir=data_dir)

    for area in areas:
        print(f"Processing area {area}")

        session_results[area] = {}

        tmp_df = pyal.restrict_to_interval(df_tr, epoch_fun=Params.perturb_epoch_long)
        rates = pyal.concat_trials(tmp_df, f"{area}_rates")
        session_results[area]["neurons"] = np.arange(5, rates.shape[1] + 1, step_size)
        session_results[area]["PR"] = part.get_pr_for_subsets_of_neurons(
            rates, n_iter=n_iter, step=step_size
        )

    with open(RESULTS_DIR + f"{session}.pkl", "wb") as f:
        pickle.dump(session_results, f)
    print(f"Saved {session}")
