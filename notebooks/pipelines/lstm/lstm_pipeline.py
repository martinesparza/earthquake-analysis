import pickle
import sys

import pyaldata as pyal
import yaml

sys.path.append("../../")

from decoding.lstm.lstm import KeypointsLSTM

from tools.dsp.preprocessing import preprocess
from tools.params import Params

# Load data
data_dir = "/data/bnd-data/raw/"
session = "M062_2025_03_21_14_00"

df = pyal.load_pyaldata(data_dir + session[:4] + "/" + session)
df = preprocess(df, only_trials=False, combine_time_bins=False)

# Outputs
outputs_angles = [col for col in df.columns if col.endswith("angle")]
outputs_keypoints = [
    col
    for col in df.columns
    if (
        col.startswith("left")
        or col.startswith("right")
        or col.startswith("shoulder")
        or col.startswith("tail")
    )
    and not col.endswith("angle")
]

# Initialize r2
r2 = {}

for outputs, keypoint_or_angle in zip(
    [outputs_keypoints, outputs_angles], ["keypoints", "angles"]
):
    r2[keypoint_or_angle] = {}

    for area in ["MOp", "CP", "SSp", "VAL"]:

        model = KeypointsLSTM(
            n_input_components=20,
            outputs=outputs,
            hidden_size=300,
            epochs=5000,
        )

        model.kfold_evaluation(
            df,
            area=area,
            condition="trial",
            k=5,
            save_example=True,
            epoch=pyal.generate_epoch_fun(
                start_point_name="idx_sol_on",
                rel_start=int(Params.WINDOW_perturb_long[0] / df.bin_size.values[0]),
                rel_end=int(Params.WINDOW_perturb_long[1] / df.bin_size.values[0]),
            ),
        )
        r2[keypoint_or_angle][area] = model.r2

results_path = "/home/me24/repos/earthquake-analysis/results/lstm/"
with open(results_path + "lstm_r2_trial.pkl", "wb") as f:
    pickle.dump(r2, f)
