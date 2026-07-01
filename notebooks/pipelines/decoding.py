"""
Encoding decoding of every keypoint
"""

import pickle
import numpy as np
import pyaldata as pyal
import sys


import scipy
from sklearn.linear_model import RidgeCV

sys.path.append("../../")

import tools.dataTools as dt
import tools.decoding as decode
import tools.dsp as dsp
import tools.subspaces as subspaces

AREAS = ["MOp", "SSp", "CP", "VAL"]
ALL_BHV_FIELDS = [
    "left_ankle",
    "left_elbow",
    "left_foot",
    "left_knee",
    "left_paw",
    "left_shoulder",
    "left_wrist",
    "right_ankle",
    "right_elbow",
    "right_foot",
    "right_knee",
    "right_paw",
    "right_shoulder",
    "right_wrist",
    "shoulder_center",
    "tail_base",
    "tail_middle",
    "tail_tip",
    "hip_center",
]


# ==== Analysis Parameters ====
WINDOW_LENGTH_BIN = 20  # window size in bins
STEP_BIN = 1  # step size in bins
CV = 5  # number of cross-validation folds

REL_START = -200
REL_END = 300

FORE_KEYWORDS = [
    "wrist",
    "paw",
    "elbow",
]

HIND_KEYWORDS = [
    "foot",
    "ankle",
    "knee",
]


def get_limb(forelimb: bool, laterality: str):
    if forelimb:
        body_parts = FORE_KEYWORDS
    else:
        body_parts = HIND_KEYWORDS

    bhv_fields = [
        field
        for field in ALL_BHV_FIELDS
        if (laterality in field) and any(body_part in field for body_part in body_parts)
    ]
    return bhv_fields


def opt_ridge_alpha(x_cv, y_cv):
    alphas = np.logspace(-4, 5, 50)
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(x_cv, y_cv)

    best_alpha = ridge_cv.alpha_
    return best_alpha


def main():

    # Load data
    data_dir = "/data/bnd-data/raw/"
    session = "M061_2025_03_06_14_00"
    df_tr, dstrb_idx = dsp.load_and_process_session(session, data_dir)
    disturb_mean_vals_s = df_tr["disturb_mean"].values
    threshold_s = -scipy.stats.sem(disturb_mean_vals_s)
    disturb_mean_sorted_s = disturb_mean_vals_s[dstrb_idx]
    idx_to_keep_s = int(np.searchsorted(disturb_mean_sorted_s, threshold_s))
    print(f"  threshold = {threshold_s:.3f}  |  trials selected: {idx_to_keep_s}")

    df_design_s = df_tr.iloc[sorted(dstrb_idx[:idx_to_keep_s])]
    perturb_td_s = pyal.restrict_to_interval(
        df_design_s, start_point_name="idx_sol_on", rel_start=REL_START, rel_end=REL_END
    )
    perturb_td_s = dt.add_bhv(perturb_td_s)
    perturb_td_shuff = perturb_td_s.sample(frac=1, random_state=42).reset_index(drop=False)

    # Empty results dictionary
    scores = {}

    # # Run all keypoints
    for bhv_field in ALL_BHV_FIELDS:
        print(f"Processing keypoint: {bhv_field}")
        bhv = np.stack(perturb_td_shuff[f"{bhv_field}"].values)
        scores[bhv_field] = {}
        for area in AREAS:
            alpha = opt_ridge_alpha(
                np.concatenate(perturb_td_shuff[f"{area}_rates_pca"].values),
                np.concatenate(perturb_td_shuff[f"{bhv_field}"].values),
            )
            print(f"\t{area} -> {bhv_field} best alpha: {alpha}")
            scores[bhv_field][area], times = decode.regression_moving_window(
                np.stack(perturb_td_shuff[f"{area}_rates_pca"].values),
                bhv,
                window_length_bin=WINDOW_LENGTH_BIN,
                step_bin=STEP_BIN,
                cv=CV,
                alpha=alpha,
                scorer=subspaces.default_scorer,
            )

    bhv_combinations = {}
    bhv_combinations["left_forelimb"] = get_limb(forelimb=True, laterality="left")
    bhv_combinations["right_forelimb"] = get_limb(forelimb=True, laterality="right")
    bhv_combinations["left_hindlimb"] = get_limb(forelimb=False, laterality="left")
    bhv_combinations["right_hindlimb"] = get_limb(forelimb=False, laterality="right")

    bhv_combinations["forelimbs"] = (
        bhv_combinations["left_forelimb"] + bhv_combinations["right_forelimb"]
    )
    bhv_combinations["hindlimbs"] = (
        bhv_combinations["left_hindlimb"] + bhv_combinations["right_hindlimb"]
    )
    bhv_combinations["tail"] = ["tail_base", "tail_middle", "tail_tip"]
    bhv_combinations["all"] = ["all"]

    for bhv_combination_key, bhv_combination_value in bhv_combinations.items():
        print(
            f"Processing keypoint combination: {bhv_combination_key}: {bhv_combination_value}"
        )

        perturb_td_shuff = dt.add_bhv(perturb_td_shuff, bhv_fields=bhv_combination_value)

        bhv = np.stack(perturb_td_shuff.bhv.values)
        scores[bhv_combination_key] = {}
        for area in AREAS:
            alpha = opt_ridge_alpha(
                np.concatenate(perturb_td_shuff[f"{area}_rates_pca"].values),
                np.concatenate(perturb_td_shuff.bhv.values),
            )
            print(f"\t{bhv_combination_key} -> {area} best alpha: {alpha}")
            scores[bhv_combination_key][area], times = decode.regression_moving_window(
                np.stack(perturb_td_shuff[f"{area}_rates_pca"].values),
                bhv,
                window_length_bin=WINDOW_LENGTH_BIN,
                step_bin=STEP_BIN,
                cv=CV,
                scorer=subspaces.default_scorer,
            )

    scores["times"] = times - (abs(REL_START) * 0.01)

    with open("scores_decoding_v2.pkl", "wb") as f:
        pickle.dump(scores, f)

    return


if __name__ == "__main__":
    main()
