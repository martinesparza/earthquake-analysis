"""
Encoding decoding of every keypoint
"""

import pickle
import numpy as np
import pyaldata as pyal
import sys

from sklearn.linear_model import RidgeCV

sys.path.append("../../")

import tools.dataTools as dt
import tools.decoding as decode


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
    "hip_center"
]

REL_START = -200
REL_END = 200

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
    alphas = np.logspace(-4, 4, 50)
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(x_cv, y_cv)

    best_alpha = ridge_cv.alpha_
    return best_alpha


def main():

    # Load data
    handler = decode.DecodingDataHandler(
        # data_dir="/data/bnd-data/raw/",
        data_dir="/data/raw/",
        session="M061_2025_03_06_14_00",
        combine_time_bins=False,
    )
    df = handler.df.copy()

    # Select trials, run pca, and drop trials without motion
    df = pyal.select_trials(df, df.trial_name == "trial")
    df = dt.add_pca_df(df)
    df = dt.remove_trials_wo_motion_before_event(df, "idx_motion", "idx_sol_on")
    perturb_td = pyal.restrict_to_interval(
        df, "idx_sol_on", rel_start=REL_START, rel_end=REL_END
    )

    # Empty results dictionary
    scores = {}

    # # Run all keypoints
    for bhv_field in ALL_BHV_FIELDS:
        print(f"Processing keypoint: {bhv_field}")
        bhv = np.stack(perturb_td[f"{bhv_field}"].values)
        scores[bhv_field] = {}
        for area in AREAS:
            alpha = opt_ridge_alpha(
                np.concatenate(perturb_td[f"{bhv_field}"].values),
                np.concatenate(perturb_td[f"{area}_rates_pca"].values)
            )
            print(f"\t{bhv_field} -> {area} best alpha: {alpha}")
            scores[bhv_field][area], times = decode.regression_moving_window(
                bhv,
                np.stack(perturb_td[f"{area}_rates_pca"].values),
                window_length_bin=20,
                step_bin=1,
                cv=5,
                alpha=alpha
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
        print(f"Processing keypoint combination: {bhv_combination_key}: {bhv_combination_value}")

        df_ = dt.add_bhv(df, bhv_fields=bhv_combination_value)
        perturb_td = pyal.restrict_to_interval(
            df_, "idx_sol_on", rel_start=REL_START, rel_end=REL_END
        )

        bhv = np.stack(perturb_td.bhv.values)
        scores[bhv_combination_key] = {}
        for area in AREAS:
            alpha = opt_ridge_alpha(
                np.concatenate(perturb_td.bhv.values),
                np.concatenate(perturb_td[f"{area}_rates_pca"].values)
            )
            print(f"\t{bhv_combination_key} -> {area} best alpha: {alpha}")
            scores[bhv_combination_key][area], times = decode.regression_moving_window(
                bhv,
                np.stack(perturb_td[f"{area}_rates_pca"].values),
                window_length_bin=20,
                step_bin=1,
                cv=5,
            )
    scores['times'] = times - (abs(REL_START) * 0.01)
    
    with open("scores_encoding_v2.pkl", "wb") as f:
        pickle.dump(scores, f)

    return


if __name__ == "__main__":
    main()
