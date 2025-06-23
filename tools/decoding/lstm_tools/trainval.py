import numpy as np
from sklearn.model_selection import KFold

from .lstm import KeypointsLSTM
from .preprocess import baseline_norm_labels, create_sliding_windows


def k_fold_eval(
    data: np.ndarray, labels: np.ndarray, area: str, keypoint_angle: list, cfg: dict
) -> dict:
    """k-fold cross validation

    Args:
        data (np.ndarray): input data
        labels (np.ndarray): kinematics
        area (str): brain area to train on
        keypoint_angle (list): names of keypoints
        cfg (dict): config

    Returns:
        dict: results
    """

    results = {}

    # Setup folds
    kf = KFold(n_splits=cfg["training"]["k_folds"], shuffle=False)

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"Fold {fold} - {keypoint_angle} - {area}")

        results[fold] = {}

        # Split data
        train_data, train_labels = data[train_idx], labels[train_idx]
        test_data, test_labels = data[test_idx], labels[test_idx]

        if cfg["preprocess"]["window_data"]:
            train_data, train_labels = create_sliding_windows(
                train_data, train_labels, len_window=cfg["preprocess"]["len_window"]
            )
            test_data, test_labels = create_sliding_windows(
                test_data, test_labels, len_window=cfg["preprocess"]["len_window"]
            )

        if cfg["preprocess"]["baseline_norm"]:
            train_labels, test_labels = baseline_norm_labels(train_labels, test_labels)

        # Initialize model
        model = KeypointsLSTM(
            input_dims=data.shape[-1],
            output_dims=labels.shape[-1],
            area=area,
            keypoint_angle=keypoint_angle,
            fold=fold,
            cfg=cfg,
        )

        # Train model
        model.train_val(train_data, train_labels)

        # Test model
        model.eval(test_data, test_labels)

        # Save outputs and plot some examples
        model.save(cfg["results"]["results_dir"])
        print(
            f"R2: {model.r2["agg_r2"]}\n--------------------------------------------------------------------"
        )

        # Save global results to dictionary
        results[fold]["agg_r2"] = model.r2["agg_r2"]
        results[fold]["agg_custom_r2"] = model.r2["agg_custom_r2"]
        results[fold]["windowed_similarity"] = model.r2["windowed_similarity"]

    return results
