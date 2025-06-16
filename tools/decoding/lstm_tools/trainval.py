from sklearn.model_selection import KFold

from .lstm import KeypointsLSTM
from .preprocess import baseline_norm_labels


def k_fold_eval(data, labels, cfg):

    results = {}

    # Setup folds
    kf = KFold(n_splits=cfg["training"]["k_folds"], shuffle=False)

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"Fold {fold}")

        # Split data
        train_data, train_labels = data[train_idx], labels[train_idx]
        test_data, test_labels = data[test_idx], labels[test_idx]
        if cfg["baseline_norm"]:
            train_labels, test_labels = baseline_norm_labels(train_labels, test_labels)

        # Initialize model
        model = KeypointsLSTM(cfg)

        # Train model
        model.train_val(train_data, train_labels)

        # Test model
        model.eval(test_data, test_labels)

        # Save outputs and plot some examples
        model.save()
        print(
            f"R2: {model.r2["agg_r2"][fold]}\n--------------------------------------------------------------------"
        )

    return results
