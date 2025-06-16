import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from .preprocess import unroll_data


def compute_moving_window_similarity(
    predictions,
    labels,
    bhv_outputs,
    window_data,
    data_window,
    epoch,
    bin_size,
    testing_window=10,
):
    # trial_length
    trial_length = int(
        np.ceil(epoch[1] / bin_size - epoch[0] / bin_size[0]),
    )

    similarity_per_output = {}

    if window_data:
        labels = unroll_data(labels, trial_length=int(trial_length - data_window))
        predictions = unroll_data(predictions, trial_length=int(trial_length - data_window))

    n_trials, n_time_, n_dims = predictions.shape
    col_counter = 0
    for output in bhv_outputs:
        sim_list = []
        for label_trial, pred_trial in zip(labels, predictions):
            sim_trial_list = []
            for t in range(n_time_ - testing_window):
                if "angle" not in output:
                    similarity = mean_squared_error(
                        label_trial[t : t + testing_window, col_counter : col_counter + 3],
                        pred_trial[t : t + testing_window, col_counter : col_counter + 3],
                    )
                else:
                    similarity = mean_squared_error(
                        label_trial[t : t + testing_window, col_counter : col_counter + 1],
                        pred_trial[t : t + testing_window, col_counter : col_counter + 1],
                    )

                sim_trial_list.append(similarity)
            sim_list.append(np.array(sim_trial_list))
        similarity_per_output[output] = np.array(sim_list)  # trials x time
        if "angle" not in output:
            col_counter = col_counter + 3
        else:
            col_counter = col_counter + 1
    return similarity_per_output


def compute_agg_r2(predictions, labels, bhv_outputs):

    n_outputs = predictions.shape[-1]

    # Concatenating batch and time for each output
    preds_flat = predictions.reshape(-1, n_outputs)  # (batch * seq_len, n_outputs)
    labels_flat = labels.reshape(-1, n_outputs)  # (batch * seq_len, n_outputs)

    r2_per_output = {}
    custom_r2_per_output = {}

    col_counter = 0
    for output in bhv_outputs:
        if "angle" not in output:
            # Use x,y,z variance weighted
            r2_per_output[output] = r2_score(
                labels_flat[:, col_counter : col_counter + 3],
                preds_flat[:, col_counter : col_counter + 3],
                multioutput="variance_weighted",
            )
            custom_r2_per_output[output] = custom_r2_func(
                labels_flat[:, col_counter : col_counter + 3],
                preds_flat[:, col_counter : col_counter + 3],
                multioutput="variance_weighted",
            )
            col_counter = col_counter + 3
        else:
            # Use angle data
            r2_per_output[output] = r2_score(
                labels_flat[:, col_counter : col_counter + 1],
                preds_flat[:, col_counter : col_counter + 1],
            )
            custom_r2_per_output[output] = custom_r2_func(
                labels_flat[:, col_counter : col_counter + 1],
                preds_flat[:, col_counter : col_counter + 1],
            )
            col_counter = col_counter + 1

    return r2_per_output, custom_r2_per_output


def custom_r2_func(y_true, y_pred, multioutput="raw_values"):
    "$R^2$ value as squared correlation coefficient, as per Gallego, NN 2020"
    c = np.corrcoef(y_true.T, y_pred.T) ** 2
    r2s = np.diag(c[-int(c.shape[0] / 2) :, : int(c.shape[1] / 2)])

    if y_true.shape[-1] > 1:
        if multioutput == "variance_weighted":
            vars = np.var(y_true, axis=0)
            r2s = np.average(r2s, weights=vars)
    else:
        r2s = r2s[0]

    return r2s
