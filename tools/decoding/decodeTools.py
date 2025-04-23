from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyaldata as pyal
import seaborn as sns
import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import GaussianNB

from tools import dataTools as dt
from tools import params
from tools.params import Params
from tools.viz import utilityTools as utility


def columnwise_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot
    return r2


def multivariate_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    ss_res = np.linalg.norm(Y_true - Y_pred, ord="fro") ** 2
    ss_tot = np.linalg.norm(Y_true - Y_true.mean(axis=0), ord="fro") ** 2
    multi_r2 = 1 - ss_res / ss_tot

    col_r2 = columnwise_r2(Y_true, Y_pred)
    return multi_r2, col_r2


def within_decoding(
    cat,
    allDFs,
    epoch,
    area="M1",
    units=None,
    model=10,
    n_components=10,
    from_bhv=False,
    bhv_fields=["all"],
    reduce_dim=False,
    control=False,
    transformation=None,
    metric=None,
    classifier_model=GaussianNB,
    ax=None,
    trial_conditions=[],
):
    """ """

    within_score = {}
    target_ids = np.unique(allDFs[0][cat])
    conf_matrices = []
    for i, df in enumerate(allDFs):
        # print(df["session"][0])
        for condition in trial_conditions:
            df = pyal.select_trials(df, condition)
        if from_bhv:
            #  for predicting from behavioural data
            AllData = dt.get_data_array_bhv(
                [df],
                cat,
                epoch=epoch,
                bhv_fields=bhv_fields,
                model=model,
                n_components=n_components,
                reduce_dim=reduce_dim,
                transformation=transformation,
                metric=metric,
            )
            # _, n_trial, n_comp = AllData1.shape
        else:
            AllData = dt.get_data_array(
                [df],
                cat,
                epoch=epoch,
                area=area,
                units=units,
                model=model,
                n_components=n_components,
            )
            AllData = AllData[0, ...]
            n_targets, n_trial, n_time, n_comp = AllData.shape
            # print(AllData.shape)
            # resizing
            X = AllData.reshape((-1, n_comp * n_time))
            AllTar = np.repeat(target_ids, n_trial)
            AllTar = np.array(AllTar, dtype=int).flatten()
            # print(AllTar)
            if control:
                np.random.shuffle(AllTar)
            if ax is not None:
                # Predictions for confusion matrix
                # print(X.shape)
                # print(AllTar.shape)

                y_pred = cross_val_predict(classifier_model(), X, AllTar, cv=5)

                # Compute confusion matrix for session
                conf_mat = confusion_matrix(AllTar, y_pred, labels=target_ids)
                conf_matrices.append(conf_mat)

                # Compute accuracy
                within_score[df.session[0]] = np.mean(y_pred == AllTar)
            else:
                _score = cross_val_score(
                    classifier_model(), X, AllTar, scoring="accuracy", cv=5
                ).mean()
                within_score[df.session[0]] = np.mean(_score)

    if ax is not None:
        avg_conf_matrix = np.mean(conf_matrices, axis=0)
        avg_conf_matrix = (
            avg_conf_matrix.astype("float") / avg_conf_matrix.sum(axis=1)[:, np.newaxis]
        )
        sns.heatmap(
            avg_conf_matrix,
            annot=False,
            fmt=".2f",
            cmap="Blues",
            xticklabels=target_ids,
            yticklabels=target_ids,
            ax=ax,
        )
        ax.set_title(
            f"Predicting {cat} from {area}, score = {np.mean(list(within_score.values())):.2f}, chance = {1/len(target_ids):.2f}"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    return within_score


def plot_decoding_over_time(
    ax,
    category,
    df_list,
    areas,
    n_components,
    model,
    idx_event="idx_sol_on",
    min_time=0,
    max_time=2,
    trial_conditions=[],
):
    """
    Increasing window not moving window!
    """
    if isinstance(areas, str):
        areas = [areas]
    if isinstance(areas, dict):
        units_per_area = list(areas.values())
        areas = list(areas.keys())
    else:
        units_per_area = None

    within_results_per_area = []
    min_timebin = int(min_time / Params.BIN_SIZE)
    max_timebin = int(max_time / Params.BIN_SIZE)
    for i, area in enumerate(areas):
        within_results_over_time = []
        for timebin in range(min_timebin, max_timebin):
            perturb_epoch = pyal.generate_epoch_fun(
                start_point_name=idx_event, rel_start=int(min_timebin), rel_end=int(timebin)
            )
            if units_per_area is not None:
                area = "all"
                units = units_per_area[i]
            else:
                units = None
            within_results = within_decoding(
                cat=category,
                allDFs=df_list,
                area=area,
                units=units,
                n_components=n_components,
                epoch=perturb_epoch,
                model=model,
                trial_conditions=trial_conditions,
            )
            within_results_over_time.append([result for result in within_results.values()])

        within_results_per_area.append(np.array(within_results_over_time))

    time_axis = np.arange(min_time, max_time, Params.BIN_SIZE) * 1000
    time_axis = time_axis[1:]
    for i, area in enumerate(areas):
        utility.shaded_errorbar(
            ax,
            time_axis,
            within_results_per_area[i],
            label=area,
            color=getattr(params.colors, area, "k"),
        )

    chance_level = 1 / len(np.unique(df_list[0][category]))
    ax.set_xlabel("Window length (ms)")
    ax.set_ylabel("Decoding accuracy (%)")
    ax.set_title(f"Decoding accuracy using increasing time intervals")
    ax.axvline(x=0, color="k", linestyle="--", label=idx_event)
    ax.axhline(y=chance_level, color="red", linestyle="--", label="Chance level")
    ax.legend()


def plot_decoding_moving_window(
    ax,
    category,
    df_list,
    areas,
    n_components,
    model,
    idx_event="idx_sol_on",
    min_time=-0.5,
    max_time=1.5,
    window_length=0.1,
    step=0.03,
    trial_conditions=[],
):
    """
    PCA model obtained on all the trials concatenated, not restricted to the moving window.
    """

    if isinstance(areas, str):
        areas = [areas]
    if isinstance(areas, dict):
        units_per_area = list(areas.values())
        areas = list(areas.keys())
    else:
        units_per_area = None

    within_results_per_area = []

    bin_size = Params.BIN_SIZE
    min_timebin = int(min_time / bin_size)
    max_timebin = int(max_time / bin_size)
    window_size_bins = int(window_length / bin_size)
    step_size_bins = int(step / bin_size)

    time_points = np.arange(min_timebin, max_timebin - window_size_bins, step_size_bins)
    
    # model_full = PCA(n_components=n_components, svd_solver="full")
    
    for i, area in enumerate(areas):
        print(area)
        modified_df_list = []
        for j,df in enumerate(df_list):
            model_full = PCA(n_components=n_components, svd_solver="full")
            rates = np.concatenate(df[f"{area}_rates"].values, axis=0)
            rates_model = model_full.fit(rates)
            modified_df_list.append(pyal.apply_dim_reduce_model(df, rates_model, f"{area}_rates", f"{area}_pca"))
        
        within_results_over_time = []

        for start_bin in time_points:
            # print(start_bin, start_bin + window_size_bins)
            perturb_epoch = pyal.generate_epoch_fun(
                start_point_name=idx_event,
                rel_start=start_bin,
                rel_end=start_bin + window_size_bins,
            )


            if units_per_area is not None:
                area = "all"
                units = units_per_area[i]
            else:
                units = None

            within_results = within_decoding(
                cat=category, allDFs=modified_df_list, area=area, units=units,
                n_components=n_components, epoch=perturb_epoch,
                model=model, trial_conditions=trial_conditions
            )
            within_results_over_time.append([result for result in within_results.values()])

        within_results_per_area.append(np.array(within_results_over_time))

    time_axis = ((time_points + window_size_bins) * bin_size) * 1000  #

    for i, area in enumerate(areas):
        utility.shaded_errorbar(
            ax,
            time_axis,
            within_results_per_area[i],
            label=area,
            color=getattr(params.colors, area, "k"),
        )

    chance_level = 1 / len(np.unique(df_list[0][category]))

    ax.set_xlabel("Time relative to event (ms)")
    ax.set_ylabel("Decoding Accuracy (%)")
    ax.set_title(f"Decoding Accuracy of {category} ({window_length*1000} ms)")
    ax.axvline(x=0, color="k", linestyle="--", label=f"Event: {idx_event}")  # Mark event
    ax.axhline(y=chance_level, color="red", linestyle="--", label="Chance level")
    ax.legend()


def plot_decoding_moving_window_per_component(
    ax,
    category,
    df_list,
    area="M1",
    max_components=20,
    model="pca",
    idx_event="idx_sol_on",
    min_time=-0.5,
    max_time=1.5,
    window_length=0.1,
    step=0.03,
    trial_conditions=[],
):
    if isinstance(area, dict):
        units_per_area = list(area.values())
        area = list(area.keys()[0])
    else:
        units_per_area = None

    bin_size = Params.BIN_SIZE
    min_timebin = int(min_time / bin_size)
    max_timebin = int(max_time / bin_size)
    window_size_bins = int(window_length / bin_size)
    step_size_bins = int(step / bin_size)

    time_points = np.arange(min_timebin, max_timebin - window_size_bins, step_size_bins)

    results_matrix = np.zeros((max_components, len(time_points)))

    for n_components in range(1, max_components + 1):
        model_full = PCA(n_components=n_components, svd_solver="full")

        for j, df in enumerate(df_list):
            rates = np.concatenate(df[f"{area}_rates"].values, axis=0)
            rates_model = model_full.fit(rates)
            df_list[j] = pyal.apply_dim_reduce_model(
                df, rates_model, f"{area}_rates", f"{area}_pca"
            )

        within_results_over_time = []

        for i, start_bin in enumerate(time_points):
            perturb_epoch = pyal.generate_epoch_fun(
                start_point_name=idx_event,
                rel_start=start_bin,
                rel_end=start_bin + window_size_bins,
            )

            if units_per_area is not None:
                area = "all"
                units = units_per_area[0]
            else:
                units = None

            within_results = within_decoding(
                cat=category,
                allDFs=df_list,
                area=area,
                units=units,
                n_components=n_components,
                epoch=perturb_epoch,
                model=model,
                trial_conditions=trial_conditions,
            )

            # Store the mean accuracy over time
            results_matrix[n_components - 1, i] = np.mean(list(within_results.values()))

    time_axis = (
        (time_points + window_size_bins) * bin_size
    ) * 1000  # Convert to milliseconds
    chance_level = 1 / len(np.unique(df_list[0][category]))
    # Plot heatmap
    zero_index = np.argmin(np.abs(time_axis))
    ax.axvline(x=zero_index, color="red", linestyle="--", linewidth=2, label=idx_event)
    sns.heatmap(
        results_matrix,
        ax=ax,
        cmap="viridis",
        vmin=chance_level,
        vmax=0.5,
        cbar=True,
        xticklabels=np.round(time_axis, 2),
        yticklabels=np.arange(1, max_components + 1),
    )

    # tick_positions = np.arange(np.min(time_axis), np.max(time_axis) + 1, 100)
    # ax.set_xticks(tick_positions)
    # ax.set_xticklabels([f"{int(t)}" for t in tick_positions])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Number of PCs")
    ax.set_title(f"Decoding Performance Over Time ({area})")
