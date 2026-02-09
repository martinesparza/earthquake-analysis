"""
Docstring for reports.quality_control
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

import pyaldata as pyal

import tools.dsp as dsp
import tools.decoding as decode
import tools.viz.utilityTools as vizutils
import tools.dataTools as dt
from tools.params import colors
import tools.kinematics as kin


def plot_frs_and_sensors_moving_window(
    df,
    areas,
    session,
    sensors=["values_MotSen1_X", "values_MotSen1_Y"],
    window_s=5,
    figsize=(15, 5),
):
    """PLots firing rates and sensory values using a moving window.

    Parameters
    ----------
    df : _type_
        _description_
    areas : _type_
        _description_
    sensors : list, optional
        _description_, by default ["values_MotSen1_X", "values_MotSen1_Y"]
    window_s : int, optional
        _description_, by default 5
    figsize : tuple, optional
        _description_, by default (15, 5)
    """
    fig, ax = plt.subplots(
        2, 1, figsize=figsize, sharex="all", gridspec_kw={"height_ratios": [3, 1]}
    )
    for area in areas:
        fr_means, time_bins = dsp.compute_moving_window_mean_on_array(
            df, f"{area}_rates", window_s=window_s
        )
        vizutils.shaded_errorbar(
            ax[0],
            x=time_bins * df.bin_size.values[0],
            y=fr_means,
            errorStat=scipy.stats.sem,
            color=getattr(colors, f"{area}_light", "k"),
            label=area,
        )
    ax[0].set_ylabel("Firing rate (Hz)")

    for sensor in sensors:
        sensor_means, time_bins = dsp.compute_moving_window_mean_on_array(
            df, sensor, window_s=window_s
        )
        ax[1].plot(
            time_bins * df.bin_size.values[0],
            sensor_means,
            label=sensor,
        )
    ax[0].legend()
    ax[1].set_ylabel("Mot. Sensor")
    ax[1].set_xlabel("Time (s)")
    plt.suptitle(f"Firing rates and mot. sensors in {window_s} s window. {session}")


def decode_sol_dir_pcs_and_bhv(
    td, areas=["bhv", "MOp", "SSp", "CP", "VAL"], step_bin=1, cv=5, window_length_bin=10
):
    """Decodes solenoid direction from pcs and behaviour

    Parameters
    ----------
    td : _type_
        _description_
    areas : list, optional
        _description_, by default ["bhv", "MOp", "SSp", "CP", "VAL"]
    step_bin : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_
    """
    scores = {}
    targets = td.values_Sol_direction.values.tolist()
    for area in areas:
        if area != "bhv":
            scores[area], times = decode.moving_window_decoding(
                np.stack(td[f"{area}_rates_pca"]),
                targets,
                window_length_bin=window_length_bin,
                step_bin=step_bin,
                cv=cv,
            )
        else:
            scores[area], times = decode.moving_window_decoding(
                np.stack(td[area]),
                targets,
                window_length_bin=window_length_bin,
                step_bin=step_bin,
                cv=cv,
            )
    return scores, times


def plot_decoding_sol_dir(td, areas, session, p=5):
    """Plots decoding solenoids direction

    Parameters
    ----------
    td : _type_
        _description_
    areas : _type_
        _description_
    """
    td = pyal.select_trials(td, td.trial_name == "trial")
    td = dt.add_pca_df(td)
    try:
        td = dt.add_bhv(td)
        areas = areas + ["bhv"]
    except Exception as e:
        print("No behaviour found")
    # td = dt.remove_trials_wo_motion_before_event(td, "idx_motion", "idx_sol_on")
    td = kin.drop_immobile_trials_from_td(td, win=40, p=p, event_onset=(100, 200))
    perturb_td = pyal.restrict_to_interval(td, "idx_sol_on", rel_start=-100, rel_end=150)

    scores, times = decode_sol_dir_pcs_and_bhv(perturb_td, areas=areas)

    fig, ax = plt.subplots()
    for area in areas:
        vizutils.shaded_errorbar(
            ax,
            times - 1,
            scores[area],
            # errorStat=scipy.stats.sem,
            label=area,
            color=getattr(colors, f"{area}_light"),
        )
        peak_time = times[np.argmax(scores[area].mean(axis=-1))] - 1
        ax.plot(peak_time, 0.47, marker="o", color=getattr(colors, f"{area}_light"))

    ax.axvline(x=0, color="r", linestyle="dashed", label="perturb")
    ax.axhline(y=1 / 12, color="k", linestyle="dashed", label="chance")
    ax.set_xlabel("Time rel. perturb. (s)")
    ax.set_ylabel("Decoding Accuracy (%)")
    ax.legend()
    ax.set_ylim([0, 0.55])
    ax.set_xlim([-0.25, 1])
    ax.set_title(session)


def plot_mot_sensors_and_kinematics_random_trials(td):
    td_tr = pyal.select_trials(td, td.trial_name == "trial")
    trials = np.random.randint(0, td_tr.shape[0], 7)
    for tr in trials:
        plot_mot_sensors_and_thresh(td_tr, tr)


def plot_mot_sensors_and_thresh(td, tr_idx, signal="left_knee"):
    fig, ax = plt.subplots(5, 1, gridspec_kw={"height_ratios": [1, 3, 1, 1, 1]}, sharex="all")
    motion_events = td.iloc[tr_idx].idx_motion
    # print(motion_events)

    # if len(motion_events) > 0:
    ax[0].plot(motion_events, np.full_like(motion_events, 0), "o")
    x, y = td.iloc[tr_idx].values_MotSen1_X, td.iloc[tr_idx].values_MotSen1_Y
    ax[1].plot(x)
    ax[1].plot(y)
    ax[1].set_ylim(-15, 15)
    ax[0].set_title(
        f"Trial {tr_idx}. Sol id: {td.iloc[tr_idx].values_Sol_direction}. {signal}"
    )

    bhv = td.iloc[tr_idx][signal]
    ax[2].plot(bhv[:, 0])
    ax[3].plot(bhv[:, 1])
    ax[4].plot(bhv[:, 2])


def run_quality_control_on_td(td, areas, session, p=5):
    """Runs quality control analyses on trial data format

    Parameters
    ----------
    td : _type_
        _description_
    areas : _type_
        _description_
    """

    # Firing rates and sensorsy
    plot_frs_and_sensors_moving_window(td, areas=areas, session=session, window_s=1)

    # # Plot random trials
    plot_mot_sensors_and_kinematics_random_trials(td)

    # Solenoid direction
    plot_decoding_sol_dir(td, areas=areas, session=session, p=p)

    return


def run_quality_control_on_session(
    data_dir, session, areas=["MOp", "SSp", "CP", "VAL", "MOs"], p=5
):
    """Entry point to quality control

    Parameters
    ----------
    data_dir : _type_
        _description_
    session : _type_
        _description_
    areas : list, optional
        _description_, by default ["MOp", "SSp", "CP", "VAL"]
    """
    td = pyal.load_pyaldata(data_dir + session[:4] + "/" + session)
    td = dsp.preprocess(
        td,
        only_trials=False,
        combine_time_bins=False,
        repair_time_varying_fields=["MotSen1_X", "MotSen1_Y"],
    )
    run_quality_control_on_td(td, areas, session, p=p)
    return
