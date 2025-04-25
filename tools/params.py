from dataclasses import dataclass

import numpy as np
import pyaldata as pyal


@dataclass
class Params:
    BIN_SIZE = 0.03  # Desired bin size for analysis.
    WINDOW_perturb = (0, 1.5)
    perturb_epoch = pyal.generate_epoch_fun(
        start_point_name="idx_sol_on",
        rel_start=int(WINDOW_perturb[0] / BIN_SIZE),
        rel_end=int(WINDOW_perturb[1] / BIN_SIZE),
    )

    WINDOW_before_perturb = (-0.6, -0.1)
    before_perturb_epoch = pyal.generate_epoch_fun(
        start_point_name="idx_sol_on",
        rel_start=int(WINDOW_perturb[0] / BIN_SIZE),
        rel_end=int(WINDOW_perturb[1] / BIN_SIZE),
    )

    WINDOW_perturb_long = (-1, 3)
    perturb_epoch_long = pyal.generate_epoch_fun(
        start_point_name="idx_sol_on",
        rel_start=int(WINDOW_perturb_long[0] / BIN_SIZE),
        rel_end=int(WINDOW_perturb_long[1] / BIN_SIZE),
    )

    # 0 = ipsi, 1 = contra
    sol_dir_to_contra_ipse = {
        0: 0,
        1: 1,
        2: 1,
        3: 0,
        4: 0,
        5: 0,
        6: 1,
        7: 1,
        8: 1,
        9: 0,
        10: 0,
        11: 1,
    }
    sol_dir_to_level = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 0,
    }


@dataclass
class colors:
    M1 = "r"
    Dls = "g"
    all = "k"
    MOp = np.array([116, 40, 129]) / 255  # "forestgreen"
    MOp_light = np.array([152, 110, 172]) / 255
    MOp_light_light = np.array([195, 164, 207]) / 255  # "forestgreen"

    CP = np.array([16, 101, 171]) / 255  # "blue"
    CP_light = np.array([58, 147, 195]) / 255
    CP_light_light = np.array([142, 196, 222]) / 255  # "blue"

    SSp_ll = "darkgreen"
    SSp = np.array([27, 121, 57]) / 255  # "limegreen"
    SSp_light = np.array([92, 174, 99]) / 255
    SSp_light_light = np.array([173, 212, 160]) / 255  # "limegreen"

    Thal = "red"
    VAL = np.array([179, 21, 41]) / 255  # "red"
    VAL_light = np.array([215, 95, 76]) / 255
    VAL_light_light = np.array([246, 164, 130]) / 255  # "red"

    corr_cmap = "viridis"
    upper = "orange"
    lower = "cornflowerblue"
    contra = "orange"
    ipsi = "cornflowerblue"
    GPe = "k"
