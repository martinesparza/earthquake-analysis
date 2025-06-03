from dataclasses import dataclass
import sys
sys.path.append("/home/zms24/Desktop")
import PyalData.pyaldata as pyal



@dataclass
class Params:
    BIN_SIZE = 0.03  # Desired bin size for analysis.
    WINDOW_perturb = (-0.1, 1.5)
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
    MOp = "forestgreen"
    CP = "blue"
    SSp_ll = "darkgreen"
    Thal = "red"
    corr_cmap = "viridis"
    upper = "orange"
    lower = "cornflowerblue"
    contra = "orange"
    ipsi = "cornflowerblue"
