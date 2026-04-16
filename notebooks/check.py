
import os
import sys
sys.path.append("../")


import pyaldata as pyal
import pandas as pd
import numpy as np


# from tools.reports.report_initial import run_initial_report
from tools import dataTools as dt
import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pyaldata as pyal
from tools.viz import mean_firing as firing
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tools.params import Params
from tools.decoding import decodeTools as decode
from tools.dsp.preprocessing import preprocess
from tools.params import colors
from tools import dataTools as dt
import torch
from tqdm.asyncio import tqdm
sessions = [
    # 'M062_2025_03_19_14_00',
    # 'M062_2025_03_20_14_00',
    # 'M062_2025_03_21_14_00',
    # 'M061_2025_03_04_10_00',
    # 'M061_2025_03_05_14_00',
    # "M061_2025_03_06_14_00",
    # "M063_2025_03_12_14_00",  
    # "M063_2025_03_13_14_00",  
    # "M063_2025_03_14_15_30",
    # "M078_2025_08_05_15_30",
    # "M078_2025_08_06_15_00",
    # "M078_2025_08_07_13_30",
    # "M078_2025_08_08_10_30",# muscimol session
    # "M086_2025_12_09_16_00",
    # "M086_2025_12_10_15_00",
    # "M086_2025_12_11_15_00"
    # "M066_2025_04_09_15_45" # bci session
    # "M103_2026_02_17_14_00" # control session
    "M103_2026_02_17_14_00",# control session
    "M103_2026_02_18_15_30",
    "M103_2026_02_19_15_30",
    "M103_2026_02_20_16_00",# muscimol session
    "M106_2026_02_24_15_00",# control session
    "M106_2026_02_25_15_00",
    "M106_2026_02_26_16_00",
    "M106_2026_02_27_16_00"# muscimol session





]

for session_idx in range(len(sessions)):
    prep_df = dt.load_sessions([sessions[session_idx]], prep = True, only_trials = False)[0]
    for area in ["MOp","SSp","CP","VAL"]:
        field = f"{area}_rates"
    
        if field not in prep_df.columns:
            continue
        n_components = 10
        # area = "MOp"
        field = f"{area}_rates"
    
        alldata = np.concatenate(prep_df[field].values,axis = 0)
        # logging.info(f"{sessions[session_idx]} total timepoints: {alldata.shape[0]}")
        # logging.info(f"{sessions[session_idx]} {area} neurons: {alldata.shape[1]}")
        model = PCA(n_components=None, svd_solver='full')
        model.fit(alldata)
        components = model.components_
        global_expl_var = model.explained_variance_/np.sum(model.explained_variance_)
        new_field = f"{area}_global_pca"
        df_ = pyal.apply_dim_reduce_model(prep_df, model, signal=field, out_fieldname=new_field)
        new_field = f"{area}_global_pca"
        root = pathlib.Path("/home/il620/earthquake-analysis/notebooks/figures/change")
        # save each figure as pdf with the name of the session and area

        # def get_var_expl(df_, model, field, trial_name):
        #     data = np.concatenate(df_[df_['trial_name'] == trial_name][field].values, axis=0)
        #     projected = model.transform(data)
        #     var_total = np.sum(np.var(data - model.mean_, axis=0))
        #     var_projected = np.var(projected, axis=0)
        #     return var_projected / var_total

        # fig,ax = plt.subplots()
        # for trial_name in ["free0","intertrial","free1","trial"]:
        #     ax.plot(np.cumsum(get_var_expl(df_,model = global_expl_var,field=field,trial_name=trial_name)),label = trial_name)
        # ax.plot(np.cumsum(global_expl_var), label="alldata")
        # ax.set_xlabel("Number of components")
        # ax.set_ylabel("Cumulative explained variance")
        # ax.set_title(f"Var explained by {new_field} in {sessions[session_idx]}")
        # ax.legend()
        # plt.show()
        # fig.savefig(root / f"{sessions[session_idx]}_{area}_explained_variance.pdf", bbox_inches='tight')
        # pc_scores = model.transform(alldata)          # shape: (n_timepoints_total, n_features_pcs)
        pc = 0                        # first principal component scores

        pc_scores = np.concatenate(df_[new_field].values, axis=0)

        # --- plot histogram ---
        fig, ax = plt.subplots()
        ax.hist(pc_scores[:, pc], bins=100, density=True)          # density=True -> probability density
        ax.set_xlabel("PC score")
        ax.set_ylabel("Density")
        ax.set_title(f"{sessions[session_idx]} {area} PC {pc} distribution (all datapoints)")
        plt.show()
        fig.savefig(root / f"{sessions[session_idx]}_{area}_PC_{pc}_distribution.pdf", bbox_inches='tight')
        conds = ["free0", "intertrial", "free1", "trial"]

        # PC1 scores per condition (from df_[new_field])
        pc1_by_cond = {
            cond: np.concatenate(df_.loc[df_["trial_name"] == cond, new_field].values, axis=0)[:, 0]
            for cond in conds
        }

        fig, ax = plt.subplots()

        # shared bins so the curves are directly comparable
        all_pc1 = np.concatenate(list(pc1_by_cond.values()), axis=0)
        bins = np.histogram_bin_edges(all_pc1, bins=100)

        for cond in conds:
            ax.hist(
                pc1_by_cond[cond],
                bins=bins,
                density=True,
                alpha=0.7,
                label=cond,
            )

        ax.set_xlabel("PC1 score")
        ax.set_ylabel("Density")
        ax.set_title(f"{sessions[session_idx]} {area} PC1 distribution by condition")
        ax.legend(frameon=False, title="Condition")
        plt.show()
        fig.savefig(root / f"{sessions[session_idx]}_{area}_PC1_distribution_by_condition.pdf", bbox_inches='tight')

        # df_ already contains PCA scores in df_[new_field]
        # and each entry is (timepoints, n_components)
        conds = ["free0", "intertrial", "free1"]
        t = np.arange(pc_scores.shape[0])             # time index (monotonic)

        fig, ax = plt.subplots()
        sc = ax.scatter(
            pc_scores[:, 0], pc_scores[:, 1],
            c=t, s=1, alpha=0.5, cmap="Purples", linewidths=0
        )
        ax.set_xlabel("PC1 score")
        ax.set_ylabel("PC2 score")
        ax.set_title(f"{sessions[session_idx]} {area} ")

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Time index (sample order)")
        plt.show()
        fig.savefig(root / f"{sessions[session_idx]}_{area}_PC1_PC2_scatter.pdf", bbox_inches='tight')
        cmaps = {
            "free0": "Blues",
            "intertrial": "Oranges",
            "free1": "Greens",
            "trial": "Purples",
        }
        fig, ax = plt.subplots()
        import matplotlib as mpl
        mappables = []
        for cond in conds:
            # concatenate all trials of that condition: (T_total_cond, n_components)
            S = np.concatenate(df_.loc[df_["trial_name"] == cond, new_field].values, axis=0)

            pc1 = S[:, 0]
            pc2 = S[:, 1]

            # time index within this condition, normalised to [0, 1]
            t = np.linspace(0.0, 1.0, S.shape[0], endpoint=True)

            sc = ax.scatter(
                pc1, pc2,
                c=t,
                cmap=cmaps[cond],
                norm=mpl.colors.Normalize(0, 1.0),
                s=1,
                alpha=1,
                linewidths=0,
                rasterized=True,
                label=cond,
            )
            mappables.append((cond, sc))

        ax.set_xlabel("PC1 score")
        ax.set_ylabel("PC2 score")
        ax.set_title(f"{sessions[session_idx]} {area}" )
        ax.legend(markerscale=10, frameon=False, title="Condition")

        # Optional: one colourbar per condition to show within-condition time (start→end)
        # for i, (cond, sc) in enumerate(mappables):
        #     cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02 + 0.04 * i)
        #     cbar.set_label(f"{cond} time (start→end)")

        plt.show()
        fig.savefig(root / f"{sessions[session_idx]}_{area}_PC1_PC2_scatter_by_condition.pdf", bbox_inches='tight')
        # trial_names = ["free1"]
        # for trial_name in trial_names:
        #     data = np.concatenate(df_.loc[df_['trial_name'] == trial_name, field].values, axis=0)
        #     model = PCA(n_components=None, svd_solver='full')
        #     model.fit(data)
        #     df_ = pyal.apply_dim_reduce_model(prep_df, model, signal=field, out_fieldname=f"{area}_{trial_name}_pca")

