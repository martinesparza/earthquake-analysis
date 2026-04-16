import numpy as np
import pandas as pd
import pyaldata
# import tables as tb
# import xarray as xr
from pyaldata import get_sig_by_trial
from sklearn.model_selection import KFold, StratifiedKFold


def get_phi(t1, t2):
    """
    Get phi for two subsequent targets

    Parameters
    ----------
    t1 : x,y point
        first target
    t2 : x,y point
        next target

    Returns
    -------

    """
    reach = t2 - t1

    alpha = np.arctan2(t1[1], t1[0])
    gamma = np.arctan2(reach[1], reach[0])

    phi = gamma - alpha

    if phi < 0:
        return phi + 2 * np.pi
    else:
        return phi


# def hdf2dataframe(path, shift_idx_fields, td_name=None):
#     """
#     Load a trial_data HDF5 file and turn it into a pandas DataFrame

#     It should have one node under root, then simple_fields and array_fields

#     Parameters
#     ----------
#     path : str
#         path to the file to load
#     td_name : str, optional
#         name of the variable under which the data was saved
#     shift_idx_fields : bool
#         whether to shift the idx fields
#         set to True if the data was exported from matlab
#         using its 1-based indexig

#     Returns
#     -------
#     df : pd.DataFrame
#         pandas dataframe replicating the trial_data format
#         each row is a trial
#     """
#     infile = tb.open_file(path, "r")

#     real_keys = [k for k in dir(infile.root) if (not k.startswith("_"))]

#     if td_name is None:
#         if len(real_keys) == 0:
#             raise ValueError("Could not find dataset name. Please specify td_name.")
#         elif len(real_keys) > 1:
#             raise ValueError("More than one datasets found. Please specify td_name.")

#         assert len(real_keys) == 1

#         td_name = real_keys[0]

#     df = pd.read_hdf(path, f"/{td_name}/simple_fields")

#     for field in infile.root[td_name]["array_fields"]:
#         fieldname = field._v_name

#         trial_IDs = sorted([int(c) for c in field._v_children])
#         df[fieldname] = [field[str(i)].read() for i in trial_IDs]

#     if shift_idx_fields:
#         df = pyaldata.data_cleaning.backshift_idx_fields(df)

#     infile.close()

#     return df


from sklearn.metrics import accuracy_score, make_scorer, r2_score
from sklearn.model_selection import cross_val_score
from tqdm.auto import tqdm

default_scorer = make_scorer(r2_score, multioutput="variance_weighted")


# def get_regr_cv_scores_through_time(
#     td,
#     regressor,
#     input_field,
#     out_field,
#     cv=KFold(10, shuffle=True),
#     scoring=make_scorer(r2_score, multioutput="variance_weighted"),
#     progress_bar=True,
#     n_jobs=None,
# ):
#     T = td[input_field].values[0].shape[0]

#     scores = []
#     for t in tqdm(np.arange(T), disable=not progress_bar):
#         X = np.stack([arr[t, :] for arr in td[input_field]])
#         y = np.row_stack(td[out_field])

#         scores.append(
#             cross_val_score(regressor, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
#         )

#     return xr.DataArray(np.stack(scores), dims=("time", "fold"))


# def get_classif_cv_scores_through_time(
#     td,
#     classifier,
#     input_field,
#     field_to_predict,
#     cv=StratifiedKFold(10, shuffle=True),
#     scoring=make_scorer(accuracy_score),
#     progress_bar=True,
#     n_jobs=None,
# ):
#     T = td[input_field].values[0].shape[0]

#     scores = []
#     for t in tqdm(np.arange(T), disable=not progress_bar):
#         X = np.stack([arr[t, :] for arr in td[input_field]])
#         y = td[field_to_predict].values

#         scores.append(
#             cross_val_score(
#                 classifier,
#                 X,
#                 y,
#                 cv=cv,
#                 scoring=scoring,
#                 n_jobs=n_jobs,
#             )
#         )

#     return xr.DataArray(np.stack(scores), dims=("time", "fold"))


# import matplotlib.pyplot as plt


# def plot_cv_scores_through_time(scores, ax=None, plot_std=True, **plot_kwargs):
#     if ax is None:
#         fig, ax = plt.subplots()

#     mean_cv_score = np.mean(scores, axis=1)
#     var_cv_score = np.var(scores, axis=1)

#     ax.plot(mean_cv_score, **plot_kwargs)
#     if plot_std:
#         ax.fill_between(
#             np.arange(len(mean_cv_score)),
#             mean_cv_score - var_cv_score,
#             mean_cv_score + var_cv_score,
#             alpha=0.1,
#         )


# def significantly_non_orthogonal(a, b):
#     """
#     Determine whether two N-dimensional vectors are significantly non-orthogonal,
#     i.e. test whether they are aligned.

#     Parameters
#     ----------
#     a, b : np.array
#         vectors to test

#     Returns
#     -------
#     True if the two vectors are aligned, false if they're not
#     """
#     v1, v2 = a.copy(), b.copy()

#     v1 = v1.flatten()
#     v2 = v2.flatten()
#     assert len(v1) == len(v2)
#     N = len(v1)

#     v1 = v1 / np.linalg.norm(v1)
#     v2 = v2 / np.linalg.norm(v2)

#     threshold = 3.3 / np.sqrt(N)

#     return bool(np.dot(v1, v2) > threshold)


# def reduce_dim(xarr, f, dim):
#     """
#     Apply function f to xr.DataArray reducing dimension dim

#     Parameters
#     ----------
#     xarr : xr.DataArray
#         dataarray to process
#     f : function
#         function to apply
#     dim : string
#         name of the dimension to reduce

#     Returns
#     -------
#     reduced xarr (without dim)
#     """
#     other_dims = tuple(d for d in xarr.dims if d != dim)
#     return xr.apply_ufunc(
#         f,
#         xarr,
#         input_core_dims=([dim, *other_dims],),
#         exclude_dims={
#             dim,
#         },
#         output_core_dims=(other_dims,),
#         kwargs={"axis": 0},
#     )


# def _apply(self, f, dim):
#     return reduce_dim(self, f, dim)


# xr.DataArray.apply = _apply


# # from https://stackoverflow.com/questions/39418380/histogram-with-equal-number-of-points-in-each-bin
# def histedges_equalN(x, nbin):
#     npt = len(x)
#     return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))


# def get_sig_by_trial_xr(
#     df: pd.DataFrame,
#     signals,
#     trial_indices=None,
#     channel_name: str = "channel",
# ):
#     """
#     Collect signal into a (time, channel, trial) array
#     """
#     X = get_sig_by_trial(df, signals, trial_indices)
#     return xr.DataArray(X, dims=("time", channel_name, "trial"))
