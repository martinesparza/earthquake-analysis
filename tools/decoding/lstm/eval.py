import numpy as np


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
