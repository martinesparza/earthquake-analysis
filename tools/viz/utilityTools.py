import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from tools.params import colors


def create_cmap_from_area(area):
    target_colour = getattr(colors, area)
    cmap = LinearSegmentedColormap.from_list("white_to_teal", ["#ffffff", target_colour])
    return cmap


# from Mo's repo
def shaded_errorbar(
    ax: plt.axes,
    x: np.array,
    y: np.array = None,
    lineStat=np.mean,
    errorStat=np.std,
    alpha=0.2,
    **props
):
    """
    ax: axis to plot into
    x,y: data, solumns in y are collapsed to calculate the errorbar
    lineStat: a function to measure the midline, must accept an `axis` argument
    errorStat: a function to measure the  symmetric errorbars, must accept an `axis` argument
    most other keyword arguments will be passed to `plt.fill_between` and *some* to `plt.plot`
    """
    if y is None:
        y = x
        x = np.arange(y.shape[0])

    line = ax.plot(x, lineStat(y, axis=1))[0]

    shadeProps = props.copy()
    for key in props.keys():
        if key == "color" or key == "c":
            line.set_color(props[key])
        elif key == "linewidth" or key == "lw":
            line.set_linewidth(props[key])
        elif key == "linestyle" or key == "ls":
            line.set_linestyle(props[key])
        elif key == "marker":
            line.set_marker(props[key])
            shadeProps.pop(key, None)
        elif key == "markersize" or key == "ms":
            line.set_markersize(props[key])
            shadeProps.pop(key, None)
        elif key == "label":
            line.set_label(props[key])
            shadeProps.pop(key, None)

    shadedY = errorStat(y, axis=1)
    shade = ax.fill_between(
        x,
        lineStat(y, axis=1) - shadedY,
        lineStat(y, axis=1) + shadedY,
        alpha=alpha,
        **shadeProps
    )

    return line, shade
