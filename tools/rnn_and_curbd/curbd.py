# imports
import numpy as np
import matplotlib.pyplot as plt

from tools.curbd import curbd

### Functions for Current Based Decomposition from trained RNN model ###
def format_curbd_output(curbd_arr, curbd_labels, n_regions, reset_points):
    all_currents = []
    max_len = curbd_arr[0, 0].shape[1]
    reset_points = [point for point in reset_points if point <= max_len]

    for iTarget in range(n_regions):
        for iSource in range(n_regions):
            num_neurons = curbd_arr[iTarget, iSource].shape[0]
            # set up space for data
            new_row = [[] for _ in range(num_neurons)]

            # iterate over neurons
            for neuron in range(num_neurons):
                curr = curbd_arr[iTarget, iSource][neuron]
                for p in range(len(reset_points)):
                    point = reset_points[p]
                    if point != 0:
                        new_row[neuron].append(curr[reset_points[p - 1]:point])

            # new_row = np.array(new_row)
            all_currents.append(new_row)

    all_currents_labels = curbd_labels.flatten()
    return all_currents, all_currents_labels

