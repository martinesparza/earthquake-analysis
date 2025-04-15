"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Performs Current-Based Decomposition (CURBD) of multi-region data. Ref:
%
% Perich MG et al. Inferring brain-wide interactions using data-constrained
% recurrent neural network models. bioRxiv. DOI: https://doi.org/10.1101/2020.12.18.423348
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import math
import random

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import numpy.random as npr
from matplotlib.gridspec import GridSpec


def trainMultiRegionRNN(
    activity,
    dtData=1,
    dtFactor=1,
    g=1.5,
    tauRNN=0.01,
    tauWN=0.1,
    ampInWN=0.01,
    nRunTrain=2000,
    nRunFree=10,
    P0=1.0,
    nonLinearity=cp.tanh,
    nonLinearity_inv=cp.arctanh,
    resetPoints=None,
    plotStatus=True,
    verbose=True,
    regions=None,
):
    r"""
    Trains a data-constrained multi-region RNN. The RNN can be used for,
    among other things, Current-Based Decomposition (CURBD).

    Parameters
    ----------

    activity: numpy.array
        N X T
    dtData: float
        time step (in s) of the training data
    dtFactor: float
        number of interpolation steps for RNN
    g: float
        instability (chaos); g<1=damped, g>1=chaotic
    tauRNN: float
        decay constant of RNN units
    tauWN: float
        decay constant on filtered white noise inputs
    ampInWN: float
        input amplitude of filtered white noise
    nRunTrain: int
        number of training runs
    nRunFree: int
        number of untrained runs at end
    P0: float
        learning rate
    nonLinearity: function
        inline function for nonLinearity
    resetPoints: list of int
        list of indeces into T. default to only set initial state at time 1.
    plotStatus: bool
        whether to plot data fits during training
    verbose: bool
        whether to print status updates
    regions: dict()
        keys are region names, values are np.array of indeces.
    """
    if dtData is None:
        print("dtData not specified. Defaulting to 1.")
        dtData = 1
    if resetPoints is None:
        resetPoints = [
            0,
        ]
    if regions is None:
        regions = {}

    # Activity is N x T
    activity = cp.asarray(activity)
    number_units = activity.shape[0]
    number_learn = activity.shape[0]

    dtRNN = dtData / float(dtFactor)
    nRunTot = nRunTrain + nRunFree

    ############### Permute units and time variables ##########################

    learnList = cp.random.permutation(number_units)
    iTarget = learnList[:number_learn]
    iNonTarget = learnList[number_learn:]
    tData = dtData * cp.arange(activity.shape[1])
    tRNN = cp.arange(0, tData[-1] + dtRNN, dtRNN)

    ############### Setup Ornstein-Uhlenbeck process and generate inputs ###################

    ampWN = math.sqrt(tauWN / dtRNN)  # Amplitude of the white noise
    iWN = ampWN * cp.random.randn(number_units, len(tRNN))  # White noise matrix
    inputWN = cp.ones((number_units, len(tRNN)))  # Initial OU variable
    for tt in range(1, len(tRNN)):
        inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt]) * cp.exp(
            -(dtRNN / tauWN)
        )  # Combine close form solution and stochastic approximation
    inputWN = ampInWN * inputWN  # Scale input h(t). Size: N x tRNN

    ############## Initialize directed interaction matrix J with variance scaling. ############
    J = (
        g * cp.random.randn(number_units, number_units) / math.sqrt(number_units)
    )  # Variance sacling to ensure stability and chaos factor g = 1.5
    J0 = J.copy()  # Save the initial values. Shape: N x N

    ############## Setup and normalize training data ############################
    Adata = activity.copy()
    Adata = Adata / Adata.max()
    Adata = cp.minimum(Adata, 0.999)
    Adata = cp.maximum(Adata, -0.999)  # Shape: N x tData
    stdData = cp.std(Adata[iTarget, :])  # get standard deviation of entire data

    # get indices for each sample of model data
    iModelSample = numpy.zeros(len(tData), dtype=cp.int32)
    for i in range(len(tData)):
        iModelSample[i] = (cp.abs(tRNN - tData[i])).argmin()

    # initialize some others
    RNN = cp.zeros((number_units, len(tRNN)))
    chi2s = []
    pVars = []

    # initialize learning update matrix (see Sussillo and Abbot, 2009)
    PJ = P0 * cp.eye(number_learn)

    if plotStatus is True:
        plt.rcParams.update({"font.size": 6})
        fig = plt.figure()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        gs = GridSpec(nrows=2, ncols=4)
    else:
        fig = None

    # start training
    # loop along training runs
    for nRun in range(0, nRunTot):

        H = Adata[
            :, 0, cp.newaxis
        ]  # Assign values to RNN passing them through the non-linear functions.
        RNN[:, 0, cp.newaxis] = nonLinearity(H)

        # variables to track when to update the J matrix since the RNN and
        # data can have different dt values
        tLearn = 0  # keeps track of current time
        iLearn = 0  # keeps track of last data point learned
        chi2 = 0.0

        for tt in range(1, len(tRNN)):
            # update current learning time
            tLearn += dtRNN
            # check if the current index is a reset point. Typically this won't
            # be used, but it's an option for concatenating multi-trial data
            if tt in resetPoints:
                timepoint = math.floor(tt / dtFactor)
                H = Adata[:, timepoint]

            ################################# Update RNN #########################################
            RNN[:, tt, cp.newaxis] = nonLinearity(H)

            ############################ Compute new hidden state #################################
            #  Compute dot product of connectivity and model activity and add noise
            JR = J.dot(RNN[:, tt]).reshape((number_units, 1)) + inputWN[:, tt, cp.newaxis]

            # Update hiddent state (pre non linear activation) using Euler
            H = H + dtRNN * (-H + JR) / tauRNN  # This is equation (4) in the paper.

            ############################ Update J connectivity matrix #############################

            # Let the model run for a series of time values before updating!! 5 time points is the default.
            # If the RNN time coincides with a data point to update J
            if tLearn >= dtData:
                tLearn = 0
                err = RNN[:, tt, cp.newaxis] - Adata[:, iLearn, cp.newaxis]
                iLearn = iLearn + 1
                # update chi2 using this error
                chi2 += cp.mean(err**2)

                if nRun < nRunTrain:
                    r_slice = RNN[iTarget, tt].reshape(number_learn, 1)
                    k = PJ.dot(r_slice)
                    rPr = (r_slice).T.dot(k)[0, 0]
                    c = 1.0 / (1.0 + rPr)
                    PJ = PJ - c * (k.dot(k.T))
                    J[:, iTarget.flatten()] = J[
                        :, iTarget.reshape((number_units))
                    ] - c * cp.outer(err.flatten(), k.flatten())

        ################################### Compute metrics #########################################

        rModelSample = RNN[iTarget, :][:, iModelSample]
        distance = cp.linalg.norm(Adata[iTarget, :] - rModelSample)
        pVar = 1 - (distance / (math.sqrt(len(iTarget) * len(tData)) * stdData)) ** 2
        pVars.append(pVar)
        chi2s.append(chi2)
        if verbose:
            print("trial=%d pVar=%f chi2=%f" % (nRun, pVar, chi2))
        if fig:
            fig.clear()
            ax = fig.add_subplot(gs[0, 0])
            ax.axis("off")
            ax.imshow(Adata[iTarget, :])
            ax.set_title("real rates")

            ax = fig.add_subplot(gs[0, 1])
            ax.imshow(RNN, aspect="auto")
            ax.set_title("model rates")
            ax.axis("off")

            ax = fig.add_subplot(gs[1, 0])
            ax.plot(pVars)
            ax.set_ylabel("pVar")

            ax = fig.add_subplot(gs[1, 1])
            ax.plot(chi2s)
            ax.set_ylabel("chi2s")

            ax = fig.add_subplot(gs[:, 2:4])
            idx = npr.choice(range(len(iTarget)))
            ax.plot(tRNN, RNN[iTarget[idx], :])
            ax.plot(tData, Adata[iTarget[idx], :])
            ax.set_title(nRun)
            fig.show()
            plt.pause(0.05)

    out_params = {}
    out_params["dtFactor"] = dtFactor
    out_params["number_units"] = number_units
    out_params["g"] = g
    out_params["P0"] = P0
    out_params["tauRNN"] = tauRNN
    out_params["tauWN"] = tauWN
    out_params["ampInWN"] = ampInWN
    out_params["nRunTot"] = nRunTot
    out_params["nRunTrain"] = nRunTrain
    out_params["nRunFree"] = nRunFree
    out_params["nonLinearity"] = nonLinearity
    out_params["resetPoints"] = resetPoints

    out = {}
    out["regions"] = regions
    out["RNN"] = RNN
    out["tRNN"] = tRNN
    out["dtRNN"] = dtRNN
    out["Adata"] = Adata
    out["tData"] = tData
    out["dtData"] = dtData
    out["J"] = J
    out["J0"] = J0
    out["chi2s"] = chi2s
    out["pVars"] = pVars
    out["stdData"] = stdData
    out["inputWN"] = inputWN
    out["iTarget"] = iTarget
    out["iNonTarget"] = iNonTarget
    out["params"] = out_params
    return out


def computeCURBD(sim):
    """
    function [CURBD,CURBDLabels] = computeCURBD(varargin)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Performs Current-Based Decomposition (CURBD) of multi-region data. Ref:
    %
    % Perich MG et al. Inferring brain-wide interactions using data-constrained
    % recurrent neural network models. bioRxiv. DOI:
    %
    % Two input options:
    %   1) out = computeCURBD(model, params)
    %       Pass in the output struct of trainMultiRegionRNN and it will do the
    %       current decomposition. Note that regions has to be defined.
    %
    %   2) out = computeCURBD(RNN, J, regions, params)
    %       Only needs the RNN activity, region info, and J matrix
    %
    %   Only parameter right now is current_type, to isolate excitatory or
    %   inhibitory currents.
    %
    % OUTPUTS:
    %   CURBD: M x M cell array containing the decomposition for M regions.
    %       Target regions are in rows and source regions are in columns.
    %   CURBDLabels: M x M cell array with string labels for each current
    %
    %
    % Written by Matthew G. Perich. Updated December 2020.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    current_type = "all"  # 'excitatory', 'inhibitory', or 'all'
    RNN = sim["RNN"]
    J = sim["J"].copy()
    regions = sim["regions"]

    if regions is None:
        raise ValueError("regions not specified")

    if current_type == "excitatory":  # take only positive J weights
        J[J < 0] = 0
    elif current_type == "inhibitory":  # take only negative J weights
        J[J > 0] = 0
    elif current_type == "all":
        pass
    else:
        raise ValueError("Unknown current type: {}".format(current_type))

    nRegions = regions.shape[0]

    # loop along all bidirectional pairs of regions
    CURBD = np.empty((nRegions, nRegions), dtype=object)  # old was np.object
    CURBDLabels = np.empty((nRegions, nRegions), dtype=object)  # old was np.object

    for idx_trg in range(nRegions):
        in_trg = regions[idx_trg, 1]
        lab_trg = regions[idx_trg, 0]
        for idx_src in range(nRegions):
            in_src = regions[idx_src, 1]
            lab_src = regions[idx_src, 0]
            sub_J = J[in_trg, :][:, in_src]
            CURBD[idx_trg, idx_src] = sub_J.dot(RNN[in_src, :])
            CURBDLabels[idx_trg, idx_src] = "{} to {}".format(lab_src, lab_trg)
    return (CURBD, CURBDLabels)
