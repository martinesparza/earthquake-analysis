"""
My own attempt at stacking trials and gpu acceleration
"""

from typing import List, Tuple

import cupy as cp
import numpy as np


class gCURBD:
    """gpu accelerated version of curbd"""

    def __init__(
        self,
        dt_data: float,
        dt_factor: int,
        train_epochs: int = 100,
        p0: float = 1.0,
        non_linear_func=cp.tanh,
        g: float = 1.5,
        tau_rnn: float = 0.1,
        tau_ou: float = 0.1,
        amp_ou: float = 0.01,
        regions: List[Tuple[str, np.ndarray]] | None = None,
        verbose: bool = True,
    ):
        self.dt_data = dt_data
        self.dt_factor = dt_factor
        self.train_epochs = train_epochs
        self.p0 = p0
        self.non_linear_func = non_linear_func
        self.g = g
        self.tau_rnn = tau_rnn
        self.tau_ou = tau_ou
        self.amp_ou = amp_ou
        self.regions = regions
        self.verbose = verbose
        return

    def setup_ou_process(self):
        amp_ou = cp.sqrt(self.tau_ou / self.dt_rnn)  # Amplitude of the white noise
        eta = amp_ou * cp.random.randn(self.n_units, len(self.t_rnn))
        self.ou_input = cp.ones((self.n_units, len(self.t_rnn)))  # Initial OU variable

        # Combine close form solution and stochastic approximation
        for tt in range(1, len(self.t_rnn)):
            self.ou_input[:, tt] = eta[:, tt] + (
                self.ou_input[:, tt - 1] - eta[:, tt]
            ) * cp.exp(-(self.dt_rnn / self.tau_ou))
        self.ou_input = self.amp_ou * self.ou_input  # Scale input h(t)
        return

    def prepare(self, a):
        # cp.random.seed(100)

        # Pass activity to gpu
        a = cp.asarray(a)

        # Definitions
        self.n_units, time = a.shape

        self.dt_rnn = self.dt_data / self.dt_factor
        self.t_data = self.dt_data * cp.arange(time)
        self.t_rnn = cp.arange(0, self.t_data[-1] + self.dt_rnn, self.dt_rnn)

        # Setup Ornstein-Uhlenbeck process
        self.setup_ou_process()

        # Initialize directed interaction matrix J with variance scaling
        self.j = (
            self.g * cp.random.randn(self.n_units, self.n_units) / cp.sqrt(self.n_units)
        )  # Variance scaling to ensure stability and chaos factor g = 1.5
        j0 = self.j.copy()  # Save the initial values. Shape: N x N

        # Setup and normalize training data
        self.a_ = a.copy()
        self.a_ = self.a_ / self.a_.max()
        self.a_ = cp.minimum(self.a_, 0.999)
        self.a_ = cp.maximum(self.a_, -0.999)  # Shape: N x t_data

        # initialize some others
        self.rnn = cp.zeros((self.n_units, len(self.t_rnn)))

        # initialize learning update matrix (see Sussillo and Abbot, 2009)
        self.pj = self.p0 * cp.eye(self.n_units)
        return

    def train(self):
        chi2s = []
        pVars = []

        for epoch in range(self.train_epochs):

            # Assign activity to model activity
            h = self.a_[:, 0]
            self.rnn[:, 0] = self.non_linear_func(h)

            # variables to track when to update the J matrix
            t_learn = 0  # time at which J update happens
            idx_learn = 0  # neurons used for error computation
            chi2 = 0.0

            for tt, _ in enumerate(self.t_rnn[1:]):

                # Update learning time
                t_learn += self.dt_rnn

                # Update RNN Eq. 4 in the paper
                self.rnn[:, tt] = self.non_linear_func(h)
                h = (
                    h
                    + self.dt_rnn
                    * (-h + cp.dot(self.j, self.rnn[:, tt]) + self.ou_input[:, tt])
                    / self.tau_rnn
                )

                # Update J
                if t_learn >= self.dt_data:
                    t_learn = 0
                    err = self.rnn[:, tt] - self.a_[:, idx_learn]
                    idx_learn = idx_learn + 1

                    # update chi2 using this errord
                    chi2 += cp.mean(err**2)

                    # compute delta_j
                    k = cp.dot(self.pj, self.rnn[:, tt])

                    c = 1.0 / (1.0 + cp.dot(self.rnn[:, tt], k))

                    # update PJ.
                    self.pj = self.pj - c * cp.outer(k, k)

                    # update J
                    delta_J = c * cp.outer(err, k)
                    self.j = self.j - delta_J

            # compute metrics
            time_values_to_compare = cp.arange(0, len(self.t_rnn), self.dt_factor)
            ss_res = cp.linalg.norm((self.a_ - self.rnn[:, time_values_to_compare]))
            ss_tot = cp.sqrt(self.n_units * len(self.t_data)) * cp.std(self.a_)
            pVar = 1 - (ss_res / ss_tot) ** 2

            # updates metrics
            pVars.append(pVar)
            chi2s.append(chi2)

            if self.verbose:
                print("trial=%d pVar=%f chi2=%f" % (epoch, pVar, chi2))
        return

    def fit(self, a):

        # Prepare variables for training
        self.prepare(a)

        # Optimize RNN and update interaction matrix J
        self.train()

        return
