import pickle
from typing import Callable

import numpy as np
import pandas as pd
import pyaldata as pyal
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

import tools.dataTools as dt
from tools.decoding.lstm.preprocess import create_sliding_windows, unroll_data
from tools.decoding.lstm.viz import plot_trial_lstm_example
from tools.dsp.preprocessing import preprocess
from tools.params import Params


def _baseline_norm_labels(train_labels: np.ndarray, test_labels: np.ndarray):
    """Baseline normalizing of data

    Args:
        train_labels (np.ndarray): training labels (keypoints)
        test_labels (np.ndarray): testing labels

    Returns:
        tuple: train and test baseline normalised labels
    """

    if test_labels.ndim > 2:
        flat_dim_train = train_labels.shape[0] * train_labels.shape[1]
        flat_dim_test = test_labels.shape[0] * test_labels.shape[1]
    else:
        flat_dim_train = train_labels.shape[0]
        flat_dim_test = test_labels.shape[0]

    means = np.mean(
        train_labels.reshape(flat_dim_train, train_labels.shape[-1]),
        axis=0,
    )
    train_labels = train_labels - means

    means = np.mean(
        test_labels.reshape(flat_dim_test, test_labels.shape[-1]),
        axis=0,
    )
    test_labels = test_labels - means
    return train_labels, test_labels


def _compute_moving_window_r2(
    predictions, labels, bhv_outputs, window_data, len_window, trial_length, r2_window=10
):
    r2_per_output = {}

    if window_data:
        labels = unroll_data(labels, trial_length=int(trial_length - len_window))
        predictions = unroll_data(predictions, trial_length=int(trial_length - len_window))

    n_trials, n_time_, n_dims = predictions.shape
    col_counter = 0
    for output in bhv_outputs:
        r2_list = []
        for label_trial, pred_trial in zip(labels, predictions):
            r2_trial_list = []
            for t in range(n_time_ - r2_window):
                if "angle" not in output:
                    r2 = r2_score(
                        label_trial[t : t + r2_window, col_counter : col_counter + 3],
                        pred_trial[t : t + r2_window, col_counter : col_counter + 3],
                    )
                else:
                    r2 = r2_score(
                        label_trial[t : t + r2_window, col_counter : col_counter + 1],
                        pred_trial[t : t + r2_window, col_counter : col_counter + 1],
                    )

                r2_trial_list.append(r2)
            r2_list.append(np.array(r2_trial_list))
        r2_per_output[output] = np.array(r2_list)  # trials x time
        if "angle" not in output:
            col_counter = col_counter + 3
        else:
            col_counter = col_counter + 1
    return r2_per_output


def _compute_agg_r2(predictions, labels, bhv_outputs):

    n_outputs = predictions.shape[-1]

    # Concatenating batch and time for each output
    preds_flat = predictions.reshape(-1, n_outputs)  # (batch * seq_len, n_outputs)
    labels_flat = labels.reshape(-1, n_outputs)  # (batch * seq_len, n_outputs)

    r2_per_output = {}
    col_counter = 0
    for output in bhv_outputs:
        if "angle" not in output:
            # Use x,y,z variance weighted
            r2_per_output[output] = r2_score(
                labels_flat[:, col_counter : col_counter + 3],
                preds_flat[:, col_counter : col_counter + 3],
                multioutput="variance_weighted",
            )
            col_counter = col_counter + 3
        else:
            # Use angle data
            r2_per_output[output] = r2_score(
                labels_flat[:, col_counter : col_counter + 1],
                preds_flat[:, col_counter : col_counter + 1],
            )
            col_counter = col_counter + 1

    return r2_per_output


def _get_trialdata_and_labels_from_df(df, bhv, area, n_components, epoch, sigma):

    arr_data, arr_bhv = dt.get_data_array(
        [pyal.select_trials(df, df.trial_name == "trial")],
        trial_cat="values_Sol_direction",
        epoch=epoch,
        area=area,
        bhv=bhv,
        n_components=n_components,
        sigma=sigma,
    )
    n_sessions, n_targets, n_trials, n_time, n_comp = arr_data.shape
    n_sessions, n_targets, n_trials, n_time, n_keypoints = arr_bhv.shape

    data = arr_data.reshape((n_targets * n_trials, n_time, n_comp))
    labels = arr_bhv.reshape((n_targets * n_trials, n_time, n_keypoints))

    return data, labels


def _get_intertrial_and_labels_from_df(df, bhv, area, n_components, min_time=300):

    intertrial_df = pyal.select_trials(df, df.trial_name == "intertrial")
    intertrial_df["intertrial_start"] = 0

    arr_data, arr_bhv = dt.get_data_array(
        [pyal.select_trials(df, df.trial_name == "intertrial")],
        trial_cat="trial_length",
        epoch=None,
        area=area,
        bhv=bhv,
        n_components=n_components,
    )
    n_sessions, n_targets, n_trials, n_time, n_comp = arr_data.shape
    n_sessions, n_targets, n_trials, n_time, n_keypoints = arr_bhv.shape

    data = arr_data.reshape((n_targets * n_trials, n_time, n_comp))
    labels = arr_bhv.reshape((n_targets * n_trials, n_time, n_keypoints))

    return


def _get_free_and_labels_from_df():
    return


class TrialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.astype(np.float32)  # ensure correct dtype
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # return torch.from_numpy(self.data[idx]), torch.tensor(
        #     self.labels[idx], dtype=torch.long
        # )
        return torch.from_numpy(self.data[idx]), torch.from_numpy(self.labels[idx])


class BaseLSTM(torch.nn.Module):
    """Base network Class"""

    def __init__(
        self, input_size, hidden_size, output_size, dropout, num_layers=2, window_data=False
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # Map final hidden state to outputs
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.window_data = window_data

    def forward(self, x):
        """Forward method

        Args:
            x (tensor): (batch, se1_len, input_dims)

        Returns:
            predictions: (batch, seq_len, output_dims). Depending on seq-to-seq
            or seq-to-point the shape changes
        """
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        if self.window_data:
            out = out[:, -1, :]
        predictions = self.fc(out)  # apply linear layer to each time step
        return predictions  # (batch, seq_len, output_size)


class KeypointsLSTM:
    """
    Keypoint class
    """

    def __init__(
        self,
        n_input_components: int,
        outputs: list,
        hidden_size: int,
        batch_size: int = 64,
        lr: float = 0.001,
        criterion: torch.nn.Module = torch.nn.MSELoss(),
        epochs: int = 100,
        device: torch.device = torch.device("cuda"),
        num_layers: int = 2,
        dropout: float = 0.1,
        sigma: float = 1.0,
    ):
        self.hidden_size = hidden_size
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.outputs = outputs
        self.n_input_components = n_input_components
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sigma = sigma

    def preprocess(
        self, df, area, condition="trial", epoch=None, window_data=False, len_window=20
    ):
        """Preprocessing method. Parses data and windows it if provided

        Args:
            df (_type_): sessions data
            area (_type_): Brain area
            condition (str, optional): Trial condition. Defaults to "trial".
            epoch (_type_, optional): Trial epoch for slicing. Defaults to None.
            window_data (bool, optional): Whether or not to window the data. Defaults to False.
            len_window (int, optional): Length of window. Defaults to 20.

        Returns:
            _type_: _description_
        """
        if condition == "trial":
            data, labels = _get_trialdata_and_labels_from_df(
                df, self.outputs, area, self.n_input_components, epoch, self.sigma
            )

        elif condition == "intertrial":
            data, labels = _get_intertrial_and_labels_from_df(
                df, self.outputs, area, self.n_input_components
            )

        if window_data:
            data, labels = create_sliding_windows(data, labels, len_window=len_window)

        return data, labels

    def train(self, n_print_epoch):
        """
        Training method
        """
        self.model.train()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)  # (batch, n_time, n_input_components)
                y_batch = y_batch.to(self.device).float()  # (batch, n_time, n_outputs)

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)  # (batch, n_time, n_outputs)

                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            if epoch % n_print_epoch == 0:  # Also print the first epoch for context
                print(f"\tEpoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f}")

    def predict(self):
        """Predict method

        Returns:
            tuple: predictions, labels
        """
        self.model.eval()  # evaluation mode

        labels = []
        preds = []

        with torch.no_grad():  # no gradients needed for inference
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch)

                preds.append(outputs.cpu().detach().numpy())
                labels.append(y_batch.cpu().detach().numpy())

            preds = np.concatenate(preds, axis=0)
            labels = np.concatenate(labels, axis=0)

        return preds, labels

    def kfold_evaluation(
        self,
        df: pd.DataFrame,
        area: str,
        condition: str,
        k: int = 5,
        save_example=False,
        epoch: Callable[[pd.Series], slice] = Params.perturb_epoch_long,
        baseline_norm_labels: bool = True,
        window_data: bool = True,
        len_window: int = 20,
        n_print_epoch: int = 250,
        results_dir: str | None = None,
        r2_window: int = 10,
        plot_example: bool = False,
        plot_epoch: list = [-1, 3],
        trial_length: int | None = None,
    ):
        """Kfold cross validation

        Args:
            df (pd.DataFrame): Session data
            area (str): brain area
            condition (str): trial condition
            k (int, optional): folds. Defaults to 5.
            save_example (bool, optional): save preds and labels for plotting. Defaults to False.
            epoch (Callable[[pd.Series], slice], optional): Trial epoch for slicing. Defaults to Params.perturb_epoch_long.
            baseline_norm_labels (bool, optional): Baseline normalising. Defaults to True.
            window_data (bool, optional): Windowing data for seq-to-point prediction. Defaults to True.
            len_window (int, optional): Length of window. Defaults to 20.
        """
        data, labels = self.preprocess(
            df,
            condition=condition,
            epoch=epoch,
            area=area,
            window_data=window_data,
            len_window=len_window,
        )

        kf = KFold(n_splits=k, shuffle=False)
        r2 = {}
        r2_agg = {}
        r2_moving_window = {}

        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            print(f"Fold {fold}\n-----------------------------------")
            self.model = BaseLSTM(
                input_size=self.n_input_components,
                hidden_size=self.hidden_size,
                output_size=labels.shape[-1],
                window_data=window_data,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )

            train_data, train_labels = data[train_idx], labels[train_idx]
            test_data, test_labels = data[test_idx], labels[test_idx]

            if baseline_norm_labels:
                train_labels, test_labels = _baseline_norm_labels(train_labels, test_labels)

            train_dataset = TrialDataset(train_data, train_labels)
            test_dataset = TrialDataset(test_data, test_labels)

            self.train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )
            self.train(n_print_epoch)
            preds_, labels_ = self.predict()
            r2_agg[fold] = _compute_agg_r2(preds_, labels_, self.outputs)
            r2_moving_window[fold] = _compute_moving_window_r2(
                preds_,
                labels_,
                self.outputs,
                window_data,
                len_window,
                trial_length,
                r2_window,
            )
            print(f"R2: {r2_agg[fold]}")

        r2["agg"] = r2_agg
        r2["window"] = r2_moving_window

        if save_example:
            self.example_preds = preds_
            self.example_labels = labels_

        if plot_example:
            if window_data:
                labels_ = unroll_data(labels_, trial_length=int(trial_length - len_window))
                preds_ = unroll_data(preds_, trial_length=int(trial_length - len_window))

            plot_trial_lstm_example(
                preds_,
                labels_,
                area,
                results_dir=results_dir,
                keypoints=self.outputs,
                bin_size=df.bin_size.values[0],
            )
        return r2


def run_experiment(cfg: dict):

    # Load data
    df = pyal.load_pyaldata(
        cfg["data"]["data_dir"] + cfg["data"]["session"][:4] + "/" + cfg["data"]["session"]
    )
    df = preprocess(df, only_trials=False, combine_time_bins=cfg["data"]["combine_time_bins"])

    WINDOW_perturb = cfg["data"]["epoch"]
    epoch = pyal.generate_epoch_fun(
        start_point_name="idx_sol_on",
        rel_start=int(WINDOW_perturb[0] / df.bin_size.values[0]),
        rel_end=int(WINDOW_perturb[1] / df.bin_size.values[0]),
    )

    model = KeypointsLSTM(
        n_input_components=cfg["model"]["input_dim"],
        outputs=cfg["model"]["outputs"],
        hidden_size=cfg["model"]["hidden_size"],
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        sigma=cfg["data"]["sigma"],
    )
    r2 = {}
    for area in cfg["training"]["areas"]:
        print(f"Processing area {area}")

        r2[area] = model.kfold_evaluation(
            df,
            area=area,
            condition=cfg["training"]["condition"],
            k=cfg["training"]["k"],
            save_example=False,  # TODO
            epoch=epoch,
            window_data=cfg["training"]["window_data"],
            len_window=cfg["training"]["len_window"],
            n_print_epoch=cfg["training"]["n_print_epoch"],
            results_dir=cfg["results"]["results_dir"],
            r2_window=cfg["results"]["r2_window"],
            plot_example=cfg["results"]["plot_example"],
            plot_epoch=cfg["data"]["epoch"],
            trial_length=np.ceil(
                WINDOW_perturb[1] / df.bin_size.values[0]
                - WINDOW_perturb[0] / df.bin_size.values[0]
            ),
        )

    if cfg["results"]["results_dir"] is not None:
        with open(cfg["results"]["results_dir"] + "results_r2.pkl", "wb") as f:
            pickle.dump(r2, f)

    return
