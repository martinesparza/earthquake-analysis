import pickle

import numpy as np
import pandas as pd
import pyaldata as pyal
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

import tools.dataTools as dt
from tools.decoding.lstm_tools.preprocess import create_sliding_windows, unroll_data
from tools.decoding.lstm_tools.viz import (
    plot_top_trials_lstm_example,
    plot_trial_lstm_example,
)

from .eval import compute_agg_r2, compute_moving_window_similarity


class TrialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.astype(np.float32)  # ensure correct dtype
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
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
        # if self.window_data:
        #     out = out[:, -1, :]
        predictions = self.fc(out)  # apply linear layer to each time step
        return predictions  # (batch, seq_len, output_size)


class KeypointsLSTM:
    def __init__(self, cfg: dict):

        # Architecture
        self.n_input_components = cfg["model"]["n_input_components"]
        self.hidden_size = cfg["model"]["hidden_size"]
        self.outputs = cfg["preprocess"]["outputs"]
        self.num_layers = cfg["model"]["num_layers"]
        self.dropout = cfg["model"]["dropout"]
        self.batch_first = cfg["model"]["batch_first"]

        # Training
        self.lr = cfg["training"]["lr"]
        self.area = cfg["preprocess"]["area"]
        self.loss = cfg["training"]["loss"]
        self.optimizer = cfg["training"]["optimizer"]
        self.n_epochs = cfg["training"]["n_epochs"]
        self.batch_size = cfg["training"]["batch_size"]
        self.n_print_epoch = cfg["training"]["n_print_epoch"]

        # Gpu
        self.device = torch.device(cfg["model"]["device"])

        # eval
        self.r2 = {}
        self.window_data = cfg["preprocess"]["window_data"]
        self.len_window = cfg["preprocess"]["len_window"]
        self.epoch = cfg["preprocess"]["epoch"]
        self.bin_size = cfg["preprocess"]["bin_size"]
        self.testing_window = cfg["eval"]["testing_window"]
        self.plot_example = cfg["eval"]["plot_example"]
        self.results_dir = cfg["results"]["results_dir"]

        # Initialize model
        self.model = BaseLSTM(
            input_size=self.n_input_components,
            hidden_size=self.hidden_size,
            output_size=self.outputs.shape[-1],
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def set_loss(self):
        if self.loss == "mse":
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError(f"criterion: {self.loss} not implemented")
        return

    def set_optimizer(self):
        if self.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"criterion: {self.optimizer} not implemented")
        return

    def train_val(self, data, labels):

        # Build data loaders
        train_dataset = TrialDataset(data, labels)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Train model
        self.model.train()
        self.model.to(self.device)
        self.set_optimizer()
        self.set_loss()

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)  # (batch, n_time, n_input_components)
                y_batch = y_batch.to(self.device).float()  # (batch, n_time, n_outputs)

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)  # (batch, n_time, n_outputs)

                loss = self.loss(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            if epoch % self.n_print_epoch == 0:  # Also print the first epoch for context
                print(f"\tEpoch {epoch}/{self.n_epochs} - Loss: {avg_loss:.4f}")

        return

    def predict(self, data, labels):
        # Build data loaders
        test_dataset = TrialDataset(data, labels)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.model.eval()  # evaluation mode

        labels = []
        preds = []

        with torch.no_grad():  # no gradients needed for inference
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch)

                preds.append(outputs.cpu().detach().numpy())
                labels.append(y_batch.cpu().detach().numpy())

            preds = np.concatenate(preds, axis=0)
            labels = np.concatenate(labels, axis=0)

        return preds, labels

    def eval(self, data, labels):

        # Aggregate r2
        preds, labels = self.predict(data, labels)
        self.r2["agg_r2"], self.r2["agg_custom_r2"] = compute_agg_r2(
            preds, labels, self.outputs
        )

        # Moving window mse
        self.r2["agg_r2"]["windowed_similarity"] = compute_moving_window_similarity(
            predictions=preds,
            labels=labels,
            bhv_outputs=self.outputs,
            window_data=self.window_data,
            data_window=self.len_window,
            epoch=self.epoch,
            testing_window=self.testing_window,
            bin_size=self.bin_size,
        )

        # Plot some examples:
        if self.plot_example:
            plot_top_trials_lstm_example(
                preds,
                labels,
                self.area,
                perturb_onset=int(abs(self.epoch[0] / self.bin_size)),
                n_trials_to_plot=1,
                keypoints=self.outputs,
                window_data=self.window_data,
                bin_size=self.bin_size,
                epoch=self.epoch,
                results_dir=self.results_dir,
                data_window=self.len_window,
            )

        return

    def save(self):
        # Save model
        # Save results
        return


# class KeypointsLSTM_:
#     """
#     Keypoint class
#     """

#     def __init__(
#         self,
#         n_input_components: int,
#         outputs: list,
#         hidden_size: int,
#         batch_size: int = 64,
#         lr: float = 0.001,
#         criterion: torch.nn.Module = torch.nn.MSELoss(),
#         epochs: int = 100,
#         device: torch.device = torch.device("cuda"),
#         num_layers: int = 2,
#         dropout: float = 0.1,
#         sigma: float = 1.0,
#     ):
#         self.hidden_size = hidden_size
#         self.criterion = criterion
#         self.lr = lr
#         self.epochs = epochs
#         self.device = device
#         self.outputs = outputs
#         self.n_input_components = n_input_components
#         self.batch_size = batch_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.sigma = sigma

#     def preprocess(
#         self, df, area, condition="trial", epoch=None, window_data=False, len_window=20
#     ):
#         """Preprocessing method. Parses data and windows it if provided

#         Args:
#             df (_type_): sessions data
#             area (_type_): Brain area
#             condition (str, optional): Trial condition. Defaults to "trial".
#             epoch (_type_, optional): Trial epoch for slicing. Defaults to None.
#             window_data (bool, optional): Whether or not to window the data. Defaults to False.
#             len_window (int, optional): Length of window. Defaults to 20.

#         Returns:
#             _type_: _description_
#         """
#         if epoch is not None:
#             epoch = pyal.generate_epoch_fun(
#                 start_point_name="idx_sol_on",
#                 rel_start=int(epoch[0] / df.bin_size.values[0]),
#                 rel_end=int(epoch[1] / df.bin_size.values[0]),
#             )

#         if condition == "trial":
#             data, labels = _get_trialdata_and_labels_from_df(
#                 df, self.outputs, area, self.n_input_components, epoch, self.sigma
#             )

#         elif condition == "intertrial":
#             data, labels = _get_intertrial_and_labels_from_df(
#                 df, self.outputs, area, self.n_input_components
#             )

#         if window_data:
#             data, labels = create_sliding_windows(data, labels, len_window=len_window)

#         return data, labels

#     def train(self, n_print_epoch):
#         """
#         Training method
#         """
#         self.model.train()
#         self.model.to(self.device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

#         for epoch in range(self.epochs):
#             epoch_loss = 0.0
#             for x_batch, y_batch in self.train_loader:
#                 x_batch = x_batch.to(self.device)  # (batch, n_time, n_input_components)
#                 y_batch = y_batch.to(self.device).float()  # (batch, n_time, n_outputs)

#                 self.optimizer.zero_grad()
#                 outputs = self.model(x_batch)  # (batch, n_time, n_outputs)

#                 loss = self.criterion(outputs, y_batch)
#                 loss.backward()
#                 self.optimizer.step()

#                 epoch_loss += loss.item()

#             avg_loss = epoch_loss / len(self.train_loader)
#             if epoch % n_print_epoch == 0:  # Also print the first epoch for context
#                 print(f"\tEpoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f}")

#     def predict(self):
#         """Predict method

#         Returns:
#             tuple: predictions, labels
#         """
#         self.model.eval()  # evaluation mode

#         labels = []
#         preds = []

#         with torch.no_grad():  # no gradients needed for inference
#             for x_batch, y_batch in self.test_loader:
#                 x_batch = x_batch.to(self.device)
#                 outputs = self.model(x_batch)

#                 preds.append(outputs.cpu().detach().numpy())
#                 labels.append(y_batch.cpu().detach().numpy())

#             preds = np.concatenate(preds, axis=0)
#             labels = np.concatenate(labels, axis=0)

#         return preds, labels

#     def kfold_evaluation(
#         self,
#         df: pd.DataFrame,
#         area: str,
#         condition: str,
#         k: int = 5,
#         epoch: list = [-1, 3],
#         similarity_metric=mean_squared_error,
#         baseline_norm_labels: bool = True,
#         window_data: bool = True,
#         len_window: int = 20,
#         n_print_epoch: int = 250,
#         results_dir: str | None = None,
#         testing_window: int = 10,
#         plot_example: bool = False,
#     ):
#         """Kfold cross validation

#         Args:
#             df (pd.DataFrame): Session data
#             area (str): brain area
#             condition (str): trial condition
#             k (int, optional): folds. Defaults to 5.
#             save_example (bool, optional): save preds and labels for plotting. Defaults to False.
#             epoch (Callable[[pd.Series], slice], optional): Trial epoch for slicing. Defaults to Params.perturb_epoch_long.
#             baseline_norm_labels (bool, optional): Baseline normalising. Defaults to True.
#             window_data (bool, optional): Windowing data for seq-to-point prediction. Defaults to True.
#             len_window (int, optional): Length of window. Defaults to 20.
#         """
#         data, labels = self.preprocess(
#             df,
#             condition=condition,
#             epoch=epoch,
#             area=area,
#             window_data=window_data,
#             len_window=len_window,
#         )

#         # trial_length
#         trial_length = int(
#             np.ceil(epoch[1] / df.bin_size.values[0] - epoch[0] / df.bin_size.values[0]),
#         )

#         # Folds
#         kf = KFold(n_splits=k, shuffle=False)

#         # General r2s
#         results = {"agg_r2": {}, "agg_custom_r2": {}, "windowed_similarity": {}}

#         for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
#             print(f"Fold {fold}")
#             self.model = BaseLSTM(
#                 input_size=self.n_input_components,
#                 hidden_size=self.hidden_size,
#                 output_size=labels.shape[-1],
#                 window_data=window_data,
#                 num_layers=self.num_layers,
#                 dropout=self.dropout,
#             )

#             train_data, train_labels = data[train_idx], labels[train_idx]
#             test_data, test_labels = data[test_idx], labels[test_idx]

#             if baseline_norm_labels:
#                 train_labels, test_labels = _baseline_norm_labels(train_labels, test_labels)

#             train_dataset = TrialDataset(train_data, train_labels)
#             test_dataset = TrialDataset(test_data, test_labels)

#             self.train_loader = DataLoader(
#                 train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
#             )
#             self.test_loader = DataLoader(
#                 test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
#             )
#             self.train(n_print_epoch)
#             preds_, labels_ = self.predict()
#             results["agg_r2"][fold], results["agg_custom_r2"][fold] = _compute_agg_r2(
#                 preds_, labels_, self.outputs
#             )
#             results["windowed_similarity"][fold] = _compute_moving_window_similarity(
#                 predictions=preds_,
#                 labels=labels_,
#                 bhv_outputs=self.outputs,
#                 window_data=window_data,
#                 data_window=len_window,
#                 trial_length=trial_length,
#                 testing_window=testing_window,
#                 similarity_metric=similarity_metric,
#             )
#             print(
#                 f"R2: {results["agg_r2"][fold]}\n--------------------------------------------------------------------"
#             )

#         if plot_example:
#             if window_data:
#                 labels_ = unroll_data(labels_, trial_length=int(trial_length - len_window))
#                 preds_ = unroll_data(preds_, trial_length=int(trial_length - len_window))

#             plot_top_trials_lstm_example(
#                 preds_,
#                 labels_,
#                 area,
#                 perturb_onset=int(abs(epoch[0] / df.bin_size.values[0])),
#                 n_trials_to_plot=1,
#                 results_dir=results_dir,
#                 keypoints=self.outputs,
#             )
#         return results
