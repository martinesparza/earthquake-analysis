from typing import Callable

import numpy as np
import pandas as pd
import pyaldata as pyal
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

import tools.dataTools as dt
from tools.params import Params


def _compute_agg_r2(predictions, labels):
    batch_size, seq_len, n_outputs = predictions.shape

    # Concatenating batch and time for each output
    preds_flat = predictions.reshape(-1, n_outputs)  # (batch * seq_len, n_outputs)
    labels_flat = labels.reshape(-1, n_outputs)  # (batch * seq_len, n_outputs)

    r2_per_output = []
    for output_idx in range(n_outputs):
        r2 = r2_score(labels_flat[:, output_idx], preds_flat[:, output_idx])
        r2_per_output.append(r2)

    return np.array(r2_per_output)  # shape: (n_outputs,)


def _get_data_and_labels_from_df(df, bhv, n_components, epoch):
    arr_data, arr_bhv = dt.get_data_array_and_pos(
        [pyal.select_trials(df, df.trial_name == "trial")],
        trial_cat="values_Sol_direction",
        epoch=epoch,
        area="MOp",
        bhv=bhv,
        n_components=n_components,
    )
    n_sessions, n_targets, n_trials, n_time, n_comp = arr_data.shape
    n_sessions, n_targets, n_trials, n_time, n_keypoints = arr_bhv.shape

    data = arr_data.reshape((n_targets * n_trials, n_time, n_comp))
    labels = arr_bhv.reshape((n_targets * n_trials, n_time, n_keypoints))

    return data, labels


class TrialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.astype(np.float32)  # ensure correct dtype
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


class BaseLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Map final hidden state to k‐step outputs
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        predictions = self.fc(out)  # apply linear layer to each time step
        return predictions  # (batch, seq_len, output_size)


class LSTM:
    def __init__(
        self,
        n_input_components: int,
        outputs: list,
        output_size: int,
        hidden_size: int,
        batch_size: int = 16,
        lr: float = 0.001,
        criterion: torch.nn.Module = torch.nn.MSELoss(),
        epochs: int = 100,
        device: torch.device = torch.device("cuda"),
        epoch: Callable[[pd.Series], slice] = Params.perturb_epoch_long,
    ):
        self.model = BaseLSTM(n_input_components, hidden_size, output_size)
        self.hidden_size = hidden_size
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.outputs = outputs
        self.epoch = epoch
        self.n_input_components = n_input_components
        self.batch_size = batch_size
        self.output_size = output_size

    def preprocess(self, df):
        return _get_data_and_labels_from_df(
            df, self.outputs, self.n_input_components, self.epoch
        )

    def train(self):
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
            if epoch % 250 == 0:  # Also print the first epoch for context
                print(f"\tEpoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f}")

    def predict(self):
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
            r2 = _compute_agg_r2(preds, labels)

        return r2, preds, labels

    def kfold_evaluation(self, df: pd.DataFrame, k: int = 5, save_example=False):
        data, labels = self.preprocess(df)

        kf = KFold(n_splits=k, shuffle=False)
        r2s = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            print(f"Fold {fold}\n-----------------------------------")
            self.model = BaseLSTM(self.n_input_components, self.hidden_size, self.output_size)

            train_data, train_labels = data[train_idx], labels[train_idx]
            test_data, test_labels = data[test_idx], labels[test_idx]

            train_dataset = TrialDataset(train_data, train_labels)
            test_dataset = TrialDataset(test_data, test_labels)

            self.train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )
            # model = BaseLSTM(input_size, hidden_size, output_size).to(device)
            self.train()
            r2, preds_, labels_ = self.predict()
            r2s.append(r2)
            print(f"R2: {r2}")

        self.r2 = r2s

        if save_example:
            self.example_preds = preds_
            self.example_labels = labels_
        return
