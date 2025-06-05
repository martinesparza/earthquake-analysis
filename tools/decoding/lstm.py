import pandas as pd
import pyaldata as pyal
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

import tools.dataTools as dt
from tools.params import Params


def _get_data_and_labels_from_df(df, epoch=Params.perturb_epoch_long):
    arr_data, arr_bhv = dt.get_data_array_and_pos(
        [pyal.select_trials(df, df.trial_name == "trial")],
        trial_cat="values_Sol_direction",
        epoch=epoch,
        area="MOp",
        bhv=["left_knee", "right_knee"],
    )
    n_sessions, n_targets, n_trials, n_time, n_comp = arr_data.shape
    n_sessions, n_targets, n_trials, n_time, n_keypoints = arr_bhv.shape

    data = arr_data.reshape((n_targets * n_trials, n_time, n_comp))
    labels = arr_bhv.reshape((n_targets * n_trials, n_time, n_keypoints))

    return data, labels


class TrialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

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
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last)  # (batch, output_horizon)


class LSTM:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        lr: float = 0.001,
        criterion: torch.nn.Module = torch.nn.MSELoss(),
        epochs: int = 100,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = BaseLSTM(input_size, hidden_size, output_size)
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.device = device

    def preprocess(self, df):
        return _get_data_and_labels_from_df(df)

    def train(self):
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0

            print(f"Epoch {epoch}\n-------------------------------")

            for batch, (X_batch, y_batch) in enumerate(self.train_loader):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.float().unsqueeze(1).to(self.device)

                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

                if batch % 1000 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X_batch)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch:>2}/{self.epochs}, Loss: {avg_loss:.4f}")

    def predict(self):
        pass

    def kfold_evaluation(self, df: pd.DataFrame, k: int = 5):
        data, labels = self.preprocess(df)

        kf = KFold(n_splits=k, shuffle=False)

        for train_idx, test_idx in kf.split(data):

            train_idx, test_idx = next(iter(kf.split(data)))  # use first fold for testing
            train_data, train_labels = data[train_idx], labels[train_idx]
            test_data, test_labels = data[test_idx], labels[test_idx]

            train_dataset = TrialDataset(train_data, train_labels)
            test_dataset = TrialDataset(test_data, test_labels)

            self.train_loader = DataLoader(
                train_dataset, batch_size=32, shuffle=True, num_workers=0
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=0
            )

            self.train()

        pass
