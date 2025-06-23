import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tools.decoding.lstm_tools.viz import plot_top_trials_lstm_example

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
        if self.window_data:
            out = out[:, -1, :]
        predictions = self.fc(out)  # apply linear layer to each time step
        return predictions  # (batch, seq_len, output_size)


class KeypointsLSTM:
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        area: str,
        keypoint_angle: list | str,
        fold: int,
        cfg: dict,
    ):

        # Architecture
        self.n_input_components = input_dims
        self.hidden_size = cfg["model"]["hidden_size"]
        self.num_layers = cfg["model"]["num_layers"]
        self.dropout = cfg["model"]["dropout"]
        self.batch_first = cfg["model"]["batch_first"]

        # Training
        self.lr = cfg["training"]["lr"]
        self.keypoint_angle = keypoint_angle
        self.area = area
        self.loss = cfg["training"]["loss"]
        self.optimizer = cfg["training"]["optimizer"]
        self.n_epochs = cfg["training"]["n_epochs"]
        self.batch_size = cfg["training"]["batch_size"]
        self.n_print_epoch = cfg["training"]["n_print_epoch"]
        self.fold = fold

        # Gpu
        self.device = torch.device(cfg["model"]["device"])

        # eval
        self.r2 = {}
        self.window_data = cfg["preprocess"]["window_data"]
        self.len_window = cfg["preprocess"]["len_window"]
        self.epoch = cfg["preprocess"]["epoch"]
        self.bin_size = cfg["data"]["bin_size"]
        self.testing_window = cfg["eval"]["testing_window"]
        self.plot_example = cfg["eval"]["plot_example"]
        self.results_dir = cfg["results"]["results_dir"]

        # Initialize model
        self.model = BaseLSTM(
            input_size=self.n_input_components,
            hidden_size=self.hidden_size,
            output_size=output_dims,
            num_layers=self.num_layers,
            dropout=self.dropout,
            window_data=self.window_data,
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
            preds, labels, self.keypoint_angle
        )

        # Moving window mse
        self.r2["windowed_similarity"] = compute_moving_window_similarity(
            predictions=preds,
            labels=labels,
            bhv_outputs=self.keypoint_angle,
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
                keypoints=self.keypoint_angle,
                window_data=self.window_data,
                bin_size=self.bin_size,
                epoch=self.epoch,
                results_dir=self.results_dir,
                data_window=self.len_window,
            )

        return

    def save(self, path: str):

        keypoint_angle_str = "-".join(self.keypoint_angle)

        folder_path = Path(f"{path}/{keypoint_angle_str}/{self.area}")
        folder_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), f"{folder_path}/model_fold_{self.fold}.pth")
        return
