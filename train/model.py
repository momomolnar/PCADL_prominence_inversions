import sys
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import HuberLoss
torch.set_default_dtype(torch.float64)


def normalize(data, props_dict):
    norm_data = torch.zeros_like(data)

    for el in range(data.shape[1]):
        norm_data[:, el] = ((data[:, el] - props_dict.item()["var_" + format(el, '02d') + "_mean"])
                             / props_dict.item()["var_" + format(el, '02d') + "_std"])

    return norm_data


def unnormalize(data, props_dict):
    unnorm_data = torch.zeros_like(data)

    for el in range(data.shape[1]):
        unnorm_data[:, el] = ((data[:, el] * props_dict.item()["var_" + format(el, '02d') + "_std"])
                              + props_dict.item()["var_" + format(el, '02d') + "_mean"])

    return unnorm_data


class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden_layers=8, n_hidden_neurons=256):
        super().__init__()
        # Activation function
        self.activation = nn.ReLU()
        # Input layer
        self.input_layer = nn.Sequential(*[nn.Linear(n_input, n_hidden_neurons), self.activation])
        # Individual hidden layer template
        hidden_layer = nn.Sequential(*[nn.Linear(n_hidden_neurons, n_hidden_neurons), self.activation])
        # Hidden layers
        self.hidden_layers = nn.Sequential(*[hidden_layer for _ in range(n_hidden_layers)])
        # Output layer
        self.output_layer = nn.Linear(n_hidden_neurons, n_output)

    # Forward pass through neural network
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class PCAModel(LightningModule):

    def __init__(self, n_input, n_output, norm,
                 n_hidden_layers=8, n_hidden_neurons=256, lr=1.e-4):
        super().__init__()
        self.norm = norm
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.lr = lr

        # Architecture
        self.model = MLP(n_input, n_output, n_hidden_layers=n_hidden_layers, n_hidden_neurons=n_hidden_neurons)
        # Loss function
        self.loss_func = HuberLoss()  # consider MSE

    def forward(self, x):
        return self.model.forward(x)

    def forward_unnormalize(self, x):
        return unnormalize(self.forward(x), self.norm)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.norm)
        y_pred = unnormalize(y_pred, self.norm)

        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100

        # Logging
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE", rae.mean(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.norm)
        y_pred = unnormalize(y_pred, self.norm)

        # computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        # Logging
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)

        y = unnormalize(y, self.norm)
        y_pred = unnormalize(y_pred, self.norm)

        # computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        # Logging
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        [self.log(f"valid_RAE_{i}", err, on_epoch=True, prog_bar=True, logger=True) for i, err in enumerate(av_rae_wl)]
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
