<<<<<<< HEAD
import torch
import pytorch_lightning as pl
import torchmetrics


class SimpleBinaryClassifier(pl.LightningModule):
    """
    This model was chosen to be as simple as possible for two purposes:
        1. To understand the architecture and interface for how to construct a model using PyTorch Lightning as shown in
            tutorial #7.
        2. To make it easier to test out the dataflow/workstream in terms of the connections between Google Colab and
            Google Drive when proving it out.
    """
    def __init__(self, n_in, n_out, activation_fn):
        super().__init__()

        model = [
            torch.nn.Linear(n_in, n_out),
            activation_fn
        ]

        self.model = torch.nn.Sequential(*model)

        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.train_accuracy(y_hat, y.type(torch.IntTensor))
        self.log('train_acc_step', self.train_accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        self.val_accuracy(y_hat, y.type(torch.IntTensor))
        self.log('val_acc_step', self.val_accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        self.test_accuracy(y_hat, y.type(torch.IntTensor))
        self.log('test_acc_step', self.test_accuracy)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_accuracy)

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.val_accuracy)

    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.test_accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=.1)
        return optimizer
=======
import torch
import pytorch_lightning as pl
import torchmetrics


class SimpleBinaryClassifier(pl.LightningModule):
    """
    This model was chosen to be as simple as possible for two purposes:
        1. To understand the architecture and interface for how to construct a model using PyTorch Lightning as shown in
            tutorial #7.
        2. To make it easier to test out the dataflow/workstream in terms of the connections between Google Colab and
            Google Drive when proving it out.
    """
    def __init__(self, n_in, n_out, activation_fn):
        super().__init__()

        model = [
            torch.nn.Linear(n_in, n_out),
            activation_fn
        ]

        self.model = torch.nn.Sequential(*model)

        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.train_accuracy(y_hat, y.type(torch.IntTensor))
        self.log('train_acc_step', self.train_accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        self.val_accuracy(y_hat, y.type(torch.IntTensor))
        self.log('val_acc_step', self.val_accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        self.test_accuracy(y_hat, y.type(torch.IntTensor))
        self.log('test_acc_step', self.test_accuracy)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_accuracy)

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.val_accuracy)

    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.test_accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=.1)
        return optimizer
>>>>>>> e1241c2eff256f868ef210ea42ef25f0ce2e1fc8
