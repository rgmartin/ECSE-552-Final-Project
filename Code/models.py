import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torchvision import transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

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

        self.train_accuracy(y_hat, y.type(torch.int))
        self.log('train_acc_step', self.train_accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        self.val_accuracy(y_hat, y.type(torch.int))
        self.log('val_acc_step', self.val_accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        # The casting here should really be fixed earlier in terms of when the
        # data is loaded but this is sufficient to prove things out from the
        # data flow standpoint.
        self.test_accuracy(y_hat, y.type(torch.int))
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

class Encoder(nn.Module):

    def __init__(self, num_input_channels : int, base_channel_size : int, latent_dim : int, act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 128x157 => 64x78
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 64x78 => 32x39
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 32x39 => 16x19
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)
    

class Decoder(nn.Module):

    def __init__(self, num_input_channels : int, base_channel_size : int, latent_dim : int, act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2*c_hid),act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x
    

class Autoencoder(pl.LightningModule):

    def __init__(self, base_channel_size: int, latent_dim: int,encoder_class : object = Encoder,decoder_class : object = Decoder,num_input_channels: int = 1,width: int = 128,height: int = 157):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        print(x.shape)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=20,min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)
