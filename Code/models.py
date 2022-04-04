import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
import pytorch_lightning as pl
from pl_bolts.models.autoencoders import AE


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


class BaselineResnetClassifier(pl.LightningModule):
    """
    Baseline model built on top of a pretrained ResNet50 architecture.
    """

    def __init__(self, num_classes):
        super().__init__()

        # Load, optionally download pre-trained Resnet.
        self.resnet50 = torchvision.models.resnet50(pretrained=True, num_classes = 1000)
        self.fc = torch.nn.Linear(1000, num_classes)

        # Log stuffs.
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        # convert the incoming data to three dimensions for processing due to the RGB expectation of the pre-trained
        # network
        x = self.resnet50(x.unsqueeze(1).repeat(1, 3, 1, 1))
        x = F.relu(x)
        x = self.fc(x)

        # Note: cross entropy loss applies softmax to output.
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.train_accuracy(y_hat, y)
        self.log('train_acc_step', self.train_accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        self.val_accuracy(y_hat, y)
        self.log('val_acc_step', self.val_accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        self.test_accuracy(y_hat, y)
        self.log('test_acc_step', self.test_accuracy)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_accuracy)

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.val_accuracy)

    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.test_accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


class Mel_ae(AE):
    # Todo: determine how logging interfaces with this as well as the metrics which we can observe...
    def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):

        backbone = torchvision.models.resnet50(pretrained=True, num_classes=1000)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        encoder = nn.Sequential(*layers)

        return encoder.eval()
    
    # def training_step(self, batch, batch_idx):
    #     return self._common_step(batch, batch_idx, "train")

    # def validation_step(self, batch, batch_idx):
    #     self._common_step(batch, batch_idx, "val")
        
    # def _common_step(self, batch, batch_idx, stage: str):
    #     x = self._prepare_batch(batch)
    #     loss = F.mse_loss(x, self(x))
    #     self.log(f"{stage}_loss", loss, on_step=True)
    #     return 
    
    
class PreTrainedResnetClassifier(pl.LightningModule):
    """
    Pretrained model on AE built on top of a pretrained ResNet50 architecture.
    """

    def __init__(self, num_classes,checkpath):
        super().__init__()
        
        AEModel = AE.load_from_checkpoint(checkpath,input_height= 128, enc_type='resnet50', first_conv=False, maxpool1=False, enc_out_dim=2048,
                       kl_coeff=0.1, latent_dim=3)
        # Load, optionally download pre-trained Resnet.
        self.resnet50 = AEModel.Encoder()
        self.fc = torch.nn.Linear(1000, num_classes)

        # Log stuffs.
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        # convert the incoming data to three dimensions for processing due to the RGB expectation of the pre-trained
        # network
        x = self.resnet50(x.unsqueeze(1).repeat(1, 3, 1, 1))
        x = F.relu(x)
        x = self.fc(x)

        # Note: cross entropy loss applies softmax to output.
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.train_accuracy(y_hat, y)
        self.log('train_acc_step', self.train_accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        self.val_accuracy(y_hat, y)
        self.log('val_acc_step', self.val_accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        self.test_accuracy(y_hat, y)
        self.log('test_acc_step', self.test_accuracy)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_accuracy)

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.val_accuracy)

    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.test_accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


class Mel_ae(AE):
    # Todo: determine how logging interfaces with this as well as the metrics which we can observe...
    def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):

        backbone = torchvision.models.resnet50(pretrained=True, num_classes=1000)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        encoder = nn.Sequential(*layers)

        return encoder.eval()


if __name__ == "__main__":

    # Generate random data.
    batch_size = 1
    num_channels = 3
    spectrogram_dimensions = [128, 128]

    dummy_data = torch.rand([batch_size, num_channels, *spectrogram_dimensions])

    model = BaselineResnetClassifier(6)
    print(model(dummy_data))
