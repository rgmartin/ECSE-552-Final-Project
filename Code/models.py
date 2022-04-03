import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
import pytorch_lightning as pl
from pl_bolts.models.autoencoders import AE


class BaselineResnetClassifier(pl.LightningModule):
    """
    Baseline model built on top of a pretrained ResNet50 architecture.
    """

    def __init__(self, num_classes):
        super().__init__()

        # Init is factored to facilitate inheritance for alternative training in 
        # child classes (e.g., prototypical networks).
        self.init_encoder(num_classes)
        self.init_log()

    def init_encoder(self, num_classes):
        # Load, optionally download pre-trained Resnet.
        self.resnet50 = torchvision.models.resnet50(pretrained=True, num_classes = 1000)
        self.fc = torch.nn.Linear(1000, num_classes)

    def init_log(self):
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


class PrototypicalResnetClassifier(BaselineResnetClassifier):
    """
    Model using a pretrained ResNet and trained with prototypical networks [1].

    [1] Snell, Jake, Kevin Swersky, and Richard Zemel. "Prototypical networks 
        for few-shot learning." Advances in neural information processing 
        systems 30 (2017).
    """

    def __init__(self, num_classes, num_support):

        # Init parent of BaselineResnetClassifier.
        pl.LightningModule.__init__(self)

        # Number of samples per class used to calculate prototype embeddings.
        self.num_support = num_support

        self.init_encoder(num_classes)
        self.init_log()

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


class Mel_ae(AE):
    # Todo: determine how logging interfaces with this as well as the metrics which we can observe...
    def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):

        backbone = torchvision.models.resnet50(pretrained=True, num_classes=1000)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        encoder = nn.Sequential(*layers)

        return encoder.eval()


if __name__ == "__main__":

    # Generate random data (mimicking 1-D spectrogram).
    batch_size = 1
    spectrogram_dimensions = [128, 128]

    dummy_data = torch.rand([batch_size, *spectrogram_dimensions])

    model = BaselineResnetClassifier(num_classes=6)
    print(model(dummy_data).shape)