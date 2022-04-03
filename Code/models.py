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

        # Load, optionally download pre-trained Resnet.
        self.resnet50 = torchvision.models.resnet50(pretrained=True, num_classes=1000)
        self.fc = torch.nn.Linear(1000, num_classes)

        self.init_log()

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

        # Store number of classes for cleaner computation later on.
        self.num_classes = num_classes

        # Basic encoder without prediction head.
        self.resnet50 = torchvision.models.resnet50(pretrained=True, num_classes=1000)

        self.init_log()

    def forward(self, x):

        # Generate embedding only.
        z = self.resnet50(x.unsqueeze(1).repeat(1, 3, 1, 1))

        return z

    def _prototype_step(self, batch):
        # Generic step to be used for train, val, and test.

        x, y = batch

        # Calculate embedding for batch.
        z = self.forward(x)

        prototypes, queries, y_queries = self.get_prototypes_and_queries(z, y)
        distances = self.get_euclidian_distance(prototypes, queries)

        # Use soft-max of _negative_ distance (smallest distance gives the
        # highest probability).
        log_p_y = F.log_softmax(-distances, dim=1)
        loss = F.nll_loss(log_p_y, y_queries)

        # Prediction.
        y_hat = torch.argmax(log_p_y, dim=1)

        return loss, y_hat, y_queries

    def training_step(self, batch, batch_idx):

        loss, y_hat, y_queries = self._prototype_step(batch)

        self.log('train_loss', loss, on_step=False, on_epoch=True)

        self.train_accuracy(y_hat, y_queries)
        self.log('train_acc_step', self.train_accuracy)

        return loss

    def validation_step(self, batch, batch_idx):

        loss, y_hat, y_queries = self._prototype_step(batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True)

        self.val_accuracy(y_hat, y_queries)
        self.log('val_acc_step', self.val_accuracy)

        return loss

    def test_step(self, batch, batch_idx):

        loss, y_hat, y_queries = self._prototype_step(batch)

        self.log('test_loss', loss, on_step=False, on_epoch=True)

        self.test_accuracy(y_hat, y_queries)
        self.log('test_acc_step', self.test_accuracy)

        return loss

    def get_euclidian_distance(self, prototypes, queries):
        # Calculate L2 ** 2 distance from prototypes to queries.
        #
        # See original paper implementation:
        #   https://github.com/jakesnell/prototypical-networks/blob/c9bb4d258267c11cb6e23f0a19242d24ca98ad8a/protonets/models/utils.py


        num_queries = queries.shape[0]
        num_prototypes = prototypes.shape[0]
        z_dim = queries.shape[1]

        # Expand with repeats to match tensor dimensions.
        queries = queries.unsqueeze(1).expand(num_queries, num_prototypes, z_dim)
        prototypes = prototypes.unsqueeze(0).expand(num_queries, num_prototypes, z_dim)

        return torch.pow(queries - prototypes, 2).sum(2)

    def get_prototypes_and_queries(self, z, y):
        # z is [batch_size, latent_dimension]
        # y is list of integers.

        embedding_dim = z.shape[1]

        prototypes = torch.zeros([self.num_classes, embedding_dim])

        # Array to keep track of support set.
        support_idx = torch.zeros_like(y)

        # TODO: Optimize this.
        for label_idx in range(self.num_classes):

            # Find first `num_support` embeddings matching this label.
            tmp_idx = torch.where(y == label_idx)[0][:self.num_support]
            support_set = z[tmp_idx]

            # Calculate and store mean embedding (i.e., the class "prototype").
            prototypes[label_idx, :] = support_set.mean(dim=0)

            # Store support set indices.
            support_idx[tmp_idx] += 1

        # Return all non-support embeddings as queries.
        queries = z[support_idx == 0, :]
        y_queries = y[support_idx == 0]

        return prototypes, queries, y_queries


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