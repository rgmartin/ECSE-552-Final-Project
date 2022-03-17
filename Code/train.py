from datetime import datetime
import timeit
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from feature_extraction import AudioDataset
from dict_logger import DictLogger
import json
import pandas as pd


def init_measurements_path():

    is_colab = 'COLAB_GPU' in os.environ

    if is_colab:
        print('Running on Colab')
        measurements_dir = '/content/drive/MyDrive/ECSE-552-FP/Measurements/'
    else:
        print('Not running on Colab')
        measurements_dir = './Measurements/'

    now = datetime.today().strftime("%b-%d-%Y")
    measurements_path = os.path.join(measurements_dir, now)

    if not os.path.isdir(measurements_dir):
        try:
            os.mkdir(measurements_dir)
        except OSError as error:
            print(error)

    if not os.path.isdir(measurements_path):
        try:
            os.mkdir(measurements_path)
        except OSError as error:
            print(error)

    return measurements_path


def make_log_filenames(comment):
    now = datetime.now().strftime("%H_%M_%S-")

    profiler_filename = f"{comment}{now}profiler_output"
    plot_filename = f"{comment}{now}Loss-Acc.png"

    return profiler_filename, plot_filename


def init_trainer(logger, max_epochs, profiler):
    is_colab = 'COLAB_GPU' in os.environ

    if is_colab:
        trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=max_epochs, profiler=profiler)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=max_epochs, profiler=profiler)

    return trainer


def plot_logger_metrics(logger, measurements_path, plot_filename):
    save_path = os.path.join(measurements_path, plot_filename+'-metrics.json')
    with open(save_path, 'w') as f:
        json.dump(logger.metrics, f, indent=4)

    f, axs = plt.subplots(1, 2, figsize=(15, 5))
    font = {'size': 14}
    matplotlib.rc('font', **font)

    axs[0].plot(logger.metrics['train_loss'], lw=3, ms=8, marker='o', color='orange', label='Train')
    axs[0].set_title("Train/Val Loss")
    axs[0].set_ylabel("Loss")
    axs[0].plot(logger.metrics['val_loss'], lw=3, ms=10, marker='^', color='purple', label='Validation')
    axs[0].set_title('Classifer\nTrain/Val Loss Over Time')
    axs[0].set_xlabel("Epochs")
    axs[0].grid()

    axs[1].plot(logger.metrics['train_acc_epoch'], lw=3, ms=8, marker='o', color='orange', label='Train')
    axs[1].set_title("Classifer\nTrain/Val Accuracy")
    axs[1].set_ylabel("Accuracy")
    axs[1].plot(logger.metrics['val_acc_epoch'], lw=3, ms=10, marker='^', color='purple', label='Validation')
    axs[1].set_title('Classifier\nTrain/Val Accuracy Over Time')
    axs[1].set_xlabel("Epochs")
    axs[1].grid()

    plt.legend(loc='lower right')
    plt.savefig(os.path.join(measurements_path, plot_filename))
    plt.show()
    

def train_SimpleBinaryClassifier():
    from models import SimpleBinaryClassifier

    def plot_layer_decision_boundaries(model: torch.nn.Sequential, X: np.ndarray, name: str, N: int):
        def boundary(X1, X2, weights, bias):
            X1 = weights[0, 0] * X1
            X2 = weights[0, 1] * X2
            return np.sign(X1 + X2 + bias)

        x1 = np.linspace(-1, 1, 1000)
        x2 = np.linspace(-1, 1, 1000)
        X1_grid, X2_grid = np.meshgrid(x1, x2)
        cs = plt.contourf(X1_grid, X2_grid,
                          boundary(X1_grid, X2_grid, model[0].weight.detach(), model[0].bias.detach()),
                          cmap='binary', levels=2)
        plt.scatter(X[:N, 0], X[:N, 1], c='r', label='C1')
        plt.scatter(X[N:, 0], X[N:, 1], c='b', label='C2')
        i = 0
        plt.title('Decision Boundary - {} Data'.format(name))
        plt.legend()
        plt.colorbar(cs, ticks=[-1, 1], label="Predicted Label")
        plt.show()


    # Set the model/parameters for the training loop here
    max_epochs = 10
    batch_size = 2
    model = SimpleBinaryClassifier(2, 1, torch.nn.Sigmoid())
    logger = DictLogger()

    # Configure the names of the output files here, these should contain enough information to help distinguish them
    # from other tests with different models/parameters to make sure we don't confuse data
    name = "SBC_training_1-"
    now = datetime.now().strftime("%H_%M_%S-")
    profiler_output_file = name + now + "profiler_output"
    image_output_file = name + now + "Loss-Acc.png"
    measurements_path = init_measurements_path()

    train_data = pd.read_csv('./test_data/synthetic_train_data.csv')
    val_data = pd.read_csv('./test_data/synthetic_validation_data.csv')

    # it is necessary to reshape the tensors to be in a column format for calculations later
    train_labels = torch.Tensor(train_data.loc[:, 'label'].values).reshape((train_data.shape[0], 1))
    val_labels = torch.Tensor(train_data.loc[:, 'label'].values).reshape((val_data.shape[0], 1))
    dataset_train = TensorDataset(torch.Tensor(train_data.loc[:, ['x0', 'x1']].values), train_labels)
    dataset_val = TensorDataset(torch.Tensor(val_data.loc[:, ['x0', 'x1']].values), val_labels)

    # Todo: Investigate how the num_workers parameter here affects efficiency
    train_loader = DataLoader(dataset_train, batch_size=batch_size)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)

    profiler = pl.profiler.SimpleProfiler(dirpath=measurements_path, filename=profiler_output_file)

    # Todo: Determine the impact and selection of the GPUs on Google Colab and make sure all tensors are on it
    if 'COLAB_GPU' in os.environ:
        trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=max_epochs, profiler=profiler)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=max_epochs, profiler=profiler)

    trainer.fit(model, train_loader, val_loader)

    plot_logger_metrics(logger, measurements_path, image_output_file)

    # this is only printed for verification purposes, hence why the file isn't saved
    plot_layer_decision_boundaries(model.model, train_data.loc[:, ['x0', 'x1']].values, 'Training', train_data.shape[0] // 2)


def train_voxforge_classifier(model, data_dir, max_epoch=5, batch_size=10, dur_seconds=5, comment=""):

    # Prepare and split dataset.
    print("Preparing and splitting dataset...")
    
    name = "Resnet50 Baseline"
    start_time = timeit.default_timer()
    dataset = AudioDataset(data_dir)
    end_time = timeit.default_timer()
    print("Dataset creation in seconds: ", end_time-start_time)

    num_samples = len(dataset)
    num_train = np.floor(num_samples * 0.8).astype(int)
    num_val = num_samples - num_train

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up logs.
    measurements_path = init_measurements_path()
    profiler_filename, plot_filename = make_log_filenames(name)

    logger = DictLogger()
    profiler = pl.profiler.SimpleProfiler(dirpath=measurements_path, filename=profiler_filename)

    print("Initializing trainer...")
    trainer = init_trainer(logger, max_epoch, profiler)

    # The main attraction: train the model.
    print("Running model...")
    trainer.fit(model, train_loader, val_loader)
    plot_logger_metrics(logger, measurements_path, plot_filename)


if __name__ == "__main__":
    # from models import BaselineResnetClassifier
    #
    # model = BaselineResnetClassifier(num_classes=3)
    # data_dir = "E:\\Temp\\Voice Data"

    # train_voxforge_classifier(model, data_dir=data_dir)
    train_SimpleBinaryClassifier()