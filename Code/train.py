from datetime import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from audio_data_loader import AudioDataset, MelSpectrogramTransform, build_annotation_file
from dict_logger import DictLogger


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
        trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=max_epochs, 
            profiler=profiler)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=max_epochs, profiler=profiler)

    return trainer


def plot_logger_metrics(logger, measurements_path, plot_filename):
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
    

def train_voxforge_classifier(model, data_dir, max_epoch=10, batch_size=10, dur_seconds=5, comment=""):

    # Prepare and split dataset.
    print("Preparing and splitting dataset...")
    
    build_annotation_file(data_dir, log_name='dataset_annotation.csv')
    annotation_path = os.path.join(data_dir, "dataset_annotation.csv")
    transform = MelSpectrogramTransform()
    dataset = AudioDataset(annotation_path, data_dir, dur_seconds=dur_seconds,
        transform=transform)

    num_samples = len(dataset)
    num_train = np.floor(num_samples * 0.8).astype(int)
    num_val = num_samples - num_train

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up logs.
    measurements_path = init_measurements_path()
    profiler_filename, plot_filename = make_log_filenames(comment)

    logger = DictLogger()
    profiler = pl.profiler.SimpleProfiler(dirpath=measurements_path, 
        filename=profiler_filename)

    print("Initializing trainer...")
    trainer = init_trainer(logger, max_epoch, profiler)

    # The main attraction: train the model.
    print("Running model...")
    trainer.fit(model, train_loader, val_loader)
    plot_logger_metrics(logger, measurements_path, plot_filename)


if __name__ == "__main__":
    from models import BaselineResnetClassifier

    model = BaselineResnetClassifier(num_classes=6)
    data_dir = "Data/"

    train_voxforge_classifier(model, data_dir=data_dir)