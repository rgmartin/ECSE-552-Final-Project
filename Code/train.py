from constants import BASELINE_RESNET_NAME, MEL_AE_NAME
from dict_logger import DictLogger
from feature_extraction import AudioDataset
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import timeit
from torch.utils.data import DataLoader, random_split
from util import (init_measurements_path, make_log_filenames, 
    plot_confusion_matrix, plot_logger_metrics)


def init_dataset(data_dir, dur_seconds, crop=None, rgb_expand=False):
    print("Creating dataset")
    start_time = timeit.default_timer()
    dataset = AudioDataset(data_dir, dur_seconds, rgb_expand, crop)
    end_time = timeit.default_timer()
    print("\nDataset creation in seconds: ", end_time - start_time)

    return dataset


def split_dataset(dataset, train_split=.8):
    num_samples = len(dataset)
    num_train = np.floor(num_samples * train_split).astype(int)
    num_val = num_samples - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    return train_dataset, val_dataset


def get_datasets(data_dir, dur_seconds, train_split=.8, crop=None, rgb_expand=False):
    dataset = init_dataset(data_dir, dur_seconds, crop, rgb_expand)
    train_dataset, val_dataset = split_dataset(dataset, train_split)
    return train_dataset, val_dataset


def init_trainer(logger, max_epochs, profiler):
    print("Initializing trainer...")

    is_colab = 'COLAB_GPU' in os.environ

    early_stopping = EarlyStopping('val_loss')

    if is_colab:
        trainer = pl.Trainer(gpus=-1, auto_select_gpus=True, callbacks=[early_stopping],
                             logger=logger, max_epochs=max_epochs, profiler=profiler)
    else:
        trainer = pl.Trainer(callbacks=[early_stopping],
                             logger=logger, max_epochs=max_epochs, profiler=profiler)

    return trainer



def train_model(model, name, train_dataset, val_dataset, max_epoch=5, batch_size=10):
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    measurements_path = init_measurements_path()
    profiler_filename, plot_filename, plot_time = make_log_filenames(name)

    logger = DictLogger()
    profiler = pl.profiler.SimpleProfiler(dirpath=measurements_path, filename=profiler_filename)

    trainer = init_trainer(logger, max_epoch, profiler)

    trainer.fit(model, train_loader, val_loader)

    if name == BASELINE_RESNET_NAME:
        plot_logger_metrics(logger, measurements_path, plot_filename)
        # plot_confusion_matrix(model, name, train_dataset, "Training", measurements_path, plot_time)
        plot_confusion_matrix(model, name, val_dataset, "Validation", measurements_path, plot_time)
    elif name == MEL_AE_NAME:
        # Todo: Determine how the logger will interact with this particular model. Metrics might need to be added in the
        #  different "end" functions to facilitate this. A more generic "plot_logger_metrics" function would help achieve
        #  this and allow these models to both be called with the same training function.
        pass
        # plot_logger_metrics(logger, measurements_path, plot_filename)


if __name__ == "__main__":
    from models import BaselineResnetClassifier, Mel_ae

    data_dir = "./Data"
    # model_name = MEL_AE_NAME
    model_name = BASELINE_RESNET_NAME

    if model_name == BASELINE_RESNET_NAME:
        model = BaselineResnetClassifier(num_classes=3)
        train_dataset, val_dataset = get_datasets(data_dir=data_dir, dur_seconds=3, train_split=.8, crop=None,
                                                  rgb_expand=False)
        train_model(model, model_name, train_dataset, val_dataset, max_epoch=2)
    elif model_name == MEL_AE_NAME:
        input_height = 128
        model = Mel_ae(input_height, enc_type='resnet50', first_conv=False, maxpool1=False, enc_out_dim=2048,
                       kl_coeff=0.1, latent_dim=3)
        train_dataset, val_dataset = get_datasets(data_dir=data_dir, dur_seconds=5, train_split=.8, crop=input_height,
                                                  rgb_expand=True)
        train_model(model, model_name, train_dataset, val_dataset, max_epoch=20, batch_size=10)
