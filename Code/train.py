import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sympy import permutedims
import tensorflow as tf
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from dict_logger import DictLogger
from models import *
from feature_extraction import *
import random

if 'COLAB_GPU' in os.environ:
    print('Running on CoLab')
    measurements_folder = '/content/drive/MyDrive/ECSE-552-FP/Measurements/'
else:
    print('Not running on CoLab')
    measurements_folder = './Measurements/'

measurements_path = measurements_folder + datetime.today().strftime("%b-%d-%Y") + '/'

if not os.path.isdir(measurements_folder):
    try:
        os.mkdir(measurements_folder)
    except OSError as error:
        print(error)

if not os.path.isdir(measurements_path):
    try:
        os.mkdir(measurements_path)
    except OSError as error:
        print(error)


def train_SimpleBinaryClassifier():
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

    def plot_logger_metrics(logger, filename):
        f, axs = plt.subplots(1, 2, figsize=(15, 5))
        font = {'size': 14}
        matplotlib.rc('font', **font)

        axs[0].plot(logger.metrics['train_loss'], lw=3, ms=8, marker='o', color='orange', label='Train')
        axs[0].set_title("Train/Val Loss")
        axs[0].set_ylabel("Loss")
        axs[0].plot(logger.metrics['val_loss'], lw=3, ms=10, marker='^', color='purple', label='Validation')
        axs[0].set_title('Simple Binary Classifer\nTrain/Val Loss Over Time')
        axs[0].set_xlabel("Epochs")
        axs[0].grid()

        axs[1].plot(logger.metrics['train_acc_epoch'], lw=3, ms=8, marker='o', color='orange', label='Train')
        axs[1].set_title("Simple Binary Classifer\nTrain/Val Accuracy")
        axs[1].set_ylabel("Accuracy")
        axs[1].plot(logger.metrics['val_acc_epoch'], lw=3, ms=10, marker='^', color='purple', label='Validation')
        axs[1].set_title('Simple Binary Classifier\nTrain/Val Accuracy Over Time')
        axs[1].set_xlabel("Epochs")
        axs[1].grid()

        plt.legend(loc='lower right')
        plt.savefig(measurements_path + image_output_file)
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

    plot_logger_metrics(logger, image_output_file)

    # this is only printed for verification purposes, hence why the file isn't saved
    plot_layer_decision_boundaries(model.model, train_data.loc[:, ['x0', 'x1']].values, 'Training',
                                   train_data.shape[0] // 2)

def train_SimpleAutoEncoder():
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

    def plot_logger_metrics(logger, filename):
        f, axs = plt.subplots(1, 2, figsize=(15, 5))
        font = {'size': 14}
        matplotlib.rc('font', **font)

        axs[0].plot(logger.metrics['train_loss'], lw=3, ms=8, marker='o', color='orange', label='Train')
        axs[0].set_title("Train/Val Loss")
        axs[0].set_ylabel("Loss")
        axs[0].plot(logger.metrics['val_loss'], lw=3, ms=10, marker='^', color='purple', label='Validation')
        axs[0].set_title('Simple Binary Classifer\nTrain/Val Loss Over Time')
        axs[0].set_xlabel("Epochs")
        axs[0].grid()

        axs[1].plot(logger.metrics['train_acc_epoch'], lw=3, ms=8, marker='o', color='orange', label='Train')
        axs[1].set_title("Simple Binary Classifer\nTrain/Val Accuracy")
        axs[1].set_ylabel("Accuracy")
        axs[1].plot(logger.metrics['val_acc_epoch'], lw=3, ms=10, marker='^', color='purple', label='Validation')
        axs[1].set_title('Simple Binary Classifier\nTrain/Val Accuracy Over Time')
        axs[1].set_xlabel("Epochs")
        axs[1].grid()

        plt.legend(loc='lower right')
        plt.savefig(measurements_path + image_output_file)
        plt.show()

    # Set the model/parameters for the training loop here
    max_epochs = 10
    batch_size = 1
    model = Autoencoder(num_input_channels=1, base_channel_size=32, latent_dim=32)
    logger = DictLogger()

    # Configure the names of the output files here, these should contain enough information to help distinguish them
    # from other tests with different models/parameters to make sure we don't confuse data
    name = "SAE_training_1-"
    now = datetime.now().strftime("%H_%M_%S-")
    profiler_output_file = name + now + "profiler_output"
    image_output_file = name + now + "Loss-Acc.png"

    # Todo: Investigate how the num_workers parameter here affects efficiency
    data_folder_name = '/content/drive/MyDrive/ECSE-552-FP/Data/Data_lang/DE'
    dataset = AudioDataset(data_folder_name)
    randind = list(range(len(dataset.data[:])))
    random.shuffle(randind)
    datarand = []
    labelsrand = []
    srrand = []
    train_loader = []
    val_loader = []
    cutoff = int(np.round(len(dataset.data[:])*0.8))
    
    print("Size Test")
    print(dataset.data[0].shape)
    
    print("cutoff")
    print(cutoff)
    print("dataset length")
    print(len(dataset))
    print("randind")
    print(randind)
    
    count = 0
    shapeddataset = np.reshape(dataset.data,(len(dataset),1,dataset.data[0].shape[0],dataset.data[0].shape[1]))
    
    print(shapeddataset.shape)
    
    for ind in randind:
        if count < cutoff:
            if len(train_loader) == 0:
                train_loader = shapeddataset[ind][:,0:128]
            else:    
                # flatten_spec = [j for sub in dataset.data[ind] for j in sub]
                # train_loader = np.vstack((train_loader,flatten_spec))
                train_loader = np.concatenate((train_loader,shapeddataset[ind][:,0:128]),axis=0)
        else:
            if len(val_loader) == 0:
                # flatten_spec = [j for sub in dataset.data[ind] for j in sub]
                val_loader = shapeddataset[ind][:,0:128]
            else:
                # flatten_spec = [j for sub in dataset.data[ind] for j in sub]
                # val_loader = np.vstack((val_loader,flatten_spec))
                # val_loader = np.dstack((val_loader,shapeddataset[ind]))
                val_loader = np.concatenate((val_loader,shapeddataset[ind][:,0:128]),axis=0)
                
        datarand.append(dataset.data[ind])
        labelsrand.append(dataset.labels[ind])
        srrand.append(dataset.sr[ind])
        count = count + 1
        
    newarray = dataset.data[0]
    train_loader = np.expand_dims(train_loader,axis=1)
    val_loader = np.expand_dims(val_loader,axis=1)
    print("train_loader shape")
    print(train_loader.shape)
    print("val_loader shape")
    print(val_loader.shape)
    
    # it is necessary to reshape the tensors to be in a column format for calculations later
    train_labels = torch.Tensor(labelsrand[:cutoff])
    val_labels = torch.Tensor(labelsrand[cutoff:])
    train_loader = torch.Tensor(train_loader)
    val_loader = torch.Tensor(val_loader)

    print("train_loader Tensor shape")
    print(train_loader.shape)
    
    # Todo: Investigate how the num_workers parameter here affects efficiency
    # train_loader = DataLoader(dataset_train, batch_size=batch_size)
    # val_loader = DataLoader(dataset_val, batch_size=batch_size)

    profiler = pl.profiler.SimpleProfiler(dirpath=measurements_path, filename=profiler_output_file)

    # Todo: Determine the impact and selection of the GPUs on Google Colab and make sure all tensors are on it
    if 'COLAB_GPU' in os.environ:
        trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=max_epochs, profiler=profiler)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=max_epochs, profiler=profiler)

    trainer.fit(model, train_loader, val_loader)

    plot_logger_metrics(logger, image_output_file)

    # this is only printed for verification purposes, hence why the file isn't saved
    # plot_layer_decision_boundaries(model.model, train_data.loc[:, ['x0', 'x1']].values, 'Training', train_data.shape[0] // 2)



if __name__ == "__main__":
    # train_SimpleBinaryClassifier()
    train_SimpleAutoEncoder()
