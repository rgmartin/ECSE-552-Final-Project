import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models import *
from feature_extraction import *
from feature_extraction import AudioDataset
import random
import time


from dict_logger import DictLogger

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


def run_ae_train(batch_size=10, max_t=5, data_dir="/content/drive/MyDrive/ECSE-552-FP/Data"):

    DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', DEVICE)

    # Hyperparameters.
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 20

    print(f"Preparing and splitting dataset...")
    dataset = AudioDataset(data_dir, max_t=max_t)

    num_samples = len(dataset)
    num_train = np.floor(num_samples * 0.8).astype(int)
    num_val = num_samples - num_train

    print(dataset.data.shape)
    print(num_train)
    print(num_val)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [num_train,
                                                                num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # print(train_dataset.dataset.shape)
    # TODO: Replace with our new AE.
    model = Mel_ae(128, #height of the input
                   enc_type='resnet50', 
                   first_conv=False, 
                   maxpool1=False, 
                   enc_out_dim=100, 
                   kl_coeff=0.1, 
                   latent_dim=50)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  
    
    trainer = pl.Trainer()
    trainer = pl.Trainer(max_epochs=5, gpus=1)
    trainer.fit(model, train_loader)#, DataLoader(val_loader))
    
    # TODO: Replace this too, obvs.
    # log_dict = train_mel_ae(num_epochs=NUM_EPOCHS, model=model,
    #                             optimizer=optimizer, device=DEVICE, 
    #                             train_loader=train_loader,
    #                             skip_epoch_stats=True,
    #                             logging_interval=250)


def train_mel_ae(num_epochs, model, optimizer, device,
                         train_loader, loss_fn=None,
                         logging_interval=100, 
                         skip_epoch_stats=False,
                         save_model=None):
    
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}
    
    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            features = features.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            loss = loss_fn(logits, features)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))

        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict


def train_mnist_ae(num_epochs, model, optimizer, device,
                         train_loader, loss_fn=None,
                         logging_interval=100, 
                         skip_epoch_stats=False,
                         save_model=None):
    
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}
    
    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            features = features.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            loss = loss_fn(logits, features)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))

        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict


def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            logits = model(features)
            loss = loss_fn(logits, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def init_trainer(logger, max_epochs, profiler):
    is_colab = 'COLAB_GPU' in os.environ

    if is_colab:
        trainer = pl.Trainer(gpus=1, logger=logger, max_epochs=max_epochs, 
            profiler=profiler)
    else:
        trainer = pl.Trainer(logger=logger, max_epochs=max_epochs, profiler=profiler)

    return trainer


def plot_logger_metrics(logger, filename, measurements_path, plot_filename):
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


def train_voxforge_classifier(model, data_dir, max_epoch=10, batch_size=10,
                              dur_seconds=5, comment=""):
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

    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [num_train,
                                                                num_val])

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
    # from models import BaselineResnetClassifier
    #
    # model = BaselineResnetClassifier(num_classes=6)
    # data_dir = "Data/"
    # train_voxforge_classifier(model, data_dir=data_dir)

    run_ae_train()
