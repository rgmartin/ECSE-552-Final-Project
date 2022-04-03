from datetime import datetime
import json
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
from torch.utils.data import DataLoader


def init_measurements_path():
    print("Creating measurements path...")

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

    return profiler_filename, plot_filename, now


def plot_logger_metrics(logger, measurements_path, plot_filename):
    save_path = os.path.join(measurements_path, plot_filename + '-metrics.json')
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


def plot_confusion_matrix(model, model_name, dataset, data_name, measurements_path, plot_time):
    print("Generating Confusion Matrices")
    class_names = dataset.dataset.dirs
    dataloader = DataLoader(dataset, batch_size=16)

    conf_mat = torch.zeros([len(class_names), len(class_names)])
    for batch in dataloader:
        x, y = batch
        y_hat = model(x)
        # convert the logit to a class prediction
        y_hat = y_hat.softmax(dim=1)
        y_hat = y_hat.argmax(dim=1)
        conf_mat += confusion_matrix(y, y_hat, labels=list(range(len(class_names))))

    title = model_name + "\nConfusion Matrix - " + data_name
    disp = ConfusionMatrixDisplay(conf_mat.numpy(), display_labels=class_names)
    png_filename = model_name + plot_time + "ConfMat" + "-" + data_name + "-raw" + ".png"
    csv_filename = model_name + plot_time + "ConfMat" + "-" + data_name + "-raw" + ".csv"
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    plt.savefig(os.path.join(measurements_path, png_filename))
    conf_mat_df = pd.DataFrame(conf_mat.numpy())
    conf_mat_df.to_csv(os.path.join(measurements_path, csv_filename))

    # normalize the data for another view
    conf_mat = conf_mat/conf_mat.sum()
    disp = ConfusionMatrixDisplay(conf_mat.numpy(), display_labels=class_names)
    png_filename = model_name + plot_time + "ConfMat" + "-" + data_name + "-norm" + ".png"
    csv_filename = model_name + plot_time + "ConfMat" + "-" + data_name + "-norm" + ".csv"
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title(title)
    plt.savefig(os.path.join(measurements_path, png_filename))
    conf_mat_df = pd.DataFrame(conf_mat.numpy())
    conf_mat_df.to_csv(os.path.join(measurements_path, csv_filename))
