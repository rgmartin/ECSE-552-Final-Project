from datetime import datetime
import timeit
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from feature_extraction import AudioDataset
from dict_logger import DictLogger
import json
import optuna
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import ConfusionMatrixDisplay

BASELINE_RESNET_NAME = "Baseline Resnet"
MEL_AE_NAME = "Mel AE"

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
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])
    return train_dataset, val_dataset


def get_datasets(data_dir, dur_seconds, train_split=.8, crop=None, rgb_expand=False):
    dataset = init_dataset(data_dir, dur_seconds, crop, rgb_expand)
    train_dataset, val_dataset = split_dataset(dataset, train_split)
    return train_dataset, val_dataset


def make_log_filenames(comment):
    now = datetime.now().strftime("%H_%M_%S-")

    profiler_filename = f"{comment}{now}profiler_output"
    plot_filename = f"{comment}{now}Loss-Acc.png"

    return profiler_filename, plot_filename, now


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


def plot_confusion_matrix(model, name, train_dataset, val_dataset, measurements_path, plot_time):
    def generate_and_save_confusion_matrices(y_targ, y_pred, model_name, data_name, class_names, save_path, plot_time):
        titles_options = [
            (model_name + "\nConfusion Matrix - " + data_name, None),
            (model_name + "\nConfusion Matrix - " + data_name, "true"),
        ]

        for title, normalize in titles_options:
            disp = ConfusionMatrixDisplay.from_predictions(
                y_targ,
                y_pred,
                display_labels=class_names,
                cmap=plt.cm.Blues,
                normalize=normalize,
            )
            disp.ax_.set_title(title)

            save_filename = model_name + plot_time + "ConfMat" + "-" + data_name
            if normalize:
                save_filename = save_filename + "-norm"
            else:
                save_filename = save_filename + "-raw"
            save_filename = save_filename + ".png"
            plt.savefig(os.path.join(save_path, save_filename))

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

    for full_set in train_loader:
        x, y_train = full_set
        y_hat_train = model(x)

    for full_set in val_loader:
        x, y_val = full_set
        y_hat_val = model(x)

    y_hat_train = y_hat_train.softmax(dim=1)
    y_hat_val = y_hat_val.softmax(dim=1)

    y_hat_train = y_hat_train.argmax(dim=1)
    y_hat_val = y_hat_val.argmax(dim=1)

    class_names = ["C0", "C1", "C2"]
    generate_and_save_confusion_matrices(y_train, y_hat_train, name, "Training", class_names, measurements_path, plot_time)
    generate_and_save_confusion_matrices(y_val, y_hat_val, name, "Valid", class_names, measurements_path, plot_time)


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
        plot_confusion_matrix(model, name, train_dataset, val_dataset, measurements_path, plot_time)
    elif name == MEL_AE_NAME:
        # Todo: Determine how the logger will interact with this particular model. Metrics might need to be added in the
        #  different "end" functions to facilitate this. A more generic "plot_logger_metrics" function would help achieve
        #  this and allow these models to both be called with the same training function.
        pass
        # plot_logger_metrics(logger, measurements_path, plot_filename)


def hp_tuning_voxforge_classifier(data_dir, max_epoch=10, batch_size=10, dur_seconds=5, comment=""):
    # Hyperparameter tuning
    def objective(trial):
        model = BaselineResnetClassifier(num_classes=3)

        logger = DictLogger()
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_acc_step",
            dirpath='./Checkpoints',
            mode='max',
            filename='{epoch:02d}-{val_acc_step:.2f}'
        )

        trainer = pl.Trainer(
            logger=logger,
            max_epochs=5,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[checkpoint_callback
                       ],
        )

        # we optimize max_t and batch_size
        max_t = trial.suggest_int("max_t", 1, 5)
        batch_size = trial.suggest_int('batch_size', 4, 64, log=True)

        # Prepare and split dataset.
        print(f"Preparing and splitting dataset...")

        name = "Resnet50 Baseline"
        start_time = timeit.default_timer()
        dataset = AudioDataset(data_dir, max_t=max_t)
        end_time = timeit.default_timer()
        print("Dataset creation in seconds: ", end_time - start_time)

        num_samples = len(dataset)
        num_train = np.floor(num_samples * 0.8).astype(int)
        num_val = num_samples - num_train

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val],
                                                                   generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        hyperparameters = dict(max_t=max_t, batch_size=batch_size)
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, train_loader, val_loader)

        return trainer.callback_metrics["val_acc_step"].item()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trainer.checkpoint_callback.best_model_path, trial.params, study


if __name__ == "__main__":
    from models import *

    data_dir = "E:\\Temp\\Voice Data"
    # model_name = MEL_AE_NAME
    model_name = BASELINE_RESNET_NAME

    if model_name == BASELINE_RESNET_NAME:
        model = BaselineResnetClassifier(num_classes=3)
        train_dataset, val_dataset = get_datasets(data_dir=data_dir, dur_seconds=5, train_split=.8, crop=None,
                                                  rgb_expand=False)
        train_model(model, model_name, train_dataset, val_dataset)
    elif model_name == MEL_AE_NAME:
        input_height = 128
        model = Mel_ae(input_height, enc_type='resnet50', first_conv=False, maxpool1=False, enc_out_dim=2048,
                       kl_coeff=0.1, latent_dim=3)
        train_dataset, val_dataset = get_datasets(data_dir=data_dir, dur_seconds=5, train_split=.8, crop=input_height,
                                                  rgb_expand=True)
        train_model(model, model_name, train_dataset, val_dataset, max_epoch=20, batch_size=10)
