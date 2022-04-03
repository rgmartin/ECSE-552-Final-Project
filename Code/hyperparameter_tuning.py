import optuna
from dict_logger import DictLogger
from models import BaselineResnetClassifier
import torch
from feature_extraction import AudioDataset
from torch.utils.data import DataLoader
import timeit
import pytorch_lightning as pl

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