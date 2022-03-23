import pickle
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
import shutil

def init_checkpoint_path():

    is_colab = 'COLAB_GPU' in os.environ

    if is_colab:
        print('Running on Colab')
        checkpoint_dir = '/content/drive/MyDrive/ECSE-552-FP/Checkpoints/'
    else:
        print('Not running on Colab')
        checkpoint_dir = './Checkpoints/'

    now = datetime.today().strftime("%b-%d-%Y")
    checkpoint_dir = os.path.join(checkpoint_dir, now)

    if not os.path.isdir(checkpoint_dir):
        try:
            os.mkdir(checkpoint_dir)
        except OSError as error:
            print(error)

    if not os.path.isdir(checkpoint_dir):
        try:
            os.mkdir(checkpoint_dir)
        except OSError as error:
            print(error)

    return checkpoint_dir


def make_chk_filename(comment):
    now = datetime.now().strftime("%H_%M_%S-")

    chk_filename = f"{comment}{now}.chk"  

    return chk_filename

def make_study_filename(comment):
    now = datetime.now().strftime("%H_%M_%S-")

    study_filename = f"{comment}{now}.pkl"  

    return study_filename



def make_params_filename(comment):
    now = datetime.now().strftime("%H_%M_%S-")

    params_filename = f"{comment}{now}.txt"  

    return params_filename


def move_checkpoint(from_path,best_params_dict,study, new_name):
  to_path = init_checkpoint_path()
  chk_file_name = make_chk_filename (new_name + '_chk_')
  to_path_chk = os.path.join(to_path, chk_file_name)
  shutil.copy(from_path, to_path_chk)

  params_file_name = make_params_filename(new_name+'_params_')
  to_path_params = os.path.join(to_path, params_file_name)
  with open(to_path_params, 'w') as convert_file:
     convert_file.write(json.dumps(best_params_dict))

  study_file_name = make_study_filename(new_name+'_study_')
  to_path_study = os.path.join(to_path, study_file_name)
  with open(to_path_study, 'wb') as handle:
    pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)




