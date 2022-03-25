import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm


def enforce_duration(x, target_dur_samples):
    # Truncate signal if too long. Otherwise repeat original signal, as per:
    #
    #   Pons, Jordi, Joan Serrà, and Xavier Serra. "Training neural audio
    #       classifiers with few data." In ICASSP 2019-2019 IEEE
    #       International Conference on Acoustics, Speech and Signal
    #       Processing (ICASSP), pp. 16-20. IEEE, 2019.

    if len(x) >= target_dur_samples:
        x = x[:target_dur_samples]
    else:
        num_pad_samples = target_dur_samples - len(x)

        # Loop if the array has to be repeated more than once.
        while num_pad_samples > 0:
            x = np.concatenate([x, x[:num_pad_samples]])
            num_pad_samples = target_dur_samples - len(x)

    return x


"""
Gets a melspectrogram using librosa libraty
Args:
    file_path: file of .wav file to process
    sr: sample rate to load the audio. If none is specified, librosa will utilise native 44.1kHz
    n_fft: how many samples are to be utilised per fft window
    hop_length: number of samples not analysed between consecutive fft windows
    n_mels: number of bins in which the spectogram is to be divided in the y-axis
    fmin, fmax: minimum and maximum frequencies to be considered in the specogram (Hz)
    top_db: highest intensity in db
    max_t: length of time to which the audio will be cropped.

Returns:
    spec_db: mel spectogram in db in the form of a np array.
    sr: sampling rate utilized.
"""


def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512,
                          n_mels=128, fmin=20, fmax=8300, top_db=80, max_t=5):
    wav, sr = librosa.load(file_path, sr=sr)

    # remove the silence from the beginning/end of a file to get more speech content
    wav, index = librosa.effects.trim(wav)
    wav = enforce_duration(wav, max_t * sr)
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels,
                                          fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db, sr


# todo implement transform to normalise the spectogram
def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


"""
Displays several spectograms 
Args:
    spec_db: list of one or more spectograms (np.arrays of dimension 2) to be displayed( only the first six will be 
    shown)
    sr: sampling rate or rates, depending on how many spectogram are considered in spec_db
    labels: labels to  be utilized in the same order as they are provided to annotate each spectogram.
"""


def display_spectogram(spec_db, sr, labels=None):
    ncols, nrows = 3, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 9))
    n_display = min(ncols * nrows, len(spec_db))
    for i in range(n_display):
        col_idx = int(i / ncols)
        row_idx = int(i % ncols)
        img = librosa.display.specshow(spec_db[i], x_axis='time', y_axis='mel',
                                       sr=sr[i], fmax=8000,
                                       ax=axs[col_idx][row_idx])
        axs[col_idx][row_idx].set_title(labels[i])
    fig.colorbar(img, ax=axs[0][2], format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectogram')
    plt.tight_layout()
    plt.show()


"""
AudioDataset class is a child of pytorch's Dataset
"""


class AudioDataset(Dataset):
    """
    Creates an AudioDataset
    Args:
        root_dir: directory where the audio files are to be read from the
                  this directory is expected to be composed of several subdirectories, one per language
                  each subdirectory will be scanned for wav files and each of the files will be added to the dataset.
                  files from the same subdirectory will have the same ordinal label.
    """

    def __init__(self, root_dir, max_t):
        self.root_dir = root_dir
        self.data = []
        self.sr = []
        self.labels = []
        self.dirs = [d for d in os.listdir(root_dir) if
                     os.path.isdir(os.path.join(root_dir, d))]

        for d in self.dirs:
            walk_dir = os.path.join(root_dir, d)
            # find recursively all files inside dir (including those that are in subdirs)
            for root, subdirs, files in tqdm(os.walk(walk_dir)):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    if filename.endswith('wav'):
                        spec, sr = get_melspectrogram_db(file_path, max_t=max_t)

                        # img = spec_to_image(spec)[np.newaxis,...]
                        self.data.append(spec)
                        self.labels.append(self.dirs.index(d))
                        self.sr.append(sr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == '__main__':
    # # 4. Display
    data_folder_name = 'Data'
    dataset = AudioDataset(data_folder_name)
    display_spectogram(list(dataset[:][0]), list(dataset.sr[:]),
                       list(dataset[:][1]))