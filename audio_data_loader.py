from email.mime import audio
import glob
import librosa
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

class AudioDataset(Dataset):

    def __init__(
        self, 
        annotation_path: str, 
        audio_dir: str, 
        dur_seconds: float = 5.0,
        target_sr: int = 16000,
        transform = None,
    ):
        self.annotations = pd.read_csv(annotation_path)
        self.audio_dir = audio_dir

        self.target_sr = target_sr
        self.target_dur_samples = np.round(dur_seconds * target_sr).astype(int)

        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int):
        audio_path = self._get_audio_path(index)
        label = self._get_label(index)
        x, sr = librosa.load(audio_path)

        # Signal conditioning.
        x = self._enforce_mono(x)
        x = librosa.resample(x, orig_sr=sr, target_sr=self.target_sr)
        x = self._enforce_duration(x)

        if self.transform:
            x = self.transform(x)

        return x, label

    def _enforce_mono(self, x):
        return x if x.ndim != 2 else x.mean(axis=0)

    def _get_audio_path(self, index: int):
        return self.annotations['filepath'][index]

    def _get_label(self, index: int):
        return self.annotations['category'][index]

    def _enforce_duration(self, x):
        # Truncate signal if too long. Otherwise repeat original signal, as per:
        #
        #   Pons, Jordi, Joan Serrà, and Xavier Serra. "Training neural audio 
        #       classifiers with few data." In ICASSP 2019-2019 IEEE 
        #       International Conference on Acoustics, Speech and Signal 
        #       Processing (ICASSP), pp. 16-20. IEEE, 2019.

        if len(x) >= self.target_dur_samples:
            x = x[:self.target_dur_samples]
        else:
            num_pad_samples = self.target_dur_samples - len(x)

            # Loop if the array has to be repeated more than once.
            while num_pad_samples > 0:
                x = np.concatenate([x, x[:num_pad_samples]])
                num_pad_samples = self.target_dur_samples - len(x)

        return x


class MelSpectrogramTransform():
    def __init__(self, sr=16000, n_fft=400, hop_length=160, n_mels=64):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, x):
        return librosa.feature.melspectrogram(
            y=x, 
            sr=self.sr, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )



def build_annotation_file(audio_dir, log_name="dataset_annotation.csv"):

    # Assuming that class is indicated by first subdirectory.
    CLASS_DIR_DEPTH = 1

    audio_pattern = os.path.join(audio_dir, "**/*.wav")
    all_filepaths = glob.glob(audio_pattern, recursive=True)

    df = pd.DataFrame()

    for filepath in all_filepaths:

        category = filepath.split("/")[CLASS_DIR_DEPTH]
        
        datum = {
            'filepath': filepath,
            'category': category,
        }

        df = df.append(datum, ignore_index=True)

    write_path = os.path.join(audio_dir, log_name)
    df.to_csv(write_path, index=False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Build summary csv file.
    audio_dir = "Data/"
    build_annotation_file(audio_dir)

    annotation_path = os.path.join(audio_dir, "dataset_annotation.csv")
    my_dataset = AudioDataset(annotation_path, audio_dir)

    print(f"This dataset has {len(my_dataset)} entries.")
    
    x, label = my_dataset[1]
    time = np.linspace(0, 5, len(x), endpoint=False)

    plt.plot(time, x)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    my_mel = MelSpectrogramTransform()
    my_spectrogram_dataset = AudioDataset(
        annotation_path, 
        audio_dir, 
        transform=my_mel
    )

    x, label = my_spectrogram_dataset[0]

    plt.imshow(20 * np.log10(np.flipud(x)))
    plt.show()