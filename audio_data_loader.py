from email.mime import audio
import glob
import librosa
import os
import pandas as pd
from torch.utils.data import Dataset

class AudioDataset(Dataset):

    def __init__(self, annotation_path, audio_dir):
        self.annotations = pd.read_csv(annotation_path)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        label = self._get_label(index)
        x, sr = librosa.load(audio_path)
        return x, label

    def _get_audio_path(self, index):
        return self.annotations['filepath'][index]

    def _get_label(self, index):
        return self.annotations['category'][index]


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
    
    x, label = my_dataset[0]
    
    plt.plot(x)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()