This tester script contains useful functions and classes that can be used as a draft for audio data exploration and analysis with the librosa python library.

The main clause contains a variable data_folder_name that must be changed, so that the script reads the audio files from the correct directory.

---	The AudioDataset class is a child of pytorch's Dataset class, which is ideal for further transformations, batch processing, etc. The constructor of this class receives data_folder_name as an argument. 

	The folder structure must be as follows:

	Data
		|
		--- EN
		     |
		     subdirs
		     |
		     subdirs
		     |--------wav files
		----ES
		|
		|
		----DE
		|
		.
		.

	That is, the first subfolders of the main Data folder (contained in data_folder_name variable) must be those corresponding to the different languages. The AudioDataset constructor will read recursively each of these language subfolders looking for wav files, and will add the corresponding mel_spectograms to the dataset, using a common label that corresponds to the given language.


--- the get_melspectrogram_db auxiliary function receives a file path (with *.wav extension) and returns a mel_spectogram (2D numpy array) and the sampling rate (in Hz) utilized. Several parameters are needed for builing a spectrogram: they are listed and explained in the corresponding documentation of the function. Also the variable max_t is passed, so that each sound is cropped to be max_t seconds long.


---- the display_spectrogram auxiliary function receives a list of spectrograms (list of 2d numpy arrays), a list of sampling rates and a list of labels. The spectrograms are then displayed in subplots (default maximum of 6 subplots) with a heatmap colorbar on the side. 
