# speech-to-text-voxforge

## Download the speech corpus
In order to download the speech corpus run

```shell
python downloader.py "voxforge-corpus"
```

You can additionally specify the amount of speaker directories to be downloaded using `-n` or the amount of threads to be used for the download using `-w`:

```shell
python downloader.py "voxforge-corpus" -n 20000 -w 15
```
