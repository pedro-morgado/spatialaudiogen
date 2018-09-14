
<img src="docs/img/header_1x3.png" style="width:100%">

# Spatial Audio Generation [[Project Page]](https://pedro-morgado.github.io/spatialaudiogen/)

This repository contains the source code accompanying our NIPS'18 paper:

**Self-Supervised Generation of Spatial Audio for 360 Video**  
[Pedro Morgado](http://www.svcl.ucsd.edu/~morgado), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno), [Timothy Langlois](http://langlo.is), [Oliver Wang](http://www.oliverwang.info/).  
In Neural Information Processing Systems, 2018.
[[Arxiv]](https://arxiv.org/abs/1809.02587)

```
@inproceedings{morgadoNIPS18,
	title={Self-Supervised Generation of Spatial Audio for 360\deg Video},
	author={Pedro Morgado, Nuno Vasconcelos, Timothy Langlois and Oliver Wang},
	booktitle={Neural Information Processing Systems (NIPS)},
	year={2018}
}
```

## Dependencies
* python 2.7 (see `requirements.txt` for required packages.)
* tensorflow 1.4
* youtube-dl
* ffmpeg (see my ffmpeg [configuration](https://pedro-morgado.github.io/spatialaudiogen/ffmpeg-config.txt))

## Dataset
Four datasets are described in the paper: `REC-Street`, `YT-Clean`, `YT-Music` and `YT-All`.
Since we do not own the copyright of 360 videos scraped from youtube, all datasets are released as a list of youtube links (`meta/spatialaudiogen_db.lst`).

The composition of each dataset can be seen in `meta/subsets/{DB}.lst`. Three train/test splits are also provided for each dataset (`meta/subsets/{DB}.{TRAIN|TEST}.{SPLIT}.lst`).

#### Download the dataset

`python scrapping/download.py meta/spatialaudiogen_db.lst`

`python scrapping/download.py -h` for help.

This script uses `youtube-dl` to download pre-selected audio and video formats for which the encoding scheme has been verified. 
Videos are downloaded into the `data/orig` directory.
Unfortunately, a small number of videos have been removed by the creators and will be skipped (36 out of 1189 at the time of writing).

**(Low-resolution)** Training uses low-resolution videos. If you only want to download the low resolution versions, please use the flag `--low_res`. However, you still need the full resolution videos for deployment of trained models, in order to create a good looking 360 video with spatial audio.

#### Preprocess videos

`python scrapping/preprocess.py meta/spatialaudiogen_db.lst`

`python scrapping/preprocess.py -h` for help.

This script pre-processes previously downloaded videos.
Video frames are resized to `(224x448)` and remapped into equirectangular projection at `10` fps. Audio channels are remapped into ACN format (`WYZX`) and resampled at `48000` kHz.
Preprocessed files are stored in `data/preproc` (as `.m4a` and `.mp4`) and `data/frames` (as `.jpg` and `.wav`). 
Training, evaluation and deployment code use the data in `data/frames`.

**IMPORTANT: (Low-resolution downloads)** If you opt to download only low-resolution videos in the previous step, please use the flag `--low_res` again.

**Preparing high-resolution videos for deployment** Assuming you downloaded high-resolution videos (i.e. without `--low_res` flag), you can preprocess them in high-resolution `(1920x1080)` using the flag `--prep_hr_video`. This is not required for training or evaluation, but it is recommended for deployment.

**Flow:** To extract flow maps, you'll need to first install [FlowNet2](https://github.com/lmb-freiburg/flownet2.git). Refer to FlowNet2 documentation for instructions. Then, simply provide the path to the FlowNet2 basefolder through `--flownet2_dir`. If no path is provided, flow computation is skipped.

**Note:** Downloading and preprocessing the entire dataset will take a long time, and requires `>500Gb` to store the full resolution videos (originals and preprocessed). Plan accordingly!

## Pre-trained models
Models pre-trained in each dataset can be downloaded from my OneDrive:

| [REC-Street](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/pmaravil_ucsd_edu/EY-SUbhyYdNFuwHXQkX3coYBrgEtVOSF4KhYN_21LfvpjA) | 
[YT-Clean](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/pmaravil_ucsd_edu/ES5xETC9aXFApPhynevZL1kBG8ejcMrp_DR4kHHmYNSHKQ) | 
[YT-Music](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/pmaravil_ucsd_edu/Earcl8uge_VAnr6aac6PhgABe5vs8rMoZNmpniBkH2a5HQ) | 
[YT-All](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/pmaravil_ucsd_edu/EdKpT8fNP7FBu__0mp-HkVwB12_Nlnducizm1xbZPJD1eQ) |

After downloading the `.tar.gz` files, extract them into `models/` directory.

##  Getting started
To test the models without downloading the entire dataset, we provide sample pre-processed videos ([link](https://ucsdcloud-my.sharepoint.com/:f:/g/personal/pmaravil_ucsd_edu/EhqgH0jRz0tFifQuXjBurToB_m-2o2c7gnw7kI-DIbbhAQ)).

Download and extract the demo data into `data/demo`. Then, run a pre-trained model using one of the following options.

**[Heatmap Visualization]** Colormap overlay with darker red indicating directions with higher audio energy.

`python deploy.py {MODEL_DIR} data/demo/{VIDEO_DIR}/ data/demo/{VIDEO_DIR}/video-hr.mp4 -output_fn data/demo/{VIDEO_DIR}/prediction-colormap.mp4 --save_video --overlay_map`.

**[Ambisonics]** Saved with actual spatial sound. The output must be watched with headphones using an 360 video player. See below for more information (section `Visualizing predictions`).

`python deploy.py {MODEL_DIR} data/demo/{VIDEO_DIR}/ data/demo/{VIDEO_DIR}.mp4 -output_fn data/demo/{VIDEO_DIR}-output.mp4 --save_video --VR`.

### Training and evaluation
`python train.py -h` and `python eval.py -h` for more info.

**Example usage**: Training and testing a model with an audio and rgb encoder (no flow) on `REC-Street` dataset:

`python train.py data/frames models/mymodel --subset_fn meta/subsets/REC-Street.train.1.lst --encoders audio video --batch_size 32 --n_iters 150000 --gpu 0`

`python eval.py models/mymodel --subset_fn meta/subsets/REC-Street.test.1.lst --batch_size 32 --gpu 0`


## Visualizing predictions
To view the output videos of `deploy.py`, we recommend one of the following players:

1. [https://www.audioease.com/360/webvr/](https://www.audioease.com/360/webvr/)

This is a web based Javascript VR video player, to play a video you can click "Choose File" for the Load local video option (ignore the local audio file box).
The only caveat is that you have to press "stop" before loading the video file, otherwise the previous video continues to play. If you hear audio crackling, refresh the page. If the audio does not start with the video, the browser might be blocking content. Press "Load unsafe scripts" in your browser, and reload the video again.

2. [http://www.kolor.com/gopro-vr-player/](http://www.kolor.com/gopro-vr-player/)

The GoPro VR Player (Windows and Mac) supports ambisonics audio. You can install the player and load the video files. 
