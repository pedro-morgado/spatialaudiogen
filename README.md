# Spatial Audio Generation [[Project Page]](https://pedro-morgado.github.io/spatialaudiogen/)

This repository contains the source code accompanying our NIPS'18 paper:

**Self-Supervised Generation of Spatial Audio for 360 Video**  
[Pedro Morgado](http://www.svcl.ucsd.edu/~morgado), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno), [Timothy Langlois](http://langlo.is), [Oliver Wang](http://www.oliverwang.info/).  
In Neural Information Processing Systems, 2018.
[[Arxiv]](https://arxiv.org/abs/1809.02587)

### Dependencies
* python 2.7 (see `requirements.txt` for required packages.)
* tensorflow 1.4
* youtube-dl
* ffmpeg (see my ffmpeg [configuration](https://pedro-morgado.github.io/spatialaudiogen/ffmpeg-config.txt))

### Tour
#### (1) Datasets
Four datasets are described in the paper: `REC-Street`, `YT-Clean`, `YT-Music` and `YT-All`.
Since we do not own the copyright of 360 videos scraped from youtube, all datasets are released as a list of youtube links (`meta/spatialaudiogen_db.lst`).

The composition of each dataset can be seen in `meta/subsets/{DB}.lst`. Three train/test splits are also provided for each dataset (`meta/subsets/{DB}.{TRAIN|TEST}.{SPLIT}.lst`).

##### Download the data with `python scrapping/download.py`. 
This script uses `youtube-dl` to download from Youtube pre-selected audio and video formats for which the encoding scheme has been verified. Unfortunately, a small number of videos have been removed by the creators (36 out of 1189 at the time of writing).
Videos are downloaded into the `data/orig` directory.

##### Preprocess videos with `python scrapping/preprocess.py`. 
This script pre-processes videos downloaded using `scrapping/download.py` script. 
Video frames are resized to `(224x448)` and remapped into equirectangular projection at `10` fps. Audio channels are remapped into ACN format (`WYZX`) and resampled at `48000` kHz.
Preprocessed files are stored in `data/preproc` (as `.m4a` and `.mp4`) and `data/frames` (as `.jpg` and `.wav`).

#### (2) Pretrained models
Run `./download_pretrained.sh` to download models pretrained in each dataset. Model weight will be stored under `models/`

(Required to run the demo.)

#### (3) Getting started
Type `python deploy.py -h` for more info.

Several demo videos are provided under `data/demo`. To run a pretrained model over one of these videos, use one of the following options.

**[Heatmap Visualization]** 
`python deploy.py data/demo/{DEMO_VIDEO}/ data/demo/{DEMO_VIDEO}.mp4 -out_base_fn data/demo/{DEMO_VIDEO}-output.mp4 --save_video --overlay_map`.

**[Ambisonics]** 
`python deploy.py data/demo/{DEMO_VIDEO}/ data/demo/{DEMO_VIDEO}.mp4 -out_base_fn data/demo/{DEMO_VIDEO}-output.mp4 --save_video --VR`.

When the ``--VR`` option is used, the output must be visualized using an 360 video player and headphones. See section (5) for more information.

**NOTE:** This script requires a version of the video preprocessed in high-resolution. For the demo examples, these files are already provided. For other videos, this can be accopmlished by setting the `prep_hr_video` to `True` in `scrapping/preprocess.py` and running it again.

#### (4) Training and evaluation
Type `python train.py -h` and `python eval.py -h` for more info.

**Example** Model with an audio and rgb encoder (no flow) on `REC-Street` dataset:

`python train.py data/frames models/mymodel --subset_fn meta/subsets/REC-Street.train.1.lst --encoders audio video --batch_size 32 --n_iters 150000 --gpu 0`

`python eval.py models/mymodel --subset_fn meta/subsets/REC-Street.test.1.lst --batch_size 32 --gpu 0`

#### (5) Visualizing predictions
To view these, we recommend one of the following players:

1. [https://www.audioease.com/360/webvr/](https://www.audioease.com/360/webvr/)

This is a web based Javascript VR video player, to play a video you can click "Choose File" for the Load local video option (ignore the local audio file box).
The only caveat is that you have to press "stop" before loading the video file, otherwise the previous video continues to play. If you hear audio crackling, refresh the page. If the audio does not start with the video, the browser might be blocking content. Press "Load unsafe scripts" in your browser, and reload the video again.

2. [http://www.kolor.com/gopro-vr-player/](http://www.kolor.com/gopro-vr-player/)

The GoPro VR Player (Windows and Mac) supports ambisonics audio. You can install the player and load the video files. 


### Citation
```
@inproceedings{morgadoNIPS18,
	title={Self-Supervised Generation of Spatial Audio for 360\deg Video},
	author={Pedro Morgado, Nuno Vasconcelos, Timothy Langlois and Oliver Wang},
	booktitle={Neural Information Processing Systems (NIPS)},
	year={2018}
}
```