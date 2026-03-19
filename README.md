# Zivid Handeye Calibration


## Introduction

This repo is ROS 1 wrapper to run ZIVID HandEye Calibration, it works in two modes:
1. Samples collection, connects to camera, and listens to TF between "root_link" and "link7"
2. Calibration, passing the dataset generated from mode 1, you get calibration in json format in dataset folder.

## How to start

first please install requirements via: `pip3 install -r requirements.txt`
```sh
$ python3 zivid_handeye_calibration/main.py

# ========================================
#       ZIVID HAND-EYE CALIBRATION
# ========================================
# [1] Capture samples (Live ROS + Zivid)
# [2] Run calibration (Process existing dataset)
# [q] Quit
# 
# Select option: 
```
please first choose 1, and hit enter, then start moving the robot with calibration board in hand (link7) to multiple poses, then for each pose (calibration board should be visible) hit `s` then enter, till you collect 10-20 samples, then finally hit `q` then enter:
```sh
# 📂 Session Directory: data/zivid_he_dataset_20260319_170109
# Commands: [s] Capture Sample | [q] Finish and Exit
# Command (s/q) > 
```

This will generate a dataset in the running directory under folder name `./data/zivid_he_{timestamp}` e.g. `zivid_he_dataset_20260319_135746` 
Please use this dataset path as dataset directory for the next stage: calibration.

Next step is to choose `EyeToHand` or `EyeInHand`:
```sh
# Mode [eth (Eye-to-Hand) / eih (Eye-in-Hand)]:
```
for example you can type `eth` or `eih` then hit enter, this will generate the calibration as a json file in the dataset directory.