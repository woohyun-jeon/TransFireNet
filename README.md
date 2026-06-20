# Burn severity mapping with Sentinel-2 images
This repository contains the implementation of burn severity mapping with bi-temporal Sentinel-2 multi-spectral images

## Prerequisites
* python >= 3.8
* torch >= 2.1.0
* torchvision >= 0.16.0

## Usage
1) Clone the repository and install the required dependencies with the following command:
```
$ git clone https://github.com/woohyun-jeon/TransFireNet.git
$ cd TransFireNet
$ pip install -r requirements.txt
```
2) Download datasets from here: https://drive.google.com/drive/folders/10jHd-mJX3e5rOBeLzjTteEKiYIENK28K?usp=drive_link

The directory structure should be as follows:
```
  image/
    after/  
        0000.tif
        0001.tif
        ...
    before/
        0000.tif
        0001.tif
        ...    
  label/
    0000.tif
    0001.tif
    ... 
  train.txt
  valid.txt
  test.txt
```
* It is important to mention that "data_path" argument in "configs.yaml" file, denoting the parent directory of image & label path, should be properly adjusted.
* Plus, "out_path" argument, indicating output directory of prediction and model files, should be properly adjusted.

3) Run main.py code with the following command:
```
$ cd src
$ python main.py
```

## Citation
If you find this work useful, please cite:
```
Jeon, W., Yi, J., & Kim, Y. (2025). TransFireNet: A novel dual-branch transformer network for burn severity estimation using bi-temporal Sentinel-2 imagery. Remote Sensing Letters, 16(11), 1191-1203.
```
```bibtex
@article{jeon2025transfirenet,
  title={TransFireNet: A novel dual-branch transformer network for burn severity estimation using bi-temporal Sentinel-2 imagery},
  author={Jeon, W. and Yi, J. and Kim, Y.},
  journal={Remote Sensing Letters},
  volume={16},
  number={11},
  pages={1191--1203},
  year={2025}
}
```
