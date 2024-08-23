# SCTFNet_IQA
## Installation

1) Clone this repository.
2) Platform: PyTorch 1.8.0
3) Language: Python 3.7.9
4) Ubuntu 18.04.6 LTS (GNU/Linux 5.4.0-104-generic x86\_64)
5) CUDA Version 11.2
6) GPU: NVIDIA GeForce RTX 3090 with 24GB memory

## Requirements

 Python requirements can installed by:

`pip install -r requirements.txt`

# Note

This model is running on n four  synthetic distortion datasets: LIVE, CSIQ, TID2013, KADID-10k.

Put the MOS label and the data python files into **data** folder. 

# Train
`python train_SCFTIQA2.py`

- Modify train dataset path, include distory image and saliency image.
- Modify validation dataset path

## Acknowledgment

Our codes partially borrowed from and [timm](https://github.com/rwightman/pytorch-image-models).

# TODO
* Cross dataset test code will be published
* Train on different distortion types on LIVE, TID2013, CSIQ will be published
* Code of evaluations on Waterloo Exploration Database (D-test, RMSE) will be published
