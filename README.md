# KRAS Mutation Detection
### SMS DataTech Corp.

This repository is a part of the Mutation Detection project I did during my internship at SMS DataTech Corp., Tokyo, Japan. The objective is to build a classifier that can automatically detect the presence of KRAS Mutation from images of lung carcinoma cell lines.

We use a pre-trained InceptionV3 architecture to 

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN

## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/ninadakolekar/KRAS-Mutation-Detection
cd KRAS-Mutation-Detection
```
- Install Tensorflow+Keras and dependencies from http://pytorch.org

### Dataset
The dataset used to train the model was provided by Dr. Daisuke Matsubara, Jichi Medical University, Tochigi, Japan. The dataset has not been made public.


### Usage
- To train and validate the model, run `train_valid.py` script under the src/ directory
- Use `--dataset-path` command-line argument to provide the path to the data folder.
```

