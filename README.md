# KRAS Mutation Detection
### SMS DataTech Corp.

This repository is a part of the Mutation Detection project I did during my internship at SMS DataTech Corp., Tokyo, Japan. The objective is to build a classifier that can automatically detect the presence of KRAS Mutation from images of lung carcinoma cell lines.

KRAS is a type of mutation found in lung cancer tumours. We selected KRAS amongst 40 others, because this particular mutation is amongst the few that are found abundantly in lung-cancer cases in Japan and have no therapy or cure as of today. Using the InceptionV3 architecture, we are able to detect these mutations in the artificially synthesized lung-cancer cell lines with more than 90% accuracy.

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
- Install Tensorflow+Keras and dependencies from http://keras.io

### Dataset
The dataset used to train the model was provided by Dr. Daisuke Matsubara, Jichi Medical University, Tochigi, Japan. The dataset has not been made public.


### Usage
- To train and validate the model, run `train_valid.py` script under the src/ directory
- Use `--dataset-path` command-line argument to provide the path to the data folder.

## Acknowledgement
I'd sincerely like to thank Dr. Daisuke Matsubara for providing the dataset for this project. I'd also like to thank Mr. Tetsuro Matsubara and Mr. Nitish Rajoria of SMS DataTech Corp. for the incredible opprtunity to work on such an exciting project.
