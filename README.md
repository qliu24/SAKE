# Semantic-Aware Knowledge Preservation for Zero-Shot Sketch-Based Image Retrieval

This project is our implementation of Semantic-Aware Knowledge prEservation (SAKE) for zero-shot sketch-based image retrieval.
More detailed descriptions and experimental results could be found in the [paper](https://arxiv.org/abs/1904.03208#).
![framework](utils/images/fig2.pdf)

If you find this project helpful, please consider citing our paper.
```
@article{liu2019semantic,
  author    = {Liu, Qing and Xie, Lingxi and Wang, Huiyu and Yuille, Alan},
  title     = {Semantic-Aware Knowledge Preservation for Zero-Shot Sketch-Based Image Retrieval},
  journal   = {arXiv preprint arXiv:1904.03208},
  year      = {2019},
}
```
## Dataset
Download the resized TUBerlin Ext and Sketchy Ext dataset and our zeroshot train/test split files from [here](https://cs.jhu.edu/~qliu24/ZSSBIR/dataset.zip).
Put the unzipped folder to the same directory of this project.
## Training
CSE-ResNet50 model with 64-d features on TUBerlin Ext:
```
python train_cse_resnet_tuberlin_ext.py
```
CSE-ResNet50 model with 64-d features on Sketchy Ext:
```
python train_cse_resnet_sketchy_ext.py
```
## Testing
CSE-ResNet50 model with 64-d features on TUBerlin Ext:
```
python test_cse_resnet_tuberlin_zeroshot.py
```
CSE-ResNet50 model with 64-d features on Sketchy Ext:
```
python test_cse_resnet_sketchy_zeroshot.py
```
## Pre-trained model
Our trained models and extracted zeroshot testing features can be downloaded from [here](https://cs.jhu.edu/~qliu24/ZSSBIR/cse_resnet50.zip).
