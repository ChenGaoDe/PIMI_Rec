# Exploring Periodicity and Interactivity in Multi-Interest Framework for Sequential Recommendation

Original implementation for paper [Exploring Periodicity and Interactivity in Multi-Interest Framework for Sequential Recommendation](http://arxiv.org/abs/2106.04415).

Accepted to IJCAI 2021!

## Requirements

- python 3.7
- tensorflow-gpu 1.13
- faiss-gpu 1.6.3
- numpy 1.19.2
- tensorboardX 2.1

## Run

### Installation

- Install faiss-gpu based on the instructions here: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md

### Dataset

- Original links of datasets are:

  - http://jmcauley.ucsd.edu/data/amazon/
  - https://tianchi.aliyun.com/dataset/dataDetail?dataId=649

- You can download the original datasets and preprocess them by yourself. You can run `python preprocess/data.py {dataset_name}` and `python preprocess/category.py {dataset_name}` to preprocess the datasets. Note the illegal timestamp.

### Training

You can use `python main.py --dataset {dataset_name} --time_span {time_threshold}` to train a specific model on a dataset. Other hyperparameters can be found in the code. (If you share the server with others or you want to use the specific GPU(s), you may need to set `CUDA_VISIBLE_DEVICES`.) 

For example, you can use `python main.py --dataset book --time_span 64` to train PIMI model on Book dataset.

## Acknowledgement

The structure of our code is based on [ComiRec](https://github.com/THUDM/ComiRec).
