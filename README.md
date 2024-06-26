# FCert: Certifiably Robust Few-Shot Classification in the Era of Foundation Models

## Introduction
This is the official PyTorch implementation of [**FCert: Certifiably Robust Few-Shot Classification in the Era of Foundation Models**](https://arxiv.org/abs/2404.08631), accepted by [IEEE S&P 2024].
In this paper, we propose the first certified defense against data poisoning attacks to few-shot classification. Below is an illustration of our method, where we utilize robust
statistics techniques to estimate a robust distance for each class. 

<p align="center">
<img src="figs/FCert_illustration.png" width="100%"/>
</p>

In this repo, we implement FCert for [CLIP](https://github.com/openai/CLIP) on three benchmark datasets. We test our code in Python 3.8, CUDA 12.3, and PyTorch 2.2.2.
## Setup
Please install [CLIP](https://github.com/openai/CLIP) by:
```
pip install git+https://github.com/openai/CLIP.git
```

## Usage

To evaluate the certification performance of FCert against individual attack for $K=15$ and $C=15$, you run:
```
python -u main.py --dataset_type cifarfs --model_type CLIP --certification_type ind --classes_per_it_val 15 --num_support_val 15
```

To evaluate the certification performance of FCert against group attack for $K=15$ and $C=15$, you run:
```
python -u main.py --dataset_type cifarfs --model_type CLIP --certification_type group --classes_per_it_val 15 --num_support_val 15
```
You can add ```--file_path``` in the command line to specify the location where you want to save the certification result.

## Citation
You can cite our paper if you use this code for your research.
```
@article{wang2024fcert,
  title={FCert: Certifiably Robust Few-Shot Classification in the Era of Foundation Models},
  author={Wang, Yanting and Zou, Wei and Jia, Jinyuan},
  journal={arXiv preprint arXiv:2404.08631},
  year={2024}
}
```

## Acknowledgement
Our code is based on [Prototypical Networks](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch) and [learn2learn](https://github.com/learnables/learn2learn/tree/master).
