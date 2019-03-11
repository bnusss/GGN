# A General Deep Learning Framework for Network Reconstruction

This repository will contain the official PyTorch implementation of:
<br>

**A General Deep Learning Framework for Network Reconstruction.**<br>
Zhang Zhang, Yi Zhao, Jing Liu, Shuo Wang, Ruyue Xin and Jiang Zhang<sup>\*</sup>(<sup>\*</sup>: Corresponding author) <br>
[https://arxiv.org/abs/1812.11482v2](https://arxiv.org/abs/1812.11482v2)<br>

<img src="./img/threekindofsys.png" width="800px" alt="">

<br>

### Abstract: 

Recovering latent network structure and dynamics from observed time series data are important tasks in network science, and host a wealth of potential applications. In this work, we introduce Gumbel Graph Network (GGN), a model-free, data-driven deep learning framework to accomplish network reconstruction and dynamics simulation. Our model consists of two jointly trained parts: a network generator that generating a discrete network with the Gumbel Softmax technique; and a dynamics learner that utilizing the generated network and one-step trajectory value to predict the states in future steps. We evaluate GGN on Kuramoto, Coupled Map Lattice, and Boolean networks, which exhibit continuous, discrete, and binary dynamics, respectively. Our results show that GGN can be trained to accurately recover the network structure and predict future states regardless of the types of dynamics, and outperforms competing network reconstruction methods.

### Requirements

- Python 3.6
- Pytorch 0.4

### Data Generation
To generate experimental data, you need to switch to the / data folder and run the corresponding file.
```
cd data
python data_generation_bn.py
python data_generation_cml.py
```

### Run Experiment
You can replicate the experiment for Boolean Network by simply running the file train_bn.py
```
python train_bn.py
```

To replicate the experiment for Coupled Map Lattice and Kuramoto model, please run the train_cml_kuramoto.py
```
python train_cml_kuramoto.py
```


### Cite
If you use this code in your own work, please cite our paper:
```
@article{kipf2018neural,
  title={Neural Relational Inference for Interacting Systems},
  author={Kipf, Thomas and Fetaya, Ethan and Wang, Kuan-Chieh and Welling, Max and Zemel, Richard},
  journal={arXiv preprint arXiv:1802.04687},
  year={2018}
}
```