# A general deep learning framework for network reconstruction and dynamics learning

This repository will contain the official PyTorch implementation of:
<br>

**A general deep learning framework for network reconstruction and dynamics learning**<br>
Zhang Zhang, Yi Zhao, Jing Liu, Shuo Wang,Ruyi Tao, Ruyue Xin and Jiang Zhang<sup>\*</sup>(<sup>\*</sup>: Corresponding author) <br>
[Download PDF](https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0194-4#citeas)<br>

<img src="./img/threekindofsys.png" width="800px" alt="">

<br>

### Abstract: 

Many complex processes can be viewed as dynamical systems on networks. However, in real cases, only the performances of the system are known, the network structure and the dynamical rules are not observed. Therefore, recovering latent network structure and dynamics from observed time series data are important tasks because it may help us to open the black box, and even to build up the model of a complex system automatically. Although this problem hosts a wealth of potential applications in biology, earth science, and epidemics etc., conventional methods have limitations. In this work, we introduce a new framework, Gumbel Graph Network (GGN), which is a model-free, data-driven deep learning framework to accomplish the reconstruction of both network connections and the dynamics on it. Our model consists of two jointly trained parts: a network generator that generating a discrete network with the Gumbel Softmax technique; and a dynamics learner that utilizing the generated network and one-step trajectory value to predict the states in future steps. We exhibit the universality of our framework on different kinds of time-series data: with the same structure, our model can be trained to accurately recover the network structure and predict future states on continuous, discrete, and binary dynamics, and outperforms competing network reconstruction methods.

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
python train_cml_kuramoto.py --simulation-type cml --dims 1 --skip 0
```
or
```
python train_cml_kuramoto.py --simulation-type kuramoto --dims 2 --skip 1
```


### Cite
If you use this code in your own work, please cite our paper:
```
Zhang, Z., Zhao, Y., Liu, J. et al. A general deep learning framework for network reconstruction and dynamics learning. Appl Netw Sci 4, 110 (2019) doi:10.1007/s41109-019-0194-4
```
