# NTT_multiGPUs

## The implementation of 3D NTT on multi-GPs

Based on the four-step NTT, this project implementes the 3-Dim NTT on CUDA GPUs, which can support the large input-scale of NTT on multi GPUs.

### Install GMP and NCCL on Ubuntu 22.04

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

sudo apt install libnccl2=2.20.5-1+cuda12.4 libnccl-dev=2.20.5-1+cuda12.4
sudo apt-get install libgmp-dev
```

### Compile and Run

```
nvcc -O3 -arch=sm_XX main.cu ./ntt/ntt.cu  ./init/init.cu ./init/fr.cu ./init/twiddle_f.cu -lgmp -lnccl -o NTT

./NTT
```

### Parameter choices

- [ ] ./parameter/parameter.cuh  :    Define the parameters on CPU
- [ ] ./parameter/parameter_g.cuh:    Define the parameters on GPU

### Tests

This code runs on BN254 curve, and the big integer implementation is on the basis of yrrid's implementation in ZPrize2022. And the ntt.py is to validate the correctness of this project.

The hardware configuration is 8 x 4090 24GB. CUDA Version: 12.3, GCC Version: 11.4.0

|        | $2^{22}$ | $2^{24}$ | $2^{26}$ | $2^{28}$ |
| ------ | -------- | -------- | -------- | -------- |
| 1 GPU  | 3.6ms    | 15.1ms   | 57.2ms   | \        |
| 4 GPUs | 4.1ms    | 15.2ms   | 59.4ms   | 239.2ms  |
| 8 GPUs | 2.7ms    | 8.8ms    | 33.6ms   | 135ms    |

​	And, the running time of 4-way NTT where the input size of every NTT reaches $2^{28}$ is 430~440ms.

## This work was mainly completed by Gao Peimin.
