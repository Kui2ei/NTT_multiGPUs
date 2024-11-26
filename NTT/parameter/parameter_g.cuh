#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>

using namespace std;
const uint32_t nDev = 8;
const uint32_t ntt_nums = 4;
const int devs[nDev] = {0,1,2,3,4,5,6,7};

__device__ const uint32_t BIT_WIDE = 256;  // Adjustment according to the actual situation
__device__ const uint32_t NUM = uint32_t((BIT_WIDE+31)/32); // Each number consists of NUM 32-bit binary unsigned integers

__device__ const uint32_t num_in_col1 = (1<<8); //N_array[2]
__device__ const uint32_t num_in_col2 = (1<<8); //N_array[1]
__device__ const uint32_t num_in_row = (1<<8); //N_array[0]

__device__ __constant__ uint32_t base_omega[NUM*(1+2+4+8+16)]={};
__device__ uint32_t np0;

__device__ const uint32_t MODULUS[NUM] = { 
    0xf0000001, 0x43e1f593, 
    0x79b97091, 0x2833e848, 
    0x8181585d, 0xb85045b6, 
    0xe131a029, 0x30644e72};
__device__ const uint32_t MODULUS_[NUM+1] = { 
    0xf0000001, 0x43e1f593, 
    0x79b97091, 0x2833e848, 
    0x8181585d, 0xb85045b6, 
    0xe131a029, 0x30644e72, 0x0};