#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
using namespace std;

const uint32_t LOGN = 24;
const uint32_t N = 1u << LOGN;
const uint32_t LOGN_array[3] = {8,8,8};
// for satisfing the CUDA hardware constraint, the max dim must being less than or equal to 11

const uint32_t N_array[3] = {1u<<LOGN_array[0], 1u<<LOGN_array[1], 1u<<LOGN_array[2]};

const uint32_t h_BIT_WIDE = 256;  // Adjustment according to the actual situation
const uint32_t h_NUM = uint32_t((h_BIT_WIDE+31)/32); // Each number consists of NUM 32-bit binary unsigned integers

const uint32_t LOGN_max = max(max(LOGN_array[0], LOGN_array[1]),LOGN_array[2]);
const uint32_t LOGN_2 = LOGN_array[0]+LOGN_array[1];
const uint32_t N_2 = 1u<<LOGN_2;
const long size_array = sizeof(uint32_t) * N * h_NUM;

const uint32_t h_MODULUS[h_NUM] = { 
    0xf0000001, 0x43e1f593, 
    0x79b97091, 0x2833e848, 
    0x8181585d, 0xb85045b6, 
    0xe131a029, 0x30644e72};