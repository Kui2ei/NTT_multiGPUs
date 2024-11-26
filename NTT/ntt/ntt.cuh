#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void NTT_COL1(uint32_t a[], uint32_t OmegaTable[], uint32_t OmegaTableStep2[], uint32_t d_output[], uint32_t ROW);
__global__ void NTT_COL2(uint32_t a[], uint32_t OmegaTable[], uint32_t OmegaTableStep2[], uint32_t d_output[], uint32_t ROW, uint32_t N_2, uint32_t Z);
__global__ void NTT_ROW(uint32_t a[], uint32_t OmegaTable[], uint32_t d_output[], uint32_t N_2, uint32_t Z, uint32_t COL);

__global__ void NTT_COL2_for_mul_gpu(uint32_t a[], uint32_t OmegaTable[], uint32_t OmegaTableStep2[], uint32_t d_output[], uint32_t ROW, uint32_t N_2, uint32_t Z);

__global__ void compute_np0(); //defined in operation.cuh
__host__ void load_base_omega(uint32_t* omega); //defined in operation.cuh
__device__ uint32_t bitReverse(uint32_t a, uint32_t bit_length);//defined in operation.cuh