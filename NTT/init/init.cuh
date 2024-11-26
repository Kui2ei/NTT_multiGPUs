#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>

__host__ void init_host_input_and_output(uint32_t** h_input, uint32_t*** h_output);
__host__ void init_host_OmegaTable(uint32_t** h_OmegaTable, uint32_t** h1_OmegaTableStep2, uint32_t** h2_OmegaTableStep2);
__host__ void init_device_input_and_output(uint32_t**** d_input, uint32_t**** d_input1, uint32_t**** d_output, uint32_t** h_input);
__host__ void init_device_OmegaTable(uint32_t*** d_OmegaTable, uint32_t*** d1_OmegaTableStep2, uint32_t*** d2_OmegaTableStep2, uint32_t** h_OmegaTable, uint32_t** h1_OmegaTableStep2, uint32_t** h2_OmegaTableStep2);
__host__ void convert_data_to_mont(uint32_t** h_input, uint32_t** h_OmegaTable, uint32_t** h1_OmegaTableStep2, uint32_t** h2_OmegaTableStep2);
__host__ void convert_data_back(uint32_t** output);
__host__ void free_ptr(uint32_t** ptr1, uint32_t*** ptr2, uint32_t*** ptr3, uint32_t**** ptr4, uint32_t len1, uint32_t len2, uint32_t len3, uint32_t len4);