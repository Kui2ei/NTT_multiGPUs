#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void h_SUB(uint32_t* self, const uint32_t* rhs, uint32_t* result);
void h_ADD(uint32_t* self, const uint32_t* rhs, uint32_t* result);
void h_MUL(uint32_t* self, const uint32_t* rhs, uint32_t* result);
void to_mont(uint32_t* data);
void mont_back(uint32_t* data);