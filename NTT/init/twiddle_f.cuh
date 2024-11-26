#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void OmegaTable(uint32_t* OmegaTable, uint32_t d_min);
void OmegaTableStep2(uint32_t* OmegaTableStep2, uint32_t ROW, uint32_t COL, uint32_t RC_SUM);