#include <iostream>
#include <cmath>
#include <inttypes.h>
#include "ntt.cuh"
#include "../operation/operation.cuh"
using namespace std;

__global__ void __launch_bounds__(num_in_col1/2) NTT_COL1(uint32_t a[], uint32_t OmegaTable[], uint32_t OmegaTableStep2[], uint32_t d_output[], uint32_t ROW)
{
    extern __shared__ uint32_t shared_array[];
    uint32_t* ptr=shared_array;
    const uint32_t temp = blockIdx.x;
    const uint32_t assist1=threadIdx.x/NUM;
    const uint32_t assist2=blockDim.x/NUM;
    const uint32_t assist3=threadIdx.x%NUM;
    const uint32_t assist4=log2f(num_in_col1);
    #pragma unroll
    for(uint32_t count=0;count<NUM;count++){
        const uint32_t temp1=assist1+assist2*count;
        #pragma unroll
        for(uint32_t i=0;i<2;i++){
            const uint32_t block_tid = temp1+num_in_col1*i/2;
            const uint32_t rk = bitReverse(block_tid, assist4);
            shared_array[rk*NUM + assist3]=a[(block_tid * uint32_t(ROW / nDev) + blockIdx.x)*NUM + assist3];
        }
    }
   __syncthreads();

    uint32_t length=1;
    const uint32_t bound=num_in_col1/2;

    #pragma unroll
    for (; length <= 16; length *= 2)
    {
        uint32_t step = length;
        uint32_t psi_step = threadIdx.x / step;
        uint32_t target_index = psi_step * step * 2 + threadIdx.x % step;
        uint32_t twiddle_id = length - 1 + threadIdx.x % step;

        uint32_t t[NUM];
        copyArrayValues(t, ptr + (target_index + step) * NUM);
        MUL(t, base_omega + twiddle_id * NUM, t);

        SUB(ptr + target_index * NUM, t, ptr + (target_index + step) * NUM);
        ADD(ptr + target_index * NUM, t, ptr + target_index * NUM);
        __syncthreads();
    }


    #pragma unroll
    for (; length < bound; length *= 2)
    {
        uint32_t step = length;
        uint32_t psi_step = threadIdx.x / step;
        uint32_t target_index = psi_step * step * 2 + threadIdx.x % step;
        uint32_t twiddle_id = length - 1 + threadIdx.x % step;

        uint32_t t[NUM];
        copyArrayValues(t, ptr + (target_index + step) * NUM);
        MUL(t, OmegaTable + twiddle_id * NUM, t);

        SUB(ptr + target_index * NUM, t, ptr + (target_index + step) * NUM);
        ADD(ptr + target_index * NUM, t, ptr + target_index * NUM);
        __syncthreads();
    }

    #pragma unroll
    for (; length < num_in_col1; length *= 2)
    {
        uint32_t step = length;
        uint32_t psi_step = threadIdx.x / step;
        uint32_t target_index = psi_step * step * 2 + threadIdx.x % step;
        uint32_t twiddle_id = length - 1 + threadIdx.x % step;

        uint32_t t[NUM];
        copyArrayValues(t, ptr + (target_index + step) * NUM);
        MUL(t, OmegaTable + twiddle_id * NUM, t);

        SUB(ptr + target_index * NUM, t, ptr + (target_index + step) * NUM);
        ADD(ptr + target_index * NUM, t, ptr + target_index * NUM);
        __syncwarp();
    }

    
    #pragma unroll
    for (uint32_t iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying back to global from shared
        uint32_t block_tid = threadIdx.x + iteration_num * num_in_col1 / 2;
        MUL(ptr+block_tid*NUM, OmegaTableStep2+(block_tid * ROW + temp)*NUM, a+(block_tid * uint32_t(ROW / nDev) + temp)*NUM);
    }
}

__global__ void __launch_bounds__(num_in_col2/2) NTT_COL2(uint32_t a[], uint32_t OmegaTable[], uint32_t OmegaTableStep2[], uint32_t d_output[], uint32_t ROW, uint32_t N_2, uint32_t Z)
{   
    extern __shared__ uint32_t shared_array[];

    uint32_t* ptr=shared_array;
    const uint32_t temp = blockIdx.x;
    const uint32_t assist1=threadIdx.x/NUM;
    const uint32_t assist2=blockDim.x/NUM;
    const uint32_t assist3=threadIdx.x%NUM;
    const uint32_t assist4=log2f(num_in_col2);
    
    #pragma unroll
    for(uint32_t count=0;count<NUM;count++){
        const uint32_t temp1=assist1+assist2*count;
        #pragma unroll
        for(uint32_t i=0;i<2;i++){
            const uint32_t block_tid = temp1+num_in_col2*i/2;
            const uint32_t rk = bitReverse(block_tid, assist4);
            shared_array[rk*NUM + assist3]=a[(blockIdx.y*N_2 + block_tid * ROW + blockIdx.x)*NUM + assist3];
        }
    }
    __syncthreads();

    uint32_t length=1;
    const uint32_t bound=num_in_col2/2;

    #pragma unroll
    for (; length <= 16; length *= 2)
    {
        uint32_t step = length;
        uint32_t psi_step = threadIdx.x / step;
        uint32_t target_index = psi_step * step * 2 + threadIdx.x % step;
        uint32_t twiddle_id = length - 1 + threadIdx.x % step;

        uint32_t t[NUM];
        copyArrayValues(t, ptr + (target_index + step) * NUM);
        MUL(t, base_omega + twiddle_id * NUM, t);

        SUB(ptr + target_index * NUM, t, ptr + (target_index + step) * NUM);
        ADD(ptr + target_index * NUM, t, ptr + target_index * NUM);
        __syncthreads();
    }

    #pragma unroll
    for (; length < bound; length *= 2)
    {
        uint32_t step = length;
        uint32_t psi_step = threadIdx.x / step;
        uint32_t target_index = psi_step * step * 2 + threadIdx.x % step;
        uint32_t twiddle_id = length - 1 + threadIdx.x % step;

        uint32_t t[NUM];
        copyArrayValues(t, ptr + (target_index + step) * NUM);
        MUL(t, OmegaTable + twiddle_id * NUM, t);

        SUB(ptr + target_index * NUM, t, ptr + (target_index + step) * NUM);
        ADD(ptr + target_index * NUM, t, ptr + target_index * NUM);
        __syncthreads();
    }

    #pragma unroll
    for (; length < num_in_col2; length *= 2)
    {
        uint32_t step = length;
        uint32_t psi_step = threadIdx.x / step;
        uint32_t target_index = psi_step * step * 2 + threadIdx.x % step;
        uint32_t twiddle_id = length - 1 + threadIdx.x % step;

        uint32_t t[NUM];
        copyArrayValues(t, ptr + (target_index + step) * NUM);
        MUL(t, OmegaTable + twiddle_id * NUM, t);

        SUB(ptr + target_index * NUM, t, ptr + (target_index + step) * NUM);
        ADD(ptr + target_index * NUM, t, ptr + target_index * NUM);
        __syncwarp();
    }

    
    #pragma unroll
    for (uint32_t iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying back to global from shared
        uint32_t block_tid = threadIdx.x + iteration_num * num_in_col2 / 2;
        MUL(ptr+block_tid*NUM, OmegaTableStep2+(block_tid * ROW + temp)*NUM, a+(blockIdx.y*N_2 + block_tid * ROW + temp)*NUM);
    }
}

__global__ void __launch_bounds__(num_in_row/2) NTT_ROW(uint32_t a[], uint32_t OmegaTable[], uint32_t d_output[], uint32_t N_2, uint32_t Z, uint32_t COL)
{   
    uint32_t local_tid = threadIdx.x;

    extern __shared__ uint32_t shared_array[];

    #pragma unroll
    for (uint32_t iteration_num = 0; iteration_num < 2; iteration_num++)
    {  
        uint32_t block_tid = local_tid + iteration_num * num_in_row / 2; 
		uint32_t rk = bitReverse(block_tid, log2f(num_in_row));
		copyArrayValues(shared_array+rk*NUM, a+(blockIdx.y*N_2 + block_tid + blockIdx.x * num_in_row)*NUM);
    }
    __syncthreads();

    uint32_t length=1;
    const uint32_t bound=num_in_row/2;

    #pragma unroll
    for (; length <= 16; length *= 2)
    {
        uint32_t step = length;
		uint32_t psi_step = local_tid / step; 
		uint32_t target_index = psi_step * step * 2 + local_tid % step;
		uint32_t twiddle_id = length - 1 + local_tid % step;

		uint32_t t[NUM];
		copyArrayValues(t, shared_array+(target_index + step)*NUM);
        MUL(t, base_omega + twiddle_id * NUM, t);

		SUB(shared_array + target_index * NUM, t, shared_array+(target_index + step)*NUM);
		ADD(shared_array+target_index*NUM, t, shared_array+target_index*NUM);
		__syncthreads();
    }

    #pragma unroll
    for (; length < bound; length *= 2)
    { 
		uint32_t step = length;
		uint32_t psi_step = local_tid / step; 
		uint32_t target_index = psi_step * step * 2 + local_tid % step;
		uint32_t twiddle_id = length - 1 + local_tid % step;

		uint32_t t[NUM];
		copyArrayValues(t, shared_array+(target_index + step)*NUM);
        MUL(t, OmegaTable + twiddle_id * NUM, t);

		SUB(shared_array + target_index * NUM, t, shared_array+(target_index + step)*NUM);
		ADD(shared_array+target_index*NUM, t, shared_array+target_index*NUM);
		__syncthreads();
	}

    #pragma unroll
    for (; length < num_in_row; length *= 2)
    { 
		uint32_t step = length;
		uint32_t psi_step = local_tid / step; 
		uint32_t target_index = psi_step * step * 2 + local_tid % step;
		uint32_t twiddle_id = length - 1 + local_tid % step;

		uint32_t t[NUM];
		copyArrayValues(t, shared_array+(target_index + step)*NUM);
        MUL(t, OmegaTable + twiddle_id * NUM, t);
        
		SUB(shared_array + target_index * NUM, t, shared_array+(target_index + step)*NUM);
		ADD(shared_array+target_index*NUM, t, shared_array+target_index*NUM);
		__syncwarp();
	}
    
    
    #pragma unroll
    for (uint32_t iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying back to global from shared
        uint32_t block_tid = local_tid + iteration_num * num_in_row / 2;
        copyArrayValues(d_output+(blockIdx.y + (blockIdx.x + block_tid * COL) * Z)*NUM, shared_array+block_tid*NUM); //tranpose
    }
}


__global__ void __launch_bounds__(num_in_col2/2) NTT_COL2_for_mul_gpu(uint32_t a[], uint32_t OmegaTable[], uint32_t OmegaTableStep2[], uint32_t d_output[], uint32_t ROW, uint32_t N_2, uint32_t Z)
{   
    extern __shared__ uint32_t shared_array[];

    uint32_t* ptr=shared_array;
    const uint32_t temp = blockIdx.x;
    const uint32_t assist1 = threadIdx.x/NUM;
    const uint32_t assist2 = blockDim.x/NUM;
    const uint32_t assist3 = threadIdx.x%NUM;
    const uint32_t assist4 = log2f(num_in_col2);
    const uint32_t assist5 = N_2 / nDev;
    const uint32_t assist6 = assist5 * num_in_col1;
    
    #pragma unroll
    for(uint32_t count=0;count<NUM;count++){
        const uint32_t temp1=assist1+assist2*count;
        #pragma unroll
        for(uint32_t i=0;i<2;i++){
            const uint32_t block_tid = temp1+num_in_col2*i/2;
            const uint32_t rk = bitReverse(block_tid, assist4);
            const uint32_t location = blockIdx.y*N_2 + block_tid * ROW + blockIdx.x;
            const uint32_t block_id = location / assist5;

            shared_array[rk*NUM + assist3]=a[(uint32_t((block_id % nDev) * assist6 / nDev) + (location % assist5) + uint32_t(block_id / nDev) * assist5)*NUM + assist3];
        }
    }
    __syncthreads();
    
    uint32_t length=1;
    const uint32_t bound=num_in_col2/2;

    #pragma unroll
    for (; length <= 16; length *= 2)
    {
        uint32_t step = length;
        uint32_t psi_step = threadIdx.x / step;
        uint32_t target_index = psi_step * step * 2 + threadIdx.x % step;
        uint32_t twiddle_id = length - 1 + threadIdx.x % step;

        uint32_t t[NUM];
        copyArrayValues(t, ptr + (target_index + step) * NUM);
        MUL(t, base_omega + twiddle_id * NUM, t);

        SUB(ptr + target_index * NUM, t, ptr + (target_index + step) * NUM);
        ADD(ptr + target_index * NUM, t, ptr + target_index * NUM);
        __syncthreads();
    }

    #pragma unroll
    for (; length < bound; length *= 2)
    {
        uint32_t step = length;
        uint32_t psi_step = threadIdx.x / step;
        uint32_t target_index = psi_step * step * 2 + threadIdx.x % step;
        uint32_t twiddle_id = length - 1 + threadIdx.x % step;

        uint32_t t[NUM];
        copyArrayValues(t, ptr + (target_index + step) * NUM);
        MUL(t, OmegaTable + twiddle_id * NUM, t);

        SUB(ptr + target_index * NUM, t, ptr + (target_index + step) * NUM);
        ADD(ptr + target_index * NUM, t, ptr + target_index * NUM);
        __syncthreads();
    }

    #pragma unroll
    for (; length < num_in_col2; length *= 2)
    {
        uint32_t step = length;
        uint32_t psi_step = threadIdx.x / step;
        uint32_t target_index = psi_step * step * 2 + threadIdx.x % step;
        uint32_t twiddle_id = length - 1 + threadIdx.x % step;

        uint32_t t[NUM];
        copyArrayValues(t, ptr + (target_index + step) * NUM);
        MUL(t, OmegaTable + twiddle_id * NUM, t);

        SUB(ptr + target_index * NUM, t, ptr + (target_index + step) * NUM);
        ADD(ptr + target_index * NUM, t, ptr + target_index * NUM);
        __syncwarp();
    }


    #pragma unroll
    for (uint32_t iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying back to global from shared
        uint32_t block_tid = threadIdx.x + iteration_num * num_in_col2 / 2;
        MUL(ptr+block_tid*NUM, OmegaTableStep2+(block_tid * ROW + temp)*NUM, d_output+(blockIdx.y*N_2 + block_tid * ROW + temp)*NUM);
    }
}
