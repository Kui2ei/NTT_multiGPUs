#pragma once
#include "bigint.cuh"

__host__ void load_base_omega(uint32_t* omega){
    cudaMemcpyToSymbol(base_omega, omega, sizeof(uint32_t) * NUM * (1+2+4+8+16));
}

__global__ void compute_np0(){
    np0 =  computeNP0(MODULUS[0]);
}

__device__ __forceinline__ void copyArrayValues(uint32_t* dest, uint32_t* src) {
    #pragma unroll
    for (int i = 0; i < NUM; i+=4) {
        *(__int128*)&dest[i] = *(__int128*)&src[i];
    }
    /*
    for (int i = 0; i < NUM; i++) {
        dest[i] = src[i];
    }*/
}

__device__ uint32_t bitReverse(uint32_t a, uint32_t bit_length)  // reverses the bits for twiddle factor calculation 将1101反转为1011
{
    uint32_t res = 0;
    
    for (int i = 0; i < bit_length; i++)
    {
        res <<= 1;
        res = (a & 1) | res;
        a >>= 1;
    }

    return res;
}

__device__ __forceinline__ void SUB(uint32_t* self, const uint32_t* rhs, uint32_t* result) {
    uint32_t a[NUM+1]={},b[NUM+1]={},c[NUM+1]={};
    copyArrayValues(a,self);
    copyArrayValues(b,(uint32_t*)rhs);

    mp_sub<NUM+1>(c,a,b);
    if(mp_comp_gt<NUM+1>(c,a)){
        mp_add<NUM+1>(c,c,MODULUS_);
    }
    copyArrayValues(result,c);
}

__device__ __forceinline__ void ADD(uint32_t* self, const uint32_t* rhs, uint32_t* result) {
    uint32_t a[NUM+1]={},b[NUM+1]={},c[NUM+1]={};
    copyArrayValues(a,self);
    copyArrayValues(b,(uint32_t*)rhs);
    
    mp_add<NUM+1>(c,a,b);
    if(mp_comp_ge<NUM+1>(c,MODULUS_)){
        mp_sub<NUM+1>(c,c,MODULUS_);
    }
    copyArrayValues(result,c);
}

__device__ __forceinline__ void MUL(uint32_t* d_a, uint32_t* d_b, uint32_t* d_r){
    uint64_t evenOdd[NUM];
	bool carry=mp_mul_red_cl<NUM>(evenOdd,d_a,d_b,MODULUS);
	mp_merge_cl<NUM>(d_r,evenOdd,carry);
}