#pragma once
#include <iostream>
#include <inttypes.h>
#include <algorithm>
#include "../communication/comm.cuh"
#include "../parameter/parameter.cuh"
#include "../parameter/parameter_g.cuh"
#include "../ntt/ntt.cuh"
#include <cassert>

#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }

__forceinline__ __host__ void global_sync(){
    #pragma unroll
    for(uint32_t i=0;i<nDev;i++){
        cudaSetDevice(devs[i]);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

__forceinline__ __host__ void warmup(uint32_t**** d_input, uint32_t**** d_input1, uint32_t**** d_output, 
    uint32_t*** d_OmegaTable, uint32_t*** d1_OmegaTableStep2, uint32_t*** d2_OmegaTableStep2, dim3 grid1, dim3 grid2){
    
    #pragma unroll
    for(uint32_t i=0;i<nDev;i++){
        cudaSetDevice(devs[i]);
        NTT_COL1<<<N_2/nDev , N_array[2]/2, sizeof(uint32_t) * N_array[2] * h_NUM, s[1][i]>>>(((*d_input)[0])[i], (*d_OmegaTable)[i], (*d2_OmegaTableStep2)[i], ((*d_output)[0])[i], N_2);
    }

    global_sync();

    if(nDev>1){
        call_All2All((*d_input)[0], (*d_output)[0], ncclUint32, size_array / (sizeof(uint32_t)*nDev*nDev), 0);
    }

    for(int num=1;num<ntt_nums;num++){
        #pragma unroll
        for(uint32_t i=0;i<nDev;i++){
            cudaSetDevice(devs[i]);
            NTT_COL1<<<N_2/nDev , N_array[2]/2, sizeof(uint32_t) * N_array[2] * h_NUM, s[1][i]>>>(((*d_input)[num])[i], (*d_OmegaTable)[i], (*d2_OmegaTableStep2)[i], ((*d_output)[num])[i], N_2);
        }
    }

    for(int num=1;num<ntt_nums;num++){
        global_sync();
        if(nDev>1){
            call_All2All((*d_input)[num], (*d_output)[num], ncclUint32, size_array / (sizeof(uint32_t)*nDev*nDev), 0);
            #pragma unroll
            for(uint32_t i=0;i<nDev;i++){
                cudaSetDevice(devs[i]);
                NTT_COL2_for_mul_gpu<<<grid1, N_array[1]/2, sizeof(uint32_t) * N_array[1] * h_NUM, s[1][i]>>>(((*d_output)[num-1])[i], (*d_OmegaTable)[i], (*d1_OmegaTableStep2)[i], ((*d_input1)[num-1])[i], N_array[0], N_2, N_array[2] / nDev);
                NTT_ROW<<<grid2, N_array[0]/2, sizeof(uint32_t) * N_array[0] * h_NUM, s[1][i]>>>(((*d_input1)[num-1])[i], (*d_OmegaTable)[i], ((*d_output)[num-1])[i], N_2, N_array[2] / nDev, N_array[1]);
            }
        }
        else{
            NTT_COL2<<<grid1, N_array[1]/2, sizeof(uint32_t) * N_array[1] * h_NUM, s[1][0]>>>(((*d_input)[num-1])[0], (*d_OmegaTable)[0], (*d1_OmegaTableStep2)[0], ((*d_input)[num-1])[0], N_array[0], N_2, N_array[2] / nDev);
            NTT_ROW<<<grid2, N_array[0]/2, sizeof(uint32_t) * N_array[0] * h_NUM, s[1][0]>>>(((*d_input)[num-1])[0], (*d_OmegaTable)[0], ((*d_output)[num-1])[0], N_2, N_array[2] / nDev, N_array[1]);
        }
    }

    if(nDev>1){
        global_sync();
        #pragma unroll
        for(uint32_t i=0;i<nDev;i++){
            cudaSetDevice(devs[i]);
            NTT_COL2_for_mul_gpu<<<grid1, N_array[1]/2, sizeof(uint32_t) * N_array[1] * h_NUM, s[1][i]>>>(((*d_output)[ntt_nums-1])[i], (*d_OmegaTable)[i], (*d1_OmegaTableStep2)[i], ((*d_input1)[ntt_nums-1])[i], N_array[0], N_2, N_array[2] / nDev);
            NTT_ROW<<<grid2, N_array[0]/2, sizeof(uint32_t) * N_array[0] * h_NUM, s[1][i]>>>(((*d_input1)[ntt_nums-1])[i], (*d_OmegaTable)[i], ((*d_output)[ntt_nums-1])[i], N_2, N_array[2] / nDev, N_array[1]);
        }
    }
    else{
        global_sync();
        NTT_COL2<<<grid1, N_array[1]/2, sizeof(uint32_t) * N_array[1] * h_NUM, s[1][0]>>>(((*d_input)[ntt_nums-1])[0], (*d_OmegaTable)[0], (*d1_OmegaTableStep2)[0], ((*d_input)[ntt_nums-1])[0], N_array[0], N_2, N_array[2] / nDev);
        NTT_ROW<<<grid2, N_array[0]/2, sizeof(uint32_t) * N_array[0] * h_NUM, s[1][0]>>>(((*d_input)[ntt_nums-1])[0], (*d_OmegaTable)[0], ((*d_output)[ntt_nums-1])[0], N_2, N_array[2] / nDev, N_array[1]);
    }
    global_sync();
}

/*
__forceinline__ __host__ void warmup(uint32_t**** d_input, uint32_t**** d_input1, uint32_t**** d_output, 
    uint32_t*** d_OmegaTable, uint32_t*** d1_OmegaTableStep2, uint32_t*** d2_OmegaTableStep2, dim3 grid1, dim3 grid2){
    
    for(int num=0;num<ntt_nums;num++)
    {
        if (N_array[2]!= 1){
            #pragma unroll
            for(uint32_t i=0;i<nDev;i++){
                cudaSetDevice(devs[i]);
                NTT_COL1<<<N_2/nDev , N_array[2]/2, sizeof(uint32_t) * N_array[2] * h_NUM, s[num][i]>>>(((*d_input)[num])[i], (*d_OmegaTable)[i], (*d2_OmegaTableStep2)[i], ((*d_output)[num])[i], N_2);
            }
        }
    }

    if(nDev>1){
        for(int num=0;num<ntt_nums;num++){
            call_All2All((*d_input)[num], (*d_output)[num], ncclUint32, size_array / (sizeof(uint32_t)*nDev*nDev), num);
        }
    }

    for(int num=0;num<ntt_nums;num++){
        if(nDev>1){
            #pragma unroll
            for(uint32_t i=0;i<nDev;i++){
                cudaSetDevice(devs[i]);
                if (N_array[1]!= 1){
                    NTT_COL2_for_mul_gpu<<<grid1, N_array[1]/2, sizeof(uint32_t) * N_array[1] * h_NUM, s[num][i]>>>(((*d_output)[num])[i], (*d_OmegaTable)[i], (*d1_OmegaTableStep2)[i], ((*d_input1)[num])[i], N_array[0], N_2, N_array[2] / nDev);
                }
            }

            #pragma unroll
            for(uint32_t i=0;i<nDev;i++){
                cudaSetDevice(devs[i]);
                if (N_array[0]!= 1){
                    NTT_ROW<<<grid2, N_array[0]/2, sizeof(uint32_t) * N_array[0] * h_NUM, s[num][i]>>>(((*d_input1)[num])[i], (*d_OmegaTable)[i], ((*d_output)[num])[i], N_2, N_array[2] / nDev, N_array[1]);
                }
            }
        }
        else{
            if (N_array[1]!= 1){
                NTT_COL2<<<grid1, N_array[1]/2, sizeof(uint32_t) * N_array[1] * h_NUM, s[num][0]>>>(((*d_input)[num])[0], (*d_OmegaTable)[0], (*d1_OmegaTableStep2)[0], ((*d_output)[num])[0], N_array[0], N_2, N_array[2] / nDev);
            }
            if (N_array[0]!= 1){
                NTT_ROW<<<grid2, N_array[0]/2, sizeof(uint32_t) * N_array[0] * h_NUM, s[num][0]>>>(((*d_input)[num])[0], (*d_OmegaTable)[0], ((*d_output)[num])[0], N_2, N_array[2] / nDev, N_array[1]);
            }
        } 
    }
    
    #pragma unroll
    for(uint32_t i=0;i<nDev;i++){
        cudaSetDevice(devs[i]);
        cudaDeviceSynchronize();
    }
}
*/