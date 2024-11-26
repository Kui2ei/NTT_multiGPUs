#pragma once
#include "all2all.cuh"
#include "../parameter/parameter_g.cuh" 

cudaStream_t s[2][nDev];
ncclComm_t comms[ntt_nums][nDev];
ncclUniqueId id;

void init_stream(){
    for(int i=0;i<2;i++){
        for(int j=0;j<nDev;j++){
            cudaSetDevice(devs[j]);
            cudaStreamCreate(s[i] + j);
        }
    }
}

void free_stream(){
    for(int i=0;i<2;i++){
        for(int j=0;j<nDev;j++){
            cudaSetDevice(devs[j]);
            cudaStreamDestroy(s[i][j]);
        }
    }
}

void init_nccl(){
    for(int i=0;i<ntt_nums;i++){
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclGetUniqueId(&id));
        for(int j=0;j<nDev;j++){
            cudaSetDevice(devs[j]);
            NCCLCHECK(ncclCommInitRank(&comms[i][j], nDev, id, j));
        }
        ncclCommInitAll(&comms[i][0], nDev, devs);
        NCCLCHECK(ncclGroupEnd());
    }  
}

void free_nccl(){
    for (int i=0;i<ntt_nums;i++){
        for(int j=0;j<nDev;j++){
            NCCLCHECK(ncclCommDestroy(comms[i][j]));
        }
    }
}

inline void call_All2All(uint32_t** sendbuff, uint32_t** recvbuff, ncclDataType_t dataType, uint32_t size, uint32_t num){
    NCCLCHECK(ncclGroupStart());
    for(int i=0;i<nDev;i++){
        cudaSetDevice(devs[i]);
        NCCLAlltoall(sendbuff[i], size, dataType, recvbuff[i], size, dataType, comms[num][i], s[num][i], num);
    }
    NCCLCHECK(ncclGroupEnd());
    /*
    for(int i=0;i<nDev;i++){
        cudaSetDevice(devs[i]);
        cudaStreamSynchronize(s[num][i]);
    }
    */
}