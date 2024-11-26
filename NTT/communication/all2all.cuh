#pragma once
#include <stdint.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nccl.h"
#include "../parameter/parameter_g.cuh"


#define CUDACHECK(cmd) do {                           \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Failed: Cuda error %s:%d '%s'\n",       \
          __FILE__,__LINE__,cudaGetErrorString(e));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)
  
  
  #define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
      printf("Failed, NCCL error %s:%d '%s'\n",       \
          __FILE__,__LINE__,ncclGetErrorString(r));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)


static __inline__ int ncclTypeSize(ncclDataType_t type) {
    switch (type) {
        case ncclInt8:
        case ncclUint8:
            return 1;
        case ncclFloat16:
    #if defined(__CUDA_BF16_TYPES_EXIST__)
        case ncclBfloat16:
    #endif
            return 2;
        case ncclInt32:
        case ncclUint32:
        case ncclFloat32:
            return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclFloat64:
            return 8;
        default:
            return -1;
    }
}

static __inline__ ncclResult_t NCCLSendrecv(void *sendbuff, size_t sendcount, ncclDataType_t datatype, int peer,
    void *recvbuff,size_t recvcount, ncclComm_t comm, cudaStream_t stream)
{   
    auto a = ncclSend(sendbuff, sendcount, datatype, peer, comm, stream);
    auto b = ncclRecv(recvbuff, recvcount, datatype, peer, comm, stream);
    if (a||b)
    {
        if(a) return a;
        return b;
    }
    return ncclSuccess;
}

ncclResult_t NCCLAlltoall(void *sendbuff, size_t sendcount, ncclDataType_t senddatatype, void *recvbuff,
    size_t recvcount, ncclDataType_t recvdatatype, ncclComm_t comm, cudaStream_t stream, uint32_t num)
{
    for (int i = 0; i < nDev; i++)
    {
        auto a = NCCLSendrecv(static_cast<std::byte*>(sendbuff) + i * ncclTypeSize(senddatatype) * sendcount, sendcount, senddatatype, i, 
        static_cast<std::byte*>(recvbuff) + i * ncclTypeSize(recvdatatype) * recvcount, recvcount, comm, stream);
        if (a) return a;
    }
    return ncclSuccess;
}