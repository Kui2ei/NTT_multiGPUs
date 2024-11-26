#include "init.cuh"
#include "fr.cuh"
#include "twiddle_f.cuh"
#include "../parameter/parameter.cuh"
#include "../parameter/parameter_g.cuh"

__host__ void init_host_input_and_output(uint32_t** h_input, uint32_t*** h_output){
    *h_input = (uint32_t *)malloc(sizeof(uint32_t)*N*h_NUM);
    *h_output = (uint32_t **)malloc(sizeof(uint32_t*)*ntt_nums);

    for(int i=0;i<ntt_nums;i++){
        (*h_output)[i] = (uint32_t *)malloc(sizeof(uint32_t)*N*h_NUM);
    }
    
    for (int i=0; i<N; i++){
        for(int j=0; j<h_NUM; j++){
            // if the code is runned on single gpus, 
            // we should make sure that the input data is arranged by row;
            // if the code is runned on multi gpus, 
            // we should make sure that the input data is arranged by column;
            (*h_input)[i*h_NUM+j] = j+1; 
        } 
    }

    for(int num=0;num<ntt_nums;num++){
        for (int i=0; i<N; i++){
            for(int j=0; j < h_NUM; j++){
                (*h_output)[num][i*h_NUM+j] = 0; 
            } 
        }
    }
}

__host__ void init_host_OmegaTable(uint32_t** h_OmegaTable, uint32_t** h1_OmegaTableStep2, uint32_t** h2_OmegaTableStep2){
    *h_OmegaTable = (uint32_t *)malloc(sizeof(uint32_t)*(1<<LOGN_max)*h_NUM);
    *h1_OmegaTableStep2 = (uint32_t *)malloc(sizeof(uint32_t) * N_2 * h_NUM);
    *h2_OmegaTableStep2 = (uint32_t *)malloc(sizeof(uint32_t) * N * h_NUM);

    OmegaTable(*h_OmegaTable, LOGN_max);
    OmegaTableStep2(*h1_OmegaTableStep2, N_array[0], N_array[1], LOGN_2);
    OmegaTableStep2(*h2_OmegaTableStep2, N_2, N_array[2], LOGN); 
}

__host__ void init_device_input_and_output(uint32_t**** d_input, uint32_t**** d_input1, uint32_t**** d_output, uint32_t** h_input){
    *d_input = new uint32_t** [ntt_nums];
    *d_input1 = new uint32_t** [ntt_nums];
    *d_output = new uint32_t** [ntt_nums];
    for(int num=0;num<ntt_nums;num++){
        (*d_input)[num] = new uint32_t* [nDev];
        (*d_input1)[num] = new uint32_t* [nDev];
        (*d_output)[num] = new uint32_t* [nDev];
    
        for(int i=0;i<nDev;i++){
            cudaSetDevice(devs[i]);
            cudaMalloc(&((*d_input)[num])[i], size_array/nDev);
            cudaMalloc(&((*d_input1)[num])[i], size_array/nDev);
            cudaMalloc(&((*d_output)[num])[i], size_array/nDev);
            cudaMemcpyAsync(((*d_input)[num])[i], *(h_input)+i*size_array/(nDev*sizeof(uint32_t)), size_array/nDev, cudaMemcpyHostToDevice, 0);
        }
    }
}

__host__ void init_device_OmegaTable(uint32_t*** d_OmegaTable, uint32_t*** d1_OmegaTableStep2, uint32_t*** d2_OmegaTableStep2, uint32_t** h_OmegaTable, uint32_t** h1_OmegaTableStep2, uint32_t** h2_OmegaTableStep2){
    *d_OmegaTable = new uint32_t* [nDev];
    *d1_OmegaTableStep2 = new uint32_t* [nDev];
    *d2_OmegaTableStep2 = new uint32_t* [nDev];
    for(int i=0;i<nDev;i++){
        cudaSetDevice(devs[i]);
        cudaMalloc(&(*d_OmegaTable)[i], sizeof(uint32_t)*(1<<LOGN_max)*h_NUM);
        cudaMalloc(&(*d1_OmegaTableStep2)[i], sizeof(uint32_t) * N_2 * h_NUM);
        cudaMalloc(&(*d2_OmegaTableStep2)[i], sizeof(uint32_t) * N * h_NUM);

        cudaMemcpy((*d_OmegaTable)[i], *h_OmegaTable, sizeof(uint32_t)*(1u<<LOGN_max)*h_NUM, cudaMemcpyHostToDevice);
        cudaMemcpy((*d1_OmegaTableStep2)[i], *h1_OmegaTableStep2, sizeof(uint32_t) * N_2 * h_NUM, cudaMemcpyHostToDevice);
        cudaMemcpy((*d2_OmegaTableStep2)[i], *h2_OmegaTableStep2, sizeof(uint32_t) * N * h_NUM, cudaMemcpyHostToDevice);
    }
}

__host__ void convert_data_to_mont(uint32_t** h_input, uint32_t** h_OmegaTable, uint32_t** h1_OmegaTableStep2, uint32_t** h2_OmegaTableStep2){
    for (int i=0; i<N; i++){
        to_mont((*h_input)+i*h_NUM);
    }
    for (int i=0; i<(1<<LOGN_max); i++){
        to_mont((*h_OmegaTable)+i*h_NUM);
    }
    for (int i=0; i<N_2; i++){
        to_mont((*h1_OmegaTableStep2)+i*h_NUM);
    }
    for (int i=0; i<N; i++){
        to_mont((*h2_OmegaTableStep2)+i*h_NUM);
    }
}

__host__ void convert_data_back(uint32_t** h_output){
    for (int i=0; i<N; i++){
        mont_back((*h_output)+i*h_NUM);
    }
}

__host__ void free_ptr(uint32_t** ptr1, uint32_t*** ptr2, uint32_t*** ptr3, uint32_t**** ptr4, uint32_t len1, uint32_t len2, uint32_t len3, uint32_t len4){
    for(int i=0; i < len1; i++){
        free(ptr1[i]);
    }
    for(int i=0; i < len2; i++){
        for(int j=0; j < nDev; j++){
            cudaFree(ptr2[i][j]);
        }
        free(ptr2[i]);
    }
    
    for(int num=0;num<ntt_nums;num++){
        for(int i=0; i < len3; i++){
            free(ptr3[i][num]);
        }
        for(int i=0; i < len4; i++){
            for(int j=0; j < nDev; j++){
                cudaFree(ptr4[i][num][j]);
            }
            free(ptr4[i][num]);
        }
    }
    for(int i=0;i<len3;i++)free(ptr3[i]);
    for(int i=0;i<len4;i++)free(ptr4[i]);
}