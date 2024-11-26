#include "./init/warmup.cuh"
#include "./init/init.cuh"

using namespace std;

int main(){
    if (N_array[0] > 1<<11 || N_array[1] > 1<<11 || N_array[2] > 1<<11) {
        std::cerr << "Error: Value out of range!" << std::endl;
        return 1;
    }  

    if (LOGN_array[0] + LOGN_array[1] + LOGN_array[2] != LOGN) {
        std::cerr << "Error: HOST Index allocation error!" << std::endl;
        return 1;
    }

    if (N_array[0] != num_in_row || N_array[1] != num_in_col2 || N_array[2] != num_in_col1) {
        std::cerr << "Error: DEVICE Index allocation error!" << std::endl;
        return 1;
    }
    
    printf("Start initializating host memory ...\n");
    uint32_t *h_input = nullptr;
    uint32_t **h_output = nullptr;
    uint32_t *h_OmegaTable = nullptr;
    uint32_t *h1_OmegaTableStep2 = nullptr;
    uint32_t *h2_OmegaTableStep2 = nullptr;
    init_host_input_and_output(&h_input, &h_output);
    init_host_OmegaTable(&h_OmegaTable, &h1_OmegaTableStep2, &h2_OmegaTableStep2);
    convert_data_to_mont(&h_input, &h_OmegaTable, &h1_OmegaTableStep2, &h2_OmegaTableStep2);

    printf("Start initializating device memory ...\n");
    uint32_t ***d_input = nullptr;
    uint32_t ***d_input1 = nullptr;
    uint32_t ***d_output = nullptr;
    uint32_t **d_OmegaTable = nullptr;
    uint32_t **d1_OmegaTableStep2 = nullptr;
    uint32_t **d2_OmegaTableStep2 = nullptr;
    init_device_OmegaTable(&d_OmegaTable, &d1_OmegaTableStep2, &d2_OmegaTableStep2, &h_OmegaTable, &h1_OmegaTableStep2, &h2_OmegaTableStep2);
    init_device_input_and_output(&d_input, &d_input1, &d_output, &h_input);
    init_stream();

    for(uint32_t i=0;i<nDev;i++){
        cudaSetDevice(devs[i]);
        load_base_omega(h_OmegaTable);
        compute_np0<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

    for(uint32_t i=0;i<nDev;i++){
        cudaSetDevice(devs[i]);
        cudaFuncSetAttribute(NTT_COL1, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(uint32_t) * max(N_array[1], N_array[2]) * h_NUM);
        if(nDev==1) cudaFuncSetAttribute(NTT_COL2, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(uint32_t) * max(N_array[1], N_array[2]) * h_NUM);
        else cudaFuncSetAttribute(NTT_COL2_for_mul_gpu, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(uint32_t) * max(N_array[1], N_array[2]) * h_NUM);
        cudaFuncSetAttribute(NTT_ROW, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(uint32_t) * N_array[0] * h_NUM);
    }

    dim3 grid1(N_array[0], N_array[2] / nDev);
    dim3 grid2(N_array[1], N_array[2] / nDev);
    
    if(nDev>1) {
        printf("Start initializating communications ...\n");
        init_nccl();
    }
    
    printf("Start warming up ...\n");
    warmup(&d_input, &d_input1, &d_output, 
        &d_OmegaTable, &d1_OmegaTableStep2, &d2_OmegaTableStep2, grid1, grid2);

    printf("Start computing ...\n");
    float elapsedTime;
    cudaEvent_t e_start, e_stop;
	cudaEventCreate(&e_start);
	cudaEventCreate(&e_stop);
    cudaEventRecord(e_start, 0);
    
    warmup(&d_input, &d_input1, &d_output, 
       &d_OmegaTable, &d1_OmegaTableStep2, &d2_OmegaTableStep2, grid1, grid2); 
    // the d_input array's data has been changed after the first warmup,
    // so second warmup's computation results will differ with first warmup, but it will not influence the running time
    
    cudaEventRecord(e_stop, 0);
	cudaEventSynchronize(e_stop);
	cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
    printf("------------------------------------------------------------\n");
    printf("------------------------------------------------------------\n");
	printf("Computing Time: %.4f ms\n",elapsedTime);
    printf("------------------------------------------------------------\n");
    printf("------------------------------------------------------------\n");

    printf("Start copying back ...\n");
    for(uint32_t num=0;num<ntt_nums;num++){
        for(uint32_t i=0;i<nDev;i++){
            cudaSetDevice(devs[i]);
            cudaMemcpy(h_output[num]+i*(size_array/(nDev*sizeof(uint32_t))), d_output[num][i], size_array/nDev, cudaMemcpyDeviceToHost);
        }
    }

    for(uint32_t num=0;num<ntt_nums;num++){
        convert_data_back(&(h_output[num]));
        cout << "the ID of ntt is: " << num << endl;
        printf("0=0x%08x\n", h_output[num][0]);
        printf("1=0x%08x\n", h_output[num][1]);
        printf("2=0x%08x\n", h_output[num][2]);
        printf("7=0x%08x\n", h_output[num][7]);
        printf("11=0x%08x\n", h_output[num][11]);
    }
    
    printf("Start freeing memory ...\n");
    uint32_t* ptr1[4] = {h_input, h_OmegaTable, h1_OmegaTableStep2, h2_OmegaTableStep2};
    uint32_t** ptr2[3] = {d_OmegaTable, d1_OmegaTableStep2, d2_OmegaTableStep2};
    uint32_t** ptr3[1] = {h_output};
    uint32_t*** ptr4[3] = {d_input, d_input1, d_output};
    free_ptr(ptr1, ptr2, ptr3, ptr4, 4, 3, 1, 3);
    free_stream();

    if(nDev > 1){
        free_nccl();
    }
            
    return 0;
}