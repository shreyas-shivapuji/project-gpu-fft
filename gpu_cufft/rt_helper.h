#define CUDA_RT_CALL(call)                                                      \
{                                                                              \
    cudaError_t cudaStatus = call;                                             \
    if (cudaStatus != cudaSuccess) {                                           \
        printf("ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
               "%s (%d).\n",                                                   \
               #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
               cudaStatus);                                                    \
        exit(1);                                                               \
    }                                                                          \
}

#define CUFFT_CALL(call)                                                        \
{                                                                              \
    cufftResult status = call;                                                 \
    if (status != CUFFT_SUCCESS) {                                             \
        printf("ERROR: CUFFT call \"%s\" in line %d of file %s failed with "   \
               "code %d.\n",                                                   \
               #call, __LINE__, __FILE__, status);                             \
        exit(1);                                                               \
    }                                                                          \
}
