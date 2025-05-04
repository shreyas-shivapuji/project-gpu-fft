#define CUDA_CHECK(call) \
{ \
    auto err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

#define CUFFT_CHECK(call) \
{ \
    auto res = call; \
    if (res != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT Error, code: " << res << std::endl; \
        exit(1); \
    } \
}
