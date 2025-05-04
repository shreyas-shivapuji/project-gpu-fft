#include <complex>
#include <iostream>
#include <vector>
#include <cufft.h> 
#include <cmath>
#include <iomanip>
#include <chrono>
#include "rt_helper.h"

using cpudata_t = std::vector<std::complex<float>>;
using dim_t = std::array<size_t, 3>;
using time_point_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

// fill charge density array for test - Gaussian dist
void fillChargeDensity(cpudata_t &rho, dim_t dims) {
    const size_t nx = dims[0], ny = dims[1], nz = dims[2];
    const float cx = nx / 2.0f;
    const float cy = ny / 2.0f;
    const float cz = nz / 2.0f;
    const float sigma = std::min(std::min(nx, ny), nz) / 6.0f;
    
    float max_charge = 0.0f;
    double total_charge = 0.0;
    
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t k = 0; k < nz; ++k) {
                float x = i - cx;
                float y = j - cy;
                float z = k - cz;
                float r2 = x*x + y*y + z*z;
                float charge = std::exp(-r2/(2*sigma*sigma));
                size_t idx = i*ny*nz + j*nz + k;
                rho[idx] = std::complex<float>(charge, 0.0f);
                
                max_charge = std::max(max_charge, charge);
                total_charge += charge;
            }
        }
    }
    
    std::cout << "Charge density initialized: max value = " << max_charge << ", total charge = " << total_charge << std::endl;
}

// poisson function on GPU = CUDA kernel
__global__ void applyPoissonOperator(cufftComplex *data, 
                                     int nx, int ny, int nz,
                                     float dx, float dy, float dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    // grid indices -> wave numbers
    float kx = (i <= nx/2) ? i : i - nx;
    float ky = (j <= ny/2) ? j : j - ny;
    float kz = (k <= nz/2) ? k : k - nz;

    kx *= 2.0f * M_PI / (nx * dx);
    ky *= 2.0f * M_PI / (ny * dy);
    kz *= 2.0f * M_PI / (nz * dz);

    float k2 = kx*kx + ky*ky + kz*kz;
    
    if (i == 0 && j == 0 && k == 0) {
        data[0].x = 0.0f;
        data[0].y = 0.0f;
        return;
    }

    const float scale = 1.0f / k2;
    
    int idx = i*ny*nz + j*nz + k;
    
    // divide by k^2
    data[idx].x *= scale;
    data[idx].y *= scale;
}

// normalize the output on GPU - CUDA knl
__global__ void scaleFFTData(cufftComplex *data, int total_elements, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

void findMinMax(const cpudata_t &data, float &min_val, float &max_val) {
    min_val = std::numeric_limits<float>::max();
    max_val = std::numeric_limits<float>::lowest();
    
    for (const auto &val : data) {
        min_val = std::min(min_val, val.real());
        max_val = std::max(max_val, val.real());
    }
}

// main driver code to solve the poisson equation
void solvePoissonSingleGPU(dim_t dims, cpudata_t &rho, cpudata_t &phi, 
                           float dx = 1.0f, float dy = 1.0f, float dz = 1.0f) {
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_checkpoint = t_start;
    
    std::cout << "Global grid: " << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;
    std::cout << "Grid spacing: dx=" << dx << ", dy=" << dy << ", dz=" << dz << std::endl;
    
    int device = 0;  // use the first GPU
    CUDA_RT_CALL(cudaSetDevice(device));
    
    size_t free_mem, total_mem;
    CUDA_RT_CALL(cudaMemGetInfo(&free_mem, &total_mem));

    size_t total_elements = dims[0] * dims[1] * dims[2];
    size_t data_size = total_elements * sizeof(cufftComplex);
    
    // allocate memory on GPU
    cufftComplex *d_data;
    CUDA_RT_CALL(cudaMalloc((void**)&d_data, data_size));
    
    auto t_mem_allocated = std::chrono::high_resolution_clock::now();
    t_checkpoint = t_mem_allocated;

    double local_charge_sum = 0.0;
    for (const auto &val : rho) {
        local_charge_sum += val.real();
    }
    
    CUDA_RT_CALL(cudaMemcpy(d_data, rho.data(), data_size, cudaMemcpyHostToDevice));
    
    auto t_copy_to_device = std::chrono::high_resolution_clock::now();
    std::cout << "Copy to device: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_copy_to_device - t_checkpoint).count() << " ms" << std::endl;
    t_checkpoint = t_copy_to_device;

    // 3d cuFFT plan
    cufftHandle plan;
    CUFFT_CALL(cufftPlan3d(&plan, dims[0], dims[1], dims[2], CUFFT_C2C));
    
    auto t_plan_created = std::chrono::high_resolution_clock::now();
    std::cout << "FFT plan creation: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_plan_created - t_checkpoint).count() << " ms" << std::endl;
    t_checkpoint = t_plan_created;

    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD)); // fwd FFT
    
    auto t_forward_fft = std::chrono::high_resolution_clock::now();
    std::cout << "Forward FFT: "<< std::chrono::duration_cast<std::chrono::microseconds>(t_forward_fft - t_checkpoint).count() << " us" << std::endl;
    t_checkpoint = t_forward_fft;
    
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(
        (dims[0] + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (dims[1] + threadsPerBlock.y - 1) / threadsPerBlock.y,
        (dims[2] + threadsPerBlock.z - 1) / threadsPerBlock.z
    );
    
    applyPoissonOperator<<<numBlocks, threadsPerBlock>>>(
        d_data, dims[0], dims[1], dims[2], dx, dy, dz);
    
    CUDA_RT_CALL(cudaDeviceSynchronize());
    
    auto t_poisson_op = std::chrono::high_resolution_clock::now();
    std::cout << "Poisson operator: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_poisson_op - t_checkpoint).count() << " ms" << std::endl;
    t_checkpoint = t_poisson_op;

    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));   // bwd FFt
    
    auto t_inverse_fft = std::chrono::high_resolution_clock::now();
    std::cout << "Backward FFT: " << std::chrono::duration_cast<std::chrono::microseconds>(t_inverse_fft - t_checkpoint).count() << " us" << std::endl;
    t_checkpoint = t_inverse_fft;

    // divide by N
    float scale = 1.0f / (dims[0] * dims[1] * dims[2]);
    
    // kernel launch parameters for scaling
    int threadsPerBlock1D = 256;
    int numBlocks1D = (total_elements + threadsPerBlock1D - 1) / threadsPerBlock1D;
    
    scaleFFTData<<<numBlocks1D, threadsPerBlock1D>>>(d_data, total_elements, scale);
    CUDA_RT_CALL(cudaDeviceSynchronize());
    
    auto t_scaling = std::chrono::high_resolution_clock::now();
    std::cout << "Scaling: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_scaling - t_checkpoint).count() << " ms" << std::endl;
    t_checkpoint = t_scaling;

    CUDA_RT_CALL(cudaMemcpy(phi.data(), d_data, data_size, cudaMemcpyDeviceToHost));
    
    auto t_copy_to_host = std::chrono::high_resolution_clock::now();
    std::cout << "Copy to host: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_copy_to_host - t_checkpoint).count() << " ms" << std::endl;
    t_checkpoint = t_copy_to_host;

    auto t_end = std::chrono::high_resolution_clock::now();
    
    CUDA_RT_CALL(cudaFree(d_data));
    CUFFT_CALL(cufftDestroy(plan));
    
    std::cout << "Total execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms" << std::endl;
    std::cout << "================================" << std::endl;
}

int main(int argc, char *argv[]) {
    auto main_start = std::chrono::high_resolution_clock::now();
    
    dim_t dims = {512, 512, 512};
    if (argc > 1) {
        int n = std::atoi(argv[1]);
        if (n > 0) {
            dims = {static_cast<size_t>(n), static_cast<size_t>(n), static_cast<size_t>(n)};
        }
    }
    
    // grid spacing dx = dy = dz = 1
    float dx = 1.0f;
    float dy = 1.0f;
    float dz = 1.0f;
    
    size_t total_elements = dims[0] * dims[1] * dims[2];
    
    std::cout << "Domain size: " << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;
    std::cout << "Total elements: " << total_elements << std::endl;
    std::cout << "Array memory: " << (total_elements * sizeof(std::complex<float>) / (1024 * 1024)) << " MB" << std::endl;
    
    cpudata_t rho(total_elements);
    cpudata_t phi(total_elements);
    
    fillChargeDensity(rho, dims);
    
    solvePoissonSingleGPU(dims, rho, phi, dx, dy, dz);
    
    auto main_end = std::chrono::high_resolution_clock::now();
    std::cout << "Total program time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(main_end - main_start).count() 
              << " ms" << std::endl;
    
    return 0;
}
