#include <iostream>
#include <vector>
#include <array>
#include <complex>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <cufftXt.h>
#include "rt_helper.h"

using cpu_data_t = std::vector<std::complex<float>>;
using dims_t = std::array<size_t, 3>;
using time_point_t = std::chrono::high_resolution_clock::time_point;

void generateCharge(cpu_data_t &rho, dims_t dims) {
    size_t nx = dims[0], ny = dims[1], nz = dims[2];
    float cx = nx / 2.0f, cy = ny / 2.0f, cz = nz / 2.0f;
    float sigma = std::min({nx, ny, nz}) / 6.0f;

    float maxVal = 0.0f;
    double total = 0.0;

    for (size_t x = 0; x < nx; ++x) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t z = 0; z < nz; ++z) {
                float dx = x - cx, dy = y - cy, dz = z - cz;
                float r2 = dx*dx + dy*dy + dz*dz;
                float value = std::exp(-r2 / (2 * sigma * sigma));
                size_t idx = x * ny * nz + y * nz + z;

                rho[idx] = std::complex<float>(value, 0.0f);
                maxVal = std::max(maxVal, value);
                total += value;
            }
        }
    }

    std::cout << "Charge initialized. Max: " << maxVal << ", Sum: " << total << "\n";
}

extern "C" __global__ void applyPoisson(cufftComplex *data, int nx, int ny, int nz,
                                        float dx, float dy, float dz,
                                        int yStart, int ySize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int jGlobal = j + yStart;
    if (i >= nx || j >= ySize || k >= nz || jGlobal >= ny) return;

    float kx = (i <= nx/2) ? i : i - nx;
    float ky = (jGlobal <= ny/2) ? jGlobal : jGlobal - ny;
    float kz = (k <= nz/2) ? k : k - nz;

    kx *= 2 * M_PI / (nx * dx);
    ky *= 2 * M_PI / (ny * dy);
    kz *= 2 * M_PI / (nz * dz);

    float k2 = kx*kx + ky*ky + kz*kz;

    int idx = i * ySize * nz + j * nz + k;

    if (i == 0 && jGlobal == 0 && k == 0) {
        data[idx].x = 0.0f;
        data[idx].y = 0.0f;
        return;
    }

    float scale = 1.0f / k2;
    data[idx].x *= scale;
    data[idx].y *= scale;
}

__global__ void scaleData(cufftComplex *data, int count, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx].x *= factor;
        data[idx].y *= factor;
    }
}

void getMinMax(const cpu_data_t &data, float &minVal, float &maxVal) {
    minVal = std::numeric_limits<float>::max();
    maxVal = std::numeric_limits<float>::lowest();
    for (const auto &v : data) {
        minVal = std::min(minVal, v.real());
        maxVal = std::max(maxVal, v.real());
    }
}

void runPoissonSolver(dims_t dims, std::vector<int> gpus, cpu_data_t &rho, cpu_data_t &phi,
                      float dx = 1.0f, float dy = 1.0f, float dz = 1.0f) {
    auto t0 = std::chrono::high_resolution_clock::now(), t1 = t0;

    std::cout << "GPUs: " << gpus.size() << "\nGrid: " << dims[0] << "x" << dims[1] << "x" << dims[2] << "\n";

    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    CUFFT_CHECK(cufftXtSetGPUs(plan, gpus.size(), gpus.data()));

    size_t ws[gpus.size()];
    CUFFT_CHECK(cufftMakePlan3d(plan, dims[0], dims[1], dims[2], CUFFT_C2C, ws));

    cudaLibXtDesc *desc;
    CUFFT_CHECK(cufftXtMalloc(plan, &desc, CUFFT_XT_FORMAT_INPLACE));

    double chargeSum = 0.0;
    for (auto &v : rho) chargeSum += v.real();
    std::cout << "Charge sum: " << chargeSum << "\n";

    CUFFT_CHECK(cufftXtMemcpy(plan, desc, rho.data(), CUFFT_COPY_HOST_TO_DEVICE));

    CUFFT_CHECK(cufftXtExecDescriptor(plan, desc, desc, CUFFT_FORWARD));

    for (size_t i = 0; i < gpus.size(); ++i) {
        CUDA_CHECK(cudaSetDevice(gpus[i]));
        cufftComplex *dptr = reinterpret_cast<cufftComplex *>(desc->descriptor->data[i]);

        int yStart = dims[1] * i / gpus.size();
        int ySize = dims[1] * (i + 1) / gpus.size() - yStart;

        dim3 block(8, 8, 8);
        dim3 grid((dims[0] + 7) / 8, (ySize + 7) / 8, (dims[2] + 7) / 8);

        applyPoisson<<<grid, block>>>(dptr, dims[0], dims[1], dims[2], dx, dy, dz, yStart, ySize);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUFFT_CHECK(cufftXtExecDescriptor(plan, desc, desc, CUFFT_INVERSE));

    float norm = 1.0f / (dims[0] * dims[1] * dims[2]);

    for (size_t i = 0; i < gpus.size(); ++i) {
        CUDA_CHECK(cudaSetDevice(gpus[i]));
        cufftComplex *dptr = reinterpret_cast<cufftComplex *>(desc->descriptor->data[i]);

        int yStart = dims[1] * i / gpus.size();
        int ySize = dims[1] * (i + 1) / gpus.size() - yStart;
        int count = dims[0] * ySize * dims[2];

        int threads = 256;
        int blocks = (count + threads - 1) / threads;

        scaleData<<<blocks, threads>>>(dptr, count, norm);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUFFT_CHECK(cufftXtMemcpy(plan, phi.data(), desc, CUFFT_COPY_DEVICE_TO_HOST));

    float vmin, vmax;
    getMinMax(phi, vmin, vmax);
    std::cout << "Potential range: " << vmin << " to " << vmax << "\n";

    size_t cx = dims[0] / 2, cy = dims[1] / 2, cz = dims[2] / 2;
    size_t center = cx * dims[1] * dims[2] + cy * dims[2] + cz;
    std::cout << "Center potential: " << std::setprecision(6) << phi[center].real() << " + " << phi[center].imag() << "i\n";

    CUFFT_CHECK(cufftXtFree(desc));
    CUFFT_CHECK(cufftDestroy(plan));

    auto tEnd = std::chrono::high_resolution_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - t0).count() << " ms\n";
}

int main(int argc, char **argv) {
    auto start = std::chrono::high_resolution_clock::now();

    dims_t dims = {512, 512, 512};
    if (argc > 1) {
        int n = std::atoi(argv[1]);
        if (n > 0) dims = {size_t(n), size_t(n), size_t(n)};
    }

    int devCount;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    std::vector<int> gpus;
    for (int i = 0; i < devCount; ++i) gpus.push_back(i);

    size_t totalElems = dims[0] * dims[1] * dims[2];
    std::cout << "Grid: " << dims[0] << "x" << dims[1] << "x" << dims[2] << ", Elements: " << totalElems << ", Memory: " << (totalElems * sizeof(std::complex<float>)) / (1024 * 1024) << " MB\n";

    cpu_data_t rho(totalElems);
    cpu_data_t phi(totalElems);

    generateCharge(rho, dims);
    runPoissonSolver(dims, gpus, rho, phi);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Program time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    return 0;
}


