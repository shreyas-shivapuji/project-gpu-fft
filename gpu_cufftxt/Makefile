NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 --extended-lambda

CUDA_PATH = /export/apps/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda
INCLUDES = -I$(CUDA_PATH)/include
LIB_PATHS = -L$(CUDA_PATH)/lib64
LIBS = -lcufft -lcudart

TARGET = gpu_cufftxt_poisson_solver_debug_v2

all: $(TARGET)

$(TARGET): gpu_cufftxt_poisson_solver_debug_v2.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIB_PATHS) $< -o $@ $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
