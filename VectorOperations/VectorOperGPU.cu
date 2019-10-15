#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "common.cuh"
#include "math.h"

//Const to set TILE_SIZE of the device
const float TILE_SIZE = 1024;

//Kernel method for vector addition
__global__ void VectoraddKernel(float* Agpu, float* Bgpu, float* Cgpu, int size)
{
	//Thread id
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < size) {
		Cgpu[tid] = Agpu[tid] + Bgpu[tid];
	}
}

//Kernel method for vector subtraction
__global__ void VectorsubtractKernel(float* Agpu, float* Bgpu, float* Cgpu, int size)
{
	//Thread id
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < size) {
		Cgpu[tid] = Agpu[tid] - Bgpu[tid];
	}
}

//Kernel method for vector scaling
__global__ void VectorscaleKernel(float* Agpu, float* Cgpu, float scaling, int size)
{
	//Thread id
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < size) {
		Cgpu[tid] = Agpu[tid] * scaling;
	}
}

// Method to allocate memory and invoke kernel method for vector addition
bool addVectorGPU(float* M, float* N, float* P, int size) {
	int bytes = size * sizeof(float);
	float* Agpu, * Bgpu, * Cgpu;
	//Page lock memory mapping
	cudaHostGetDevicePointer((void**)&Agpu, M, 0);
	cudaHostGetDevicePointer((void**)&Bgpu, N, 0);
	cudaHostGetDevicePointer((void**)&Cgpu, P, 0);
	//Set the block and grid dimens
	dim3 dimBlock(TILE_SIZE);
	dim3 dimGrid((int)ceil((float)size / (float)TILE_SIZE));
	// Launch the kernel on a size-by-size block of threads
	VectoraddKernel << <dimGrid, dimBlock >> > (Agpu, Bgpu, Cgpu, size);
	cudaThreadSynchronize();
	//Return error if any 
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) {
		printf("Kernel failed: %s", cudaGetErrorString(status));
		return false;
	}
	return true;
}

// Method to allocate memory and invoke kernel method for vector subtraction
bool subtractVectorGPU(float* M, float* N, float* P, int size) {
	int bytes = size * sizeof(float);
	float* Agpu, * Bgpu, * Cgpu;
	//Page lock memory mapping
	cudaHostGetDevicePointer((void**)&Agpu, M, 0);
	cudaHostGetDevicePointer((void**)&Bgpu, N, 0);
	cudaHostGetDevicePointer((void**)&Cgpu, P, 0);
	//Set the block and grid dimens
	dim3 dimBlock(TILE_SIZE);
	dim3 dimGrid((int)ceil((float)size / (float)TILE_SIZE));
	// Launch the kernel on a size-by-size block of threads
	VectorsubtractKernel << <dimGrid, dimBlock >> > (Agpu, Bgpu, Cgpu, size);
	cudaThreadSynchronize();
	//Return error if any 
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) {
		printf("Kernel failed: %s", cudaGetErrorString(status));
		return false;
	}

	return true;
}

// Method to allocate memory and invoke kernel method for vector scaling
bool scaleVectorGPU(float* M, float* P, float scaling, int size) {
	int bytes = size * sizeof(float);
	float* Agpu, * Cgpu;
	//Page lock memory mapping
	cudaHostGetDevicePointer((void**)&Agpu, M, 0);
	cudaHostGetDevicePointer((void**)&Cgpu, P, 0);
	//Set the block and grid dimens
	dim3 dimBlock(TILE_SIZE);
	dim3 dimGrid((int)ceil((float)size / (float)TILE_SIZE));
	// Launch the kernel on a size-by-size block of threads
	VectorscaleKernel << <dimGrid, dimBlock >> > (Agpu, Cgpu, scaling, size);
	cudaThreadSynchronize();
	//Return error if any 
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) {
		printf("Kernel failed: %s", cudaGetErrorString(status));
		return false;
	}

	return true;
}
