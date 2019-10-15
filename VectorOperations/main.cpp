#include <cstdlib> // malloc(), free()
#include <stdio.h> 
#include "common.cuh"
#include <ctime>
#include <cmath>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

//Size of vector
const int SIZE = 20000;
//Scale factor
float scaleFactor = 5;
//No of iters for time calculation
const int ITERS = 1000;
//Method to calculate normalization error
float errorCalc(float* c_cpu, float* c_gpu);

int main() {
	cudaSetDeviceFlags(cudaDeviceMapHost);
	//bool to return status of call
	bool success;
	//Error value
	float error;
	clock_t start, end;
	float timeCpu, timeGpu;
	float* a;
	float* b;
	float* c_cpu = new float[SIZE];
	float* c_gpu;
	//Memory pinned to be shared between CPU and GPU
	cudaHostAlloc((void**)&a, SIZE * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&b, SIZE * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&c_gpu, SIZE * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	// Initialize a and b to random integers
	for (int i = 0; i < SIZE; i++) {
		a[i] = ((float)rand() / (RAND_MAX));
		b[i] = ((float)rand() / (RAND_MAX));
	}
	//Clock started
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		//CPU vector addition method invoke
		addVectorCPU(a, b, c_cpu, SIZE);
	}
	//End of clock
	end = clock();
	//CPU time for vector addition
	timeCpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	printf("Addition by CPU took %f\n", timeCpu);
	success = addVectorGPU(a, b, c_gpu, SIZE);
	if (!success) {
		printf("\n * Device error! * \n");
		return 1;
	}
	//Clock started
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		//CPU vector addition method invoke
		addVectorGPU(a, b, c_gpu, SIZE);
	}
	//End of clock
	end = clock();
	//GPU time for vector addition
	timeGpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	printf("Addition by GPU took %f\n", timeGpu);
	printf("Addition speedup = %f\n", timeCpu / timeGpu);
	error = errorCalc(c_cpu, c_gpu);
	printf("Addition Error = %f\n", error);
	printf("\n");
	//Clock started
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		//CPU vector subtraction method invoke
		subtractVectorCPU(a, b, c_cpu, SIZE);
	}
	//End of clock
	end = clock();
	//CPU time for vector addition
	timeCpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	printf("Subtraction by CPU took %f\n", timeCpu);

	success = subtractVectorGPU(a, b, c_gpu, SIZE);
	if (!success) {
		printf("\n * Device error! * \n");
		return 1;
	}
	//Clock started
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		//GPU vector subtraction method invoke
		subtractVectorGPU(a, b, c_gpu, SIZE);
	}
	//End of clock
	end = clock();
	//GPU time for vector subtraction
	timeGpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	printf("Subtraction by GPU took %f\n", timeGpu);
	printf("Subtraction speedup = %f\n", timeCpu / timeGpu);
	error = errorCalc(c_cpu, c_gpu);
	printf("Subtraction Error = %f\n", error);
	printf("\n");
	//Clock started
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		//CPU vector scaling method invoke
		scaleVectorCPU(a, c_cpu, scaleFactor, SIZE);
	}
	//End of clock
	end = clock();
	//CPU time for vector scaling
	timeCpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	printf("Scaling by CPU took %f\n", timeCpu);

	success = scaleVectorGPU(a, c_gpu, scaleFactor, SIZE);
	if (!success) {
		printf("\n * Device error! * \n");
		return 1;
	}
	//Clock started
	start = clock();
	for (int i = 0; i < ITERS; i++) {
		//GPU vector scaling method invoke
		scaleVectorGPU(a, c_gpu, scaleFactor, SIZE);
	}
	//End of clock
	end = clock();
	//GPU time for vector scaling
	timeGpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	printf("Scaling by GPU took %f\n", timeGpu);
	printf("Scaling speedup = %f\n", timeCpu / timeGpu);
	error = errorCalc(c_cpu, c_gpu);
	printf("Scaling Error = %f\n", error);

	//Free the allocated memory pointers
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c_gpu);
	delete[] c_cpu;
	return 0;
}

//Method to calculate normalization error
float errorCalc(float* c_cpu, float* c_gpu) {
	float sum = 0, delta = 0;
	for (int i = 0; i < SIZE; i++) {
		delta += (c_cpu[i] - c_gpu[i]) * (c_cpu[i] - c_gpu[i]);
		sum += (c_cpu[i] * c_gpu[i]);
	}
	float L2norm = sqrt(delta / sum);
	return L2norm;
}