#include "common.cuh"
//CPU vector addition method
void addVectorCPU(float* a, float* b, float* c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}

//CPU vector subtraction method
void subtractVectorCPU(float* a, float* b, float* c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] - b[i];
	}
}

//CPU vector scaling method
void scaleVectorCPU(float* a, float* c, float scaleFactor, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] * scaleFactor;
	}
}