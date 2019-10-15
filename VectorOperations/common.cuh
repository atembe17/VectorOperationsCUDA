/*******************************************************************************/
/* Jason Lowden                                                                */
/* High Performance Architectures - Vector Operations                          */
/* Sunday, February 24, 2013                                                   */
/*                                                                             */
/* common.h                                                                    */
/* This file contains the function prototypes that will be called by the       */
/* application.                                                                */
/*******************************************************************************/

#ifndef __COMMON_H__
#define __COMMON_H__

/**
* Computes the CPU addition algorithm
* a - Input vector 1
* b - Input vector 2
* c - Vector to store the output to
* size - length of the input vectors
*/
void addVectorCPU(float* a, float* b, float* c, int size);

/**
* Computes the CPU subtraction algorithm
* a - Input vector 1
* b - Input vector 2
* c - Vector to store the output to
* size - length of the input vectors
*/
void subtractVectorCPU(float* a, float* b, float* c, int size);

/**
* Computes the CPU scaling algorithm
* a - Input vector
* c - Vector to store the output to
* scaleFactor - the value to scale all entries of the vector by
* size - length of the input vectors
*/
void scaleVectorCPU(float* a, float* c, float scaleFactor, int size);

/**
* Computes the GPU addition algorithm
* a - Input vector 1
* b - Input vector 2
* c - Vector to store the output to
* size - length of the input vectors
*/
bool addVectorGPU(float* a, float* b, float* c, int size);

/**
* Computes the GPU addition algorithm
* a - Input vector 1
* b - Input vector 2
* c - Vector to store the output to
* size - length of the input vectors
*/
bool subtractVectorGPU(float* a, float* b, float* c, int size);

/**
* Computes the GPU scaling algorithm
* a - Input vector
* c - Vector to store the output to
* scaleFactor - the value to scale all entries of the vector by
* size - length of the input vectors
*/
bool scaleVectorGPU(float* a, float* c, float scaleFactor, int size);

#endif