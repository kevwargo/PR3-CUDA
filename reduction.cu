#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// This example demonstrates a parallel sum reduction
// using two kernel launches

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <iostream>

typedef unsigned int uint;

template <typename T>
__global__ void reduce1(T *g_idata, T *g_odata)
{
	extern __shared__  T sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) 
	{
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <typename T>
__global__ void reduce2(T *g_idata, T *g_odata)
{
	extern __shared__  T sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) 
	{
		unsigned int index = 2 * s * tid;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
template <typename T>
__global__ void reduce3(T *g_idata, T *g_odata)
{
	extern __shared__  T sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) 
	{
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template <typename T>
void test( void reduce(T *, T *), int elements, int threadsperblock)
{
	cudaError_t error;

	

	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	// create array of 256k elements
	const uint num_elements = (1 << elements);

	// generate random input on the host
	std::vector<T> h_input(num_elements);
	for (unsigned int i = 0; i < h_input.size(); ++i)
	{
		h_input[i] = (T)1;
	}

	const T host_result = std::accumulate(h_input.begin(), h_input.end(), (T)0);
	//std::cerr << "Host sum: " << host_result << std::endl;

	// move input to device memory
	T *d_input = (T*)0;
	cudaMalloc((void**)&d_input, sizeof(T)* num_elements);
	cudaMemcpy(d_input, &h_input[0], sizeof(T)* num_elements, cudaMemcpyHostToDevice);

	const size_t block_size = 1 << threadsperblock;
	const size_t num_blocks = (num_elements / block_size);

	// allocate space to hold one partial sum per block, plus one additional
	// slot to store the total sum
	T *d_partial_sums_and_total = 0;
	cudaMalloc((void**)&d_partial_sums_and_total, sizeof(T)* (num_blocks + 1));
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	reduce << <num_blocks, block_size, block_size * sizeof(T) >> >(d_input, d_partial_sums_and_total);
	reduce << <1, num_blocks, num_blocks * sizeof(T) >> >(d_partial_sums_and_total, d_partial_sums_and_total + num_blocks);
	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Compute and print the performance
	int nIter = 1;
	float msecPerVector = msecTotal / nIter;
	double gigaFlops = ((num_elements - 1) * 1.0e-9f) / (msecPerVector / 1000.0f);
	printf(
		"Performance= %.2f GFlop/s, Time= %.3f msec\n",
		gigaFlops,
		msecPerVector);
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// copy the result back to the host
	T device_result = 0;
	cudaMemcpy(&device_result, d_partial_sums_and_total + num_blocks, sizeof(T), cudaMemcpyDeviceToHost);

	//std::cout << "Device sum: " << device_result << std::endl;

	// deallocate device memory
	cudaFree(d_input);
	cudaFree(d_partial_sums_and_total);
	std::cout << std::endl;
}
int main(int argc, char **argv)
{
	cudaError_t error = cudaSetDevice(0);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to set CUDA device (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	if (argc < 3)
	{
		fprintf(stderr, "usage: %s number_of_elements threads_per_block\n");
		cudaDeviceReset();
		return 1;
	}

	int elements = atoi(argv[1]);
	int tpb = atoi(argv[2]);

	test<double>(reduce1<double>, elements, tpb);
	test<double>(reduce2<double>, elements, tpb);
	test<double>(reduce3<double>, elements, tpb);

	cudaDeviceReset();

	return 0;
	
}