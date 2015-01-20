// CUDA Runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <algorithm>




__global__ void reduce0(double *g_idata, double *g_odata, unsigned int n)
{
	__shared__ double sdata[];

	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n) ? g_idata[i] : 0;

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		// modulo arithmetic is slow!
		if ((tid % (2 * s)) == 0)
		{
			sdata[tid] += sdata[tid + s];
		}

		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

cudaError_t reduceWithCuda(double *hdata, int size)
{

}

void cleanup(cudaError_t status)
{
	cudaDeviceReset();
	exit(status == cudaSuccess ? 0 : 1);
}

int main(int argc, char **argv)
{
	int size = 1024;
	int threads = 1024;
	double *hostData = (double *)malloc(size * sizeof(double));
	for (int i = 0; i<size; i++)
		hostData[i] = (rand() & 0xFF) / (double)RAND_MAX;
	
	cudaError_t exitcode;

	exitcode = cudaSetDevice(0);
	if (exitcode != cudaSuccess)
	{
		printf("Error setting device 0: %d\n", exitcode);
		cleanup(exitcode);
	}
	exitcode = reduceWithCuda(hostData, size);
	if (exitcode != cudaSuccess)
	{
		printf("Error in reduceWithCuda %d!!!\n", exitcode);
		cleanup(exitcode);
	}
	cleanup(cudaSuccess);
}