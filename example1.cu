#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <ctime>
#include <time.h>
#include <sstream>
#include <string>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

//Kernel
__global__ void reduce0(int *g_idata, int *g_odata, int size) {
	
	//Tablica wartości do zsumowania w pamięci współdzielonej (w obrębie bloku)
	extern __shared__ int sdata[];
	
	//Pobierz id wątku (w bloku)
	unsigned int tid = threadIdx.x;
	//Oblicz globalny id wątku (we wszystkich blokach)
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	//Wyzeru pozycję w tablicy odpowiadającej wątkowi
	sdata[tid] = 0;
	if(i < size)
		//Pobierz wartość do pamięci współdzielonej
		sdata[tid] = g_idata[i];
	//Synchronizacja (aby wszystkie wątki zdążyły pobrać dane)
	__syncthreads();
	
	//Sumowanie
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		//Jeżeli id wątku jest parzysty ...
        if (tid % (2*s) == 0) {
			//... zlicz wartość swoją i nieparzystego sąsiada
			sdata[tid] += sdata[tid + s];
        }
		//Synchronizacja (aby wszystkie wątki zdążyły posumować)
        __syncthreads();
    }
	
	//Zapisz rezultat z tego bloku do pamięci
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main(void) {
	//Deklaracja rozmiaru
	int size = 939289;
	//Wektor wejściowy hosta
	thrust::host_vector<int> data_h_i(size, 1);

	//Liczba wątków na blok
	int threadsPerBlock = 1024;
	//Liczba bloków (na początku)
	int totalBlocks = (size+(threadsPerBlock-1))/threadsPerBlock;

	//Wektor wejściowy i wyjściowy device
	thrust::device_vector<int> data_v_i = data_h_i;
	thrust::device_vector<int> data_v_o(totalBlocks);

	//Wskaźniki na wektory device
	int* output = thrust::raw_pointer_cast(data_v_o.data());
	int* input = thrust::raw_pointer_cast(data_v_i.data());
	
	//Czy kolej na zamianę tablicy wejściowej z wyjściową?
	bool turn = true;
	
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

    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
	
	while(true) {	
		if(turn) {
			//Odpal kernel (tablica wejściowa jako input, wyjściowa jako output
			reduce0<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(input, output, size);
			turn = false;
		} else {
			//Odpal kernel (tablica wyjściowa jako input, wejściowa jako output
			reduce0<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(int)>>>(output, input, size);
			turn = true;
		}
		
		//Jeżeli został jeden blok, to obliczenia zostały zakończone
		if(totalBlocks == 1) break;
		
		//Korzystaj tylko z zakresu tablicy odpowiadającemu liczbie bloków z poprzedniej iteracji
		size = totalBlocks;
		//Oblicz nową liczbę bloków
		totalBlocks = ceil((double)totalBlocks/threadsPerBlock);
	}
	
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
	
	//Wektor wyjściowy hosta
	thrust::host_vector<int> data_h_o;
	  
	//Pobierz wynik
	if(turn)
		//Wynik w tablicy wejściowej device
		data_h_o = data_v_i;
	else
		//Wynik w tablicy wyjściowej device
		data_h_o = data_v_o;
	
	//Wyczyść wektory
	data_v_i.clear();
	data_v_i.shrink_to_fit();
	  
	data_v_o.clear();
	data_v_o.shrink_to_fit();
	  
	//Wypisz wynik
	cout<< "Wynik: " << data_h_o[0] << endl << "W czasie:" << msecTotal << endl;

	return 0;
}
