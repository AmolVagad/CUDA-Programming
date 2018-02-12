#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

__global__ void hist_kernel(uint32_t * input, size_t height, size_t width,uint32_t* histo);

uint32_t *d_input;
uint32_t *device_bins;

uint32_t* AllocateDevice(  size_t size)
{
        uint32_t* data;
        cudaMalloc((void**)&data, size);
        return data;

}

void opt_2dhisto(uint32_t *input, size_t height, size_t width, uint32_t *bins)
{
    /* This function should only contain grid setup 
       code and a call to the GPU histogramming kernel. 
       Any memory allocations and transfers must be done 
       outside this function */

    cudaMemset(device_bins, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t));
    dim3 block, grid;
    block.x = HISTO_WIDTH*HISTO_HEIGHT;
    block.y = 1;
    block.z = 1;
    
    if(width%(HISTO_WIDTH*HISTO_HEIGHT) == 0)
	grid.x = width/(HISTO_WIDTH*HISTO_HEIGHT);
    else
	grid.x = width/(HISTO_WIDTH*HISTO_HEIGHT) + 1;

    //printf("Number of blocks %d \n",grid.x);
    grid.y = 1;
    grid.z = 1;

    hist_kernel<<<grid,block>>>(d_input, height, width, device_bins);    
    cudaThreadSynchronize(); 	

}

/* Include below the implementation of any other functions you need */
__global__ void hist_kernel(uint32_t * input, size_t height, size_t width,uint32_t* histo)
{
   
    __shared__ uint32_t private_histo[HISTO_HEIGHT*HISTO_WIDTH];
    
    if (threadIdx.x < HISTO_WIDTH*HISTO_HEIGHT) 
	private_histo[threadIdx.x] = 0;
    
    __syncthreads();

    int i = threadIdx.x + blockDim.x*blockIdx.x;
    
    // stride is total number of threads
    int stride = blockDim.x * gridDim.x;

    while (i < height*width) {
         atomicAdd( &(private_histo[input[i]]), 1);
         i += stride;
    }
   
    __syncthreads();

    if (threadIdx.x < HISTO_WIDTH*HISTO_HEIGHT) 
        atomicAdd( &(histo[threadIdx.x]),private_histo[threadIdx.x] );

}



// Function to copy from deice to host

void DeviceToHost(uint32_t* h_data, uint32_t* d_data, size_t size)
{

        cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
        
}



//Function to free device memory 

void DeviceFree(uint32_t *d_data)
{
        cudaFree(d_data);
}


// Function to copy from host to deice 

void HostToDevice(uint32_t* d_data, uint32_t* h_data, size_t size)
{
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

}

//Function for zeroing out global histogram bins 

void setmemory(uint32_t* data, int value , size_t count)
{

        cudaMemset(data, value, count);
}




