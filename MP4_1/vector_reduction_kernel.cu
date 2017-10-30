
#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_


#define BLOCK_SIZE 16
// **===--------------------- Modify this function -----------------------===**
//! @param g_data  input data in global memory
//                  result is expected in index 0 of g_data
//! @param n        input number of elements to reduce from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(unsigned int *g_data, int n)
{
 	// Declare variable for shared memory 
 	
        __shared__ unsigned int partialSum[2*BLOCK_SIZE];


        unsigned int t = threadIdx.x;
        unsigned int start = 2*blockDim.x*blockIdx.x;

    	// Allocate input data into shared memory 

        partialSum[t] = g_data[start + t];

  
        partialSum[blockDim.x + t] = g_data[start + blockDim.x + t];
       
       	// Carry out vector addition 
       	
        for( unsigned int stride = blockDim.x; stride >= 1 ; stride >>= 1)
        {
                __syncthreads();   //to ensure that all elements of each round of partial sums have been generated before we proceed to the next round
                if(t < stride)
                        partialSum[t] += partialSum[ t + stride];
        }


        g_data[blockIdx.x] = partialSum[0];

}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_



