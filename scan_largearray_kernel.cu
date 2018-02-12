#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define TILE_SIZE 1024
// You can use any other block size you wish.
#define BLOCK_SIZE 256


unsigned int** BlockSums;
unsigned int** BlockSumsSummed;
int *size;

// Host Helper Functions (allocate your own data structure...)



// Device Functions



// Kernel Functions



// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE



// Function to preallocate the block sums
void preallocBlockSums(int num_elements)
{

	int n = num_elements;
	int i = 1;

	if(n%(2*BLOCK_SIZE) == 0)
		n = n/(2*BLOCK_SIZE) +1;
	else
		n = n/(2*BLOCK_SIZE) + 1;



	while((n) > 1)
	{
		i++;
		if(n%(2*BLOCK_SIZE) == 0)
			n = n/(2*BLOCK_SIZE) +1;
		else 
			n = n/(2*BLOCK_SIZE) + 1;
	}
	


	// Allocate host memory to block sum variables
	BlockSums = (unsigned int**)malloc(sizeof(unsigned int*)*i);
	BlockSumsSummed = (unsigned int**)malloc(sizeof(unsigned int*)*i);
	size = ( int*)malloc(sizeof( int*)*i);
	n = num_elements;
	i = 0;
	 if(n%(2*BLOCK_SIZE) == 0)
                n = n/(2*BLOCK_SIZE) +1;
        else
                n = n/(2*BLOCK_SIZE) + 1;


	while (n > 1)
	{
		// Allocating memory on the device 

		cudaMalloc((void**)&(BlockSums[i]), n * sizeof(unsigned int));
		cudaMalloc((void**)&(BlockSumsSummed[i]), n * sizeof(unsigned int));
		size[i] = n;
	 if(n%(2*BLOCK_SIZE) == 0)
                n = n/(2*BLOCK_SIZE) +1;
        else
                n = n/(2*BLOCK_SIZE) + 1;



		//storing the number of elements per reduction 
		
		i++;
	}
	//Allocating memory on device

	 cudaMalloc((void**)&(BlockSums[i]), n * sizeof(unsigned int));
         cudaMalloc((void**)&(BlockSumsSummed[i]), n * sizeof(unsigned int));
	size[i] = n;
	
}
	

/*
void deallocBlockSums()
{

	// Empty the device variables 
	cudaFree(&BlockSums);
	cudaFree(&BlockSumsSummed);
}

*/

// Prescan kernel function 

__global__ void prescanArray_kernel(unsigned int *outArray , unsigned int *inArray,unsigned int *BlockSums, int numElements)
{
// Declare  shared memory 

	 __shared__ unsigned int scan_array[2*BLOCK_SIZE];
	int t = threadIdx.x;
	int stride = 1;
	int n = numElements ;
	int start = 2*blockIdx.x*blockDim.x;

// Copy dara into the shared memory 
	 if(start + t < n)

                scan_array[ t] = inArray[start+t];
        else
		scan_array[t] = 0;
	if(start + blockDim.x + t<n)
                scan_array[blockDim.x + t] = inArray[blockDim.x + t + start];
	else
		scan_array[blockDim.x + t] = 0;


// Carry out up-sweep ( prescan operation)

	__syncthreads();
	while (stride <= BLOCK_SIZE)
	{
		int index = (t + 1)*stride*2-1;
		if(index < 2*BLOCK_SIZE)
			scan_array[index] += scan_array[index - stride];
		stride = stride*2;
		__syncthreads();
	}
// Carrying out downsweep  (post scan)  operation 


	if(t == 0)
	{
		BlockSums[blockIdx.x] = scan_array[2*BLOCK_SIZE -1];
		scan_array[2*BLOCK_SIZE -1] = 0;
	}
	 stride = BLOCK_SIZE;

	while(stride > 0)

	{
	
		int index = (t + 1)*stride*2 -1;
		if(index < 2*BLOCK_SIZE)
		{
			unsigned int temp = scan_array[index];
			scan_array[index] += scan_array[index - stride];
			scan_array[index - stride] = temp;
		}
	

	stride = stride /2;
	
	__syncthreads();
	}

// Copy the data back to from shared to global memory

	if(start + t < n)

		outArray[start + t] = scan_array[t];
	if(start + blockDim.x + t<n)
		outArray[start + blockDim.x + t] = scan_array[blockDim.x + t];
 

	

	
}

// Function to adjust the increments of Block Sums



__global__ void BlockSum_Increment(unsigned int *sum_array,unsigned int *increment, int n) 
{
         if(blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 + 1 < n)
         {
                 sum_array[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 + 1] += increment[blockIdx.x];
                 sum_array[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2] += increment[blockIdx.x];
         }
         else if (blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 < n)
         {
                 sum_array[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2] += increment[blockIdx.x];
         }
 } 






// Prescan array function to call the kernel 

 void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements )
{
/*
	printf("hi");
	int i = 0;
	dim3 block_Dim;
	dim3 grid_Dim;
	block_Dim.x = BLOCK_SIZE;
	block_Dim.y = block_Dim.z = 1;

	 if(numElements %(2*BLOCK_SIZE) == 0)
                        grid_Dim.x = numElements/(2*BLOCK_SIZE);
         else
                        grid_Dim.x = numElements/(2*BLOCK_SIZE) +1 ;
	printf("dim %d",grid_Dim.x);
	//grid_Dim.x = ceil((float)(numElements/(float)(block_Dim.x *2)));
	grid_Dim.y = grid_Dim.z = 1;
	
//	prescanArray_kernel<<<grid_Dim,block_Dim>>>(outArray, inArray, BlockSums[i], numElements);
	
	while(grid_Dim.x >1)
	{

		if(grid_Dim.x %(2*BLOCK_SIZE) == 0)
			grid_Dim.x = grid_Dim.x/(2*BLOCK_SIZE);
		else
			grid_Dim.x = grid_Dim.x/(2*BLOCK_SIZE) +1 ;
//		prescanArray_kernel<<<grid_Dim,block_Dim>>>(BlockSumsSummed[i], BlockSums[i], BlockSums[i+1], size[i]);
	//	prescanArray_helper(BlockSumsSummed[index], BlockSums[index], grid_Dim.x, index +1);
	
	i++;
	}

	for(int k = i-1; k>=0; k++)
	{
		grid_Dim.x = size[k+1];
		BlockSum_Increment<<<grid_Dim,block_Dim>>>(BlockSumsSummed[k], BlockSumsSummed[k+1], size[k]);
	}
	
	if(numElements %(2*BLOCK_SIZE) == 0)
   */
//Initializing the variables

                 
     int i = 0;

    dim3 Dimblock, Dimgrid;

    Dimblock.x = BLOCK_SIZE;

    Dimblock.y = 1;

    Dimblock.z = 1;

// Computing the number of blocks

    if(numElements%(2*BLOCK_SIZE) == 0)

       Dimgrid.x = numElements/(2*BLOCK_SIZE);

    else

        Dimgrid.x = numElements/(2*BLOCK_SIZE) + 1;

    Dimgrid.y = 1;

    Dimgrid.z = 1;

    
//Initially  Calling the kernel function

    prescanArray_kernel<<<Dimgrid,Dimblock>>>(outArray, inArray, BlockSums[i], numElements);

    



    while(Dimgrid.x > 1)

    {

        if(Dimgrid.x%(2*BLOCK_SIZE) == 0)

            Dimgrid.x = Dimgrid.x/(2*BLOCK_SIZE);

        else

            Dimgrid.x = Dimgrid.x/(2*BLOCK_SIZE) + 1;

// Calling the kernel function for every block

	prescanArray_kernel<<<Dimgrid,Dimblock>>>(BlockSumsSummed[i],BlockSums[i], BlockSums[i+1], size[i]);

	i++;

    }

    
// Adjusting the increments of block sums
    for(int k = i-1;k >=0 ; k--)

    {

	Dimgrid.x = size[k+1];

	BlockSum_Increment<<<Dimgrid, Dimblock>>>(BlockSumsSummed[k],BlockSumsSummed[k+1], size[k]);

    }



    if(numElements%(2*BLOCK_SIZE) == 0)

        Dimgrid.x = numElements/(2*BLOCK_SIZE);

    else

        Dimgrid.x = numElements/(2*BLOCK_SIZE) + 1;  

// Calling the block sum adjustment function 


    BlockSum_Increment<<<Dimgrid, Dimblock>>>(outArray, BlockSumsSummed[0], numElements);


}


/*i
__global__ void BlockSum_Increment(unsigned int *sum_array,unsigned int increment, int n)
{
	if(blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 + 1 < n)
	{
		sum_array[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 + 1] += increment[blockIdx.x];
		sum_array[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2] += increment[blockIdx.x];
	}
	else if (blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2 < n)
	{
		sum_array[blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x * 2] += increment[blockIdx.x];
	}
} 
*/
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
