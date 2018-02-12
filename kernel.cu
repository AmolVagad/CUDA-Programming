#include <stdio.h>
#define BLOCK_SIZE 32

__global__ void spmv_csr_kernel(unsigned int dim, unsigned int *csrRowPtr, 
    unsigned int *csrColIdx, float *csrData, float *inVector, 
    float *outVector) {
    // INSERT KERNEL CODE HERE
    
	// Creating the row
	int row = blockDim.x*blockIdx.x + threadIdx.x;
	
	if(row < dim)
	{
		float dot = 0;
		int row_start = csrRowPtr[row];
		int row_end = csrRowPtr[row + 1];

		for(int jj = row_start; jj < row_end; jj++)
		{
			dot += csrData[jj]*inVector[csrColIdx[jj]];
		}
	
		outVector[row] += dot;
	}
		

}

__global__ void spmv_jds_kernel(unsigned int dim, unsigned int *jdsRowPerm, 
    unsigned int *jdsRowNNZ, unsigned int *jdsColStartIdx, 
    unsigned int *jdsColIdx, float *jdsData, float* inVector,
    float *outVector) {

    // INSERT KERNEL CODE HERE

	int row = blockDim.x*blockIdx.x + threadIdx.x;
	
	if(row < dim )
	{
		float dot = 0;

		for(int i = 0; i < jdsRowNNZ[row]; i++)
		{

			dot += jdsData[jdsColStartIdx[i] +row]*inVector[jdsColIdx[jdsColStartIdx[i]+ row]];
		} 

	// transferring the output to out vector

	outVector[jdsRowPerm[row]] = dot;

	}
         

}

void spmv_csr(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx, 
    float *csrData, float *inVector, float *outVector) {

    // INSERT CODE HERE

	// Calling the kernel
	int gridsize = dim/BLOCK_SIZE;

	if(dim%BLOCK_SIZE != 0)
	{
		gridsize++;
	}
	

	dim3 dimGrid(gridsize, 1,1);
	
        dim3 dimBlock(BLOCK_SIZE,1,1);
	spmv_csr_kernel<<<dimGrid,dimBlock>>>(dim,csrRowPtr,csrColIdx,csrData, inVector, outVector);

}

void spmv_jds(unsigned int dim, unsigned int *jdsRowPerm, unsigned int *jdsRowNNZ, 
    unsigned int *jdsColStartIdx, unsigned int *jdsColIdx, float *jdsData, 
    float* inVector, float *outVector) {

    // INSERT CODE HERE
	int gridsize = dim/BLOCK_SIZE;

        if(dim%BLOCK_SIZE != 0)
        {
                gridsize++;
        }


        dim3 dimGrid(gridsize, 1,1);

        dim3 dimBlock(BLOCK_SIZE,1,1);


	spmv_jds_kernel<<<dimGrid,dimBlock>>>( dim,jdsRowPerm,jdsRowNNZ,jdsColStartIdx,jdsColIdx,jdsData, inVector,outVector) ;

}






