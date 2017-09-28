/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
#include <stdio.h>
#include "matrixmul.h"

#define TILE_WIDTH 16

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	// Create Matrices in the shared memory 
	__shared__ float M_s[TILE_WIDTH][TILE_WIDTH];
	__shared__ float N_s[TILE_WIDTH][TILE_WIDTH];
	
	// Thread allocation 
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x ; int ty = threadIdx.y;

	// Generate rows and cols of P 
	int row = by*TILE_WIDTH + ty;
	int col = bx*TILE_WIDTH + tx;
	float pvalue = 0.00;

	//Loop over M and N tiles to compute P_d elements 
	for ( int m = 0; m <= (M.width /TILE_WIDTH)  ; ++m)
	{
		if(row < M.height && (m*TILE_WIDTH + tx) < M.width)   //Checking the boundary conditions for matrix M               
			M_s[ty][tx] = M.elements[row*M.width + m*TILE_WIDTH + tx];
		else
			 M_s[ty][tx] = 0;
		if((m*TILE_WIDTH + ty) < N.height && col < N.width)   //Checking the boundary conditions for matrix N
			N_s[ty][tx] = N.elements[(m*TILE_WIDTH + ty)*N.width + col];
		else 
			N_s[ty][tx] = 0; 		 
		

		__syncthreads();         // To ensure all elements of tile are loaded and consumed 

		for(int k = 0; k < TILE_WIDTH; ++k)
		{	
			pvalue += M_s[ty][k]*N_s[k][tx];
	        
	
		}
		__syncthreads();        // To ensure all elements of tile are loaded and consumed 


        }
	if(row < M.height && col < N.width)                    //Checking the boundary conditions for matrix P
		 P.elements[row*P.width + col] = pvalue;

	 

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
