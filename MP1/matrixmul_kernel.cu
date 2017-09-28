/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	//Calculate row index of P 
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	//Calculate col index of P
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	// Multiplication of 2 input matrices 

	if((row < M.width) && ( col < M.width))
	{
		float Pvalue = 0.0;

		for(int k = 0; k < M.width ; ++k)
		{
			Pvalue += M.elements[row*M.width + k] * N.elements[k*N.width + col];
			P.elements[row*M.width + col] = Pvalue;
		}
	}
  

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
