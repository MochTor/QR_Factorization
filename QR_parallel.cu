/**
 * C file for parallel QR factorization program usign CUDA
 * See header for more infos.
 *
 * 2016 Marco Tieghi - marco01.tieghi@student.unife.it
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>

#include "QR_parallel.h"

#define THREADS_PER_BLOCK 512   //I'll use 512 threads for each block (as required in the assignment)

int main(int argc, char const *argv[]) {
    //--------------- Defining variables ---------------
    int m = 400;    //matrix rows number (first set)
    int n = 300;    //matrix columns number (first set)
    // int m = 1000;    //matrix rows number (second set)
    // int n = 800;     //matrix columns number (second set)

    double *A_h, *R_h;  //A is the initial matrix, R the upper triangular matrix. Copy on host (_h)

    cudaEvent_t start, stop;    //CUDA events to record start and stop time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;  //it memorizes the overall elapsed time since the start of CUDA code
    //---------------------------------------------------

    //--------------- Starting algoritm -----------------
    cudaEventRecord(start, 0);    //memorizing starting time
    A_h = (double*) malloc (m * n * sizeof(double));  //allocating memory for matrix A
    R_h = (double*) malloc (n * n * sizeof(double));  //allocating memory for matrix R
    bzero(A_h, m*n);  //cleaning A's memory, init matrix A with 0s
    bzero(R_h, n*n);  //cleaning R's memory, init matrix R with 0s
    initMatrix(A_h, n);    //init matrix A diagonal with values
    gram(A_h, m, n, R_h);   //applying Gram-Schmidt algorithm
    free(A_h);    //deallocating A's memory
    free(R_h);    //deallocating R's memory
    cudaEventRecord(stop, 0); //memorizing stop time
    cudaEventSynchronize(stop);
    //---------------------------------------------------

    //------------ Printing results on screen -----------
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time: %7.5f s\n", elapsedTime/1000);   //elapsedTime keeps time in milliseconds
    printf("Bandwidth: %7.5f GB/s\n", (m*n * sizeof(double)) / (elapsedTime/1000));
    //---------------------------------------------------

    cudaEventDestroy(start);
	cudaEventDestroy(stop);

    return 0;
}

void initMatrix(double *A, int n) {
    for (int ii = 0; ii < n; ii++) {
        A[ii*n + ii] = (double)ii + 1;
    }
}

void gram(double* A, int m, int n, double *R) {
    double *A_d, *R_d;  //A is the initial matrix, R the upper triangular matrix. Copy on device (_d)
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1); //as required, each block has 512 threads
    dim3 dimGrid(THREADS_PER_BLOCK, (THREADS_PER_BLOCK -1 + m)/THREADS_PER_BLOCK, 1);   //for bigger matrix
    //dim3 dimGrid(m, 1, 1);  //for small matrix

    checkCudaErrors(cudaMalloc((void **) &A_d, m * n *sizeof(double))); //allocating A's memory on device
    checkCudaErrors(cudaMalloc((void **) &R_d, n *n *sizeof(double)) ); //allocating R's memory on device

    checkCudaErrors(cudaMemcpy(A_d, A, m * n *sizeof(double), cudaMemcpyHostToDevice)); //copying A's data into A_d's space
    checkCudaErrors(cudaMemcpy(R_d, R, n * n *sizeof(double), cudaMemcpyHostToDevice)); //copying R's data into R_d's space

    for (int ii = 0; ii < n; ii++) {
        xTA <<< n-ii, dimBlock >>> (&R_d[ii*n + ii], n-ii, &A_d[ii], m, n, &A_d[ii], n);   //1
        scale <<< m, dimBlock >>> (&A_d[ii], m, n, &R_d[ii*n + ii]);    //2-3
        scale <<< n - ii, dimBlock >>> (&R_d[ii*n + ii], n-ii, 1, &R_d[ii*n + ii]);   //2-4
        r1_update <<< dimGrid, dimBlock >>> (&A_d[ii], m, n-ii-1, n, &A_d[ii], n, &R_d[ii]);    //5
    }

    checkCudaErrors(cudaMemcpy(A, A_d, m * n *sizeof(double), cudaMemcpyDeviceToHost)); //copying A_d's data into A's space
    checkCudaErrors(cudaMemcpy(R, R_d, n * n *sizeof(double), cudaMemcpyDeviceToHost)); //copying R_d's data into R's space

    cudaFree(A_d);  //deallocating A's memory on device
    cudaFree(R_d);  //deallocating R's memory on device
}

__global__ void xTA (double *y, int k, double*A, int m, int lda, double *x, int ldx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double s;   //It memorizes the sum

    if (idx < k) {
        for (int ii = 0; ii < m; ii++) {    //Moving through rows
            s += x[ii * ldx] * A[idx + ii*lda];
        }
        y[idx] = s;  //Adding the sum to result vector
    }
}

/**
 * @param  double *s: s is now a pointer, not a value, as required in the assignment
 *                    it's not yet scaled, sqrt needs to be applied inside function (not as in serial code)
 */
__global__ void scale(double *d, int m, int ld, double *s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m) {
        d[idx*ld] = d[idx*ld] / sqrt(*s);    //Applying scale
    }
}

__global__ void r1_update(double *A, int m, int n, int lda, double *col, int ldc, double *row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    //A(:,ii+1:n−1)=A(:,ii+1:n−1)−A(:,ii)*R(ii,ii+1:n−1)
    if (idx < m && idy < m) {
        for (int ii=0; ii < n-1; ii++) {
            A[idx*lda + ii+1] = A[idx*lda + ii+1] - col[idy*ldc] * row[ii+1];
        }
    }
}
