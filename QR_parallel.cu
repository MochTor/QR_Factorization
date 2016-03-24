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
    A = (double*) malloc (m * n * sizeof(double));  //allocating memory for matrix A
    R = (double*) malloc (n * n * sizeof(double));  //allocating memory for matrix R
    bzero(A, m*n);  //cleaning A's memory
    bzero(R, n*n);  //cleaning R's memory
    initMatrixZero(A, m, n);    //init matrix A to all zeros
    initMatrix(A, n);    //init matrix A diagonal with values
    initMatrixZero(R, n, n);    //init matrix R to all zeros
    gram(A, m, n, R);   //applying Gram-Schmidt algorithm
    free(A);    //deallocating A's memory
    free(R);    //deallocating R's memory
    cudaEventRecord(stop, 0); //memorizing stop time
    cudaEventSynchronize(stop);
    //---------------------------------------------------

    //------------ Printing results on screen -----------
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time %7.5f [s]\n", elapsedTime/1000);   //elapsedTime keeps time in milliseconds
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

void initMatrixZero(double *A, int m, int n) {
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < m; jj++) {
            A[jj*n + ii] = 0;
        }
    }
}

void gram(double* A, int m, int n, double *R) {
    double *A_d, *R_d;  //A is the initial matrix, R the upper triangular matrix. Copy on device (_d)
    dim3 dimGrid(THREADS_PER_BLOCK, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, (THREADS_PER_BLOCK + m - 1)/THREADS_PER_BLOCK, 1);

    checkCudaErrors(cudaMalloc((void **) &A_d, m * n *sizeof(double))); //allocating A's memory on device
    checkCudaErrors(cudaMalloc((void **) &R_d, n *n *sizeof(double)) ); //allocating R's memory on device

    checkCudaErrors(cudaMemcpy(A_d, A, m * n *sizeof(double), cudaMemcpyHostToDevice)); //copying A's data into A_d's space
    checkCudaErrors(cudaMemcpy(R_d, R, n * n *sizeof(double), cudaMemcpyHostToDevice)); //copying R's data into R_d's space

    for (int ii = 0; ii < n; ii++) {
        xTA <<< n-1, dimBlock >>> (&R_d[ii*n + ii], n - ii, &A_d[ii], m, n, &A_d[ii], n);   //1
        scale <<< m, dimBlock >>> (&A_d[ii], m, n, &R_d[ii*n + ii]);    //2-3
        scale <<< n - ii, dimBlock >>> (&R_d[ii*n + ii], n - ii, 1, &R_d[ii*n + ii]);   //2-4
        //     last step
    }

    checkCudaErrors(cudaMemcpy(A, A_d, m * n *sizeof(double), cudaMemcpyDeviceToHost)); //copying A_d's data into A's space
    checkCudaErrors(cudaMemcpy(R, R_d, n * n *sizeof(double), cudaMemcpyDeviceToHost)); //copying R_d's data into R's space

    cudaFree(A_d);  //deallocating A's memory on device
    cudaFree(R_d);  //deallocating R's memory on device
}

__global__ void xTA (double *y, int k, double*A, int m, int lda, double *x, int ldx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    //it selects which threads is working on row vector (a row of R matrix)
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
 */
__global__ void scale(double *d, int m, int ld, double *s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    //it selects which threads is working on row vector (a row of R matrix)

    if (idx < m) {
        d[idx*ld] = d[idx*ld] / sqrt(*s);    //Applying scale
    }
}

void r1_update(double *A, int m, int n, int lda, double *col, int ldc, double *row) {
    //Does it work? Recheck!
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < m && idy < m) {
        //converted algorithm
    }
//     for (int jj = 0; jj < n-1; jj++)  //Moving through columns
//         for (int ii = 0; ii < m; ii++)   //Moving through rows
//             A[lda*ii + jj+1] -= row[jj] * col[ldc*ii];
// }
