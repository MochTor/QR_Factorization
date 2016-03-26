/**
 * Header file for parallel QR factorization program used for Parallel Computing class
 *
 * 2016 Marco Tieghi - marco01.tieghi@student.unife.it
 *
 */

#ifndef QRPARALLEL_H
#define QRPARALLEL_H

/**
 * Initialize a matrix, according to text
 * @param A is the matrix
 * @param m is the number of rows
 * @param n is the number of columns
 */
void initMatrix(double *A, int n);

/**
  * Rank 1 update of columns of A
  * A     m x n lda
  * col   m x 1 ldc
  * coeff 1 x n
  */
__global__ void r1_update(double *A, int m, int n, int lda, double *col, int ldc, double *row);


/**
  * Matrix vector product
  * Performs y =  x'A
  * A : m x k, leading dim lda
  * x : m, leading dim. ldx
  * y : 1 x k
  */
__global__ void xTA (double *y, int k, double*A, int m, int lda, double *x, int ldx);

/**
  * Mult. for constant s
  * d vector
  * m number of elements to change
  * ld leading dimension (distance from elements)
  *
  */
__global__ void scale(double *d, int m, int ld, double *s);

/**
  * Performs Modified Gram Schmidt
  * ortogonalization of columns of A
  * A m x n
  * Q m x n
  * R n x n
  * m >= n
  */
void gram(double* A, int m, int n, double *R);

#endif
