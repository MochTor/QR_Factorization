/**
 * Header file for serial QR factorization program used for Parallel Computing class
 *
 * 2016 Marco Tieghi - marco01.tieghi@student.unife.it
 *
 */

#ifndef QRSERIAL_H
#define QRSERIAL_H

/**
  * Rank 1 update of columns of A
  * A     m x n lda
  * col   m x 1 ldc
  * coeff 1 x n
  */
void r1_update(double *A, int m, int n, int lda, double *col, int ldc, double *row);


/**
  * Matrix vector product
  * Performs y =  x'A
  * A : m x k, leading dim lda
  * x : m, leading dim. ldx
  * y : 1 x k
  */
void xTA (double *y, int k, double*A, int m, int lda, double *x, int ldx);

/**
  * Mult. for constant s
  * d vector
  * m number of elements to change
  * ld leading dimension (distance from elements)
  *
  */
void scale(double *d, int m, int ld, double s);

/**
  * Performs Modified Gram Schmidt
  * ortogonalization of columns of A
  * A m x n
  * Q m x n
  * R n x n
  * m >= n
  */
void gram(double* A, int M, int N, double *R);

#endif
