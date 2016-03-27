/**
 * C file for serial QR factorization program.
 * See header for more infos.
 *
 * 2016 Marco Tieghi - marco01.tieghi@student.unife.it
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "QR_serial.h"

int main(int argc, char const *argv[]) {
    //--------------- Defining variables ---------------
    clock_t start, stop;    //timers for calculating CPU times

    // int m = 400;    //matrix rows number (first set)
    // int n = 300;    //matrix columns number (first set)
    int m = 1000;    //matrix rows number (second set)
    int n = 800;     //matrix columns number (second set)

    double *A, *R;  //A is the initial matrix, R the upper triangular matrix
    //---------------------------------------------------

    //--------------- Starting algoritm -----------------
    start = clock();    //memorizing starting time
    A = (double*) malloc (m * n * sizeof(double));  //allocating memory for matrix A
    R = (double*) malloc (n * n * sizeof(double));  //allocating memory for matrix R
    bzero(A, m*n);  //cleaning A's memory, init matrix A with 0s
    bzero(R, n*n);  //cleaning R's memory, init matrix R with 0s
    initMatrix(A, n);    //init matrix A diagonal with values
    gram(A, m, n, R);   //applying Gram-Schmidt algorithm
    free(A);    //deallocating A's memory
    free(R);    //deallocating R's memory
    stop = clock(); //memorizing stop time
    //---------------------------------------------------

    printf("Elapsed time %7.5f s\n", (stop-start)/(double)CLOCKS_PER_SEC);

    return 0;
}

void initMatrix(double *A, int n) {
    for (int ii = 0; ii < n; ii++) {
        A[ii*n + ii] = (double)ii + 1;
    }
}

void gram(double* A, int m, int n, double *R) {
    double sf;  //Scale factor

    for (int ii = 0; ii < n-1; ii++) {
        xTA(&R[ii*n + ii], n-ii, &A[ii], m, n, &A[ii], n);  //1
        sf = sqrt(R[ii*n + ii]);                            //2
        scale(&A[ii], m, n, sf);                            //3
        scale(&R[ii*n + ii], n, 1, sf);                     //4
        r1_update(&A[ii+1], m, n-ii-2, n, &A[ii], n, &R[ii]);//5
    }
}

void xTA (double *y, int k, double*A, int m, int lda, double *x, int ldx) {
    double s;   //It memorizes the sum

    for (int jj = 0; jj < k; jj++) {    //Moving through columns
        s = 0;
        for (int ii = 0; ii < m; ii++) {    //Moving through rows
            s += x[ii * ldx] * A[ii*lda + jj];
        }
        y[jj] = s;  //Adding the sum to result vector
    }
}

void scale(double *d, int m, int ld, double s) {
    for (int ii = 0; ii < m; ii++)    //Moving through rows
        d[ii*ld] = d[ii*ld] / s;    //Applying scale
}

void r1_update(double *A, int m, int n, int lda, double *col, int ldc, double *row) {
    //A(:,ii+1:n−1)=A(:,ii+1:n−1)−A(:,ii)∗R(ii,ii+1:n−1)
    for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < m; jj++) {
            A[jj*lda + ii] = A[jj*lda + ii] - col[jj*ldc] * row[ii];
        }
    }
}
