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

int int main(int argc, char const *argv[]) {
    //--------------- Defining variables ---------------
    clock_t start, stop;    //timers fof calculating CPU times

    int m = 400;    //matrix rows number (first set)
    int n = 300;    //matrix columns number (first set)
    // int m = 1000;    //matrix rows number (second set)
    // int n = 800;     //matrix columns number (second set)

    double *A, *R;  //A is the initial matrix, R the upper triangular matrix
    //---------------------------------------------------

    //--------------- Starting algoritm -----------------
    start = clock();    //memorizing starting time
    A = (double*) malloc (m * n * sizeof(double));  //allocating memory for matrix A
    R = (double*) malloc (n * n * sizeof(double));  //allocating memory for matrix R
    bzero(A, m*n);  //cleaning A's memory
    bzero(R, n*n);  //cleaning R's memory
    initMatrix(A, m, n);    //init A matrix with random values
    gram(A, m, n, R);   //applying Gram-Schmidt algorithm
    free(A);    //deallocating A's memory
    free(R);    //deallocating R's memory
    stop = clock(); //memorizing stop time
    //---------------------------------------------------

    printf("Elapsed time %7.3f [s]\n", (stop-start)/(double)CLOCKS_PER_SEC);

    return 0;
}

void initMatrix(double *A, int m, int n) {
    srand((unsigned int) 123);
    for (int i = 0; i < m; i++) {
        for (int ii = 0; i < n; i++) {
            A[i][ii] = rand() % 100;
        }
    }
}

void gram(double* A, int M, int N, double *R) {
    
}
