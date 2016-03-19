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
    bzero(A, m*n);
    bzero(R, n*n);
    //---------------------------------------------------
    return 0;
}
