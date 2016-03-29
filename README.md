**DISCLAIMER**<br/>
This project was written for the Algorithms for Parallel Computing class as part of the exam.
It is focused on didatic purposes.

I don't accept merge requests or new issues, because I'll work on this project no more after the exam.

For any question, write to me.

# QR Factorization
That's the final project I discussed in the final exam for my Parallel Computing class.
The goal was to develop a program for QR factorization (Gram-Schmidt modified algorithm),
both in serial and usign CUDA (Compute Unified Device Architecture) tools for parallel computation. <br/><br/>

**CUDA** is a framework developed by _NVIDIA_ for its GPU that is used for massively parallel computation.
CUDA consists of libraries and toolkits available for free.

##Compiling
To compile the serial program you just need a C compiler (as gcc), and compile it just any other C program. <br/>
To compile the parallel program, you need to download, install and configure CUDA toolkits from NVIDIA site. Then you can compile it using _nvcc_ comand.
