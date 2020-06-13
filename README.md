# Parallel_Computing_Project
Solving massive linear equations using CUDA

The main objective is to efficiently solve massive linear equations in parallel.
A system of n homogeneous or non-homogeneous linear system of equations in n variables x1, x2,..xn or simply a linear system is a set of n linear equations, each in n variables.
The linear system could be represented in matrix form as

Ax = B

where A is called the coefficient matrix of order n⨯n, x is any solution vector of order n⨯1 and b is any vector of order n⨯1.
The most common method to solve simultaneous linear equations is the Gaussian elimination method, the main disadvantage of this is that it is often impractical to solve larger problems on serial systems since it takes a lot of time and money.
Hence, parallel methods implement a lot of resources working together to reduce the time and cut down the potential costs.

The two methods implemented as part of this project include:

First, Row partitioning solution using openmp and second, implementation of the same on CUDA. Further, the runtimes are analysed from the plot and the speedup is calculated.
