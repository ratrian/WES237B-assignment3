# Tiled Matrix Multiplication
## Objective
Implement a tiled dense matrix multiplication routine using shared memory.

## Instructions
Edit the code in the code tab to perform the following:

* Allocate device memory
* Copy host memory to device
* Initialize work group dimensions
* Invoke OpenCL kernel
* Copy results from device to host
* Deallocate device memory
* Implement the matrix-matrix multiplication routine using shared memory and tiling

Recommended Block dimensions are $16 \times 16$.

## How to Test
Use the make run command to test your program. here are a total of 9 tests on which your program will be evaluated for (functional) correctness. We will verify if your programs meet the speedup requirements that you should get using shared memory.

## Dataset Generation (Optional)
The dataset required to test the program is already generated. If you are interested in how the dataset is generated please refer to the `datagen.c` file. To recreate the dataset, run the `datagen.sh` script.