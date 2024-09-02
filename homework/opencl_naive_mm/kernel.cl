__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns)
{
  //@@ Insert code to implement matrix multiplication here
  unsigned int row = get_global_id(0);
  unsigned int col = get_global_id(1);
  float sum = 0;
  for (unsigned int i = 0; i < numAColumns; i++)
  {
    sum += (A[row * numAColumns + i] * B[i * numBColumns + col]);
  }
  C[row * numCColumns + col] = sum;
}