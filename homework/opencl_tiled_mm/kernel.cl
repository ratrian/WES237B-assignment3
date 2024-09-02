#define TILE_SIZE 16

__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns)
{
  //@@ Insert code to implement matrix multiplication here
  unsigned int num_tiles = ((numAColumns + TILE_SIZE) / TILE_SIZE);
  unsigned int row_pr = get_local_id(0);
  unsigned int col_pr = get_local_id(1);
  unsigned int row = get_group_id(0) * TILE_SIZE + row_pr;
  unsigned int col = get_group_id(1) * TILE_SIZE + col_pr;
  __local float A_pr[TILE_SIZE][TILE_SIZE];
  __local float B_pr[TILE_SIZE][TILE_SIZE];
  float sum = 0;
  for (unsigned int tile = 0; tile < num_tiles; tile++)
  {
    unsigned int tile_row = (tile * TILE_SIZE) + row_pr;
    unsigned int tile_col = (tile * TILE_SIZE) + col_pr;
    if (tile_col < numAColumns)
    {
      A_pr[row_pr][col_pr] = A[row * numAColumns + tile_col];
    }
    else
    {
      A_pr[row_pr][col_pr] = 0;
    }
    if (tile_row < numBRows)
    {
      B_pr[row_pr][col_pr] = B[tile_row * numBColumns + col];
    }
    else
    {
      B_pr[row_pr][col_pr] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int i = 0; i < TILE_SIZE; i++)
    {
      sum += (A_pr[row_pr][i] * B_pr[i][col_pr]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((row < numCRows) && (col < numCColumns))
  {
    C[row * numCColumns + col] = sum;
  }
}