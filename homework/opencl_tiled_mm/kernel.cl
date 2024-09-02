__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns)
{
  //@@ Insert code to implement matrix multiplication here
  unsigned int tile_size = get_local_size();
  unsigned int num_tiles = ((numAColumns + tile_size) / tile_size);
  unsigned int row' = get_local_id(0);
  unsigned int col' = get_local_id(1);
  unsigned int row = get_group_id(0) * tile_size + row';
  unsigned int col = get_group_id(1) * tile_size + col';
  __local float A'[tile_size][tile_size];
  __local float B'[tile_size][tile_size];
  float sum = 0;
  for (unsigned int tile = 0; tile < num_tiles; tile++)
  {
    unsigned int tile_row = (tile * tile_size) + row';
    unsigned int tile_col = (tile * tile_size) + col';
    if (tile_col < numAColumns)
    {
      A'[row'][col'] = A[row * numAColumns + tile_col];
    }
    else
    {
      A'[row'][col'] = 0;
    }
    if (tile_row < numBRows)
    {
      B'[row'][col'] = B[tile_row * numBColumns + col];
    }
    else
    {
      B'[row'][col'] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int i = 0; i < tile_size; i++)
    {
      sum += (A'[row'][i] * B'[i][col']);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((row < numCRows) && (col < numCColumns))
  {
    C[row * numCColumns + col] = sum;
  }
}