__kernel void vectorAdd(__global const float *a, __global const float *b,
                        __global float *result, const unsigned int size) {
  //@@ Insert code to implement vector addition here
  //printf("Hi, I'm in the kernel!");
  unsigned int i = get_global_id(0) % size;
  result[i] = a[i] + b[i];
}