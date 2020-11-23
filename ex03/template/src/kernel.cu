/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : 04
 *
 *                            File : main.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/

//
// Kernels
//

__global__ void 
globalMemCoalescedKernel(int bytes, int *x, int *y)
{
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while ( tid < bytes/sizeof(int) ) // check the boundry condition for the threads
      {
            y[tid] = x[tid];
            tid+= blockDim.x * gridDim.x ;
      }
}

void 
globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, int bytes, int *x, int *y) {
	globalMemCoalescedKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(bytes, x, y);
}

__global__ void 
globalMemStrideKernel(int bytes, int *x, int *y, int stride)
{
  unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
  if ( i < bytes/sizeof(int)) y[i] = x[i];
}

void 
globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim, int bytes, int *x, int *y, int stride) {
	globalMemStrideKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(bytes, x, y, stride);
}

__global__ void 
globalMemOffsetKernel(int bytes, int *x, int *y, int offset)
{
    unsigned int i = (blockId.x * blockDim.x + threadIdx.x) + offset;
    if ( i < bytes/sizeof(int)) y[i] = x[i];
}

void 
globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim, int bytes, int *x, int *y, int offset) {
	globalMemOffsetKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(bytes, x, y, offset);
}

