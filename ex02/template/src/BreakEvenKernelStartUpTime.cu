

#include <stdio.h>
#include "chTimer.h"


#define NUM_BLOCKS 64
#define NUM_THREADS 256
#define NUM_CYCLES 600

__global__
void
NullKernel()
{
}

__global__ void Kernel(clock_t *runtime)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if (tid == 0) runtime[bid] = clock();

  if (tid + bid == 10)
  {
    int busyint = 0;
    while (clock() - runtime[bid] < NUM_CYCLES) busyint++;
  }
  if (tid == 0) runtime[bid + gridDim.x] = clock();

}


int
main()
{
  NullKernel<<<64,256>>>();
  cudaDeviceSynchronize();

  clock_t *druntime = NULL;
  clock_t runtime[NUM_BLOCKS * 2];

  cudaMalloc((void **)&druntime, sizeof(clock_t) * NUM_BLOCKS * 2);

  chTimerTimestamp start, stop;
  chTimerGetTime( &start );

  Kernel<<<NUM_BLOCKS, NUM_THREADS>>>(druntime);
  cudaDeviceSynchronize();

  chTimerGetTime( &stop );

  cudaMemcpy(runtime, druntime, sizeof(clock_t) * NUM_BLOCKS * 2, cudaMemcpyDeviceToHost);
  cudaFree(druntime);

  long double avgElapsedClocks = 0;

  for (int i = 0; i < NUM_BLOCKS; i++)
    {
      avgElapsedClocks += (long double) (runtime[i + NUM_BLOCKS] - runtime[i]);
    }

  double microseconds = 1e6*chTimerElapsedTime( &start, &stop );

  avgElapsedClocks = avgElapsedClocks/NUM_BLOCKS;
  printf("Average clocks/block = %Lf\n", avgElapsedClocks);
  printf("Async Kernel Startup Time %f\n", microseconds);
  return 0;
}
