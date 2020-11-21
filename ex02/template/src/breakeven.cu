#include <stdio.h>
#include <time.h>
#include "chTimer.h"
#include <iostream>

__global__
void
NullKernel()
{
}

__device__ long ddiff;

__global__
void
WaitKernel(const long wait, long *hwaitreturn, clock_t *timer)
{
    const int tid = threadIdx.x;
    if (tid == 0) timer[0] = clock();
    for(long i=0;i<wait;i++){
        clock();
    }
    if (tid == 0) timer[1] = clock();
    if (tid == 0) ddiff = timer[1]-timer[0];
    *hwaitreturn = ddiff;
}


int
main()
{
    NullKernel<<<1,1>>>();
    cudaDeviceSynchronize();

    const int cIterations = 1000000;

    clock_t *dtimer = NULL;
    chTimerTimestamp startt, stopt;
    clock_t start = clock();
    chTimerGetTime( &startt );
    for ( int i = 0; i < cIterations; i++ ) {
        NullKernel<<<1,1>>>();
    }
    cudaDeviceSynchronize();
    chTimerGetTime( &stopt );
    clock_t stop = clock();

    long itime = stop-start;
    double microseconds = 1e6*chTimerElapsedTime( &startt, &stopt ) /(float) cIterations;
    printf("Asynchronous Launch %i cycles\n", itime);
    printf("Asynchronous Launch %.2f us\n", microseconds);

    long waitcycles = 100000000;
    long hwaitret = 0;
    long *dhwaitret;
    cudaMalloc((void**) &dhwaitret, sizeof(long));
    cudaMemcpy(dhwaitret,&hwaitret,sizeof(long), cudaMemcpyHostToDevice);
    clock_t startw = clock();
    chTimerGetTime( &startt );
    for ( int i = 0; i < cIterations; i++ ) {
        WaitKernel<<<1,1>>>(waitcycles, dhwaitret, dtimer);
    }
    cudaDeviceSynchronize();
    chTimerGetTime( &stopt );
    clock_t stopw = clock();
    cudaMemcpy(&hwaitret,dhwaitret,sizeof(long), cudaMemcpyDeviceToHost);
    microseconds = 1e6*chTimerElapsedTime( &startt, &stopt ) /(float) cIterations;
    long waittime = stopw-startw;
    printf("Asynchronous Launch with %i idling: %i cycles\n", waitcycles, waittime);
    printf("Asynchronous Launch %.2f us\n", microseconds);
    std::cout<<hwaitret<<std::endl;
}
