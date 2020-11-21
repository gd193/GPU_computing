#include <stdio.h>
#include "chTimer.h"
#include <iostream>
#include <fstream>


__global__
void
CopyKernel(long bytes, float *x, float *y) // Kernel that simply multiplies a given array by two (was used to check whether correct data was written in y)
{   
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < bytes) y[i] = 2*x[i];
}

int main(){
    for(int j = 10; j< 31; j++){                                            //loop through data sizes
        long bytes = (1<<j) / sizeof(float);   //define data size
        double Gbytes = double(bytes*sizeof(float))/double(1<<30);
        //printf("Data size = %ld Bytes: \n",bytes*sizeof(float));
        printf("%f\t",Gbytes);                                          //loop for pageable and pinned memory
        float *x, *d_x, *y, *d_y, *d_z;
        cudaMallocHost((void **) &x, bytes*sizeof(float));
        cudaMallocHost((void **) &y, bytes*sizeof(float));
        
        cudaMalloc(&d_x, bytes*sizeof(float));
        cudaMalloc(&d_y, bytes*sizeof(float));
        cudaMalloc(&d_z, bytes*sizeof(float));

        for (int i = 0; i < bytes; i++) {
            x[i] = i/2.;
        };
        chTimerTimestamp start, stop;
        //Timing with clock and chTimer just to check them
        chTimerGetTime( &start );
        cudaMemcpy(d_x, x, bytes*sizeof(float), cudaMemcpyHostToDevice);
        chTimerGetTime( &stop );
        double seconds = chTimerElapsedTime( &start, &stop );
        //printf("Copying HostToDevice took %i cycles / %.2d us \n", diff, microseconds);
        printf("%f\t",Gbytes/seconds); //the output is directly in a form in which it can be easily accessed an plotted (but bit harder to read)
        CopyKernel<<<4,256>>>(bytes, d_x, d_y);
        cudaDeviceSynchronize();

        chTimerTimestamp start2, stop2;
        chTimerGetTime( &start2 );
        cudaMemcpy(d_z, d_y, bytes*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        chTimerGetTime( &stop2 );
        double seconds2 = chTimerElapsedTime( &start2, &stop2 );
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        printf("%f\n",Gbytes/seconds2);
        cudaFreeHost(x);
        cudaFreeHost(y);
    };
}
