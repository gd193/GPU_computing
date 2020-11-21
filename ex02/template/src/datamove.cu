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
        //printf("Data size = %ld Bytes: \n",bytes*sizeof(float));
        printf("%ld\t",bytes*sizeof(float));
        for(int k = 0; k<2; k++){                                           //loop for pageable and pinned memory
            float *x, *d_x, *y, *d_y;
            if (k==0){
                x = (float*)malloc(bytes*sizeof(float));
                y = (float*)malloc(bytes*sizeof(float));
            }
            else {
                cudaMallocHost((void **) &x, bytes*sizeof(float));
                cudaMallocHost((void **) &y, bytes*sizeof(float));
            };
            
            cudaMalloc(&d_x, bytes*sizeof(float));
            cudaMalloc(&d_y, bytes*sizeof(float));

            for (int i = 0; i < bytes; i++) {
                x[i] = i/2.;
            };

            //Timing with clock and chTimer just to check them
            chTimerTimestamp start, stop;
            clock_t begin = clock();
            chTimerGetTime( &start );
            cudaMemcpy(d_x, x, bytes*sizeof(float), cudaMemcpyHostToDevice);
            chTimerGetTime( &stop );
            clock_t end = clock();
                        
            long diff = end - begin;
            double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
            //printf("Copying HostToDevice took %i cycles / %.2d us \n", diff, microseconds);
            printf("%ld\t%f\t",diff,microseconds); //the output is directly in a form in which it can be easily accessed an plotted (but bit harder to read)

            CopyKernel<<<4,256>>>(bytes, d_x, d_y);
            cudaDeviceSynchronize();

            chTimerTimestamp start2, stop2;
            begin = clock();
            chTimerGetTime( &start2 );
            cudaMemcpy(y, d_y, bytes*sizeof(float), cudaMemcpyDeviceToHost);
            //cudaDeviceSynchronize();
            chTimerGetTime( &stop2 );
            end = clock();
            
            diff = end - begin;
            double microseconds2 = 1e6*chTimerElapsedTime( &start2, &stop2 ); //For some reason this is always 1 which is wrong
            cudaFree(d_x);
            cudaFree(d_y);
            if (k==0){
                printf("%ld\t%f\t",diff,microseconds2);
                free(x);
                free(y);
            }
            else {
                printf("%ld\t%f\n",diff,microseconds2);
                cudaFreeHost(x);
                cudaFreeHost(y);
            };
            
        };
    };
}
