/*
 *
 * nullKernelAsync.cu
 *
 * Microbenchmark for throughput of asynchronous kernel launch.
 *
 * Build with: nvcc -I ../chLib <options> nullKernelAsync.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 
 *
 * 1. Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright 
 *    notice, this list of conditions and the following disclaimer in 
 *    the documentation and/or other materials provided with the 
 *    distribution. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include <math.h>
#include "chTimer.h"

#include <iostream>
#include <fstream>

__global__
void
NullKernel()
{
}

int
main()
{
    const int cIterations = 10000;
    printf( "Measuring launch time... \n" ); fflush( stdout );
    std::ofstream myfile;

    //output for plots in file launchgrid.txt
    myfile.open ("launchgrid.txt", std::ios::trunc);

    //Otherwise for some reason the first measured value is wrong
    NullKernel<<<1,1>>>();
    cudaDeviceSynchronize();

    //loop through different gridsizes
    for ( int j = 0; j<15; j++) {
        int NumBlocks = pow(2,j);
        myfile << NumBlocks << "\t";
        chTimerTimestamp start, stop;
        chTimerGetTime( &start );
        for ( int i = 0; i < cIterations; i++ ) {
            NullKernel<<<NumBlocks,1>>>();
        }
        cudaDeviceSynchronize();
        chTimerGetTime( &stop );
        {
            double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
            double usPerLaunch = microseconds / (float) cIterations;
            printf("Grid size %i : \t \t", NumBlocks);
            printf( "asynchronous: %.2f us \t", usPerLaunch );
            myfile << usPerLaunch << "\t";
        }
        chTimerGetTime( &start );
        for ( int i = 0; i < cIterations; i++ ) {
            NullKernel<<<NumBlocks,1>>>();
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();
        chTimerGetTime( &stop );
        {
            double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
            double usPerLaunch = microseconds / (float) cIterations;
            printf( "synchronous: %.2f us\n", usPerLaunch );
            myfile << usPerLaunch << "\n";
        }
    }
    myfile.close();

    //output for plots in file launchblock.txt
    myfile.open ("launchblock.txt", std::ios::trunc);
    //loop through different blocksizes
    for ( int j = 0; j<11; j++) {
        int threadsPerBlock = pow(2,j);
        myfile << threadsPerBlock << "\t";
        chTimerTimestamp start, stop;
        chTimerGetTime( &start );
        for ( int i = 0; i < cIterations; i++ ) {
            NullKernel<<<1,threadsPerBlock>>>();
            //cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();
        chTimerGetTime( &stop );

        {
            double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
            double usPerLaunch = microseconds / (float) cIterations;
            printf("Block size %i : \t \t", threadsPerBlock);
            printf( "asynchronous: %.2f us\t", usPerLaunch );
            myfile << usPerLaunch << "\t";
        }
        chTimerGetTime( &start );
        for ( int i = 0; i < cIterations; i++ ) {
            NullKernel<<<1,threadsPerBlock>>>();
            cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();
        chTimerGetTime( &stop );

        {
            double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
            double usPerLaunch = microseconds / (float) cIterations;
            printf( "asynchronous: %.2f us\n", usPerLaunch );
            myfile << usPerLaunch << "\n";
        }
        
    }
    myfile.close();
    return 0;
}
