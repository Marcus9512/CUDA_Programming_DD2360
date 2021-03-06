#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <random>
#include <math.h>
#include <time.h>

#include <curand_kernel.h>
#include <curand.h>

#define SEED     921
#define NUM_ITER 1000000000

#define NUM_THREADS 256
#define NUM_BLOCKS 1

//Double precision kernel implementation
__global__ void pi_kernel(curandState* states, int *res, int iterations) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    int seed = id; // different seed per thread
    curand_init(seed, id, 0, &states[id]);  // 	Initialize CURAND

	int count = 0;
	for(int i = 0; i < iterations; i++){
        
		double x = curand_uniform(&states[id]);
		double y = curand_uniform(&states[id]);

		//printf("%f %f\n",x,y);
		double z = sqrt((x * x) + (y * y));

		// Check if point is in unit circle
		if (z <= 1.0)
		{
			count ++;
		}
        
	}
	atomicAdd(res, count);  
}

//Single precision kernel implementation
__global__ void pi_kernel_single_prec(curandState* states, int* res, int iterations) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    int seed = id; // different seed per thread
    curand_init(seed, id, 0, &states[id]);  // 	Initialize CURAND

    int count = 0;
    for (int i = 0; i < iterations; i++) {

        float x = curand_uniform(&states[id]);
        float y = curand_uniform(&states[id]);


        float z = sqrt((x * x) + (y * y));

        // Check if point is in unit circle
        if (z <= 1.0)
        {
            count++;
        }
    }
    atomicAdd(res, count);
}

// The original CPU code, it is not used, only here for comparision 
void originalCode() {
    int count = 0;
    double x, y, z, pi;

    srand(SEED); // Important: Multiply SEED by "rank" when you introduce MPI!

    // Calculate PI following a Monte Carlo method
    for (int iter = 0; iter < NUM_ITER; iter++)
    {
        // Generate random (X,Y) points
        x = (double)rand() / (double)RAND_MAX;
        y = (double)rand() / (double)RAND_MAX;
        z = sqrt((x * x) + (y * y));

        // Check if point is in unit circle
        if (z <= 1.0)
        {
            count++;
        }
    }

    // Estimate Pi and display the result
    pi = ((double)count / (double)NUM_ITER) * 4.0;

    printf("The result is %f\n", pi);
}


int gpu_solution(bool singlePrec) {

    int* res = (int*)malloc(sizeof(int));
	
	dim3 numberOfBlocks(NUM_BLOCKS);
	dim3 numberOfThreads(NUM_THREADS);

	// calculate total number of threads
	float total_amount_of_threads = NUM_BLOCKS * NUM_THREADS;
	
	// calculate how many iterations each thread should do
	int iterationsPerCudaThread = NUM_ITER / total_amount_of_threads;
    int* cuda_res;

    //init random
    curandState* dev_random;
    if (cudaMalloc((void**)&dev_random, total_amount_of_threads * sizeof(curandState)) != cudaSuccess) {
        printf("Error in cudamalloc 1 \n");
        exit(-1);
    }

    if (cudaMalloc(&cuda_res, sizeof(int)) != cudaSuccess) {
        printf("Error in cudamalloc 2 \n");
        exit(-1);
    }	

	// set cuda_res to 0
    cudaMemset(cuda_res, 0, sizeof(int));

    if (singlePrec) {
        printf("Using single precission\n");
        pi_kernel_single_prec << <numberOfBlocks, numberOfThreads >> > (dev_random, cuda_res, iterationsPerCudaThread);
    }
    else {
        printf("Using double precission\n");
        pi_kernel << <numberOfBlocks, numberOfThreads >> > (dev_random, cuda_res, iterationsPerCudaThread);
    }
    
	cudaDeviceSynchronize();
    cudaMemcpy(res, cuda_res, sizeof(int), cudaMemcpyDeviceToHost);

    // Estimate Pi and display the result
    double pi = ((double)*res / (double)NUM_ITER) * 4.0;

    printf("The result is %f\n", pi);

	// Free memory
    cudaFree(cuda_res);
	cudaFree(dev_random);
    free(res);
    return 0;
}

int main(int argc, char* argv[])
{   
    gpu_solution(true);
    return 0;
}

