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

__global__ void pi_kernel(curandState* states, int *res, int iterations) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    //if (id >= iter) return;

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

    printf("%d\n",res);

	atomicAdd(res, count);  

}

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

int gpu_solution() {

    int* res = (int*)malloc(sizeof(int));

	
	dim3 numberOfBlocks(NUM_BLOCKS);
    dim3 numberOfThreads(NUM_THREADS);

	float total_amount_of_threads = NUM_BLOCKS * NUM_THREADS;

    int iterationsPerCudaThread = NUM_ITER / total_amount_of_threads;
    int* cuda_res;


    //init random
    curandState* dev_random;
    cudaMalloc((void**)&dev_random, total_amount_of_threads  * sizeof(curandState));

    cudaMemset(cuda_res, 0, sizeof(int));

    
    pi_kernel << <numberOfBlocks, numberOfThreads >> > (dev_random, cuda_res, iterationsPerCudaThread);
	cudaDeviceSynchronize();

    cudaMemcpy(res, cuda_res, sizeof(int), cudaMemcpyDeviceToHost);

    // Estimate Pi and display the result
    printf("%d",*res);
    double pi = ((double)*res / (double)NUM_ITER) * 4.0;

    printf("The result is %f\n", pi);

    cudaFree(cuda_res);
    free(res);
    return 0;
}

int main(int argc, char* argv[])
{
   
    gpu_solution();
    return 0;
}

