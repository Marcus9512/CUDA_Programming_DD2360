#include <stdio.h>
#include <random>
#include <time.h>

#define ARRAY_SIZE 100000000
#define THREADS 256
#define RANDOM_MAX 100000


__global__ void saxpy_kernal(float *x, float *y, const float a, int length) {
	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= length) return;
	y[id] = x[id] * a + y[id];

}

// Calculates SAXPY on CPU
void saxpy_on_cpu(float* x, float* y, const float a, int length) {
	clock_t start = clock();
	for (int i = 0; i < length; i++) {
		y[i] = x[i] * a + y[i];
	}
	double time = (double)(clock() - start)/CLOCKS_PER_SEC;
	printf("Computing SAXPY on the CPU Done in %f seconds\n", time);
}

// Checks if a1 and a2 have the same elements
bool isSame(float* a1, float* a2, int length) {
	float margin = 0.0001;
	for (int i = 0; i < length; i++) {
		//printf("a1 %f a2 %f \n", a1[i], a2[i]);
		if (fabs(a1[i] - a2[i]) > margin) {			
			return false;
		}
	}
	return true;
}

int main() {
	
	// Malloc x and y
	float* x = (float*) malloc(ARRAY_SIZE * sizeof(float));
	float* y = (float*) malloc(ARRAY_SIZE * sizeof(float));
	const float a = 2;

	//Store the result from gpu here
	float* parallel_results = (float*)malloc(ARRAY_SIZE * sizeof(float));

	//Fill random values in x and y
	srand((unsigned int)time(NULL));
	for (int i = 0; i < ARRAY_SIZE; i++) {
		x[i] = ((float)rand() / (float)RAND_MAX) * RANDOM_MAX;
		y[i] = ((float)rand() / (float)RAND_MAX) * RANDOM_MAX;
	}

	//Specify the amout of blocks and threads

	//To ensure number of blocks is rounded up 
	dim3 numberOfBlocks((ARRAY_SIZE + THREADS -1) / THREADS);
	dim3 numberOfThreads(THREADS);	
	
	float* x_parallel = 0;
	float* y_parallel = 0;

	//Start timer
	clock_t start = clock();

	//Allocate gpu memory
	if (cudaMalloc(&x_parallel, sizeof(float) * ARRAY_SIZE) != cudaSuccess) {
		printf("Error in cudamalloc 1 \n");
		exit(-1);
	}
	if (cudaMalloc(&y_parallel, sizeof(float) * ARRAY_SIZE) != cudaSuccess) {
		printf("Error in cudamalloc 2 \n");
		exit(-1);
	}	
	
	//Transfer to gpu memory
	cudaMemcpy(x_parallel, x, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(y_parallel, y, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	//Start kernel and calculate SAXPY
	saxpy_kernal <<<numberOfBlocks, numberOfThreads >>> (x_parallel, y_parallel, a, ARRAY_SIZE);
	cudaDeviceSynchronize();
	cudaMemcpy(parallel_results, y_parallel, sizeof(float) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	//Stop timer
	double time = (double)(clock() - start) / CLOCKS_PER_SEC;
	printf("Computing SAXPY on the GPU Done in %f seconds\n",time);

	//Calculate SAXPY on cpu and store result in y
	saxpy_on_cpu(x, y, a, ARRAY_SIZE);

	//Check if the results from the gpu and cpu are the same
	bool results = isSame(y, parallel_results, ARRAY_SIZE);

	if (results) {
		printf("Comparing the output for each implementation, Correct!\n");
	}
	else {
		printf("Comparing the output for each implementation, Wrong \n");
	}

	//Free memory
	cudaFree(x_parallel);
	cudaFree(y_parallel);
	free(x);
	free(y);
	free(parallel_results);

}