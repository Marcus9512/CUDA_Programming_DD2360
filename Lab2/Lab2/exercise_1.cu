#include <stdio.h>

#define BLOCKS 1
#define THREADS 256

//Create a kernal to perform the wanted task
__global__ void kernal() {
	//Get the tread id and print it
	printf("Hello world, I'm thread number %d \n", threadIdx.x + blockIdx.x*blockDim.x);
}
int main() {

	//Specify the amout of blocks and threads
	dim3 numberOfBlocks(BLOCKS);
	dim3 numberOfThreads(THREADS);

	//Launch kernal
	kernal <<<numberOfBlocks, numberOfThreads>>> ();

	//Wait for kernal to terminate
	cudaDeviceSynchronize();

}