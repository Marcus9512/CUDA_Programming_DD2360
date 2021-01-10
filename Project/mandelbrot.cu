 #include <stdio.h>
 #include <math.h>
#include <time.h>

#define BLOCK_SIZE 16

#define MAX_DWELL 1024

void setColor(unsigned char* color, int r, int g, int b) {
	color[0] =(int) r ;
	color[1] =(int) g ;
	color[2] =(int) b ;
}

void getColor(unsigned char* color, int index) {
	// colors suggested by https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia
	switch (index) {
	case -1:
		setColor(color, 0, 0, 0);
		break;
	case 0:
		setColor(color, 66, 30, 15);
		//setColor(color, 155, 30, 15);
		break;
	case 1:
		setColor(color, 25, 7, 26);
		break;
	case 2:
		setColor(color, 9, 1, 47);
		break;
	case 3:
		setColor(color, 4, 4, 73);
		break;
	case 4:
		setColor(color, 0, 7, 100);
		break;
	case 5:
		setColor(color, 12, 44, 138);
		break;
	case 6:
		setColor(color, 24, 82, 177);
		break;
	case 7:
		setColor(color, 57, 125, 209);
		break;
	case 8:
		setColor(color, 138, 182, 229);
		break;
	case 9:
		setColor(color, 211, 236, 248);
		break;
	case 10:
		setColor(color, 241, 233, 191);
		break;
	case 11:
		setColor(color, 248, 201, 95);
		break;
	case 12:
		setColor(color, 255, 170, 0);
		break;
	case 13:
		setColor(color, 204, 128, 0);
		break;
	case 14:
		setColor(color, 153, 87, 0);
		break;
	case 15:
		setColor(color, 106, 52, 3);
		break;
	}
}
/*
Set the color on a pixel depending on the dwell value in results.
*/
void writeImage(char *name, int *image, int width, int height){
	FILE * filepointer = fopen(name, "wb");
	
	// number of colors
	int numColors = 16;

	// we have 3 colors, colorindex defines the range of each
	int colorIndex = MAX_DWELL / numColors;
	int length = width * height;    


	//print header, see https://en.wikipedia.org/wiki/Netpbm#File_formats
	fprintf(filepointer,"P6\n %s\n %d\n %d\n %d\n","# ",width,height,255);

	for (int i = 0; i < length; i++) {

		//calculates which color index the pixel belongs to
		int index = image[i] / colorIndex;
		int index2 = index - 1;

		if (image[i] >= MAX_DWELL || image[i] <= 0) {
			index = -1;
			index2 = -1;
		} 
		//calculates how "strong" the color is (in %)
		double scale = (image[i] % colorIndex) / ((double)colorIndex);
		
		// declare color
		unsigned char color1[3];
		unsigned char color2[3];
		unsigned char finalColor[3];

		getColor(color1, index);
		getColor(color2, index2);

		//calculate final color, linear scaling between color1 and color2
		for (int c = 0; c < 3; c++) {
			finalColor[c] = color1[c] + (color2[c] - color1[c]) * scale;
		}
		
		fwrite(finalColor, 1, 3, filepointer);
		
	}	
	fclose(filepointer);
}

/*
	Using the dwell alorithm (The Escape Time Algorithm)
	Code is inspired by the pseudocode from wikipedia:
	https://en.wikipedia.org/wiki/Mandelbrot_set
*/
void cpuImplementation(int *results, int width, int height){
	
	clock_t start = clock();
	for (int j = 0; j < height; j++){
		float cy = -1.0f + j*(2.0f / (float)height);
		
		for (int i = 0; i < width; i++){
		
			float cx = -1.5f + i * (2.0f / (float)width);
		
			int currentDwel = 0;
			float x = 0;
			float y = 0;
			
			for(; currentDwel < MAX_DWELL && (x*x + y*y)<= 4.0f ; currentDwel++){
				/*if(j == 38 && i == 714){
					printf("y: %f x: %f %d\n",y,x,currentDwel);
				}*/
				float temp = x*x - y*y + cx;
				y = 2.0f*x*y + cy;
				x = temp;
			}
			//printf("%d %d\n", (j + i * height), (width * height));
			results[i + j * width] = currentDwel;
			
		}
		
	}
	double time = (double)(clock() - start) / CLOCKS_PER_SEC;
	printf("CPU done in %f seconds, width %d height %d\n", time, width, height);
}

/*
	My naive escape time algorithm implementation of mandelbrot based on the wikipedia solution
	https://en.wikipedia.org/wiki/Mandelbrot_set
*/
__global__ void naiveKernel(int *results, int width, int height){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;	
	//printf("IDX %d %d %d IDY %d %d %d\n", threadIdx.x, blockIdx.x, blockDim.x, threadIdx.y, blockIdx.y, blockDim.y);
	if (idx >= width || idy >= height) return;
	//printf("IDX %d %d %d IDY %d %d %d\n",threadIdx.x, blockIdx.x,blockDim.x, threadIdx.y, blockIdx.y,blockDim.y);		
	
	int currentDwel = 0;
	float x = 0;
	float y = 0;

	float cy = -1.0f + idy * (2.0f / (float)height); // 1 + 1
	float cx = -1.5f + idx * (2.0f / (float)width); // 0.5 + 1.5

	for(; currentDwel < MAX_DWELL && (x*x + y*y)<= 4.0f ; currentDwel++){
		/*if(idy == 38 && idx == 714){
			printf("GPU y: %f x: %f %d\n",y,x,currentDwel);
		}*/
		float temp = x*x - y*y + cx;
		y = 2.0f*x*y + cy;
		x = temp;
	}	

	results[idx + idy * width] = currentDwel;
	
}

/*
	Uses the naive kernel to generate a madelbrot image.
*/
void naiveKernelImplementation(int *results, int width, int height, int blockSize){

	dim3 blockGrid (
					((width + (blockSize - 1)) / blockSize),
					((height + (blockSize - 1)) / blockSize)
				   );

	dim3 block(blockSize, blockSize);
	
	//printf("Blockgrid %d %d block %d %d\n", blockGrid.x, blockGrid.y, block.x, block.y);
	//double check if you can cudamalloc a double array like this.
	int* gpuResultMemory;
	//Allocate gpu memory
	if (cudaMalloc(&gpuResultMemory, sizeof(int) * width * height) != cudaSuccess) {
		printf("Error in cudamalloc, naive-kernel\n");
		exit(-1);
	}
	
	clock_t start = clock();
	//Transfer to gpu memory
	cudaMemcpy(gpuResultMemory, results, sizeof(int) * width * height, cudaMemcpyHostToDevice);
	
	naiveKernel <<<blockGrid, block>>> (gpuResultMemory, width, height);
	cudaDeviceSynchronize();
	
	
	cudaMemcpy(results, gpuResultMemory, sizeof(int) * width * height , cudaMemcpyDeviceToHost);

	double time = (double)(clock() - start) / CLOCKS_PER_SEC;
	printf("Naive Gpu done in %f seconds, width %d height %d block %d\n", time, width, height, blockSize);
	
	cudaFree(gpuResultMemory);
}

bool isEqual(int* a, int* b, int length) {
	bool res = true;
	int numWrong = 0;
	for (int i = 0; i < length; i++) {
		if (abs(a[i] - b[i]) > 10) {
			printf("Not equal, %d %d i= %d\n", a[i],b[i], i);
			numWrong++;
			res = false;
		}
	}
	printf("Number of unequal dwells: %f %%\n", ((double) numWrong/length));
	return res;
}

void test() {
	int testpointsCPU = 5;
	int testpointsGPU = 6;

	int blockpoints = 5;
	int pointsCPU[] = {
				 1024,
				 2048,
				 4096,
				 8192,
				 16384
	};
	int pointsGPU[] = { 
					 1024,
					 2048,
					 4096,
					 8192,
					 16384,
					 32768
	};

	int blocksize[] = {
					 4,
					 8,
					 10,
					 16,
					 32					
	};
	for (int i = 0; i < testpointsCPU; i++) {
		int size = pointsCPU[i];
		int* resultsCpu = (int*)malloc(size * size * sizeof(int));
		cpuImplementation(resultsCpu, size, size);
		free(resultsCpu);
	}
	for (int i = 0; i < testpointsGPU; i++) {
		int size = pointsGPU[i];
		for (int j = 0; j < blockpoints; j++) {
			int* resultsNaive = (int*)malloc(size * size * sizeof(int));
			int bs = blocksize[j];
			naiveKernelImplementation(resultsNaive, size, size, bs);
			free(resultsNaive);
		}
	}
}

void generateImage() {

	bool checkEqual = false;
	bool useCPU = false;

	int width  =  1024;
	int height =  1024;

	int* resultsCpu = (int*)malloc(width * height * sizeof(int));
	int* resultsNaive = (int*)malloc(width * height * sizeof(int));

	if (!resultsCpu || !resultsNaive) {
		printf("Failiur in malloc");
		exit(-2);
	}
	
	if (useCPU) {
		// CPU implementation
		cpuImplementation(resultsCpu, width, height);
		printf("CPU Done \n");
		writeImage("CPU_Image.ppm", resultsCpu, width, height);
	}	

	// Naive CUDA implementation
	naiveKernelImplementation(resultsNaive, width, height, BLOCK_SIZE);
	printf("Naive GPU Done \n");
	writeImage("Naive_GPU_Image.ppm", resultsNaive, width, height);

	if (checkEqual) {
		if (isEqual(resultsCpu, resultsNaive, width * height)) {
			printf("CPU and GPU equal\n");
		}
		else {
			printf("CPU and GPU not equal\n");
		}
	}

	free(resultsCpu);
	free(resultsNaive);
}

int main(int argc, char **argv){
	
	bool image = true;
	bool runTest = false;

	if (image) {
		generateImage();
	}
	if (runTest) {
		test();
	}
	
	return 0;

}