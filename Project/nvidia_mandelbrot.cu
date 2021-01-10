/*
	IMPORTENT NOTE:

	The following code about mandelbrot dynamic parallelism is taken from:
    https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/

    With the assosiated github page:
    https://github.com/canonizer/mandelbrot-dyn/blob/master/mandelbrot-dyn/mandelbrot-dyn.cu

	This code is used to compare the execution time for dynamic parallellism with my naive implementation
	located in mandelbrot.cu. Small modifications have been done in order to get the wanted results.

	The author of the original code is Andrew V. Adinetz
*/

///////////////////START OF DYNAMIC IMPLEMENTATION FROM NVIDIA///////////////////////

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#define MAX_DWELL 1024

#define BSX 16
#define BSY 16

/** maximum recursion depth */
#define MAX_DEPTH 4
/** size below which we should call the per-pixel kernel */
#define MIN_SIZE 32
/** subdivision factor along each axis */
#define SUBDIV 4
/** subdivision factor when launched from the host */
#define INIT_SUBDIV 32

#define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (-1)

/** a simple complex type */
struct complex {
	__host__ __device__ complex(float re, float im = 0) {
		this->re = re;
		this->im = im;
	}
	/** real and imaginary part */
	float re, im;
}; // struct complex

// operator overloads for complex numbers
inline __host__ __device__ complex operator+
(const complex& a, const complex& b) {
	return complex(a.re + b.re, a.im + b.im);
}
inline __host__ __device__ complex operator-
(const complex& a) {
	return complex(-a.re, -a.im);
}
inline __host__ __device__ complex operator-
(const complex& a, const complex& b) {
	return complex(a.re - b.re, a.im - b.im);
}
inline __host__ __device__ complex operator*
(const complex& a, const complex& b) {
	return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
}
inline __host__ __device__ float abs2(const complex& a) {
	return a.re * a.re + a.im * a.im;
}
inline __host__ __device__ complex operator/
(const complex& a, const complex& b) {
	float invabs2 = 1 / abs2(b);
	return complex((a.re * b.re + a.im * b.im) * invabs2,
		(a.im * b.re - b.im * a.re) * invabs2);
}  // operator/


#define cucheck_dev(call)                                   \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
    assert(0);                                              \
  }                                                         \
}

#define cucheck(call) \
	{\
	cudaError_t res = (call);\
	if(res != cudaSuccess) {\
	const char* err_str = cudaGetErrorString(res);\
	fprintf(stderr, "%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call);	\
	exit(-1);\
	}\
	}

__device__ int same_dwell(int d1, int d2) {
	if (d1 == d2)
		return d1;
	else if (d1 == NEUT_DWELL || d2 == NEUT_DWELL)
		return min(d1, d2);
	else
		return DIFF_DWELL;
}

/** find the dwell for the pixel */
__device__ int pixel_dwell(int w, int h, complex cmin, complex cmax, int x, int y) {
	complex dc = cmax - cmin;
	float fx = (float)x / w, fy = (float)y / h;
	complex c = cmin + complex(fx * dc.re, fy * dc.im);
	int dwell = 0;
	complex z = c;
	while (dwell < MAX_DWELL && abs2(z) <= 2 * 2) {
		z = z * z + c;
		dwell++;
	}
	return dwell;
}  // pixel_dwell


/** a useful function to compute the number of threads */
__host__ __device__ int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

/** evaluates the common border dwell, if it exists */
__device__ int border_dwell
(int w, int h, complex cmin, complex cmax, int x0, int y0, int d) {
	// check whether all boundary pixels have the same dwell
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int bs = blockDim.x * blockDim.y;
	int comm_dwell = NEUT_DWELL;
	// for all boundary pixels, distributed across threads
	for (int r = tid; r < d; r += bs) {
		// for each boundary: b = 0 is east, then counter-clockwise
		for (int b = 0; b < 4; b++) {
			int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
			int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
			int dwell = pixel_dwell(w, h, cmin, cmax, x, y);
			comm_dwell = same_dwell(comm_dwell, dwell);
		}
	}  // for all boundary pixels
	// reduce across threads in the block
	__shared__ int ldwells[BSX * BSY];
	int nt = min(d, BSX * BSY);
	if (tid < nt)
		ldwells[tid] = comm_dwell;
	__syncthreads();
	for (; nt > 1; nt /= 2) {
		if (tid < nt / 2)
			ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2]);
		__syncthreads();
	}
	return ldwells[0];
}  // border_dwell


/** the kernel to fill in per-pixel values of the portion of the Mandelbrot set*/
__global__ void mandelbrot_pixel_k
(int* dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < d && y < d) {
		
		x += x0, y += y0;
		dwells[y * w + x] = pixel_dwell(w, h, cmin, cmax, x, y);
		//printf("Mandelbrot %d %d %d\n", x, y, dwells[y * w + x]);
	}
}

/** the kernel to fill the image region with a specific dwell value */
__global__ void dwell_fill
(int* dwells, int w, int x0, int y0, int d, int dwell) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < d && y < d) {		
		
		x += x0, y += y0;
		dwells[y * w + x] = dwell;
		//printf("Dwell fill %d %d %d\n", x, y, dwells[y * w + x]);
	}
}  // dwell_fill

__global__ void mandelbrot_block_k(int* dwells,
	int w, int h,
	complex cmin, complex cmax,
	int x0, int y0,
	int d, int depth) {

	x0 += d * blockIdx.x, y0 += d * blockIdx.y;
	int common_dwell = border_dwell(w, h, cmin, cmax, x0, y0, d);

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		if (common_dwell != DIFF_DWELL) {
			// uniform dwell, just fill
			dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
			dwell_fill << <grid, bs >> > (dwells, w, x0, y0, d, common_dwell);
		}
		else if (depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
			// subdivide recursively
			dim3 bs(blockDim.x, blockDim.y), grid(SUBDIV, SUBDIV);
			mandelbrot_block_k << <grid, bs >> >
				(dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth + 1);
		}
		else {
			// leaf, per-pixel kernel
			dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
			mandelbrot_pixel_k << <grid, bs >> >
				(dwells, w, h, cmin, cmax, x0, y0, d);
		}
		cucheck_dev(cudaGetLastError());
	}
}  // mandelbrot_block_k


/*
A launch method for the dynamic solution, slightly adjusted by me
*/


void launchDynamicSolution(int* results, int width, int height) {

	//double check if you can cudamalloc a double array like this.
	int* gpuResultMemory;
	//Allocate gpu memory
	if (cudaMalloc(&gpuResultMemory, sizeof(int) * width * height) != cudaSuccess) {
		printf("Error in cudamalloc, naive-kernel\n");
		exit(-1);
	}
	
	clock_t start = clock();

	//Transfer to gpu memory
	cucheck(cudaMemcpy(gpuResultMemory, results, sizeof(int) * width * height, cudaMemcpyHostToDevice));

	mandelbrot_block_k << <dim3(INIT_SUBDIV, INIT_SUBDIV), dim3(BSX, BSY) >> >
		(gpuResultMemory, width, height,
			complex(-1.5, -1), complex(0.5, 1), 0, 0, width / INIT_SUBDIV, 1);


	cucheck(cudaThreadSynchronize());
	
	cucheck(cudaMemcpy(results, gpuResultMemory, sizeof(int) * width * height , cudaMemcpyDeviceToHost));

	double time = (double)(clock() - start) / CLOCKS_PER_SEC;
	printf("Dynamic Gpu done in %f seconds, width %d height %d BSY %d BSX %d\n", time, width, height, BSY, BSX);

	/*
	Sanity check
	for (int i = 0; i < width*height;i++){
		if (results[i] != 0){
			printf("NOT 0 \n");
			break;
		}
	}
	*/
	cudaFree(gpuResultMemory);
}

///////////////////END OF DYNAMIC IMPLEMENTATION FROM NVIDIA///////////////////////

void setColor(unsigned char* color, int r, int g, int b) {
	color[0] = (int)r;
	color[1] = (int)g;
	color[2] = (int)b;
}

void getColor(unsigned char* color, int index) {
	// colors suggested by https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia
	switch (index) {
	case -1:
		setColor(color, 0, 0, 0);
		break;
	case 0:
		setColor(color, 66, 30, 15);
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
void writeImage(char* name, int* image, int width, int height) {
	FILE* filepointer = fopen(name, "wb");

	// number of colors
	int numColors = 16;

	// we have 3 colors, colorindex defines the range of each
	int colorIndex = MAX_DWELL / numColors;
	int length = width * height;


	//print header, see https://en.wikipedia.org/wiki/Netpbm#File_formats
	fprintf(filepointer, "P6\n %s\n %d\n %d\n %d\n", "# ", width, height, 255);

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

int main(){
	int width = 1024;
	int height = 1024;

	int* resultsDynamic = (int*)malloc(width * height * sizeof(int));

	launchDynamicSolution(resultsDynamic, width, height);

	printf("Dynamic Done \n");	
	writeImage("Dynamic.ppm", resultsDynamic, width, height);

	free(resultsDynamic);

	return 0;
}
