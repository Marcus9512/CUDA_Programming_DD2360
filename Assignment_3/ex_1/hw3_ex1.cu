#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE  16
#define HEADER_SIZE 122
#define BLOCK_SIZE_SH 18


#ifdef _WIN32
/////////////////////////////////////WINDOWS FIX

/*
 * Taken from https://gist.github.com/ugovaretto/5875385
 * Author: Ugo Varetto - ugovaretto@gmail.com
 * This code is distributed under the terms of the Apache Software License version 2.0
 * https://opensource.org/licenses/Apache-2.0
*/

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
 #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
 #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

#include < time.h >
#include < windows.h>

struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};


int gettimeofday(struct timeval* tv, struct timezone* tz)
{
    FILETIME ft;
    unsigned __int64 tmpres = 0;
    static int tzflag = 0;

    if (NULL != tv)
    {
        GetSystemTimeAsFileTime(&ft);

        tmpres |= ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |= ft.dwLowDateTime;

        tmpres /= 10;  /*convert into microseconds*/
        /*converting file time to unix epoch*/
        tmpres -= DELTA_EPOCH_IN_MICROSECS;
        tv->tv_sec = (long)(tmpres / 1000000UL);
        tv->tv_usec = (long)(tmpres % 1000000UL);
    }

    if (NULL != tz)
    {
        if (!tzflag)
        {
            _tzset();
            tzflag++;
        }
        tz->tz_minuteswest = _timezone / 60;
        tz->tz_dsttime = _daylight;
    }

    return 0;
}

////////////////////////////////////////////END OF WINDOWS FIX
#elif __linux__ 
#include <sys/time.h>
#endif

typedef unsigned char BYTE;

/**
 * Structure that represents a BMP image.
 */
typedef struct
{
    int   width;
    int   height;
    float *data;
} BMPImage;

typedef struct timeval tval;

BYTE g_info[HEADER_SIZE]; // Reference header

/**
 * Reads a BMP 24bpp file and returns a BMPImage structure.
 * Thanks to https://stackoverflow.com/a/9296467
 */
BMPImage readBMP(char *filename)
{
    BMPImage bitmap = { 0 };
    int      size   = 0;
    BYTE     *data  = NULL;
    FILE     *file  = fopen(filename, "rb");
    
    // Read the header (expected BGR - 24bpp)
    fread(g_info, sizeof(BYTE), HEADER_SIZE, file);

    // Get the image width / height from the header
    bitmap.width  = *((int *)&g_info[18]);
    bitmap.height = *((int *)&g_info[22]);
    size          = *((int *)&g_info[34]);
    
    // Read the image data
    data = (BYTE *)malloc(sizeof(BYTE) * size);
    fread(data, sizeof(BYTE), size, file);
    
    // Convert the pixel values to float
    bitmap.data = (float *)malloc(sizeof(float) * size);
    
    for (int i = 0; i < size; i++)
    {
        bitmap.data[i] = (float)data[i];
    }
    
    fclose(file);
    free(data);
    
    return bitmap;
}

/**
 * Writes a BMP file in grayscale given its image data and a filename.
 */
void writeBMPGrayscale(int width, int height, float *image, char *filename)
{
    FILE *file = NULL;
    
    file = fopen(filename, "wb");
    
    // Write the reference header
    fwrite(g_info, sizeof(BYTE), HEADER_SIZE, file);
    
    // Unwrap the 8-bit grayscale into a 24bpp (for simplicity)
    for (int h = 0; h < height; h++)
    {
        int offset = h * width;
        
        for (int w = 0; w < width; w++)
        {
            BYTE pixel = (BYTE)((image[offset + w] > 255.0f) ? 255.0f :
                                (image[offset + w] < 0.0f)   ? 0.0f   :
                                                               image[offset + w]);
            
            // Repeat the same pixel value for BGR
            fputc(pixel, file);
            fputc(pixel, file);
            fputc(pixel, file);
        }
    }
    
    fclose(file);
}

/**
 * Releases a given BMPImage.
 */
void freeBMP(BMPImage bitmap)
{
    free(bitmap.data);
}

/**
 * Checks if there has been any CUDA error. The method will automatically print
 * some information and exit the program when an error is found.
 */
void checkCUDAError()
{
    cudaError_t cudaError = cudaGetLastError();
    
    if(cudaError != cudaSuccess)
    {
        printf("CUDA Error: Returned %d: %s\n", cudaError,
                                                cudaGetErrorString(cudaError));
        exit(-1);
    }
}

/**
 * Calculates the elapsed time between two time intervals (in milliseconds).
 */
double get_elapsed(tval t0, tval t1)
{
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}

/**
 * Stores the result image and prints a message.
 */
void store_result(int index, double elapsed_cpu, double elapsed_gpu,
                     int width, int height, float *image)
{
    char path[255];
    
    sprintf(path, "images/hw3_result_%d.bmp", index);
    writeBMPGrayscale(width, height, image, path);
    
    printf("Step #%d Completed - Result stored in \"%s\".\n", index, path);
    printf("Elapsed CPU: %fms / ", elapsed_cpu);
    
    if (elapsed_gpu == 0)
    {
        printf("[GPU version not available]\n");
    }
    else
    {
        printf("Elapsed GPU: %fms\n", elapsed_gpu);
    }
}

/**
 * Converts a given 24bpp image into 8bpp grayscale using the CPU.
 */
void cpu_grayscale(int width, int height, float *image, float *image_out)
{
    for (int h = 0; h < height; h++)
    {
        int offset_out = h * width;      // 1 color per pixel
        int offset     = offset_out * 3; // 3 colors per pixel
        
        for (int w = 0; w < width; w++)
        {
            float *pixel = &image[offset + w * 3];
            
            // Convert to grayscale following the "luminance" model
            image_out[offset_out + w] = pixel[0] * 0.0722f + // B
                                        pixel[1] * 0.7152f + // G
                                        pixel[2] * 0.2126f;  // R
        }
    }
}

/**
 * Converts a given 24bpp image into 8bpp grayscale using the GPU.
 */
__global__ void gpu_grayscale(int width, int height, float *image, float *image_out)
{
    ////////////////
    // TO-DO #4.2 /////////////////////////////////////////////
    // Implement the GPU version of the grayscale conversion //
    ///////////////////////////////////////////////////////////
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx > width || idy > height) return;

    // Same code as the inner loop from cpu_grayscale
    int offset_out = idy * width;      // 1 color per pixel
    int offset = offset_out * 3;       // 3 colors per pixel

    float* pixel = &image[offset + idx * 3];

    // Convert to grayscale following the "luminance" model
    image_out[offset_out + idx] = pixel[0] * 0.0722f + // B
        pixel[1] * 0.7152f + // G
        pixel[2] * 0.2126f;  // R
}

/**
 * Applies a 3x3 convolution matrix to a pixel using the CPU.
 */
float cpu_applyFilter(float *image, int stride, float *matrix, int filter_dim)
{
    float pixel = 0.0f;
    
    for (int h = 0; h < filter_dim; h++)
    {
        int offset        = h * stride;
        int offset_kernel = h * filter_dim;
        
        for (int w = 0; w < filter_dim; w++)
        {
            pixel += image[offset + w] * matrix[offset_kernel + w];
        }
    }
    
    return pixel;
}

/**
 * Applies a 3x3 convolution matrix to a pixel using the GPU.
 */
__device__ float gpu_applyFilter(float *image, int stride, float *matrix, int filter_dim)
{
    ////////////////
    // TO-DO #5.2 ////////////////////////////////////////////////
    // Implement the GPU version of cpu_applyFilter()           //
    //                                                          //
    // Does it make sense to have a separate gpu_applyFilter()? //
    //////////////////////////////////////////////////////////////

    // This method is called by threads from gpu_gaussian, withother words only one
    // Thread will calculate the value for 1 pixel, thus the code can be the same as cpu_applyFilter().
    // However, we can't call cpu_applyFilter from the gpu and thus need a __device__ method with
    // the same functionality. 

    // Code is copied from cpu_applyFilter
    float pixel = 0.0f;

    for (int h = 0; h < filter_dim; h++)
    {
        int offset = h * stride;
        int offset_kernel = h * filter_dim;

        for (int w = 0; w < filter_dim; w++)
        {
            pixel += image[offset + w] * matrix[offset_kernel + w];
        }
    }

    return pixel;
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the CPU.
 */
void cpu_gaussian(int width, int height, float *image, float *image_out)
{
    float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                          2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                          1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };
    
    for (int h = 0; h < (height - 2); h++)
    {
        int offset_t = h * width;
        int offset   = (h + 1) * width;
        
        for (int w = 0; w < (width - 2); w++)
        {
            image_out[offset + (w + 1)] = cpu_applyFilter(&image[offset_t + w],
                                                          width, gaussian, 3);
        }
    }
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the GPU.
 */
__global__ void gpu_gaussian(int width, int height, float *image, float *image_out)
{
    __shared__ float sh_block[BLOCK_SIZE_SH * BLOCK_SIZE_SH];

    float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                          2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                          1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };
    
    // Gives id x and y for the entire image
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	//From original implementation
    int offset_t = index_y * width + index_x;
    int offset   = (index_y + 1) * width + (index_x + 1);

    //From tutorial in shared memory
    int shared_pos = threadIdx.y * BLOCK_SIZE_SH + threadIdx.x;

	bool less_then_width = index_x < (width - 2);
	bool less_then_height = index_y < (height - 2);

    sh_block[shared_pos] = image[offset_t];

	//Make sure to copy data to position [14][14], [14][15], [15][14] and [15][15] in sh_block
    if(less_then_width && less_then_height){
        if (threadIdx.y >= (BLOCK_SIZE - 2) && (threadIdx.x >= (BLOCK_SIZE - 2))) {
			//calculates adjustments
			int adjustment_sh = 2 + BLOCK_SIZE_SH * 2;
			int adjustment_im = width * 2 + 2;
            sh_block[shared_pos + adjustment_sh] = image[offset_t + adjustment_im];
        }
    }

	//Make sure to copy data to the colums [x][14] and [x][15] where 0<=x<=15
    if(less_then_height){
        if (threadIdx.y >= (BLOCK_SIZE - 2)) {
			//calculates adjustments
			int adjustment_sh = BLOCK_SIZE_SH * 2;
			int adjustment_im = width * 2;
            sh_block[shared_pos+ adjustment_sh] = image[offset_t + adjustment_im];
        }
    }

	//Make sure to copy data to the colums [14][y] and [15][y] where 0<=y<=15
    if(less_then_width){
        if (threadIdx.x >= (BLOCK_SIZE - 2)) {
			//calculates adjustments
			int adjustment_sh_im = 2;
            sh_block[shared_pos + adjustment_sh_im] = image[offset_t + adjustment_sh_im];
        }
    }

    __syncthreads();
	
	//make sure that position is defined for gpu_applyFilter
    if (!less_then_width && !less_then_height) return;

	image_out[offset] = gpu_applyFilter(&sh_block[shared_pos],
            BLOCK_SIZE_SH, gaussian, 3);   
}

/**
 * Calculates the gradient of an image using a Sobel filter on the CPU.
 */
void cpu_sobel(int width, int height, float *image, float *image_out)
{
    float sobel_x[9] = { 1.0f,  0.0f, -1.0f,
                         2.0f,  0.0f, -2.0f,
                         1.0f,  0.0f, -1.0f };
    float sobel_y[9] = { 1.0f,  2.0f,  1.0f,
                         0.0f,  0.0f,  0.0f,
                        -1.0f, -2.0f, -1.0f };
    
    for (int h = 0; h < (height - 2); h++)
    {
        int offset_t = h * width;
        int offset   = (h + 1) * width;
        
        for (int w = 0; w < (width - 2); w++)
        {
            float gx = cpu_applyFilter(&image[offset_t + w], width, sobel_x, 3);
            float gy = cpu_applyFilter(&image[offset_t + w], width, sobel_y, 3);
            
            // Note: The output can be negative or exceed the max. color value
            // of 255. We compensate this afterwards while storing the file.
            image_out[offset + (w + 1)] = sqrtf(gx * gx + gy * gy);
        }
    }
}

/**
 * Calculates the gradient of an image using a Sobel filter on the GPU.
 */
__global__ void gpu_sobel(int width, int height, float *image, float *image_out)
{
    ////////////////
    // TO-DO #6.1 /////////////////////////////////////
    // Implement the GPU version of the Sobel filter //
    ///////////////////////////////////////////////////

    __shared__ float sh_block[BLOCK_SIZE_SH * BLOCK_SIZE_SH];

    float sobel_x[9] = { 1.0f,  0.0f, -1.0f,
                        2.0f,  0.0f, -2.0f,
                        1.0f,  0.0f, -1.0f };
    float sobel_y[9] = { 1.0f,  2.0f,  1.0f,
                         0.0f,  0.0f,  0.0f,
                        -1.0f, -2.0f, -1.0f };

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
   
    //From original implementation
    int offset_t = idy * width + idx;
    int offset = (idy + 1) * width + (idx+1);

    //From tutorial in shared memory
    int shared_pos = threadIdx.y * BLOCK_SIZE_SH + threadIdx.x;

	bool less_then_width = idx < (width - 2)
	bool less_then_height = idy < (height - 2)

	sh_block[shared_pos] = image[offset_t];

    //Make sure to copy data to position [14][14], [14][15], [15][14] and [15][15] in sh_block
    if(less_then_width && less_then_height){
        if (threadIdx.y >= (BLOCK_SIZE - 2) && (threadIdx.x >= (BLOCK_SIZE - 2))) {
			//calculates adjustments
			int adjustment_sh = 2 + BLOCK_SIZE_SH * 2;
			int adjustment_im = width * 2 + 2;
            sh_block[shared_pos + adjustment_sh] = image[offset_t + adjustment_im];
        }
    }

	//Make sure to copy data to the colums [x][14] and [x][15] where 0<=x<=15
    if(less_then_height){
        if (threadIdx.y >= (BLOCK_SIZE - 2)) {
            //calculates adjustments
			int adjustment_sh = BLOCK_SIZE_SH * 2;
			int adjustment_im = width * 2;
            sh_block[shared_pos+ adjustment_sh] = image[offset_t + adjustment_im];
        }
    }

	//Make sure to copy data to the colums [14][y] and [15][y] where 0<=y<=15
    if(less_then_width){
        if (threadIdx.x >= (BLOCK_SIZE - 2)) {
            //calculates adjustments
			int adjustment_sh_im = 2;
            sh_block[shared_pos + adjustment_sh_im] = image[offset_t + adjustment_sh_im];
        }
    }

    __syncthreads();

	//make sure that position is defined for gpu_applyFilter
    if (!less_then_width && !less_then_height) return;

    float gx = gpu_applyFilter(&sh_block[shared_pos], BLOCK_SIZE_SH, sobel_x, 3);
    float gy = gpu_applyFilter(&sh_block[shared_pos], BLOCK_SIZE_SH, sobel_y, 3);

    // Note: The output can be negative or exceed the max. color value
    // of 255. We compensate this afterwards while storing the file.
    image_out[offset] = sqrtf(gx * gx + gy * gy);

}

int main(int argc, char **argv)
{
    BMPImage bitmap          = { 0 };
    float    *d_bitmap       = { 0 };
    float    *image_out[2]   = { 0 };
    float    *d_image_out[2] = { 0 };
    int      image_size      = 0;
    tval     t[2]            = { 0 };
    double   elapsed[2]      = { 0 };
    dim3     grid(1);                       // The grid will be defined later
    dim3     block(BLOCK_SIZE, BLOCK_SIZE); // The block size will not change
    
    // Make sure the filename is provided
    if (argc != 2)
    {
        fprintf(stderr, "Error: The filename is missing!\n");
        return -1;
    }
    
    // Read the input image and update the grid dimension
    bitmap     = readBMP(argv[1]);
    image_size = bitmap.width * bitmap.height;
    grid       = dim3(((bitmap.width  + (BLOCK_SIZE - 1)) / BLOCK_SIZE),
                      ((bitmap.height + (BLOCK_SIZE - 1)) / BLOCK_SIZE));
    
    printf("Image opened (width=%d height=%d).\n", bitmap.width, bitmap.height);
    
    // Allocate the intermediate image buffers for each step
    for (int i = 0; i < 2; i++)
    {
        image_out[i] = (float *)calloc(image_size, sizeof(float));
        
        cudaMalloc(&d_image_out[i], image_size * sizeof(float));
        cudaMemset(d_image_out[i], 0, image_size * sizeof(float));
    }

    cudaMalloc(&d_bitmap, image_size * sizeof(float) * 3);
    cudaMemcpy(d_bitmap, bitmap.data,
               image_size * sizeof(float) * 3, cudaMemcpyHostToDevice);
    
    // Step 1: Convert to grayscale
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_grayscale(bitmap.width, bitmap.height, bitmap.data, image_out[0]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);
        
        // Launch the GPU version
        gettimeofday(&t[0], NULL);
        gpu_grayscale<<<grid, block>>>(bitmap.width, bitmap.height, d_bitmap, d_image_out[0]);
        
        cudaMemcpy(image_out[0], d_image_out[0], image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Store the result image in grayscale
        store_result(1, elapsed[0], elapsed[1], bitmap.width, bitmap.height, image_out[0]);
    }
    
    // Step 2: Apply a 3x3 Gaussian filter
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        //cpu_gaussian(bitmap.width, bitmap.height, image_out[0], image_out[1]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);
        
        // Launch the GPU version
        gettimeofday(&t[0], NULL);
        gpu_gaussian<<<grid, block>>>(bitmap.width, bitmap.height,
                                       d_image_out[0], d_image_out[1]);
        
        cudaMemcpy(image_out[1], d_image_out[1],
                    image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Store the result image with the Gaussian filter applied
        store_result(2, elapsed[0], elapsed[1], bitmap.width, bitmap.height, image_out[1]);
    }
    
    // Step 3: Apply a Sobel filter
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        //cpu_sobel(bitmap.width, bitmap.height, image_out[1], image_out[0]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);
        
        // Launch the GPU version
        gettimeofday(&t[0], NULL);
        gpu_sobel<<<grid, block>>>(bitmap.width, bitmap.height,
                                    d_image_out[1], d_image_out[0]);
        
        cudaMemcpy(image_out[0], d_image_out[0],
                   image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Store the final result image with the Sobel filter applied
        store_result(3, elapsed[0], elapsed[1], bitmap.width, bitmap.height, image_out[0]);
    }
    
    // Release the allocated memory
    for (int i = 0; i < 2; i++)
    {
        free(image_out[i]);
        cudaFree(d_image_out[i]);
    }
    
    freeBMP(bitmap);
    cudaFree(d_bitmap);
    
    return 0;
}

