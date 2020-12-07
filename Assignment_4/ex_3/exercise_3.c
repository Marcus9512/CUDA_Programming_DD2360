// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>

#define NUM_PARTICLES 100000
#define NUM_ITERATIONS 100
#define BLOCK_SIZE 256  //Number of threads

#define RANDOM_C 1000
#define RANDOM_V 10

#define VELOCITY_DEC 0.0001

// A particle
typedef struct {
    cl_float3 pos;
    cl_float3 velocity;
}Particle;


// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));

// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

//kerneal SAXPY
const char* particleKernel =
   
    "typedef struct { \n"
    "float3 pos; \n"
    "float3 velocity;\n"
    "}Particle;\n"
    "\n"
    "void updateVelocity(__global Particle* par, int index, float vel)\n"
    "{par[index].velocity.x -= vel; \n"
    "par[index].velocity.y -= vel; \n"
    "par[index].velocity.z -= vel;}\n"
    "\n"
    "void updatePos(__global Particle* par, int index)\n"
    "{par[index].pos += par[index].velocity;}\n"
    "\n"
    "__kernel           \n"
    "void particleSim(__global Particle* par, \n"
    "                 int iterations, \n"
    "                 int length, \n"
    "                 float vel)"
    "{int id = get_global_id(0);\n"
    "if (id >= length) return;"
    "for(int i=0 ; i < iterations; i++)\n"
    "{updateVelocity(par, id, vel);\n"
    "updatePos(par, id);}}\n";


// Calculate simulation on CPU
void particleCPU(Particle * par, int len) {
    for (int i = 0; i < len; i++) {
        //update velocity
        par[i].velocity.x -= VELOCITY_DEC;
        par[i].velocity.y -= VELOCITY_DEC;
        par[i].velocity.z -= VELOCITY_DEC;

        //update position
        par[i].pos.x += par[i].velocity.x;
        par[i].pos.y += par[i].velocity.y;
        par[i].pos.z += par[i].velocity.z;
        
    }
}

// Evaluate if the cpu and gpu solution are the same
bool equivalent(Particle* p_cpu, Particle* p_gpu, int len) {
    float margin = 0.00001;
    for (int i = 0; i < len; i++) {
        //printf("X: %f %f, Y: %f %f Z: %f %f \n", p_gpu[i].pos.x, p_cpu[i].pos.x, p_gpu[i].pos.y,p_cpu[i].pos.y , p_gpu[i].pos.z, p_cpu[i].pos.z);
        //Check position
        if (fabs(p_gpu[i].pos.x - p_cpu[i].pos.x) > margin ||
            fabs(p_gpu[i].pos.y - p_cpu[i].pos.y) > margin ||
            fabs(p_gpu[i].pos.z - p_cpu[i].pos.z) > margin) {
            return false;
        }
        //Check velocity
        if (fabs(p_gpu[i].velocity.x - p_cpu[i].velocity.x) > margin ||
            fabs(p_gpu[i].velocity.y - p_cpu[i].velocity.y) > margin ||
            fabs(p_gpu[i].velocity.z - p_cpu[i].velocity.z) > margin) {
            return false;
        }
    }
    return true;
}


int main(int argc, char* argv) {
    cl_platform_id* platforms; cl_uint     n_platform;

    // Find OpenCL Platforms
    cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * n_platform);
    err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

    // Find and sort devices
    cl_device_id* device_list; cl_uint n_devices;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices);CHK_ERROR(err);
    device_list = (cl_device_id*)malloc(sizeof(cl_device_id) * n_devices);
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);CHK_ERROR(err);

    // Create and initialize an OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &err);CHK_ERROR(err);

    // Create a command queue
    cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);CHK_ERROR(err);

    //Init
    int array_size_byte = NUM_PARTICLES * sizeof(Particle);
    Particle* particles = (Particle*)malloc(array_size_byte);
    Particle* res_gpu = (Particle*)malloc(array_size_byte);
    int len = NUM_PARTICLES;
    int iterations = NUM_ITERATIONS;
    float vel = VELOCITY_DEC;

    //Fill random values particles
    srand((unsigned int)time(NULL));
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].pos.x = ((float)rand() / (float)RAND_MAX) * RANDOM_C;
        particles[i].pos.y = ((float)rand() / (float)RAND_MAX) * RANDOM_C;
        particles[i].pos.z = ((float)rand() / (float)RAND_MAX) * RANDOM_C;

        particles[i].velocity.x = ((float)rand() / (float)RAND_MAX) * RANDOM_V;
        particles[i].velocity.y = ((float)rand() / (float)RAND_MAX) * RANDOM_V;
        particles[i].velocity.z = ((float)rand() / (float)RAND_MAX) * RANDOM_V;
    }
   
    //Init buffers, inspierd by slides
    cl_mem particles_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, array_size_byte, NULL, &err);CHK_ERROR(err); 

    clock_t startk = clock();

    //Transfer data to host
    err = clEnqueueWriteBuffer(cmd_queue, particles_dev, CL_TRUE, 0, array_size_byte, particles,0, NULL, NULL);CHK_ERROR(err);
   
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&particleKernel, NULL, &err);CHK_ERROR(err);

    err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    //check error (given code from leacutre)
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build error: %s\n", buffer); return 0;
    }

   
    //create kernel
    cl_kernel kernel = clCreateKernel(program, "particleSim", &err);CHK_ERROR(err);

    //set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&particles_dev);CHK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), (void*)&iterations);CHK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), (void*)&len);CHK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(float), (void*)&vel);CHK_ERROR(err);
  
    size_t n_workitem = NUM_PARTICLES + (BLOCK_SIZE - (NUM_PARTICLES % BLOCK_SIZE));
    size_t workgroup_size = BLOCK_SIZE;

    //start gpu-timer
   
    clock_t startk2 = clock();

    err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &n_workitem, &workgroup_size, 0, NULL, NULL);CHK_ERROR(err);
    
    err = clEnqueueReadBuffer(cmd_queue, particles_dev, CL_TRUE, 0, array_size_byte, res_gpu, 0, NULL, NULL);CHK_ERROR(err);
    
    err = clFlush(cmd_queue);CHK_ERROR(err);
    clFinish(cmd_queue);

    double timek = (double)(clock() - startk) / CLOCKS_PER_SEC;
    double timek2 = (double)(clock() - startk2) / CLOCKS_PER_SEC;
    printf("Simulation on the GPU Done in %f seconds, only kernel time %f\n", timek, timek2);

    //Run simulation on CPU
    clock_t start = clock();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        //printf("%d\n",i);
        particleCPU(particles, NUM_PARTICLES);
    }
    double time = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("CPU done in %f seconds!\n", time);

    bool res = equivalent(particles, res_gpu, NUM_PARTICLES);

    if (res) {
        printf("Comparing the output for each implementation, Correct!\n");
    }
    else {
        printf("Comparing the output for each implementation, Wrong \n");
    }

    // Finally, release all that we have allocated.
    err = clReleaseCommandQueue(cmd_queue);CHK_ERROR(err);
    err = clReleaseContext(context);CHK_ERROR(err);
    free(platforms);
    free(device_list);
    free(particles);
    free(res_gpu);

    return 0;
}



// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clGetErrorString(int errorCode) {
    switch (errorCode) {
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
    case -69: return "CL_INVALID_PIPE_SIZE";
    case -70: return "CL_INVALID_DEVICE_QUEUE";
    case -71: return "CL_INVALID_SPEC_ID";
    case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
    case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
    case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
    case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
    case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
    case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
    case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
    case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
    case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
    case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
    case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
    case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
    case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
    case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
    case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
    case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
    case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
    case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
    default: return "CL_UNKNOWN_ERROR";
    }
}
