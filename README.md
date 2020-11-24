# CUDA_Programming_DD2360
This is a repository containing my solutions to the lab assignmensts for DD2360

# Cuda commands
* compile: `nvcc -O1 -arch=sm_75 file.cu -o out`
* Nvprof: `nvprof ./out`
* Mem-check: `nvcc -lineinfo -Xcompiler -rdynamic -o out file.cu` `cuda-memcheck ./out`
* Nvvp: `nvprof --output-profile profile_mycode.nvprof ./out`
