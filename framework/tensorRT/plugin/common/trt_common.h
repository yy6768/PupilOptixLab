#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>
#include <iostream>

namespace nvinfer1 {
//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

const int NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N) {
    return (N + NUM_THREADS - 1) / NUM_THREADS;
}
}// namespace nvinfer1
