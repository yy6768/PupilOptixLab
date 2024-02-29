// Copyright(c) 2022 Alex S.Fu All rights reserved.
// Inspired by https://github.com/IwakuraRein/KernelFilter-PyTorch/tree/main/_KernelFilter
// Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "kpn_plugin.h"
#include "tensorRT/plugin/common/trt_common.h"
#include <cuda.h>
#include <cuda_fp16.h>

namespace nvinfer1 {


template<typename scalar_t>
__global__ void kernelFilterKernel(
    const scalar_t *__restrict__ kernel,  // [B, H, W, k, k]
    const scalar_t *__restrict__ radiance,// [B, H, W, C]
    scalar_t *__restrict__ output,        // [B, H, W, C]
    const int batch,
    const int channel,
    const int height,
    const int width,
    const int k0,    // filter size
    const int half_k,// k0 / 2
    const int dilation_h,
    const int dilation_w) {

    // batch index
    const int n = blockIdx.y;
    // column index
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    const int xi = col / width;// Kernel height index
    const int yi = col % width;// Kernel width index

    // radiance offset
    const int off_r = n * height * width * channel;

    // kernel radiance offset
    const int off_k = ((n * height + xi) * width + yi) * k0 * k0;

    if (n >= batch || xi >= height || yi >= width) return;
    scalar_t color[3] = { scalar_t(0), scalar_t(0), scalar_t(0) };
    
    for (int io = -half_k; io <= half_k; io++) {
        int xo = xi + io * dilation_h;// True output height index
        if (xo < 0 || xo >= height) continue;
        for (int jo = -half_k; jo <= half_k; jo++) {
            int yo = yi + jo * dilation_w;// True output width index
            if (yo < 0 || yo >= width) continue;
            int radiance_idx = off_r + (xo * width + yo) * channel;
            int kernel_idx = off_k + (io + half_k) * k0 + jo + half_k;
            #pragma unroll
            for (int c = 0; c < 3; c++) {
                if (radiance[radiance_idx + c] > scalar_t(0)) {
                    color[c] += radiance[radiance_idx + c] * kernel[kernel_idx];
                }
            }
        }
    }
    int output_idx = off_r + (xi * width + yi) * channel;
    output[output_idx] = color[0];
    output[output_idx + 1] = color[1];
    output[output_idx + 2] = color[2];
}

/**
  *
  */
template<typename scalar_t>
int kernelFilterKernelLauncher(
    const scalar_t *__restrict__ kernel,  // [B, H, W, k, k]
    const scalar_t *__restrict__ radiance,// [B, C, H, W]
    scalar_t *__restrict__ output,        // [B, C, H, W]
    const int batch,
    const int channel,
    const int height,
    const int width,
    const int k0,    // filter size
    const int half_k,// k0 / 2
    const int dilation_h,
    const int dilation_w) {
    const dim3 blocks(GET_BLOCKS(height * width), batch);

    kernelFilterKernel<scalar_t><<<blocks, NUM_THREADS>>>(
        kernel,
        radiance,
        output,
        batch,
        channel,
        height,
        width,
        k0,
        half_k,
        dilation_h,
        dilation_w);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

// TODO: Int8
int32_t KPNPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                              const nvinfer1::PluginTensorDesc *outputDesc,
                              const void *const *inputs, void *const *outputs,
                              void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {

    int k = inputDesc[0].dims.d[3];

    int rb = inputDesc[1].dims.d[0];
    int rc = inputDesc[1].dims.d[1];
    int rh = inputDesc[1].dims.d[2];
    int rw = inputDesc[1].dims.d[3];

    const void *kernel = inputs[0];
    const void *radiance = inputs[1];

    void *output = outputs[0];

    // quantization
    float inputScale = inputDesc->scale;
    float outputScale = outputDesc->scale;

    auto data_type = inputDesc[0].type;
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            return kernelFilterKernelLauncher<float>(
                (float *)kernel,
                (float *)radiance,
                (float *)output,
                rb, rc, rh, rw, k, k / 2, mDilation.d[0], mDilation.d[1]);
            break;
        case nvinfer1::DataType::kHALF:
            return kernelFilterKernelLauncher<__half>(
                (__half *)kernel,
                (__half *)radiance,
                (__half *)output,
                rb, rc, rh, rw, k, k / 2, mDilation.d[0], mDilation.d[1]);
        case nvinfer1::DataType::kINT8:// TODO quantization
            return 1;
        default:
            return 1;
    }
}

}

