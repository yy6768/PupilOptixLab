#pragma once

#include <iostream>
#include <NvInfer.h>
#include <cuda/stream.h>

namespace Pupil::tensorRT {
class TRTDenoiser {
public:
    enum class ImportMode {
        Onnx,
        Torch2TRT,
        Cache
    };

    explicit TRTDenoiser(const std::string &onnx_file, ImportMode mode = ImportMode::Onnx);

    ~TRTDenoiser();

    TRTDenoiser(const TRTDenoiser &) = delete;
    TRTDenoiser operator=(const TRTDenoiser &) = delete;
    // Neural network infer
    float* operator() (const float* ) const;


private:
    // Build the tensorRT engine from an onnx file(PyTorch)
    void BuildEngineFromOnnx(const std::string &);
    // Build the tensorRT engine from a torch2trt file (torch2trt)
    void BuildEngineFromTRT(const std::string &);
    // Build the tensorRT engine from a cache file
    void BuildEngineFromCache(const std::string &);
    // Export the cache
    void CacheEngine(const std::string &);

    nvinfer1::ICudaEngine *m_engine;
    nvinfer1::IExecutionContext *m_context;
    void *m_device_buffer[3];
    float *m_output_buffer;
    cudaStream_t m_stream;
    int m_input_idx, m_output_idx;
    // unknown input
    int m_extra_input_idx;
    size_t m_input_sz, m_output_sz;
};
};
