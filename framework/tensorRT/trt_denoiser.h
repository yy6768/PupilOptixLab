#pragma once

#include <iostream>
#include <NvInfer.h>

#include "cuda/stream.h"

namespace Pupil::tensorRT {
class TRTDenoiser {
public:
    enum class ImportMode {
        Onnx,
        Torch2TRT,
        Cache
    };

    explicit TRTDenoiser(const std::string &onnx_file, 
        ImportMode mode = ImportMode::Onnx,
        bool use_kpn=false);

    ~TRTDenoiser();

    TRTDenoiser(const TRTDenoiser &) = delete;
    TRTDenoiser operator=(const TRTDenoiser &) = delete;
    // Neural network infer
    float* operator() (const float* ) const;

private:
    // Buffer size (1 input, 1 output)
    static const int BUFFER_SIZE = 3;
    // FP16 enable
    static const bool FP16_ENABLE = true;
    // Build the tensorRT engine from an onnx file(PyTorch)
    void BuildEngineFromOnnx(const std::string &);
    // Build the tensorRT engine from a torch2trt file (torch2trt)
    void BuildEngineFromTRT(const std::string &);
    // Build the tensorRT engine from a cache file
    void BuildEngineFromCache(const std::string &);
    // Export the cache
    void CacheEngine(const std::string &);

    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    void *m_device_buffer[BUFFER_SIZE];
    float *m_output_buffer;
    float *m_extra_buffer;

    std::unique_ptr<cuda::Stream> m_stream;
    // Input & output idx
    int m_input_idx, m_output_idx;
    // Optional Extra input
    int m_extra_input_idx;
    size_t m_input_sz, m_output_sz, m_extra_sz;
    // Use Kernel prediction plugin
    bool m_use_plugin;
};
};
