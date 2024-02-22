#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>

#include "tensorRT/logger.h"

namespace {
static inline size_t GetDimsSize(const nvinfer1::Dims &dims) {
    size_t sz = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
        sz *= dims.d[i];
    return sz;
}
}

void test_import_custom_kpn() {

    nvinfer1::ILogger &logger = Pupil::tensorRT::gLogger;
    // 创建TensorRT的builder
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

    // 创建ONNX解析器
    auto parser = nvonnxparser::createParser(*network, logger);

    // 解析ONNX文件
    std::filesystem::path root_path = std::filesystem::current_path()
                                          .parent_path()
                                          .parent_path()
                                          .parent_path();
    std::filesystem::path data_path = root_path / "data";
    std::filesystem::path onnx_path = data_path / "onnx" / "custom_kpn.onnx";
    parser->parseFromFile(onnx_path.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    // 构建推理引擎
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    for (int i = 0; i < engine->getNbBindings(); i++) {
        Pupil::Log::Info("Binding {} : name : {}, Isinput:{}", i, engine->getBindingName(i), (engine->bindingIsInput(i) ? "Yes" : "No"));
    }

    auto context = engine->createExecutionContext();
    int input_idx[3];
    int output_idx;
    void * device_buffer[4];

    input_idx[0] = engine->getBindingIndex("kernel");
    input_idx[1] = engine->getBindingIndex("radiance");
    input_idx[2] = engine->getBindingIndex("onnx::Cast_2");
    output_idx = engine->getBindingIndex("output0");

    int input_sz[3];
    int output_sz;



    for (int i = 0; i < 3; i++) {
        auto input_dim = engine->getBindingDimensions(input_idx[i]);
        input_sz[i] = GetDimsSize(input_dim);
        cudaMalloc(&device_buffer[input_idx[i]], input_sz[i] * sizeof(float));
    }

    auto output_dim = engine->getBindingDimensions(output_idx);
    output_sz = GetDimsSize(output_dim);
    cudaMalloc(&device_buffer[output_idx], output_sz * sizeof(float));
       
    cudaStream_t stream{};
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);
    const char *input_name[3];
    for (int i = 0; i < 3; i++) {
        input_name[i] = engine->getIOTensorName(input_idx[i]);
        context->setTensorAddress(input_name[i], device_buffer[input_idx[i]]);
    }
    auto output_name = engine->getIOTensorName(output_idx);

    context->setTensorAddress(output_name, device_buffer[output_idx]);
    context->enqueueV3(stream);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float total_time;
    cudaEventElapsedTime(&total_time, start, end);
    Pupil::Log::Info("total_time:{}", total_time);
    // 释放资源
    parser->destroy();
}


