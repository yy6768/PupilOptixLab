#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>

#include "tensorRT/logger.h"

void test_custom_kpn() {

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
    int input_idx[2];
    int output_idx;
    input_idx[0] = engine->getBindingIndex("input0");
    input_idx[1] = engine->getBindingIndex("input1");
    output_idx = engine->getBindingIndex("output0");
    
    cudaStream_t stream;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);
    auto input_name = engine->getIOTensorName(input_idx[0]);
    auto extra_input_name = engine->getIOTensorName(input_idx[1]);
    auto output_name = engine->getIOTensorName(output_idx);
    void *device_buffer[3];
    context->setTensorAddress(input_name, device_buffer[input_idx[0]]);
    context->setTensorAddress(extra_input_name, device_buffer[input_idx[1]]);
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


