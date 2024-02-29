#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "tensorRT/logger.h"
#include "tensorRT/plugin/common/plugin_register.h"
#include "tensorRT/plugin/kpn/kpn_plugin.h"

namespace {
inline size_t GetDimsSize(const nvinfer1::Dims &dims) {
    size_t sz = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
        sz *= dims.d[i];
    return sz;
}


std::vector<float> loadInputData(const std::string &inputFilePath, size_t size) {
    std::vector<float> inputData(size);
    std::ifstream file(inputFilePath, std::ios::binary);
    if (file.is_open()) {
        file.read(reinterpret_cast<char *>(inputData.data()), size * sizeof(float));
        file.close();
    } else {
        std::cerr << "Error opening file: " << inputFilePath << std::endl;
    }
    return inputData;
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

    // Build Infer engine
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


void test_kpn_plugin() {
    std::filesystem::path root_path = std::filesystem::current_path()
                                          .parent_path()
                                          .parent_path()
                                          .parent_path();
    std::filesystem::path data_path = root_path / "data";
    std::filesystem::path kernel_path = data_path / "input" / "kernel_data.bin";
    std::filesystem::path radiance_path = data_path / "input" / "radiance_data.bin";
    std::filesystem::path output_path = data_path / "input" / "output_data_trt.bin";
    nvinfer1::Dims4 kernelDims{ 1, 49, 1080, 1920 };
    std::vector<float> kernel = loadInputData(kernel_path.string(), GetDimsSize(kernelDims));
    nvinfer1::Dims4 radianceDims{ 1, 3, 1080, 1920 };
    std::vector<float> radiance = loadInputData(radiance_path.string(), GetDimsSize(radianceDims));
    
    int k = 7, h = 1080, w = 1920, c = 3;

    auto builder = nvinfer1::createInferBuilder(Pupil::tensorRT::gLogger);
     uint32_t flag = 1U << static_cast<uint32_t>(
                        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    auto network = builder->createNetworkV2(flag);

    // Load Plugin
    bool isPluginLoaded = initLibKPNPlugins(&Pupil::tensorRT::gLogger, "");
    if (!isPluginLoaded) {
        std::cerr << "Load plugin library error!";
        abort();
    }
    // Input
    auto kernelT = network->addInput("kernel", nvinfer1::DataType::kFLOAT, kernelDims);
    auto radianceT = network->addInput("radiance", nvinfer1::DataType::kFLOAT, radianceDims);
    // Softmax
    auto *softmaxLayer = network->addSoftMax(*kernelT);
    softmaxLayer->setAxes(1 << 1);

    // View + Permutation
    auto *softmaxOutput = softmaxLayer->getOutput(0);
    auto *kernelShuffleLayer = network->addShuffle(*softmaxOutput);
    kernelShuffleLayer->setReshapeDimensions({ 5, { 1, k, k, h, w } });
    nvinfer1::Permutation permutation = { 0, 3, 4, 1, 2 };
    kernelShuffleLayer->setSecondTranspose(permutation);

    auto *radianceShuffleLayer = network->addShuffle(*radianceT);
    radianceShuffleLayer->setFirstTranspose({ 0, 2, 3, 1 });

    
    // Create kpn plugin
    nvinfer1::IPluginCreator *pluginCreator =
        nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kDEFAULT)
            ->getPluginCreator("KPNPluginDynamic", "1");

    // plugin field names & values
    const char *dilationFieldName = "dilation";
    int dilation[2] = { 1, 1 };

    // PluginField
    nvinfer1::PluginField dilationField(dilationFieldName, dilation, nvinfer1::PluginFieldType::kINT32, 2);

    // add PluginField
    nvinfer1::PluginFieldCollection *pluginFieldCollection = new nvinfer1::PluginFieldCollection();
    pluginFieldCollection->nbFields = 1;
    pluginFieldCollection->fields = &dilationField;

    nvinfer1::IPluginV2 *plugin = pluginCreator->createPlugin("kernel prediction", pluginFieldCollection);
    nvinfer1::ITensor *inputTensors[] = { kernelShuffleLayer->getOutput(0), radianceShuffleLayer->getOutput(0) };
    nvinfer1::IPluginV2Layer *pluginLayer = network->addPluginV2(inputTensors, 2, *plugin);


    auto *outputShuffleLayer = network->addShuffle(*(pluginLayer->getOutput(0)));
    outputShuffleLayer->setFirstTranspose({ 0, 3, 1, 2 });
    outputShuffleLayer->getOutput(0)->setName("output");
    network->markOutput(*(outputShuffleLayer->getOutput(0))); 
    //kernelShuffleLayer->getOutput(0)->setName("output");
    //network->markOutput(*(kernelShuffleLayer->getOutput(0)));
    
    // build
    auto config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);// 为构建器设置工作空间大小
    if (builder->platformHasFastFp16()) {
        Pupil::Log::Info("Enable fp16");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
        Pupil::Log::Info("Builfer not support fp16");
    }

    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    for (int i = 0; i < engine->getNbBindings(); i++) {
        Pupil::Log::Info("Binding {} : name : {}, Isinput:{}", i, engine->getBindingName(i), (engine->bindingIsInput(i) ? "Yes" : "No"));
    }

    auto context = engine->createExecutionContext();
    int input_idx[2];
    int output_idx;
    void *device_buffer[3];

    input_idx[0] = engine->getBindingIndex("kernel");
    input_idx[1] = engine->getBindingIndex("radiance");
    //input_idx[2] = engine->getBindingIndex("onnx::Cast_2");
    output_idx = engine->getBindingIndex("output");

    int input_sz[2];
    int output_sz;

    for (int i = 0; i < 2; i++) {
        auto input_dim = engine->getBindingDimensions(input_idx[i]);
        input_sz[i] = GetDimsSize(input_dim);
        cudaMalloc(&device_buffer[input_idx[i]], input_sz[i] * sizeof(float));
    }

    auto output_dim = engine->getBindingDimensions(output_idx);
    output_sz = GetDimsSize(output_dim);
    cudaMalloc(&device_buffer[output_idx], output_sz * sizeof(float));

    // Organize input data
    cudaStream_t stream{};
    cudaMemcpyAsync(device_buffer[input_idx[0]], 
        kernel.data(), input_sz[0] * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_buffer[input_idx[1]],
                    radiance.data(), input_sz[1] * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);
    const char *input_name[2];
    for (int i = 0; i < 2; i++) {
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
    
    std::vector<float> output(output_sz);
    cudaMemcpy(output.data(), device_buffer[output_idx], output_sz * sizeof(float), cudaMemcpyDeviceToHost);

    // 保存到bin文件
    std::ofstream outputFile(output_path.string(), std::ios::binary);
    outputFile.write(reinterpret_cast<char *>(output.data()), output_sz * sizeof(float));
    outputFile.close();

    cudaFree(device_buffer[input_idx[0]]);
    cudaFree(device_buffer[input_idx[1]]);
    cudaFree(device_buffer[output_idx]);
}




void test_kpn() {
    std::filesystem::path root_path = std::filesystem::current_path()
                                          .parent_path()
                                          .parent_path()
                                          .parent_path();
    std::filesystem::path data_path = root_path / "data";
    std::filesystem::path kernel_path = data_path / "input" / "kernel_data.bin";
    std::filesystem::path radiance_path = data_path / "input" / "radiance_data.bin";
    std::filesystem::path output_path = data_path / "input" / "expect_output_data_trt.bin";
    nvinfer1::Dims4 kernelDims{ 1, 49, 1080, 1920 };
    std::vector<float> kernel = loadInputData(kernel_path.string(), GetDimsSize(kernelDims));
    nvinfer1::Dims4 radianceDims{ 1, 3, 1080, 1920 };
    std::vector<float> radiance = loadInputData(radiance_path.string(), GetDimsSize(radianceDims));

    int k = 7, h = 1080, w = 1920;

    auto builder = nvinfer1::createInferBuilder(Pupil::tensorRT::gLogger);
    uint32_t flag = 1U << static_cast<uint32_t>(
                        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(flag);

     // onnx parser
    auto parser = nvonnxparser::createParser(*network, Pupil::tensorRT::gLogger);

    std::filesystem::path onnx_path = data_path / "onnx" / "kpn.onnx";
    parser->parseFromFile(onnx_path.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
  
    // build
    auto config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16()) {
        Pupil::Log::Info("Enable fp16");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
        Pupil::Log::Info("Builfer not support fp16");
    }

    size_t free_size, total_size;
    cuMemGetInfo(&free_size, &total_size);
    Pupil::Log ::Info("Total GPU mem: {} MB, free GPU mem: {} MB", total_size >> 20, free_size >> 20);
    config->setMaxWorkspaceSize(free_size);
    auto plan = builder->buildSerializedNetwork(*network, *config);
    auto runtime = createInferRuntime(Pupil::tensorRT::gLogger);

    auto engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    for (int i = 0; i < engine->getNbBindings(); i++) {
        Pupil::Log::Info("Binding {} : name : {}, Isinput:{}", i, engine->getBindingName(i), (engine->bindingIsInput(i) ? "Yes" : "No"));
    }

   auto context = engine->createExecutionContext();
   int input_idx[3];
   int output_idx;
   void *device_buffer[4];

   input_idx[0] = engine->getBindingIndex("kernel");
   input_idx[1] = engine->getBindingIndex("radiance");
   input_idx[2] = engine->getBindingIndex("onnx::Cast_2");
   output_idx = engine->getBindingIndex("output");

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

    // Organize input data
    
    cudaStream_t stream{};
    cudaMemcpyAsync(device_buffer[input_idx[0]],
                    kernel.data(), input_sz[0] * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_buffer[input_idx[1]],
                    radiance.data(), input_sz[1] * sizeof(float), cudaMemcpyHostToDevice, stream);
    float *extra_input = new float[input_sz[2]];
    cudaMemcpyAsync(device_buffer[input_idx[2]],
                    extra_input, input_sz[2] * sizeof(float), cudaMemcpyHostToDevice, stream);

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

    std::vector<float> output(output_sz);
    cudaMemcpy(output.data(), device_buffer[output_idx], output_sz * sizeof(float), cudaMemcpyDeviceToHost);

    // 保存到bin文件
    std::ofstream outputFile(output_path.string(), std::ios::binary);
    outputFile.write(reinterpret_cast<char *>(output.data()), output_sz * sizeof(float));
    outputFile.close();

    cudaFree(device_buffer[input_idx[0]]);
    cudaFree(device_buffer[input_idx[1]]);
    cudaFree(device_buffer[input_idx[2]]);
    cudaFree(device_buffer[output_idx]);
    delete[] extra_input;
    parser->destroy();
}
