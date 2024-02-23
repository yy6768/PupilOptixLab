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


void test_kpn_plugin() {
    nvinfer1::Dims kernelDims{ 1, 49, 1080, 1920 };
    std::vector<float> kernel = loadInputData("kernel_data.bin", GetDimsSize(kernelDims));
    nvinfer1::Dims radianceDims{ 1, 3, 1080, 1920 };
    std::vector<float> radiance = loadInputData("radiance_data.bin", GetDimsSize(radianceDims));
    
    int k = 7, h = 1080, w = 1920;

    auto builder = nvinfer1::createInferBuilder(Pupil::tensorRT::gLogger);
    auto network = builder->createNetworkV2(0U);

    // Input
    auto kernelT = network->addInput("kernel", nvinfer1::DataType::kFLOAT, kernelDims);
    auto radianceT = network->addInput("radiance", nvinfer1::DataType::kFLOAT, radianceDims);
     // Softmax
    auto *softmaxLayer = network->addSoftMax(*kernelT);
    softmaxLayer->setAxes(1 << 1);

    // View + Permutation
    auto *softmaxOutput = softmaxLayer->getOutput(0);
    auto *shuffleLayer = network->addShuffle(*softmaxOutput);
    shuffleLayer->setReshapeDimensions(nvinfer1::Dims{ -1, k, k, h, w });
    nvinfer1::Permutation permutation = { 0, 3, 4, 1, 2 };
    shuffleLayer->setSecondTranspose(permutation);

    // Create kpn plugin
    nvinfer1::IPluginCreator *pluginCreator =
        nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kDEFAULT)
            ->getPluginCreator("KPN", "1");

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
    nvinfer1::ITensor *inputTensors[] = { shuffleLayer->getOutput(0), radianceT };
    nvinfer1::IPluginV2Layer *pluginLayer = network->addPluginV2(inputTensors, 2, *plugin);
    pluginLayer->getOutput(0)->setName("output");
    
    // build
    auto config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);// 为构建器设置工作空间大小
    auto engine = builder->buildEngineWithConfig(*network, *config);


}
