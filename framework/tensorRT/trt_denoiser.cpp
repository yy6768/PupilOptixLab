#include "trt_denoiser.h"
#include "logger.h"

#include "plugin/kpn_plugin.h"

#include "util/timer.h"

#include <fstream>
#include <filesystem>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntimeCommon.h>


namespace {
#define TRT_ASSERT(condition, message)                                                                                \
    do {                                                                                                              \
        if (!(condition)) {                                                                                           \
            Pupil::Log::Error("Assertion failed, condition:{}, in {}, line : {}, message: {}", #condition, __FILE__, __LINE__, message); \
            std::abort();                                                                                             \
        }                                                                                                             \
    } while (false)

static inline size_t GetDimsSize(const nvinfer1::Dims& dims) {
    size_t sz = 1;
    for (size_t i = 0; i < dims.nbDims; ++i) 
        sz *= dims.d[i];
    return sz;
}
} // namespace 

namespace Pupil::tensorRT {
TRTDenoiser::TRTDenoiser(const std::string &model_file, ImportMode mode, bool use_plugin): m_use_plugin(use_plugin) {
    std::filesystem::path model_path(model_file);
    Pupil::Log::Info("extension:{}", model_path.extension().string());
    auto cache_file_path = model_path;
    cache_file_path.replace_extension("cache");
    Log::Info("Cache file:{}, exist:{}", cache_file_path.string(), std::filesystem::exists(cache_file_path));
    if (std::filesystem::exists(cache_file_path)) {
        TRT_ASSERT(std::filesystem::exists(cache_file_path), "Don't exist cache file");
		BuildEngineFromCache(cache_file_path.string());

    } else {
		if (mode == ImportMode::Onnx) {
			BuildEngineFromOnnx(model_path.string());
			CacheEngine(cache_file_path.string());// add cache file
		}
		else {// From trt model file
            BuildEngineFromCache(model_path.string());
            CacheEngine(cache_file_path.string()); // add cache file
		}
    }
    // Create context
    TRT_ASSERT(m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext()), "Context generation fails");
    // Bind index
    // ***** Debug ******
    for (int i = 0; i < m_engine->getNbBindings(); i++) {
        Pupil::Log::Info("Binding {} : name : {}, Isinput:{}",i, m_engine->getBindingName(i), (m_engine->bindingIsInput(i) ? "Yes" : "No"));
    }
    // ***** Debug ******
    //TRT_ASSERT((m_input_idx = m_engine->getBindingIndex("input0")) == 0, "Input index wrong");
    m_input_idx = m_engine->getBindingIndex("input0");
    m_output_idx = m_engine->getBindingIndex("output0");
    //TRT_ASSERT((m_output_idx = m_engine->getBindingIndex("output0")) == 2, "output index wrong");
    // Get & calculate input
    auto input_dims = m_engine->getBindingDimensions(m_input_idx);
    auto output_dims = m_engine->getBindingDimensions(m_output_idx);
    m_input_sz = GetDimsSize(input_dims);
    m_output_sz = GetDimsSize(output_dims);

    TRT_ASSERT(cudaMalloc(&m_device_buffer[m_input_idx], m_input_sz * sizeof(float)) == 0, "No device memory");
    TRT_ASSERT(cudaMalloc(&m_device_buffer[m_output_idx], m_output_sz * sizeof(float)) == 0, "No device memory");

    //Extra input
    m_extra_input_idx = 1;
    auto extra_input_dims = m_engine->getBindingDimensions(m_extra_input_idx);
    m_extra_sz = GetDimsSize(extra_input_dims);
    Log::Info("extra_sz:{}", m_extra_sz);
    TRT_ASSERT(cudaMalloc(&m_device_buffer[m_extra_input_idx], m_extra_sz * sizeof(float)) == 0, "No device memory");
    
    // Cuda Stream
    m_stream = std::make_unique<cuda::Stream>();
    // Output Buffer
    m_output_buffer = new float[m_output_sz];
    TRT_ASSERT(m_output_buffer != nullptr, "Output allocation fails");
    m_extra_buffer = new float[m_extra_sz];
    TRT_ASSERT(m_extra_buffer != nullptr, "Extra allocation fails");
    
}

TRTDenoiser::~TRTDenoiser() {
	delete[] m_output_buffer;
	cudaFree(m_device_buffer[m_output_idx]);
	cudaFree(m_device_buffer[m_extra_input_idx]);
	cudaFree(m_device_buffer[m_input_idx]);
}
/**
* The module is exported by torch.onnx.export
**/
void TRTDenoiser::BuildEngineFromOnnx(const std::string &onnx_file) {
    Pupil::Log::Info("Build tensorRT from onnx");
    // Build the builder
    auto builder = nvinfer1::createInferBuilder(gLogger);
    TRT_ASSERT(builder != nullptr, "Generate builder fails");
    // Build the network & parser
    uint32_t flag = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    auto network = builder->createNetworkV2(flag);
    auto parser = nvonnxparser::createParser(*network, gLogger);
    TRT_ASSERT(parser != nullptr, "Generate parser fails");
    parser->parseFromFile(onnx_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));

    if (m_use_plugin) {
        // Get output
        nvinfer1::ITensor *kernelT = network->getOutput(0);
        nvinfer1::ITensor *radianceT = network->getOutput(1);
        auto dim = kernelT->getDimensions();
        int k = sqrt(dim.d[1]);
        int h = dim.d[2];
        int w = dim.d[3];
        // Softmax
        auto *softmaxLayer = network->addSoftMax(*kernelT);
        softmaxLayer->setAxes(1 << 1);

        // View + Permutation
        auto *softmaxOutput = softmaxLayer->getOutput(0);
        auto *shuffleLayer = network->addShuffle(*softmaxOutput);
        shuffleLayer->setReshapeDimensions(nvinfer1::Dims{ 5, { -1, k, k, h, w } });
        nvinfer1::Permutation permutation = { 0, 3, 4, 1, 2 };
        shuffleLayer->setSecondTranspose(permutation);

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
        nvinfer1::ITensor *inputTensors[] = { shuffleLayer->getOutput(0), radianceT };
        nvinfer1::IPluginV2Layer *pluginLayer = network->addPluginV2(inputTensors, 2, *plugin);
        pluginLayer->getOutput(0)->setName("output0");
        network->markOutput(*(pluginLayer->getOutput(0)));
    }

    // Build each layer of the network
    // **********
    // DEBUG
   /* for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }*/
    // ***********
    // Try to transform to FP16
    auto config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16() && FP16_ENABLE) {
        Pupil::Log::Info("Enable fp16");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
        Pupil::Log::Info("Builfer not support fp16");
    }


    size_t free_size, total_size;
    cuMemGetInfo(&free_size, &total_size);
    Pupil::Log ::Info("Total GPU mem: {} MB, free GPU mem: {} MB", total_size >> 20, free_size >> 20);
    config->setMaxWorkspaceSize(free_size);
    TRT_ASSERT(m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config)), "Build engine fail");
    // Destroy the local variable
    parser->destroy();
}

/**
* Use the torch2trt file(.pth) file to build the network
*/
void TRTDenoiser::BuildEngineFromTRT(const std::string& trt_file) {
    Pupil::Log::Info("Build engine from trt engine");
    std::ifstream ifs(trt_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    auto buffer = std::make_unique<char[]>(sz);
    ifs.read(buffer.get(), sz);
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    TRT_ASSERT(runtime != nullptr, "Runtime build fails");
    TRT_ASSERT(m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.get(), sz)) , "Engine build fails");
    Pupil::Log::Info("Engine has been built");
}

/*
* If the cache file was generated, it can be used to build the network
*/
void TRTDenoiser::BuildEngineFromCache(const std::string &cache_file){
    Pupil::Log::Info("Build engine from cache");
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    auto buffer = std::make_unique<char[]>(sz);
    ifs.read(buffer.get(), sz);
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    TRT_ASSERT(runtime != nullptr, "Runtime build fails");
    TRT_ASSERT(m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.get(), sz)), "Engine build fails");
    Pupil::Log::Info("Engine has been built");
}

void TRTDenoiser::CacheEngine(const std::string& cache_file) {
    auto engine_buffer = m_engine->serialize();
    TRT_ASSERT(engine_buffer != nullptr, "Engine buffer is empty");
    std::ofstream ofs(cache_file, std::ios::binary);
    ofs.write(static_cast<const char *>(engine_buffer->data()), engine_buffer->size());
    engine_buffer->destroy();
}


/**
* Denoise function
*/ 
 float* TRTDenoiser::operator()(const float* input) const {
    // Preprocess for cuda
    cudaMemcpyAsync(m_device_buffer[m_input_idx], input, 
        m_input_sz * sizeof(float), cudaMemcpyHostToDevice, m_stream->GetStream());
    
    cudaMemcpyAsync(m_device_buffer[m_extra_input_idx], m_extra_buffer, 
        m_extra_sz * sizeof(float), cudaMemcpyHostToDevice, m_stream->GetStream() );
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, m_stream->GetStream());
    
#if NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    auto input_name = m_engine->getIOTensorName(m_input_idx);
    auto extra_input_name = m_engine->getIOTensorName(m_extra_input_idx);
    auto output_name = m_engine->getIOTensorName(m_output_idx);
    TRT_ASSERT(m_context->setTensorAddress(input_name, m_device_buffer[m_input_idx]), "Set input tensor address fails");
    TRT_ASSERT(m_context->setTensorAddress(extra_input_name, m_device_buffer[m_extra_input_idx]), "Set extra input tensor address fails");
    TRT_ASSERT(m_context->setTensorAddress(output_name, m_device_buffer[m_output_idx]), "Set output tensor address fails");
    TRT_ASSERT(m_context->enqueueV3(m_stream->GetStream()), "EnqueueV3 fails");
#else // NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR < 85
    // Infer
    TRT_ASSERT(m_context->enqueueV2(m_device_buffer, m_stream->GetStream(), nullptr), "EnqueueV2 fails");
#endif //NV_TENSORRT_MAJOR * 10 + NV_TENSORRT_MINOR >= 85
    cudaEventRecord(end, m_stream->GetStream());
     //m_context->executeV2(m_device_buffer);
    //cudaStreamSynchronize(m_stream);
    cudaEventSynchronize(end);
    float total_time;
    cudaEventElapsedTime(&total_time, start, end);
    Pupil::Log::Info("Cost time:{} ms", total_time);

    cudaMemcpyAsync(m_output_buffer, m_device_buffer[m_output_idx], 
        m_output_sz * sizeof(float), cudaMemcpyDeviceToHost, m_stream->GetStream());
    return m_output_buffer;
 }

} // namespace Pupil::tensorRT
