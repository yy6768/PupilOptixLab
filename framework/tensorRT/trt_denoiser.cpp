#include "trt_denoiser.h"
#include "ilogger.h"

#include "util/timer.h"

#include <fstream>
#include <filesystem>
#include <assert.h>

#include <cuda.h>
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
TRTDenoiser::TRTDenoiser(const std::string &model_file, ImportMode mode) {
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
    TRT_ASSERT((m_context = m_engine->createExecutionContext()) != nullptr, "Context generation fails");
    // Bind index
    // ***** Debug ******
    for (int i = 0; i < m_engine->getNbBindings(); i++) {
        Pupil::Log::Info("Binding {} : name : {}, Isinput:{}",i, m_engine->getBindingName(i), (m_engine->bindingIsInput(i) ? "Yes" : "No"));
    }
    // ***** Debug ******
    TRT_ASSERT((m_input_idx = m_engine->getBindingIndex("input0")) == 0, "Input index wrong");
   
    TRT_ASSERT((m_output_idx = m_engine->getBindingIndex("output0")) == 2, "output index wrong");
    // Get & calculate input
    auto input_dims = m_engine->getBindingDimensions(m_input_idx);
    auto output_dims = m_engine->getBindingDimensions(m_output_idx);
    m_input_sz = GetDimsSize(input_dims);
    m_output_sz = GetDimsSize(output_dims);

    TRT_ASSERT(cudaMalloc(&m_device_buffer[m_input_idx], m_input_sz * sizeof(float)) == 0, "No device memory");
    TRT_ASSERT(cudaMalloc(&m_device_buffer[m_output_idx], m_output_sz * sizeof(float)) == 0, "No device memory");

    //Extra input
    int extra_input_idx = 1;
    auto extra_input_dims = m_engine->getBindingDimensions(extra_input_idx);
    auto extra_sz = GetDimsSize(extra_input_dims);
    TRT_ASSERT(cudaMalloc(&m_device_buffer[extra_input_idx], extra_sz * sizeof(float)) == 0, "No device memory");
    
    // Cuda Stream
    TRT_ASSERT(cudaStreamCreate(&m_stream) == 0, "Create strean fails");
    m_output_buffer = new float[m_output_sz];
    TRT_ASSERT(m_output_buffer != nullptr, "Output allocation fails");
}

TRTDenoiser::~TRTDenoiser() {
    delete[] m_output_buffer;
    cudaStreamDestroy(m_stream);
    cudaFree(m_device_buffer[m_output_idx]);
    cudaFree(m_device_buffer[m_input_idx]);
    m_context->destroy();
    m_engine->destroy();
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
    // Build each layer of the network
    // **********
    // DEBUG
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    // ***********
    // Try to transform to FP16
    auto config = builder->createBuilderConfig();
   /* if (builder->platformHasFastFp16()) {
        Pupil::Log::Info("Enable fp16");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
        Pupil::Log::Info("Builfer not support fp16");
    }*/


    size_t free_size, total_size;
    cuMemGetInfo(&free_size, &total_size);
    Pupil::Log ::Info("Total GPU mem: {} MB, free GPU mem: {} MB", total_size >> 20, free_size >> 20);
    config->setMaxWorkspaceSize(free_size);
    TRT_ASSERT((m_engine = builder->buildEngineWithConfig(*network, *config)) != nullptr, "Build engine fail");
    // Destroy the local variable
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
    Pupil::Log::Info("Complete");
}

/**
* Use the torch2trt file(.pth) file to build the network
*/
void TRTDenoiser::BuildEngineFromTRT(const std::string& trt_file) {
    Pupil::Log::Info("Build engine from cache");
    std::ifstream ifs(trt_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    auto buffer = std::make_unique<char[]>(sz);
    ifs.read(buffer.get(), sz);
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    TRT_ASSERT(runtime != nullptr, "Runtime build fails");
    TRT_ASSERT((m_engine = runtime->deserializeCudaEngine(buffer.get(), sz)) != nullptr, "Engine build fails");
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
    TRT_ASSERT((m_engine = runtime->deserializeCudaEngine(buffer.get(), sz)) != nullptr, "Engine build fails");
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
    // preprocess for cuda
    cudaMemcpyAsync(m_device_buffer[m_input_idx], input, 
        m_input_sz * sizeof(float), cudaMemcpyHostToDevice, m_stream);
    // infer
    // *** debug
    Timer timer;
    timer.Start();
    // *** debug
    m_context->enqueueV2(m_device_buffer, m_stream, nullptr);
    cudaMemcpyAsync(m_output_buffer, m_device_buffer[m_output_idx], 
        m_output_sz * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);
    timer.Stop();
    Pupil::Log::Info("Cost time:{} ms", timer.ElapsedMilliseconds());
    return m_output_buffer;
 }


} // namespace Pupil::tensorRT
