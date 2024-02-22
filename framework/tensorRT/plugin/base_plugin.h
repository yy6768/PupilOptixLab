// Copyright (c) OpenMMLab. All rights reserved.
// https://github.com/open-mmlab/mmdeploy/blob/main/csrc/mmdeploy/backend_ops/tensorrt/common/trt_plugin_base.hpp
#ifndef TRT_BASE_PLUGIN_H
#define TRT_BASE_PLUGIN_H
#include "NvInferRuntime.h"
#include "NvInferVersion.h"
#include <iostream>
#include <vector>

#if NV_TENSORRT_MAJOR > 7
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

namespace {
// Enumerator for status
typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

}// namespace

namespace nvinfer1 {
class TRTPluginBase : public nvinfer1::IPluginV2DynamicExt {
public:
    TRTPluginBase(const std::string &name) : mLayerName(name) {}
    // IPluginV2 Methods
    const char *getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
    int initialize() TRT_NOEXCEPT override { return STATUS_SUCCESS; }
    void terminate() TRT_NOEXCEPT override {}
    void destroy() TRT_NOEXCEPT override { delete this; }
    void setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override {
        mNamespace = pluginNamespace;
    }
    const char *getPluginNamespace() const TRT_NOEXCEPT override { return mNamespace.c_str(); }

    virtual void configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs,
                                 DynamicPluginTensorDesc const *out, int32_t nbOutputs) TRT_NOEXCEPT override {}

    virtual size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs,
                                    const nvinfer1::PluginTensorDesc *outputs,
                                    int32_t nbOutputs) const TRT_NOEXCEPT override {
        return 0;
    }

    virtual void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                                 nvinfer1::IGpuAllocator *gpuAllocator) TRT_NOEXCEPT override {}

    virtual void detachFromContext() TRT_NOEXCEPT override {}

protected:
    const std::string mLayerName;
    std::string mNamespace;

#if NV_TENSORRT_MAJOR < 8
protected:
    // To prevent compiler warnings.
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
#endif
};

class TRTPluginCreatorBase : public nvinfer1::IPluginCreator {
public:
    const char *getPluginVersion() const TRT_NOEXCEPT override { return "1"; };

    const nvinfer1::PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return &mFC; }

    void setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override {
        mNamespace = pluginNamespace;
    }

    const char *getPluginNamespace() const TRT_NOEXCEPT override { return mNamespace.c_str(); }

protected:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

inline unsigned int getElementSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kFP8:
            return 1;
        default:
            throw std::runtime_error("Invalid DataType");
    }
}

inline size_t getAlignedSize(size_t origin_size, size_t aligned_number = 16) {
    return size_t((origin_size + aligned_number - 1) / aligned_number) * aligned_number;
}

#ifndef DEBUG

#define PLUGIN_CHECK(status)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
            abort();                                                                                                   \
    } while (0)

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_BAD_PARAM;                                                                                   \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_FAILURE;                                                                                     \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)

#else

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__);                                      \
            return STATUS_BAD_PARAM;                                                                                   \
        }                                                                                                              \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__);                                        \
            return STATUS_FAILURE;                                                                                     \
        }                                                                                                              \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));                        \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define PLUGIN_CHECK(status)                                                                                           \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            DEBUG_PRINTF("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(status));                      \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        printf(__VA_ARGS__);                                                                                           \
    } while (0)

#endif // DEBUG

} // namespace nvinfer1
#endif // TRT_BASE_PLUGIN_H

