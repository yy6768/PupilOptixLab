#pragma once
#ifndef TRT_KPN_PLUGIN_H
#define TRT_KPN_PLUGIN_H
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <vector>
#include <iostream>

#include "base_plugin.h"

namespace nvinfer1 {
class KPNPluginDynamic : public TRTPluginBase {
public:
    KPNPluginDynamic(const std::string &name, const nvinfer1::Dims dilation);

    KPNPluginDynamic(const std::string &name, void const *data, size_t length);

    KPNPluginDynamic() = delete;

    ~KPNPluginDynamic() TRT_NOEXCEPT override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, 
        DimsExprs const* inputs, 
        int32_t nbInputs, 
        IExprBuilder& exprBuilder)
        TRT_NOEXCEPT override;

    bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *ioDesc, int32_t nbInputs,
                                   int32_t nbOutputs) TRT_NOEXCEPT override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc *out,
                         int32_t nbOutputs) TRT_NOEXCEPT override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs,
                            const nvinfer1::PluginTensorDesc *outputs,
                            int32_t nbOutputs) const TRT_NOEXCEPT override;

    int32_t enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override;

    void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                         nvinfer1::IGpuAllocator *gpuAllocator) TRT_NOEXCEPT override;
    void detachFromContext() TRT_NOEXCEPT override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType *inputTypes,
                                         int32_t nbInputs) const TRT_NOEXCEPT override;

    // IPluginV2 Methods
    const char *getPluginType() const TRT_NOEXCEPT override;
    const char *getPluginVersion() const TRT_NOEXCEPT override;
    int getNbOutputs() const TRT_NOEXCEPT override;
    size_t getSerializationSize() const TRT_NOEXCEPT override;
    void serialize(void *buffer) const TRT_NOEXCEPT override;

private:
    const std::string mLayerName;
    nvinfer1::Dims mDilation;
    
    void deserialize(uint8_t const *data, size_t length) TRTNOEXCEPT;
    cublasHandle_t mCublasHandle;
};

class KPNPluginDynamicCreator : public TRTPluginCreatorBase {
public:
    KPNPluginDynamicCreator();

    const char *getPluginName() const TRT_NOEXCEPT override;

    const char *getPluginVersion() const TRT_NOEXCEPT override;

    nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
        TRT_NOEXCEPT override;

    nvinfer1::IPluginV2 *deserializePlugin(const char *name, void const * serialData,
                                           size_t serialLength) TRT_NOEXCEPT override; 

};

}// namespace nvinfer1
#endif// !TRT_KPN_PLUGIN_H

