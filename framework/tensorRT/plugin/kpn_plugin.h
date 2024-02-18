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

namespace Pupil::tensorRT {
class KPNPluginDynamic : public TRTPluginBase {
public:
    KPNPluginDynamic(const std::string name);

    KPNPluginDynamic() = delete;

    ~KPNPluginDynamic() TRT_NOEXCEPT override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                            int nbInputs, nvinfer1::IExprBuilder &exprBuilder)
        TRT_NOEXCEPT override;

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs,
                                   int nbOutputs) TRT_NOEXCEPT override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc *out,
                         int nbOutputs) TRT_NOEXCEPT override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                            const nvinfer1::PluginTensorDesc *outputs,
                            int nbOutputs) const TRT_NOEXCEPT override;
    int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                void *const *outputs, void *workspace, cudaStream_t stream) TRT_NOEXCEPT override;
    void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                         nvinfer1::IGpuAllocator *gpuAllocator) TRT_NOEXCEPT override;
    void detachFromContext() TRT_NOEXCEPT override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                         int nbInputs) const TRT_NOEXCEPT override;

    // IPluginV2 Methods
    const char *getPluginType() const TRT_NOEXCEPT override;
    const char *getPluginVersion() const TRT_NOEXCEPT override;
    int getNbOutputs() const TRT_NOEXCEPT override;
    size_t getSerializationSize() const TRT_NOEXCEPT override;
    void serialize(void *buffer) const TRT_NOEXCEPT override;

private:
    const std::string mLayerName;
    nvinfer1::Dims mStride;
    nvinfer1::Dims mPadding;
    nvinfer1::Dims mDilation;
    

    cublasHandle_t mCublasHandle;
};

class KPNPluginDynamicCreator : public TRTPluginCreatorBase {
public:
    KPNPluginDynamicCreator();

    const char *getPluginName() const TRT_NOEXCEPT override;

    const char *getPluginVersion() const TRT_NOEXCEPT override;

    nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
        TRT_NOEXCEPT override;

    nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                           size_t serialLength) TRT_NOEXCEPT override; 

};

}// namespace Pupil
#endif// !TRT_KPN_PLUGIN_H

