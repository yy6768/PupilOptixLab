#include "kpn_plugin.h"
#include "trt_serialize.h"

using namespace nvinfer1;

namespace {
static const char *PLUGIN_VERSION{ "1" };
static const char *PLUGIN_NAME{ "KPN" };
} // namespace

namespace Pupil::tensorRT {

KPNPluginDynamic::KPNPluginDynamic(
    const std::string &name,
    const nvinfer1::Dims stride,
    const nvinfer1::Dims padding,
    const nvinfer1::Dims dilation) 
    : TRTPluginBase(name), mStride(stride), 
    mPadding(padding), mDilation(dilation) {}

KPNPluginDynamic::KPNPluginDynamic(const std::string &name, void const* data, size_t length) : TRTPluginBase(name) {
    deserialize(static_cast<uint8_t const*>(data), length);
}

KPNPluginDynamic::~KPNPluginDynamic() {}

nvinfer1::IPluginV2DynamicExt *KPNPluginDynamic::clone() const TRT_NOEXCEPT {
    KPNPluginDynamic *plugin = new KPNPluginDynamic(mLayerName,
        mStride, 
        mPadding, 
        mDilation);
    plugin->setPluginNamespace(getPluginNamespace());

    return plugin;
}

nvinfer1::DimsExprs KPNPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {

    nvinfer1::DimsExprs ret;

    ret.nbDims = 4;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[0].d[1];
    ret.d[2] = inputs[0].d[2];
    ret.d[3] = inputs[0].d[3];

    return ret;
}

// TODO Int8?
bool KPNPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *ioDesc,
    int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
    if (pos == 0) {
        return ((ioDesc[0].type == nvinfer1::DataType::kFLOAT || ioDesc[0].type == nvinfer1::DataType::kHALF)
                // || ioDesc[0].type == nvinfer1::DataType::kInt8)
                && ioDesc[0].format == nvinfer1::TensorFormat::kLINEAR);
    } else {
        return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
    }
}

void KPNPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs,
                                       int nbInputs,
                                       const nvinfer1::DynamicPluginTensorDesc *outputs,
                                       int nbOutputs) TRT_NOEXCEPT {}

// TODO
size_t KPNPluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                          int nbInputs,
                                          const nvinfer1::PluginTensorDesc *outputs,
                                          int nbOutputs) const TRT_NOEXCEPT {
    return 0;
}
// TODO
int KPNPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                              const nvinfer1::PluginTensorDesc *outputDesc,
                              const void *const *inputs, void *const *outputs,
                              void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {

    return 0;
}

nvinfer1::DataType KPNPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *inputTypes,
    int nbInputs) const TRT_NOEXCEPT {
    return inputTypes[0];
}

// ******************************
// IPluginV2 methods
// ******************************
 
const char* KPNPluginDynamic::getPluginType() const TRT_NOEXCEPT {
    return PLUGIN_NAME;
}

const char* KPNPluginDynamic::getPluginVersion() const {
    return PLUGIN_VERSION;
}

int KPNPluginDynamic::getNbOutputs() const TRT_NOEXCEPT {
    return 1;
}

size_t KPNPluginDynamic::getSerializationSize() const TRTNOEXCEPT {
    return serialized_size(mStride) +
           serialized_size(mPadding) +
           serialized_size(mDilation);
}

void KPNPluginDynamic::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {
    mCublasHandle = cublasContext;
}

void KPNPluginDynamic::detachFromContext() TRT_NOEXCEPT {}

// **********************
// Creator methods
// **********************

KPNPluginDynamicCreator::KPNPluginDynamicCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("stride"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("padding"));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("dilation"));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *KPNPluginDynamicCreator::getPluginName() const TRT_NOEXCEPT {
    return PLUGIN_NAME;
}

const char *KPNPluginDynamicCreator::getPluginVersion() const TRT_NOEXCEPT {
    return PLUGIN_VERSION;
}

nvinfer1::IPluginV2* KPNPluginDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
    nvinfer1::Dims stride{ 2, { 1, 1 } };
    nvinfer1::Dims padding{ 2, { 0, 0 } };
    nvinfer1::Dims dilation{ 2, { 1, 1 } };

    for (int i = 0; i < fc->nbFields; i++) {
        if (fc->fields[i].data == nullptr) {
            continue;
        }
        std::string field_name(fc->fields[i].name);

        if (field_name.compare("stride") == 0) {
            stride.nbDims = 2;
            stride.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
            stride.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
        }

        if (field_name.compare("padding") == 0) {
            padding.nbDims = 2;
            padding.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
            padding.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
        }

        if (field_name.compare("dilation") == 0) {
            dilation.nbDims = 2;
            dilation.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
            dilation.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
        }
    }

    KPNPluginDynamic *plugin =
        new KPNPluginDynamic(name, stride, padding, dilation);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

nvinfer1::IPluginV2 *KPNPluginDynamicCreator::deserializePlugin(
    const char *name, void const *serialData, size_t serialLength) TRT_NOEXCEPT {
    auto plugin = new KPNPluginDynamic(name, serialData, serialLength);
    plugin->setPluginNamespace(getPluginNamespace());
    return plugin;
}

REGISTER_TENSORRT_PLUGIN(KPNPluginDynamicCreator);

} // namespace Pupil::tensorRT