#pragma once

#include <NvInfer.h>

#include "util/util.h"

namespace Pupil::tensorRT {
class Context : public util::Singleton<Context>  {
public:

private:
    nvinfer1::IExecutionContext *context;
};

}
