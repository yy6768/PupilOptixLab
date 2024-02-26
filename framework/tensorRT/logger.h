#pragma once
#include <NvInfer.h>
#include "util/log.h"



namespace Pupil::tensorRT{

class Logger : public nvinfer1::ILogger {
public:
	virtual void log(Severity severity, const char *msg) noexcept override {

		if (severity == Severity::kINTERNAL_ERROR) {
            Pupil::Log::Error("NVInfer INTERNAL_ERROR:{}", msg);
			abort();
		} else if (severity == Severity::kERROR) {
            Pupil::Log::Error("NVInfer INTERNAL_ERROR:{}", msg);
		} else if (severity == Severity::kWARNING) {
            Pupil::Log::Warn("NVInfer WARNING:{}", msg);
		} else if (severity == Severity::kINFO) {
            Pupil::Log::Info("NVInfer INFO:{}", msg);
		} else {
            Pupil::Log::Info("NVInfer DEBUG:{}", msg);
		}
	}
};

static Logger gLogger;
}; // namespace Pupil::tensorRT


