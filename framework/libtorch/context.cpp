#include "context.h"

#include "cuda/context.h"

#include "util/log.h"

#include <assert.h>

namespace Pupil::libtorch {
void Context::Init() noexcept {
    if (IsInitialized()) {
        Pupil::Log::Warn("Torch is initialized repeatedly.");
        return;
    }

    auto cuda_ctx = util::Singleton<Pupil::cuda::Context>::instance();
    assert(cuda_ctx->IsInitialized() && "cuda should be initialized before libtorch.");

	if (torch::cuda::is_available()) {
        m_device = torch::Device(torch::kCUDA, cuda_ctx->cuda_device_id);
    } else {
        Pupil::Log::Warn("Torch cuda is not available");
        m_device = torch::Device(torch::kCPU);
    }

    m_init_flag = true;
}

void Context::Destroy() noexcept {
    if (IsInitialized()) {
        m_init_flag = false;
    }
}
} // namespace Pupil::libtorch