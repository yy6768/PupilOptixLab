#pragma once
#include <torch/torch.h>

#include "util/util.h"

namespace Pupil::libtorch {
class Context: public Pupil::util::Singleton<Context> {
public:
    
    void Init() noexcept;
    void Destroy() noexcept;

    [[nodiscard]] bool IsInitialized() noexcept { return m_init_flag; }
    
    inline torch::Device GetDevice() { return m_device; }

private:
    torch::Device m_device{torch::kCPU};
    bool m_init_flag = false;
};

} // namespace Pupil::libtorch