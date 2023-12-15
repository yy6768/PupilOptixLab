#include <memory>

#include "system/pass.h"
#include "util/timer.h"

#include "torch/torch.h"

class TorchPass : public Pupil::Pass {
public:
    TorchPass(std::string_view name = "Torch") noexcept;
    virtual void OnRun() noexcept override;

private:
    std::unique_ptr<torch::jit::Module> model;
};