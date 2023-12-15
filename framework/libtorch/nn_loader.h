#pragma once
#include <memory>
#include <torch/torch.h>
#include <torch/script.h>
#include "util/util.h"
#include "context.h"

namespace Pupil::libtorch {
class NetworkLoader : public util::Singleton<NetworkLoader> {
public:
    static std::unique_ptr<torch::jit::script::Module> load_denoise(const std::string &model_path){
        try {
            return std::make_unique<torch::jit::script::Module>(std::move(torch::jit::load(model_path, Context::instance()->GetDevice())));
        }
        catch (const c10::Error& e) {
            std::cerr << e.what() << std::endl;
            throw std::runtime_error(
                "Error when loading torchscript model from " + model_path);
        }
    }

};
}