//#include "nn_loader.h"
//
//namespace Pupil::libtorch {
//std::unique_ptr<torch::jit::Module> NetworkLoader::load(const std::string& model_path) {
//    auto module = std::make_unique<torch::jit::Module>(torch::jit::load(model_path));
//    module->to(Context::instance()->GetDevice());
//    return module;
//}
//} // namespace Pupil::libtorch
