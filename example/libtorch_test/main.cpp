#include <iostream>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#include "libtorch/nn_loader.h"
#include "libtorch/data_loader.h"

#include "system/system.h"

#include "util/log.h"
#include "util/timer.h"

#include "static.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);
    { // 加载网络， 渲染
        // 获取当前可执行文件所在目录
        std::filesystem::path executable_path = std::filesystem::current_path();
        // 构建相对路径，假设模型文件在项目的 output 目录下
        std::filesystem::path model_path = executable_path / "model.pth";
        auto model(std::move(Pupil::libtorch::NetworkLoader::load_denoise(model_path.string())));
         
        // 获得数据集
        std::filesystem::path data_path = "C:\\Users\\12587\\Desktop\\code\\graduation_design\\PupilOptixLab\\data";
        std::filesystem::path dataset_path = data_path / "test";
        Pupil::libtorch::DenoiseDataset dataset(dataset_path.string());

        Pupil::Timer timer;
        const auto &[data, target] = dataset.get(0);
        const auto &[data1, target1] = dataset.get(1);
        const auto &[data2, target2] = dataset.get(2);

        
        torch::jit::Stack stack;
        torch::jit::Stack stack1;
        torch::jit::Stack stack2;
        torch::NoGradGuard no_grad;
        //从4到16
        at::set_num_threads(64);
        //从4到16
        at::set_num_interop_threads(64);
        std::cout << "userEnabledCuDNN: " << at::globalContext().userEnabledCuDNN() << std::endl;
        std::cout << "userEnabledMkldnn: " << at::globalContext().userEnabledMkldnn() << std::endl;
        std::cout << "benchmarkCuDNN: " << at::globalContext().benchmarkCuDNN() << std::endl;
        std::cout << "deterministicCuDNN: " << at::globalContext().deterministicCuDNN() << std::endl; 

        at::globalContext().setUserEnabledCuDNN(true);
        at::globalContext().setUserEnabledMkldnn(true);
        at::globalContext().setBenchmarkCuDNN(true);
        //at::globalContext().setDeterministicCuDNN(true);        
        stack.emplace_back(data.to(torch::kCUDA));
        stack1.emplace_back(data1.to(torch::kCUDA));
        stack2.emplace_back(data2.to(torch::kCUDA));
        std::filesystem::path prof_path = data_path / "output" / "model.pth.trace.json";
        torch::autograd::profiler::RecordProfile prof_guard(prof_path.string());
       
        try {
            at::Tensor output;
            for (int i = 0; i < 49; i++) 
                output = model->forward(stack).toTensor();
            at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

            AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            timer.Start();
            output = model->forward(stack1).toTensor();
            AT_CUDA_CHECK(cudaStreamSynchronize(stream));
            timer.Stop();
            Pupil::Log::Info("Forward time is {}s", timer.ElapsedSeconds());
            output = output.to(torch::kCPU);
            output.squeeze_();
            auto output_alpha = torch::cat({ output, torch::ones({ 1, output.size(1), output.size(2) }) }, 0);

            auto output_ = output_alpha.permute({1, 2, 0}).contiguous();
            
            float *f_output = output_.data_ptr<float>();
            std::filesystem::path output_path = data_path / "output" / "output.exr";
            
            
            // **************
            std::filesystem::path target_path = data_path / "output" / "target.exr";
            
            target.squeeze_();
            auto target_size = target.sizes();
            auto target_ = target.permute({ 1, 2, 0 }).contiguous();
            float *f_target = target_.data_ptr<float>();
            Pupil::util::BitmapTexture::Save(f_target, target_size.at(1), target_size.at(2),
                                             target_path.string(), Pupil::util::BitmapTexture::FileFormat::EXR);
            // **************
           /* std::vector<int64_t> sizes = output_.sizes().vec();
            for (int i = 0; i < 128; ++i) {
                for (int j = 0; j < 128; ++j) {
                    int index = i * 128 + j;
                    for (int k = 0; k < 4; ++k) {
                        std::cout << f_output[index * 4 + k] << " "; 
                    }
                    std::cout << std::endl;
                }
                std::cout << "**************************" << std::endl;
            }*/
            // **************
            Pupil::util::BitmapTexture::Save(f_output, output_.size(0), output_.size(1), 
                                             output_path.string(), Pupil::util::BitmapTexture::FileFormat::EXR);
            
            Pupil::Log::Info("Succeed!");
        } catch (std::runtime_error &e) {
            Pupil::Log::Error("Forward error!,{}", e.what());
            exit(-1);
        }
        catch (const c10::Error& e) {
            Pupil::Log::Error("Forward error!,{}", e.what());
            exit(-1);
        }
        
    }
    system->Destroy();
    return 0;
}