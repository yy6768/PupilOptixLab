#include <iostream>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

#include "tensorRT/trt_denoiser.h"

#include "libtorch/data_loader.h"

#include "system/system.h"

#include "util/log.h"
#include "util/timer.h"

#include "static.h"

int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);

    { // 加载模型
        // 获得onnx路径
        std::filesystem::path data_path = "C:\\Users\\12587\\Desktop\\code\\graduation_design\\PupilOptixLab\\data";
        std::filesystem::path onnx_path = data_path / "onnx" / "model.onnx";
        Pupil::tensorRT::TRTDenoiser denoiser(onnx_path.string());
        // 加载数据集
        std::filesystem::path dataset_path = data_path / "test";
        Pupil::libtorch::DenoiseDataset dataset(dataset_path.string());

        const auto &[data, _] = dataset.get(0);
        const auto &[data1, _1] = dataset.get(1);
        float* f_data = data.data_ptr<float>();
        for (int i = 0; i < 49; i++) {
            float *output = denoiser(f_data);
        }
        float *f_data1 = data1.data_ptr<float>();
        float *output1 = denoiser(f_data1);

        
    }

    system->Destroy();
    return 0;
} 