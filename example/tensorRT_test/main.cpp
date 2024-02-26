#include <iostream>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>


#include "tensorRT/trt_denoiser.h"

#include "libtorch/data_loader.h"

#include "system/system.h"

#include "util/log.h"
#include "util/timer.h"

#include "test_custom_kpn.h"


void test_main() {
	// Load Model 
	// Get onnx path
	// data_path = "..\\PupilOptixLab\\data";
	std::filesystem::path root_path = std::filesystem::current_path()
		.parent_path().parent_path().parent_path();
	std::filesystem::path data_path = root_path / "data";
	std::filesystem::path onnx_path = data_path / "onnx" / "model116.onnx";
	Pupil::tensorRT::TRTDenoiser denoiser(onnx_path.string());
	// Load Dataset
	std::filesystem::path dataset_path = data_path / "test1";
	Pupil::libtorch::DenoiseDataset dataset(dataset_path.string());
	const auto &[data, target] = dataset.get(0);
	// Warm up
	// Inference
	const float* f_data = data.contiguous().const_data_ptr<float>();
	float *output = nullptr;
	for (int i = 0; i < 1; i++) {
		output = denoiser(f_data);
	}
	torch::Tensor t_output = torch::from_blob(output, { 3, data.size(2), data.size(3) });
	t_output = torch::cat({ t_output, torch::ones({ 1, t_output.size(1), t_output.size(2) }) }, 0);
	auto output_ = t_output.permute({ 1, 2, 0 }).flip({0}).contiguous();
	float *f_output = output_.data_ptr<float>();
	std::filesystem::path output_path = data_path / "output" / "output_trt.exr";
	Pupil::util::BitmapTexture::Save(f_output, output_.size(1), output_.size(0),
									 output_path.string(), Pupil::util::BitmapTexture::FileFormat::EXR);
	target.squeeze_();
	auto target_ = target.permute({ 1, 2, 0 }).flip({0}).contiguous();
	float *f_target = target_.data_ptr<float>();
	Pupil::Log::Info("{}, {}", target_.size(0), target_.size(1));
	std::filesystem::path target_path = data_path / "output" / "target_trt.exr";
	Pupil::util::BitmapTexture::Save(f_target, target_.size(1), target_.size(0),
									 target_path.string(), Pupil::util::BitmapTexture::FileFormat::EXR);
	Pupil::Log::Info("Succeed!");

}


int main() {
    auto system = Pupil::util::Singleton<Pupil::System>::instance();
    system->Init(true);
    {   
		switch (2) {
			// Main denoise network
            case 0: test_main();	   break;
			// Try to import pytorch custom kernel prediction （fail）
            case 1: test_import_custom_kpn(); break;
			// Test plugin Kpn
            case 2: test_kpn_plugin(); break;
		}
	}
    system->Destroy();
    return 0;
} 