#pragma once
#include "util/texture.h"
#include "util/log.h"

#include <torch/torch.h>
#include <iostream>
#include <filesystem>

namespace Pupil::libtorch {
class DenoiseDataset : public torch::data::Dataset<DenoiseDataset> {
public:
    DenoiseDataset(const std::string &dataset_dir) 
        : dataset_dir_(dataset_dir) {

    // 初始化数据集
    for (const auto &scene_name : std::filesystem::directory_iterator(dataset_dir_)) {
        const auto& scene_path = scene_name.path();
        std::unordered_map<std::string, std::string> exrname;
        for (const auto& filename : std::filesystem::directory_iterator(scene_name)) {
            exrname.emplace(filename.path().stem().string(), filename.path().string());
        }
        imgs_.emplace_back(exrname);
        Pupil::Log::Info("{}", scene_path.string());
    }

}

torch::data::Example<> get(size_t index) override {
    std::unordered_map<std::string, std::string> exrname = imgs_[index];
    util::BitmapTexture color, target, normal, depth, albedo;

    // 加载数据集
    color = util::BitmapTexture::Load(exrname["color"]);
    target = util::BitmapTexture::Load(exrname["target"]);
    normal = util::BitmapTexture::Load(exrname["normal"]);
    depth = util::BitmapTexture::Load(exrname["depth"]);
    albedo = util::BitmapTexture::Load(exrname["albedo"]);

    
    // 将数据转换为 Tensor
    torch::Tensor rgba_tensor = torch::from_blob(color.data, { static_cast<long long>(color.h), static_cast<long long>(color.w) , 4}, torch::kFloat);
    torch::Tensor color_tensor = rgba_tensor.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3) });
    torch::Tensor normal_tensor = torch::from_blob(normal.data, { static_cast<long long>(normal.h), static_cast<long long>(normal.w), 4 }, torch::kFloat);
    normal_tensor = normal_tensor.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3) });
    torch::Tensor depth_tensor = torch::from_blob(depth.data, { static_cast<long long>(depth.h), static_cast<long long>(depth.w), 4 }, torch::kFloat);
    depth_tensor = depth_tensor.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1) });
    torch::Tensor albedo_tensor = torch::from_blob(albedo.data, { static_cast<long long>(albedo.h), static_cast<long long>(albedo.w), 4 }, torch::kFloat);
    albedo_tensor = albedo_tensor.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3) });
    
 
    normal_tensor = normal_tensor * 0.5 + 0.5;
    depth_tensor = (depth_tensor - torch::min(depth_tensor)) / (torch::max(depth_tensor) - torch::min(depth_tensor)); 

    torch::Tensor data_complete = torch::cat({ color_tensor, albedo_tensor, normal_tensor, depth_tensor }, 2).permute({ 2, 0, 1 });
    torch::Tensor data_tensor = data_complete.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 512), torch::indexing::Slice(0, 512) })
                                             .unsqueeze(0);

    torch::Tensor target_complete = torch::from_blob(target.data, { static_cast<long long>(target.h), static_cast<long long>(target.w), 4 }, torch::kFloat)
                                    .permute({ 2, 0, 1 });
    torch::Tensor target_tensor = target_complete.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 512), torch::indexing::Slice(0, 512) })
                                    .unsqueeze(0);

    return { data_tensor, target_tensor };
}

torch::optional<size_t> size() const override {
    return imgs_.size();
}

private:
    std::string dataset_dir_;
    std::vector<std::unordered_map<std::string, std::string>> imgs_;
};

} // namespace Pupil::libtorch
