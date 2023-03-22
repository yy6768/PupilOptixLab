#include "static.h"

#include "path_tracer/type.h"

#include "gui/dx12_backend.h"
#include "gui/window.h"
#include "imgui.h"

#include "device/optix_device.h"
#include "device/dx12_device.h"
#include "device/cuda_texture.h"
#include "device/optix_wrap/module.h"
#include "device/optix_wrap/pipeline.h"
#include "device/optix_wrap/pass.h"

#include "scene/scene.h"
#include "scene/texture.h"
#include "material/optix_material.h"

#include "optix_util/emitter.h"
#include "optix_util/camera.h"

#include <memory>
#include <iostream>
#include <fstream>

std::unique_ptr<device::Optix> g_optix_device;

std::unique_ptr<optix_wrap::Module> g_sphere_module;
std::unique_ptr<optix_wrap::Module> g_pt_module;

static char s_scene_name[256];
std::unique_ptr<scene::Scene> g_scene;

OptixLaunchParams g_params;

optix_util::CameraDesc g_camera_init_desc;
std::unique_ptr<optix_util::CameraHelper> g_camera;

std::unique_ptr<optix_util::EmitterHelper> g_emitters;

int g_max_samples = 32;
bool g_render_flag = true;

struct SBTTypes {
    using RayGenDataType = RayGenData;
    using MissDataType = MissData;
    using HitGroupDataType = HitGroupData;
};

std::unique_ptr<optix_wrap::Pass<SBTTypes, OptixLaunchParams>> g_pt_pass;

void ConfigOptixPipeline() noexcept;
void LoadScene(std::string_view, std::string_view default_scene = "default.xml") noexcept;
void InitGuiAndEventCallback() noexcept;

int main() {
    auto gui_window = util::Singleton<gui::Window>::instance();
    gui_window->Init();

    InitGuiAndEventCallback();

    auto backend = gui_window->GetBackend();

    g_optix_device = std::make_unique<device::Optix>(backend->GetDevice());
    g_pt_pass = std::make_unique<optix_wrap::Pass<SBTTypes, OptixLaunchParams>>(g_optix_device->context, g_optix_device->cuda_stream);

    ConfigOptixPipeline();
    std::string scene_name;
    std::ifstream scene_config_file(std::string{ ROOT_DIR } + "/pt_config.ini", std::ios::in);
    if (scene_config_file.is_open()) {
        std::getline(scene_config_file, scene_name);
        scene_config_file.close();
    }
    LoadScene(scene_name, "default.xml");

    do {
        if (g_render_flag) {
            if ((int)g_params.sample_cnt >= g_max_samples) {
                g_render_flag = false;
            } else {

                g_params.camera.SetData(g_camera->GetCudaMemory());
                g_params.frame_buffer = reinterpret_cast<float4 *>(backend->GetCurrentFrameResource().src->cuda_buffer_ptr);
                g_pt_pass->Run(g_params, g_params.config.frame.width, g_params.config.frame.height);

                if (g_params.config.accumulated_flag)
                    ++g_params.sample_cnt;
                else
                    g_params.sample_cnt = 1;

                ++g_params.frame_cnt;
            }
        }

        auto msg = gui_window->Show();
        if (msg == gui::GlobalMessage::Quit)
            break;
        else if (msg == gui::GlobalMessage::Minimized)
            g_render_flag = false;

    } while (true);

    g_emitters.reset();
    g_camera.reset();
    g_pt_module.reset();
    g_sphere_module.reset();
    g_optix_device.reset();
    gui_window->Destroy();
    return 0;
}

void ConfigOptixPipeline() noexcept {
    g_sphere_module = std::make_unique<optix_wrap::Module>(g_optix_device->context, OPTIX_PRIMITIVE_TYPE_SPHERE);
    g_pt_module = std::make_unique<optix_wrap::Module>(g_optix_device->context, "path_tracer/main.ptx");
    optix_wrap::PipelineDesc pipeline_desc;
    {
        optix_wrap::ProgramDesc desc{
            .module = g_pt_module.get(),
            .ray_gen_entry = "__raygen__main",
            .hit_miss = "__miss__default",
            .shadow_miss = "__miss__shadow",
            .hit_group = { .ch_entry = "__closesthit__default" },
            .shadow_grop = { .ch_entry = "__closesthit__shadow" }
        };
        pipeline_desc.programs.push_back(desc);
    }

    {
        optix_wrap::ProgramDesc desc{
            .module = g_pt_module.get(),
            .hit_group = { .ch_entry = "__closesthit__default", .intersect_module = g_sphere_module.get() },
            .shadow_grop = { .ch_entry = "__closesthit__shadow", .intersect_module = g_sphere_module.get() }
        };
        pipeline_desc.programs.push_back(desc);
    }
    g_pt_pass->InitPipeline(pipeline_desc);
}

void ConfigSBT() {
    optix_wrap::SBTDesc<SBTTypes> desc{};
    desc.ray_gen_data = {
        .program_name = "__raygen__main",
        .data = SBTTypes::RayGenDataType{}
    };
    {
        int emitter_index_offset = 0;
        using HitGroupDataRecord = decltype(desc)::Pair<SBTTypes::HitGroupDataType>;
        for (auto &&shape : g_scene->shapes) {
            HitGroupDataRecord hit_default_data{};
            hit_default_data.program_name = "__closesthit__default";
            hit_default_data.data.mat.LoadMaterial(shape.mat);
            hit_default_data.data.geo.LoadGeometry(shape);
            if (shape.is_emitter) {
                hit_default_data.data.emitter_index_offset = emitter_index_offset;
                emitter_index_offset += shape.sub_emitters_num;
            }

            desc.hit_datas.push_back(hit_default_data);

            HitGroupDataRecord hit_shadow_data{};
            hit_shadow_data.program_name = "__closesthit__shadow";
            hit_shadow_data.data.mat.type = shape.mat.type;
            desc.hit_datas.push_back(hit_shadow_data);
        }
    }
    {
        decltype(desc)::Pair<SBTTypes::MissDataType> miss_data = {
            .program_name = "__miss__default",
            .data = SBTTypes::MissDataType{}
        };
        desc.miss_datas.push_back(miss_data);
        decltype(desc)::Pair<SBTTypes::MissDataType> miss_shadow_data = {
            .program_name = "__miss__shadow",
            .data = SBTTypes::MissDataType{}
        };
        desc.miss_datas.push_back(miss_shadow_data);
    }
    g_pt_pass->InitSBT(desc);
}

void InitLaunchParams() {
    g_params.config.frame.width = g_scene->sensor.film.w;
    g_params.config.frame.height = g_scene->sensor.film.h;
    g_params.config.max_depth = g_scene->integrator.max_depth;
    g_params.config.accumulated_flag = true;
    g_params.config.use_tone_mapping = false;

    g_params.frame_cnt = 0;
    g_params.sample_cnt = 0;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&g_params.accum_buffer),
        g_params.config.frame.height * g_params.config.frame.width * sizeof(float4)));

    g_params.frame_buffer = nullptr;
    g_params.handle = g_optix_device->ias_handle;
}

void LoadScene(std::string_view scene_file, std::string_view default_scene) noexcept {
    std::filesystem::path scene_file_path{ DATA_DIR };
    scene_file_path /= scene_file;
    if (!std::filesystem::exists(scene_file_path)) {
        std::cout << std::format("warning: scene file [{}] does not exist.\n", scene_file_path.string());
        if (default_scene.empty()) return;
        scene_file = "default.xml";
        scene_file_path = scene_file_path.parent_path() / "default.xml";
    }
    memcpy(s_scene_name, scene_file.data(), scene_file.size() * sizeof(char));

    if (g_scene == nullptr)
        g_scene = std::make_unique<scene::Scene>();

    util::Singleton<device::CudaTextureManager>::instance()->Clear();

    g_scene->LoadFromXML(scene_file_path);
    g_optix_device->InitScene(g_scene.get());

    if (!g_emitters) {
        g_emitters = std::make_unique<optix_util::EmitterHelper>(g_scene.get());
    } else {
        g_emitters->Reset(g_scene.get());
    }
    g_params.emitters = g_emitters->GetEmitterGroup();

    auto &&sensor = g_scene->sensor;
    optix_util::CameraDesc desc{
        .fov_y = sensor.fov,
        .aspect_ratio = static_cast<float>(sensor.film.w) / sensor.film.h,
        .near_clip = sensor.near_clip,
        .far_clip = sensor.far_clip,
        .to_world = sensor.transform
    };
    g_camera_init_desc = desc;
    g_camera = std::make_unique<optix_util::CameraHelper>(desc);

    ConfigSBT();
    InitLaunchParams();

    util::Singleton<scene::ShapeDataManager>::instance()->Clear();
    util::Singleton<scene::TextureManager>::instance()->Clear();

    auto gui_window = util::Singleton<gui::Window>::instance();
    gui_window->Resize(g_scene->sensor.film.w, g_scene->sensor.film.h, true);
    g_optix_device->ClearSharedFrameResource();
    gui_window->GetBackend()->SetScreenResource(g_optix_device->GetSharedFrameResource());
}

void InitGuiAndEventCallback() noexcept {
    auto gui_window = util::Singleton<gui::Window>::instance();

    gui_window->AppendGuiConsoleOperations(
        "Path Tracer Option",
        []() {
            ImGui::SeparatorText("scene");
            {
                ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.5f);
                ImGui::InputText("scene file", s_scene_name, 256);
                ImGui::SameLine();
                if (ImGui::Button("Load")) {
                    LoadScene(s_scene_name, "");
                    g_params.frame_cnt = 0;
                    g_params.sample_cnt = 0;
                    g_render_flag = true;
                }
                ImGui::PopItemWidth();

                if (ImGui::Button("Reset Camera")) {
                    g_camera->Reset(g_camera_init_desc);
                    g_params.frame_cnt = 0;
                    g_params.sample_cnt = 0;
                }
            }

            ImGui::SeparatorText("render options");
            ImGui::InputInt("max samples", &g_max_samples, 1, 32);
            if (g_max_samples <= 0) g_max_samples = 1;

            ImGui::Text("sample count: %d", g_params.sample_cnt);
            ImGui::SameLine();
            if (ImGui::Button(g_render_flag ? "Stop" : "Continue")) {
                g_render_flag ^= 1;
                if (g_render_flag == false) {
                    util::Singleton<gui::Backend>::instance()->SynchronizeFrameResource();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset")) {
                g_params.sample_cnt = 0;
            }

            int depth = g_params.config.max_depth;
            ImGui::InputInt("trace depth", &depth);
            depth = clamp(depth, 1, 128);
            if (g_params.config.max_depth != depth) {
                g_params.config.max_depth = (unsigned int)depth;
                g_params.frame_cnt = 0;
                g_params.sample_cnt = 0;
            }

            if (ImGui::Checkbox("accumulate radiance", &g_params.config.accumulated_flag)) {
                g_params.sample_cnt = 0;
            }
            ImGui::Checkbox("ACES tone mapping", &g_params.config.use_tone_mapping);
        });

    gui_window->SetWindowMessageCallback(
        gui::GlobalMessage::Resize,
        [&]() {
            g_params.frame_cnt = 0;
            g_params.sample_cnt = 0;
            g_render_flag = true;

            unsigned int &w = g_params.config.frame.width;
            unsigned int &h = g_params.config.frame.height;
            gui_window->GetWindowSize(w, h);

            CUDA_FREE(g_params.accum_buffer);

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_params.accum_buffer), w * h * sizeof(float4)));

            g_optix_device->ClearSharedFrameResource();
            gui_window->GetBackend()->SetScreenResource(g_optix_device->GetSharedFrameResource());

            float aspect = static_cast<float>(w) / h;
            g_camera->SetAspectRatio(aspect);
        });

    gui_window->SetWindowMessageCallback(
        gui::GlobalMessage::MouseLeftButtonMove,
        [&]() {
            float scale = util::Camera::sensitivity * util::Camera::sensitivity_scale;
            float dx = gui_window->GetMouseLastDeltaX() * scale;
            float dy = gui_window->GetMouseLastDeltaY() * scale;

            g_camera->Rotate(dx, dy);

            g_params.frame_cnt = 0;
            g_params.sample_cnt = 0;
            g_render_flag = true;
        });

    gui_window->SetWindowMessageCallback(
        gui::GlobalMessage::MouseWheel,
        [&]() {
            float fov_delta = 1.f / 120.f * gui_window->GetMouseWheelDelta();
            g_camera->SetFovDelta(fov_delta);

            g_params.frame_cnt = 0;
            g_params.sample_cnt = 0;
            g_render_flag = true;
        });

    gui_window->SetWindowMessageCallback(
        gui::GlobalMessage::KeyboardMove,
        [&]() {
            auto right = util::Camera::X;
            auto up = util::Camera::Y;
            auto forward = util::Camera::Z;

            util::Float3 translation{ 0.f };
            if (gui_window->IsKeyPressed('W') || gui_window->IsKeyPressed(VK_UP)) {
                translation += forward;
            }
            if (gui_window->IsKeyPressed('S') || gui_window->IsKeyPressed(VK_DOWN)) {
                translation -= forward;
            }

            if (gui_window->IsKeyPressed('A') || gui_window->IsKeyPressed(VK_LEFT)) {
                translation += right;
            }
            if (gui_window->IsKeyPressed('D') || gui_window->IsKeyPressed(VK_RIGHT)) {
                translation -= right;
            }

            if (gui_window->IsKeyPressed('Q')) {
                translation += up;
            }
            if (gui_window->IsKeyPressed('E')) {
                translation -= up;
            }

            g_camera->Move(translation * util::Camera::sensitivity * util::Camera::sensitivity_scale);

            g_params.frame_cnt = 0;
            g_params.sample_cnt = 0;
            g_render_flag = true;
        });
}
