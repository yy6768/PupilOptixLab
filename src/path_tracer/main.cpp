#include "static.h"

#include "path_tracer/type.h"

#include "gui/dx12_backend.h"
#include "gui/window.h"
#include "imgui.h"

#include "device/optix_device.h"
#include "device/dx12_device.h"
#include "device/optix_wrap/module.h"
#include "device/optix_wrap/pipeline.h"
#include "device/optix_wrap/pass.h"

#include "scene/scene.h"
#include "material/optix_material.h"

#include "optix_util/emitter.h"

#include <memory>
#include <iostream>

std::unique_ptr<optix_wrap::Module> g_sphere_module;
std::unique_ptr<optix_wrap::Module> g_pt_module;

std::unique_ptr<scene::Scene> g_scene;

OptixLaunchParams g_params;

CUdeviceptr g_camera_cuda_memory = 0;
optix_util::Camera g_camera;

CUdeviceptr g_emitters_cuda_memory = 0;
std::vector<optix_util::Emitter> g_emitters;

struct SBTTypes {
    using RayGenDataType = RayGenData;
    using MissDataType = MissData;
    using HitGroupDataType = HitGroupData;
};

std::unique_ptr<optix_wrap::Pass<SBTTypes, OptixLaunchParams>> g_pt_pass;

void ConfigOptix(device::Optix *device);
void InitGuiEventCallback();

int main() {
    auto gui_window = util::Singleton<gui::Window>::instance();
    gui_window->Init();

    bool exit_flag = true;
    gui_window->SetWindowMessageCallback(
        gui::GlobalMessage::Quit,
        [&exit_flag]() { exit_flag = false; });

    gui_window->AppendGuiConsoleOperations([]() {
        ImGui::Text("test Text.");
    });

    auto backend = gui_window->GetBackend();
    std::unique_ptr<device::Optix> optix_device = std::make_unique<device::Optix>(backend->GetDevice());

    g_pt_pass = std::make_unique<optix_wrap::Pass<SBTTypes, OptixLaunchParams>>(optix_device->context, optix_device->cuda_stream);

    gui_window->SetWindowMessageCallback(
        gui::GlobalMessage::Resize,
        [&]() {
            g_params.frame_cnt = 0;

            unsigned int &w = g_params.config.frame.width;
            unsigned int &h = g_params.config.frame.height;
            gui_window->GetWindowSize(w, h);

            CUDA_FREE(g_params.accum_buffer);

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_params.accum_buffer), w * h * sizeof(float4)));

            optix_device->ClearSharedFrameResource();
            backend->SetScreenResource(optix_device->GetSharedFrameResource());

            float aspect = static_cast<float>(w) / h;
            g_camera.SetCameraTransform(g_scene->sensor.fov, aspect);
            cuda::CudaMemcpy(g_camera_cuda_memory, &g_camera, sizeof(g_camera));
        });

    ConfigOptix(optix_device.get());
    gui_window->Resize(g_scene->sensor.film.w, g_scene->sensor.film.h, true);

    backend->SetScreenResource(optix_device->GetSharedFrameResource());

    do {
        // TODO: handle minimize event
        g_params.frame_buffer = reinterpret_cast<float4 *>(backend->GetCurrentFrameResource().src->cuda_buffer_ptr);
        g_pt_pass->Run(g_params, g_params.config.frame.width, g_params.config.frame.height);

        gui_window->Show();

        ++g_params.frame_cnt;
    } while (exit_flag);

    CUDA_FREE(g_camera_cuda_memory);
    CUDA_FREE(g_emitters_cuda_memory);

    g_pt_module.reset();
    g_sphere_module.reset();
    gui_window->Destroy();
    return 0;
}

void ConfigPipeline(device::Optix *device) {
    g_sphere_module = std::make_unique<optix_wrap::Module>(device->context, OPTIX_PRIMITIVE_TYPE_SPHERE);
    g_pt_module = std::make_unique<optix_wrap::Module>(device->context, "path_tracer/main.ptx");
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

void ConfigScene(device::Optix *device) {
    g_scene = std::make_unique<scene::Scene>();
    std::string scene_name = "staircase/scene_v3.xml";
    // scene_name = "veach-ajar/scene_v3.xml";
    // scene_name = "veach-mis/scene_v3.xml";
    scene_name = "cornell-box/scene_v3.xml";
    g_scene->LoadFromXML(scene_name, DATA_DIR);
    device->InitScene(g_scene.get());

    g_emitters = optix_util::GenerateEmitters(g_scene.get());
    g_emitters_cuda_memory = cuda::CudaMemcpy(g_emitters.data(), g_emitters.size() * sizeof(optix_util::Emitter));
    g_params.emitters.SetData(g_emitters_cuda_memory, g_emitters.size());
}

void ConfigSBT(device::Optix *device) {
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

void InitLaunchParams(device::Optix *device) {
    g_params.config.frame.width = g_scene->sensor.film.w;
    g_params.config.frame.height = g_scene->sensor.film.h;
    g_params.config.max_depth = g_scene->integrator.max_depth;

    g_params.frame_cnt = 0;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&g_params.accum_buffer),
        g_params.config.frame.height * g_params.config.frame.width * sizeof(float4)));

    g_params.frame_buffer = nullptr;
    g_params.handle = device->ias_handle;

    float aspect = static_cast<float>(g_scene->sensor.film.w) / g_scene->sensor.film.h;
    g_camera.SetCameraTransform(g_scene->sensor.fov, aspect);
    g_camera.SetWorldTransform(g_scene->sensor.transform.matrix);

    g_camera_cuda_memory = cuda::CudaMemcpy(&g_camera, sizeof(g_camera));
    g_params.camera.SetData(g_camera_cuda_memory);
}

void ConfigOptix(device::Optix *device) {
    ConfigPipeline(device);
    ConfigScene(device);
    ConfigSBT(device);
    InitLaunchParams(device);
}

void InitGuiEventCallback() {
    auto gui_window = util::Singleton<gui::Window>::instance();
    gui_window->SetWindowMessageCallback(
        gui::GlobalMessage::MouseLeftButtonMove,
        [&camera = g_params.camera, &gui_window]() {
            float dx = 0.25f * gui_window->GetMouseLastDeltaX();
            float dy = 0.25f * gui_window->GetMouseLastDeltaY();
            //camera
        });
}
