#pragma once

#include "cuda_util/preprocessor.h"
#include "common/camera.h"

#include <vector_types.h>
#include <cuda.h>

namespace optix_util {
struct Camera {
    struct {
        float4 r0;
        float4 r1;
        float4 r2;
        float4 r3;
    } sample_to_camera, camera_to_world;
};

#ifdef PUPIL_OPTIX_LAUNCHER_SIDE
struct CameraDesc {
    float fov_y;
    float aspect_ratio;
    float near_clip = 0.01f;
    float far_clip = 10000.f;

    util::Transform to_world;
};

class CameraHelper {
private:
    bool m_dirty;
    CameraDesc m_desc;
    util::Camera m_camera;
    optix_util::Camera m_optix_camera;

    CUdeviceptr m_camera_cuda_memory = 0;

public:
    CameraHelper(const CameraDesc &desc) noexcept;
    ~CameraHelper() noexcept;

    void Reset(const CameraDesc &desc) noexcept;

    void SetFov(float fov) noexcept;
    void SetFovDelta(float fov_delta) noexcept;
    void SetAspectRatio(float aspect_ration) noexcept;
    void SetNearClip(float near_clip) noexcept;
    void SetFarClip(float far_clip) noexcept;
    void SetWorldTransform(util::Transform to_world) noexcept;

    void Rotate(float delta_x, float delta_y) noexcept;
    void Move(util::Float3 translation) noexcept;

    CUdeviceptr GetCudaMemory() noexcept;

    std::tuple<util::Float3, util::Float3, util::Float3> GetCameraCoordinateSystem() const noexcept;
};
#endif
}// namespace optix_util