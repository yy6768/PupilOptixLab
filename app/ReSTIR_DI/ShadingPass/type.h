#pragma once

#include "optix/geometry.h"
#include "optix/scene/camera.h"
#include "material/optix_material.h"
#include "cuda/data_view.h"

#include "../reservoir.h"

struct ShadingPassLaunchParams {
    struct {
        unsigned int width;
        unsigned int height;
    } frame;

    unsigned int debug_type;
    OptixTraversableHandle handle;
    Pupil::cuda::ConstArrayView<float4> position;
    Pupil::cuda::ConstArrayView<float4> normal;
    Pupil::cuda::ConstArrayView<float4> albedo;

    Pupil::cuda::ConstArrayView<Reservoir> reservoirs;

    Pupil::cuda::RWArrayView<float4> frame_buffer;
};
