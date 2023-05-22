#pragma once

#include "predefine.h"
#include "cuda/texture.h"
#include "ggx.h"

namespace Pupil::optix::material {

struct RoughConductor {
    cuda::Texture alpha;
    cuda::Texture eta;
    cuda::Texture k;
    cuda::Texture specular_reflectance;

    struct Local {
        float alpha;
        float3 eta;
        float3 k;
        float3 specular_reflectance;

        CUDA_HOSTDEVICE void GetBsdf(BsdfSamplingRecord &record) const noexcept {
            float3 f = make_float3(0.f);
            if (record.wi.z > 0.f && record.wo.z > 0.f) {
                float3 wh = record.wi + record.wo;
                if (!optix::IsZero(wh)) {
                    wh = normalize(wh);
                    float3 fresnel_o = fresnel::ConductorReflectance(eta, k, dot(record.wo, wh));
                    f = specular_reflectance * ggx::D(wh, alpha) * fresnel_o * ggx::G(record.wi, record.wo, alpha) / (4.f * record.wi.z * record.wo.z);
                }
            }
            record.f = f;
        }

        CUDA_HOSTDEVICE void GetPdf(BsdfSamplingRecord &record) const noexcept {
            float pdf = 0.f;
            if (record.wi.z > 0.f && record.wo.z > 0.f) {
                float3 wh = record.wi + record.wo;
                if (!optix::IsZero(wh)) {
                    wh = normalize(wh);
                    pdf = ggx::Pdf(record.wo, wh, alpha) / (4.f * dot(record.wo, wh));
                }
            }
            record.pdf = pdf;
        }

        CUDA_HOSTDEVICE void Sample(BsdfSamplingRecord &record) const noexcept {
            float2 xi = record.sampler->Next2();
            record.wi = optix::Reflect(record.wo, ggx::Sample(record.wo, alpha, xi));
            GetPdf(record);
            GetBsdf(record);
            record.sampled_type = EBsdfLobeType::DiffuseReflection;
        }
    };

    CUDA_HOSTDEVICE Local GetLocal(float2 sampled_tex) const noexcept {
        Local local_bsdf;
        local_bsdf.alpha = alpha.Sample(sampled_tex).x;
        local_bsdf.eta = eta.Sample(sampled_tex);
        local_bsdf.k = k.Sample(sampled_tex);
        local_bsdf.specular_reflectance = specular_reflectance.Sample(sampled_tex);
        return local_bsdf;
    }
};

}// namespace Pupil::optix::material