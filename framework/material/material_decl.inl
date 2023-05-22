// clang-format off
// #define PUPIL_MATERIAL_TYPE_DEFINE(type)
#ifdef PUPIL_MATERIAL_TYPE_DEFINE
PUPIL_MATERIAL_TYPE_DEFINE(Diffuse)
PUPIL_MATERIAL_TYPE_DEFINE(Conductor)
PUPIL_MATERIAL_TYPE_DEFINE(Dielectric)
PUPIL_MATERIAL_TYPE_DEFINE(RoughConductor)
PUPIL_MATERIAL_TYPE_DEFINE(Plastic)
PUPIL_MATERIAL_TYPE_DEFINE(RoughPlastic)
#endif

#ifdef PUPIL_MATERIAL_ATTR_DEFINE
PUPIL_MATERIAL_ATTR_DEFINE(diffuse)
PUPIL_MATERIAL_ATTR_DEFINE(conductor)
PUPIL_MATERIAL_ATTR_DEFINE(dielectric)
PUPIL_MATERIAL_ATTR_DEFINE(rough_conductor)
PUPIL_MATERIAL_ATTR_DEFINE(plastic)
PUPIL_MATERIAL_ATTR_DEFINE(rough_plastic)
#endif

#ifdef PUPIL_MATERIAL_NAME_DEFINE
PUPIL_MATERIAL_NAME_DEFINE(diffuse)
PUPIL_MATERIAL_NAME_DEFINE(conductor)
PUPIL_MATERIAL_NAME_DEFINE(dielectric)
PUPIL_MATERIAL_NAME_DEFINE(roughconductor)
PUPIL_MATERIAL_NAME_DEFINE(plastic)
PUPIL_MATERIAL_NAME_DEFINE(roughplastic)
#endif

#ifdef PUPIL_MATERIAL_TYPE_ATTR_DEFINE
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(Diffuse,         diffuse)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(Conductor,       conductor)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(Dielectric,      dielectric)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(RoughConductor,  rough_conductor)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(Plastic,         plastic)
PUPIL_MATERIAL_TYPE_ATTR_DEFINE(RoughPlastic,    rough_plastic)
#endif
// clang-format on