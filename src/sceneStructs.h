#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>
#include <cstdint>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum class GeomType
{
    SPHERE,
    CUBE,
    Mesh
};

enum class AlphaMode
{
    Opaque,
    Mask,
    Blend
};

enum class WrapMode
{
    Repeat,
    ClampToEdge,
    MirroredRepeat
};

enum class FilterMode
{
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    int primitiveId = -1;
};

struct Vertex
{
    glm::vec3 position{ 0.0f };
    glm::vec3 normal{ 0.0f };
    glm::vec4 tangent{ 0.0f };
    glm::vec4 color{ 1.0f };
};

struct Triangle
{
    uint32_t indices[3]{};
};

struct MeshPrimitive
{
    int vertexOffset = 0;
    int vertexCount = 0;

    int triangleOffset = 0;
    int triangleCount = 0;

    int texcoordOffset = 0;
    int texcoordSetCount = 0;

    bool hasNormals = false;
    bool hasTangents = false;
};

struct TextureRef
{
    int textureId = -1;
    int texCoord = 0;

    glm::mat3 uvTransform{ 1.0f };
};

struct TextureImage
{
    int width = 0;
    int height = 0;

    int texelOffset = 0;
};

struct Texture
{
    int imageId = -1;

    WrapMode wrapU = WrapMode::Repeat;
    WrapMode wrapV = WrapMode::Repeat;

    FilterMode minFilter = FilterMode::Linear;
    FilterMode magFilter = FilterMode::Linear;
};

struct Material
{
    glm::vec3 albedo = glm::vec3{ 1.0f };
    float alpha = 1.0f;

    float metalness = 1.0f;
    float roughness = 1.0f;

    glm::vec3 emittance = glm::vec3(0.0f);

    TextureRef baseColorTexture;
    TextureRef metallicRoughnessTexture;
    TextureRef normalTexture;
    TextureRef occlusionTexture;
    TextureRef emissiveTexture;
    TextureRef transmissionTexture;

    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;
    AlphaMode alphaMode = AlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    bool doubleSided = false;

    float transmission = 0.0f;
    float ior = 1.5f;
    bool thinWalled = false;
    glm::vec3 attenuationColor = glm::vec3(1.0f);
    float attenuationDistance = INFINITY;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 radiance;
    glm::vec3 throughput;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t;
    glm::vec3 surfaceNormal;
    int materialId;

    int geomId = -1;
    int triangleId = -1;
    glm::vec2 barycentrics{ 0.0f };
    glm::vec3 geometricNormal{ 0.0f };
};