#pragma once

#include "sceneStructs.h"

__host__ __device__ int wrapTexel(int i, int size, WrapMode mode);

__host__ __device__ glm::vec4 readTexel(
    int x, int y, const Texture& texture, const TextureImage& image,
    const uint32_t* texels, const glm::vec4* floatTexels, bool srgb);

__host__ __device__ glm::vec2 textureUV(
    const TextureRef& textureRef, const MeshPrimitive& primitive, int vertex, const glm::vec2* texcoords);

__host__ __device__ glm::vec4 sampleTexture(
    const TextureRef& textureRef, const MeshPrimitive& primitive, const Triangle& triangle, glm::vec3 weights,
    const glm::vec2* texcoords, const Texture* textures, const TextureImage* images,
    const uint32_t* texels, const glm::vec4* floatTexels, bool srgb = false);

__host__ __device__ void evaluateMaterial(
    Material& material, glm::vec3& normal, const ShadeableIntersection& hit,
    const Geom* geoms, const MeshPrimitive* primitives, const Vertex* vertices,
    const Triangle* triangles, const glm::vec2* texcoords, const Texture* textures,
    const TextureImage* images, const uint32_t* texels, const glm::vec4* floatTexels);
