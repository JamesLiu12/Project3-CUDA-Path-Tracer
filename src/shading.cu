#pragma once

#include "shading.h"

__host__ __device__ int wrapTexel(int i, int size, WrapMode mode)
{
    if (mode == WrapMode::ClampToEdge) {
        return glm::clamp(i, 0, size - 1);
    }

    int period = mode == WrapMode::MirroredRepeat ? size * 2 : size;
    i = (i % period + period) % period;
    return i < size ? i : period - 1 - i;
}

__host__ __device__ glm::vec4 readTexel(
    int x, int y, const Texture& texture, const TextureImage& image,
    const glm::vec4* texels, bool srgb)
{
    glm::vec4 color = texels[image.texelOffset + wrapTexel(y, image.height, texture.wrapV) * image.width
        + wrapTexel(x, image.width, texture.wrapU)];

    if (srgb) {
        for (int i = 0; i < 3; ++i) {
            color[i] = color[i] <= 0.04045f ? color[i] / 12.92f : powf((color[i] + 0.055f) / 1.055f, 2.4f);
        }
    }

    return color;
}

__host__ __device__ glm::vec2 textureUV(
    const TextureRef& textureRef, const MeshPrimitive& primitive, int vertex, const glm::vec2* texcoords)
{
    int offset = primitive.texcoordOffset + textureRef.texCoord * primitive.vertexCount;
    return glm::vec2(textureRef.uvTransform * glm::vec3(texcoords[offset + vertex - primitive.vertexOffset], 1));
}

__host__ __device__ glm::vec4 sampleTexture(
    const TextureRef& textureRef, const MeshPrimitive& primitive, const Triangle& triangle, glm::vec3 weights,
    const glm::vec2* texcoords, const Texture* textures, const TextureImage* images,
    const glm::vec4* texels, bool srgb)
{
    if (textureRef.textureId < 0 || textureRef.texCoord >= primitive.texcoordSetCount) {
        return glm::vec4(1);
    }

    const Texture& texture = textures[textureRef.textureId];

    if (texture.imageId < 0) {
        return glm::vec4(1);
    }

    const TextureImage& image = images[texture.imageId];
    glm::vec2 uv(0);

    for (int i = 0; i < 3; i++) {
        uv += weights[i] * textureUV(textureRef, primitive, triangle.indices[i], texcoords);
    }
    
    // TODO: mip chain
    glm::vec2 p = uv * glm::vec2(image.width, image.height);
    if (texture.magFilter == FilterMode::Nearest) {
        return readTexel(int(floorf(p.x)), int(floorf(p.y)), texture, image, texels, srgb);
    }
        
    p -= glm::vec2(0.5f);

    int x = int(floorf(p.x));
    int y = int(floorf(p.y));

    glm::vec2 f = p - glm::vec2(x, y);
    return glm::mix(glm::mix(readTexel(x, y, texture, image, texels, srgb),
        readTexel(x + 1, y, texture, image, texels, srgb), f.x),
        glm::mix(readTexel(x, y + 1, texture, image, texels, srgb),
            readTexel(x + 1, y + 1, texture, image, texels, srgb), f.x), f.y);
}

__host__ __device__ void evaluateMaterial(
    Material& material, glm::vec3& normal, const ShadeableIntersection& hit,
    const Geom* geoms, const MeshPrimitive* primitives, const Vertex* vertices,
    const Triangle* triangles, const glm::vec2* texcoords, const Texture* textures,
    const TextureImage* images, const glm::vec4* texels)
{
    if (hit.triangleId < 0) {
        return;
    }

    const Geom& geom = geoms[hit.geomId];
    const MeshPrimitive& primitive = primitives[geom.primitiveId];
    const Triangle& triangle = triangles[hit.triangleId];

    glm::vec3 w(1 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics);
    glm::vec4 color(0);

    for (int i = 0; i < 3; i++) {
        color += w[i] * vertices[triangle.indices[i]].color;
    }

    color *= sampleTexture(material.baseColorTexture, primitive, triangle, w, texcoords, textures, images, texels, true);

    material.albedo *= glm::vec3(color);
    material.alpha *= color.a;

    glm::vec4 mr = sampleTexture(material.metallicRoughnessTexture, primitive, triangle, w, texcoords, textures, images, texels);

    material.roughness *= mr.g;
    material.metalness *= mr.b;
    material.emittance *= glm::vec3(sampleTexture(material.emissiveTexture, primitive, triangle, w, texcoords, textures, images, texels, true));

    const TextureRef& textureRef = material.normalTexture;

    if (textureRef.textureId < 0 || textureRef.texCoord >= primitive.texcoordSetCount) {
        return;
    }

    if (textures[textureRef.textureId].imageId < 0) {
        return;
    }

    glm::vec3 n = glm::dot(normal, hit.geometricNormal) < 0 ? -normal : normal;
    glm::vec3 tangent, bitangent;

    if (primitive.hasTangents && textureRef.texCoord == 0) {
        glm::vec4 t(0);

        for (int i = 0; i < 3; ++i) {
            t += w[i] * vertices[triangle.indices[i]].tangent;
        }

        tangent = glm::vec3(geom.transform * glm::vec4(glm::vec3(t), 0));
        tangent -= n * glm::dot(n, tangent);

        if (glm::dot(tangent, tangent) < 1e-12f) {
            return;
        }

        tangent = glm::normalize(tangent);
        float sign = glm::determinant(glm::mat3(geom.transform)) < 0 ? -1.0f : 1.0f;
        bitangent = glm::cross(n, tangent) * (t.w < 0 ? -sign : sign);
    }
    else {
        glm::vec3 p0 = vertices[triangle.indices[0]].position;
        glm::vec3 e1 = glm::vec3(geom.transform * glm::vec4(vertices[triangle.indices[1]].position - p0, 0));
        glm::vec3 e2 = glm::vec3(geom.transform * glm::vec4(vertices[triangle.indices[2]].position - p0, 0));
        glm::vec2 uv0 = textureUV(textureRef, primitive, triangle.indices[0], texcoords);
        glm::vec2 d1 = textureUV(textureRef, primitive, triangle.indices[1], texcoords) - uv0;
        glm::vec2 d2 = textureUV(textureRef, primitive, triangle.indices[2], texcoords) - uv0;

        float det = d1.x * d2.y - d1.y * d2.x;

        if (fabsf(det) < 1e-8f) {
            return;
        }

        tangent = (e1 * d2.y - e2 * d1.y) / det;
        bitangent = (e2 * d1.x - e1 * d2.x) / det;
        tangent -= n * glm::dot(n, tangent);

        if (glm::dot(tangent, tangent) < 1e-12f) return;

        tangent = glm::normalize(tangent);
        bitangent = glm::cross(n, tangent) * (glm::dot(glm::cross(n, tangent), bitangent) < 0 ? -1.0f : 1.0f);
    }

    glm::vec3 mapped = 2.0f * glm::vec3(sampleTexture(textureRef, primitive, triangle, w, texcoords, textures, images, texels)) - 1.0f;
    mapped.x *= material.normalScale;
    mapped.y *= material.normalScale;
    mapped = tangent * mapped.x + bitangent * mapped.y + n * mapped.z;

    if (glm::dot(mapped, mapped) > 1e-12f) {
        normal = glm::normalize(mapped) * (glm::dot(normal, hit.geometricNormal) < 0 ? -1.0f : 1.0f);
    }
}
