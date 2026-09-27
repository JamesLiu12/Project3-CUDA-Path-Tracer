#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <cmath>

__host__ __device__ inline bool intersectMeshTriangle(
    const glm::vec3& origin, const glm::vec3& direction,
    const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
    glm::vec3& hit)
{
    const glm::vec3 edge1 = b - a;
    const glm::vec3 edge2 = c - a;
    const glm::vec3 p = glm::cross(direction, edge2);
    const float det = glm::dot(edge1, p);
    const float scale = glm::length(edge1) * glm::length(edge2) * glm::length(direction);
    if (fabsf(det) <= 1e-7f * scale) return false;

    const float invDet = 1.0f / det;
    const glm::vec3 fromA = origin - a;
    hit.y = glm::dot(fromA, p) * invDet;
    if (hit.y < 0.0f || hit.y > 1.0f) return false;

    const glm::vec3 q = glm::cross(fromA, edge1);
    hit.z = glm::dot(direction, q) * invDet;
    if (hit.z < 0.0f || hit.y + hit.z > 1.0f) return false;

    hit.x = glm::dot(edge2, q) * invDet;
    return true;
}


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Ray r,
    const MeshPrimitive* primitives,
    const Vertex* vertices,
    const Triangle* triangles,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    ShadeableIntersection& hitInfo);
