# pragma once

#include "sceneStructs.h"

#include <vector>

enum class PrimitiveType {
    Triangle,
    Sphere,
    Cube
};

struct AABB {
    glm::vec3 min = glm::vec3(FLT_MAX);
    glm::vec3 max = glm::vec3(-FLT_MAX);

    __host__ __device__ void expandBounds(const glm::vec3& p);

    __host__ __device__ bool intersectBounds(
        const Ray& ray, float tMax, float& tNear) const;
};

struct PrimitiveRef {
    PrimitiveType type;

    int geomId;
    int triangleId;

    AABB bounds;
};

struct BVHNode {
    AABB bounds;

    int leftChild;
    int rightChild;

    int start; 
    int count;
};

class Scene;

class BVH
{
public:
    std::vector<BVHNode> nodes;
    std::vector<PrimitiveRef> primitiveRefs;

    void build(const Scene& scene);

private:
    int buildRecursive(int start, int end);
};

__device__ ShadeableIntersection bvhIntersectionTest(
    const Ray& ray,
    const BVHNode* nodes,
    const PrimitiveRef* refs,
    int nodeCount,
    const Geom* geoms,
    const MeshPrimitive* primitives,
    const Vertex* vertices,
    const Triangle* triangles);
