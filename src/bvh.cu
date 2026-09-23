#include "bvh.h"
#include "scene.h"
#include "intersections.h"

#include <algorithm>

__host__ __device__ void AABB::expandBounds(const glm::vec3& p)
{
    min = glm::min(min, p);
    max = glm::max(max, p);
}

__host__ __device__ bool AABB::intersectBounds(
    const Ray& ray, float tMax, float& tNear) const
{
    tNear = 0.0f;

    for (int axis = 0; axis < 3; ++axis) {
        float origin = ray.origin[axis];
        float direction = ray.direction[axis];

        if (direction == 0.0f) {
            if (origin < min[axis] || origin > max[axis]) {
                return false;
            }
            continue;
        }

        float a = (min[axis] - origin) / direction;
        float b = (max[axis] - origin) / direction;

        tNear = glm::max(tNear, glm::min(a, b));
        tMax = glm::min(tMax, glm::max(a, b));

        if (tNear > tMax) {
            return false;
        }
    }

    return true;
}

void BVH::build(const Scene& scene)
{
    nodes.clear();
    primitiveRefs.clear();

    for (int geomId = 0; geomId < scene.geoms.size(); geomId++) {
        const Geom& geom = scene.geoms[geomId];

        if (geom.type == GeomType::Mesh) {
            const MeshPrimitive& mesh = scene.primitives[geom.primitiveId];

            for (int i = 0; i < mesh.triangleCount; ++i) {
                int triangleId = mesh.triangleOffset + i;
                const Triangle& triangle = scene.triangles[triangleId];

                PrimitiveRef ref{
                    PrimitiveType::Triangle, geomId, triangleId, AABB{}
                };

                for (int j = 0; j < 3; j++) {
                    glm::vec3 p = scene.vertices[triangle.indices[j]].position;
                    ref.bounds.expandBounds(glm::vec3(geom.transform * glm::vec4(p, 1.0f)));
                }

                primitiveRefs.push_back(ref);
            }
        }
        else {
            PrimitiveRef ref{
                geom.type == GeomType::SPHERE ? PrimitiveType::Sphere : PrimitiveType::Cube,
                geomId, -1, AABB{}};

            for (int i = 0; i < 8; i++) {
                glm::vec3 p(
                    (i & 1) ? 0.5f : -0.5f,
                    (i & 2) ? 0.5f : -0.5f,
                    (i & 4) ? 0.5f : -0.5f);

                ref.bounds.expandBounds(glm::vec3(geom.transform * glm::vec4(p, 1.0f)));
            }

            primitiveRefs.push_back(ref);
        }
    }

    for (PrimitiveRef& ref : primitiveRefs) {
        ref.bounds.min -= glm::vec3(1e-4f);
        ref.bounds.max += glm::vec3(1e-4f);
    }

    nodes.reserve(primitiveRefs.size());
    if (!primitiveRefs.empty()) {
        buildRecursive(0, primitiveRefs.size());
    }
}

int BVH::buildRecursive(int start, int end)
{
    AABB bounds = AABB{};
    AABB centers = AABB{};

    for (int i = start; i < end; i++) {
        const AABB& b = primitiveRefs[i].bounds;
        bounds.expandBounds(b.min);
        bounds.expandBounds(b.max);
        centers.expandBounds(0.5f * (b.min + b.max));
    }

    int nodeId = nodes.size();
    int count = end - start;
    nodes.push_back({ bounds, -1, -1, start, count });

    if (count <= 4) {
        return nodeId;
    }

    glm::vec3 size = centers.max - centers.min;
    int axis = size.y > size.x ? 1 : 0;
    if (size.z > size[axis]) {
        axis = 2;
    }

    int mid = start + count / 2;
    std::nth_element(
        primitiveRefs.begin() + start,
        primitiveRefs.begin() + mid,
        primitiveRefs.begin() + end,
        [axis](const PrimitiveRef& a, const PrimitiveRef& b) {
            return a.bounds.min[axis] + a.bounds.max[axis]
                < b.bounds.min[axis] + b.bounds.max[axis];
        });

    int left = buildRecursive(start, mid);
    int right = buildRecursive(mid, end);

    nodes[nodeId].leftChild = left;
    nodes[nodeId].rightChild = right;
    nodes[nodeId].count = 0;

    return nodeId;
}

static __device__ float intersectTriangle(
    const Geom& geom,
    const Ray& ray,
    const Triangle& triangle,
    const Vertex* vertices,
    bool hasNormals,
    ShadeableIntersection& hit)
{
    const Vertex& a = vertices[triangle.indices[0]];
    const Vertex& b = vertices[triangle.indices[1]];
    const Vertex& c = vertices[triangle.indices[2]];

    Ray local;
    local.origin = glm::vec3(
        geom.inverseTransform * glm::vec4(ray.origin, 1.0f));
    local.direction = glm::vec3(
        geom.inverseTransform * glm::vec4(ray.direction, 0.0f));

    glm::vec3 tuv;
    if (!glm::intersectLineTriangle(
        local.origin, local.direction,
        a.position, b.position, c.position, tuv) || tuv.x <= 0.0f) {
        return -1.0f;
    }

    glm::vec3 faceNormal = glm::cross(b.position - a.position, c.position - a.position);
    glm::vec3 normal = hasNormals
        ? (1.0f - tuv.y - tuv.z) * a.normal + tuv.y * b.normal + tuv.z * c.normal
        : faceNormal;

    if (glm::dot(normal, faceNormal) < 0.0f) {
        normal = -normal;
    }

    hit.barycentrics = glm::vec2(tuv.y, tuv.z);
    hit.geometricNormal = glm::normalize(
        glm::vec3(geom.invTranspose * glm::vec4(faceNormal, 0.0f)));
    hit.surfaceNormal = glm::normalize(
        glm::vec3(geom.invTranspose * glm::vec4(normal, 0.0f)));

    if (glm::dot(ray.direction, hit.geometricNormal) >= 0.0f) {
        hit.surfaceNormal = -hit.surfaceNormal;
    }

    return tuv.x;
}

__device__ ShadeableIntersection bvhIntersectionTest(
    const Ray& ray,
    const BVHNode* nodes,
    const PrimitiveRef* refs,
    int nodeCount,
    const Geom* geoms,
    const MeshPrimitive* primitives,
    const Vertex* vertices,
    const Triangle* triangles)
{
    ShadeableIntersection result{};
    result.t = -1.0f;

    if (nodeCount == 0) {
        return result;
    }

    float closest = FLT_MAX;
    int stack[64];
    int stackSize = 0;
    stack[stackSize++] = 0;

    while (stackSize > 0) {
        const BVHNode& node = nodes[stack[--stackSize]];

        float tNear;
        if (!node.bounds.intersectBounds(ray, closest, tNear)) {
            continue;
        }

        if (node.count == 0) {
            int left = node.leftChild;
            int right = node.rightChild;

            float leftNear, rightNear;
            bool hitLeft = nodes[left].bounds.intersectBounds(ray, closest, leftNear);
            bool hitRight = nodes[right].bounds.intersectBounds(ray, closest, rightNear);

            if (hitLeft && hitRight) {
                if (rightNear < leftNear) {
                    int temp = left;
                    left = right;
                    right = temp;
                }

                stack[stackSize++] = right;
                stack[stackSize++] = left;
            }
            else if (hitLeft) {
                stack[stackSize++] = left;
            }
            else if (hitRight) {
                stack[stackSize++] = right;
            }

            continue;
        }

        for (int i = node.start; i < node.start + node.count; i++) {
            const PrimitiveRef& ref = refs[i];
            const Geom& geom = geoms[ref.geomId];
            ShadeableIntersection hit{};

            if (ref.type == PrimitiveType::Triangle) {
                hit.t = intersectTriangle(
                    geom, ray, triangles[ref.triangleId], vertices,
                    primitives[geom.primitiveId].hasNormals, hit);
            }
            else {
                glm::vec3 point;
                bool outside;

                if (ref.type == PrimitiveType::Cube) {
                    hit.t = boxIntersectionTest(
                        geom, ray, point, hit.surfaceNormal, outside);
                }
                else {
                    hit.t = sphereIntersectionTest(
                        geom, ray, point, hit.surfaceNormal, outside);
                }

                hit.geometricNormal = hit.surfaceNormal;
            }

            if (hit.t > 0.0f && hit.t < closest) {
                closest = hit.t;
                hit.geomId = ref.geomId;
                hit.triangleId = ref.triangleId;
                hit.materialId = geom.materialid;
                result = hit;
            }
        }
    }

    return result;
}