#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n(0.0f);
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(q.origin + tmin * q.direction, 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = rt.origin + t * rt.direction;

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Ray r,
    const MeshPrimitive* primitives,
    const Vertex* vertices,
    const Triangle* triangles,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    ShadeableIntersection& hitInfo)
{
    Ray q;
    q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    const MeshPrimitive& primitive = primitives[mesh.primitiveId];
    float tmin = 1e38f;
    glm::vec3 tmin_n, faceNormal;

    for (int i = 0; i < primitive.triangleCount; ++i)
    {
        const Triangle& triangle = triangles[primitive.triangleOffset + i];

        const Vertex& v0 = vertices[triangle.indices[0]];
        const Vertex& v1 = vertices[triangle.indices[1]];
        const Vertex& v2 = vertices[triangle.indices[2]];

        glm::vec3 hit;
        if (!glm::intersectLineTriangle(q.origin, q.direction, v0.position, v1.position, v2.position, hit)
            || hit.x <= 0.0f || hit.x >= tmin)
            continue;

        tmin = hit.x;
        hitInfo.triangleId = primitive.triangleOffset + i;
        hitInfo.barycentrics = glm::vec2(hit.y, hit.z);
        faceNormal = glm::cross(v1.position - v0.position, v2.position - v0.position);
        tmin_n = primitive.hasNormals
            ? (1.0f - hit.y - hit.z) * v0.normal + hit.y * v1.normal + hit.z * v2.normal
            : faceNormal;

        if (glm::dot(tmin_n, tmin_n) == 0.0f) tmin_n = faceNormal;
        if (glm::dot(tmin_n, faceNormal) < 0.0f) tmin_n = -tmin_n;
    }

    if (tmin == 1e38f) return -1;

    hitInfo.geometricNormal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(faceNormal, 0)));
    outside = glm::dot(r.direction, hitInfo.geometricNormal) < 0.0f;
    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(q.origin + tmin * q.direction, 1.0f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(tmin_n, 0.0f)));

    if (!outside) normal = -normal;

    return glm::length(r.origin - intersectionPoint);
}
