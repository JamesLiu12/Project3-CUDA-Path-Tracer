#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ static void computeTangents2(glm::vec3& tangent1, glm::vec3& tangent2, const glm::vec3& normal)
{
    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

   // Use not-normal direction to generate two perpendicular directions
   tangent1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
   tangent2 =
        glm::normalize(glm::cross(normal, tangent1));
}

__host__ __device__ glm::vec3 sampleCosineWeightedHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    glm::vec3 tangent1;
    glm::vec3 tangent2;
    computeTangents2(tangent1, tangent2, normal);

    return up * normal
        + cos(around) * over * tangent1
        + sin(around) * over * tangent2;
}

__host__ __device__ static float ggxD(float NdotH, float alpha)
{
    if (NdotH <= 0.0f) return 0.0f;

    float alpha2 = alpha * alpha;
    float val = NdotH * NdotH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * val * val);
}

__host__ __device__ static glm::vec3 fresnelSchlick(float VdotH, const glm::vec3& F0)
{
    float val = 1.0f - glm::clamp(VdotH, 0.0f, 1.0f);

    return F0 + (glm::vec3(1.0f) - F0) * (val * val * val * val * val);
}

__host__ __device__ static float smithG1(float cosAngle, float alpha)
{
    if (cosAngle <= 0.0f) return 0.0f;

    float alpha2 = alpha * alpha;
    return 2.0f * cosAngle / (cosAngle + sqrtf(alpha2 + (1.0f - alpha2) * cosAngle * cosAngle));
}

__host__ __device__ static glm::vec3 sampleGGXNormal(const glm::vec3& normal, float alpha, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float f1 = u01(rng);
    float f2 = u01(rng);

    float phi = TWO_PI * f1;
    float cosTheta = sqrtf(
        (1.0f - f2) /
        (1.0f + (alpha * alpha - 1.0f) * f2)
    );
    float sinTheta = sqrtf(
        glm::max(0.0f, 1.0f - cosTheta * cosTheta)
    );

    glm::vec3 half(
        sinTheta * cosf(phi),
        sinTheta * sinf(phi),
        cosTheta
    );

    glm::vec3 tangent1;
    glm::vec3 tangent2;
    computeTangents2(tangent1, tangent2, normal);

    return glm::normalize(
        tangent1 * half.x + tangent2 * half.y + normal * half.z
    );
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    float metallic = glm::clamp(m.metalness, 0.0f, 1.0f);
    float roughness = glm::clamp(m.roughness, 0.0f, 1.0f);

    // TODO: refraction
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.albedo, metallic);

    float alpha = roughness * roughness;

    // TODO: improve?
    float pSpecular = metallic == 1.0f ? 1.0f : 0.5f;

    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 view = -glm::normalize(pathSegment.ray.direction);

    if (glm::dot(normal, view) < 0.0f) {
        normal = -normal;
    }

    glm::vec3 light;

    if (alpha < 1e-3f)
    {
        float NdotV = glm::dot(normal, view);

        if (NdotV <= 0.0f)
        {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
            return;
        }

        if (u01(rng) < pSpecular)
        {
            light = glm::normalize(glm::reflect(-view, normal));
            glm::vec3 F = fresnelSchlick(NdotV, F0);
            pathSegment.color *= F / pSpecular;
        }
        else
        {
            light = sampleCosineWeightedHemisphere(normal, rng);

            float NdotL = glm::dot(normal, light);

            if (NdotL <= 0.0f)
            {
                pathSegment.color = glm::vec3(0.0f);
                pathSegment.remainingBounces = 0;
                return;
            }

            glm::vec3 half = glm::normalize(view + light);
            glm::vec3 F = fresnelSchlick(glm::dot(view, half), F0);

            pathSegment.color *= (1.0f - metallic) * (glm::vec3(1.0f) - F) * m.albedo / (1.0f - pSpecular);
        }

        pathSegment.ray.direction = light;
        pathSegment.ray.origin = intersect + 1e-3f * normal;
        return;
    }

    if (u01(rng) < pSpecular) {
        glm::vec3 half = sampleGGXNormal(normal, alpha, rng);

        if (glm::dot(view, half) <= 0.0f) {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
            return;
        }

        light = glm::normalize(glm::reflect(-view, half));
    }
    else {
        light = sampleCosineWeightedHemisphere(normal, rng);
    }

    glm::vec3 half = glm::normalize(view + light);

    float NdotV = glm::dot(normal, view);
    float NdotL = glm::dot(normal, light);
    float NdotH = glm::dot(normal, half);
    float VdotH = glm::dot(view, half);

    if (NdotV <= 0.0f || NdotL <= 0.0f) {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
        return;
    }

    float D = ggxD(NdotH, alpha);
    glm::vec3 F = fresnelSchlick(VdotH, F0);
    float G = smithG1(NdotV, alpha) * smithG1(NdotL, alpha);

    glm::vec3 diffuse = (1.0f - metallic) * (glm::vec3(1.0f) - F) * m.albedo / PI;
    glm::vec3 specular = D * G * F / (4.0f * NdotV * NdotL);

    float pdfDiffuse = NdotL / PI;
    float pdfSpecular = D * NdotH / (4.0f * VdotH);

    float pdf = (1.0f - pSpecular) * pdfDiffuse + pSpecular * pdfSpecular;

    if (pdf > 0)
    {
        pathSegment.color *= (diffuse + specular) * NdotL / pdf;

        pathSegment.ray.direction = light;
        pathSegment.ray.origin = intersect + 1e-3f * normal;
    }
    else
    {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
    }
}
