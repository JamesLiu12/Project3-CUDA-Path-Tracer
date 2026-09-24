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

__host__ __device__ static float fresnelDielectric(float cosThetaIn, float eta)
{
    if (eta == 0.0f) return 1.0f;
    if (eta == 1.0f) return 0.0f;

    cosThetaIn= glm::clamp(cosThetaIn, 0.0f, 1.0f);
    float sin2Trans = (1.0f - cosThetaIn* cosThetaIn) / (eta * eta);

    if (sin2Trans >= 1.0f) return 1.0f;

    float cosTrans = sqrtf(1.0f - sin2Trans);
    float reflP = (eta * cosThetaIn- cosTrans) / (eta * cosThetaIn + cosTrans);
    float reflS = (cosThetaIn - eta * cosTrans) / (cosThetaIn + eta * cosTrans);
    return 0.5f * (reflP * reflP + reflS * reflS);
}

__host__ __device__ static glm::vec3 surfaceFresnel(
    float cosThetaIn, const Material& mat, float eta)
{
    return mat.metalness == 1.0f
        ? fresnelSchlick(cosThetaIn, mat.albedo)
        : glm::vec3(fresnelDielectric(cosThetaIn, eta));
}

struct DielectricSample
{
    glm::vec3 dir{ 0.0f };
    glm::vec3 bsdf{ 0.0f };
    float pdf = 0.0f;
    bool isTransmit = false;
    bool isDelta = false;
};

__host__ __device__ static DielectricSample sampleDielectric(
    glm::vec3 outDir, glm::vec3 normal, const Material& mat, float eta,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    DielectricSample result;

    float alpha = mat.roughness * mat.roughness;
    result.isDelta = alpha < 1e-3f || eta == 1.0f;

    glm::vec3 half = result.isDelta ? normal : sampleGGXNormal(normal, alpha, rng);
    float outHalf = glm::min(glm::dot(outDir, half), 1.0f);
    if (outHalf <= 0.0f) return {};

    float F = fresnelDielectric(outHalf, eta);
    result.isTransmit = u01(rng) >= F;
    float probability = result.isTransmit ? 1.0f - F : F;

    if (!result.isTransmit) {
        result.dir = glm::reflect(-outDir, half);
    }
    else if (mat.thinWalled) {
        result.dir = glm::reflect(glm::reflect(-outDir, half), normal);
    }
    else {
        float cosTrans = sqrtf(glm::max(
            0.0f, 1.0f - (1.0f - outHalf * outHalf) / (eta * eta)));

        result.dir = -outDir / eta + (outHalf / eta - cosTrans) * half;
    }

    result.dir = glm::normalize(result.dir);

    float cosOut = glm::dot(normal, outDir);
    float cosIn = glm::dot(normal, result.dir);

    if (result.isTransmit ? cosIn >= 0.0f : cosIn <= 0.0f)
        return {};

    cosIn = fabsf(cosIn);
    glm::vec3 color = result.isTransmit ? mat.albedo : glm::vec3(1.0f);

    if (result.isDelta) {
        result.pdf = probability;
        result.bsdf = color * probability / cosIn;

        if (result.isTransmit && !mat.thinWalled)
            result.bsdf /= eta * eta;

        return result;
    }

    float NdotH = glm::dot(normal, half);
    float D = ggxD(NdotH, alpha);
    float G = smithG1(cosOut, alpha) * smithG1(cosIn, alpha);
    float pdfH = D * NdotH;

    if (!result.isTransmit || mat.thinWalled) {
        result.pdf = probability * pdfH / (4.0f * outHalf);
        result.bsdf = color * probability * D * G / (4.0f * cosOut * cosIn);
    }
    else {
        float inHalf = glm::dot(result.dir, half);
        float denom = inHalf + outHalf / eta;
        denom *= denom;

        result.pdf = probability * pdfH * fabsf(inHalf) / denom;
        result.bsdf = color * probability * D * G * fabsf(inHalf * outHalf)
            / (cosOut * cosIn * denom * eta * eta);
    }

    return result;
}

__host__ __device__ static void scatterOpaque(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& mat,
    float eta,
    thrust::default_random_engine& rng)
{
    float metallic = glm::clamp(mat.metalness, 0.0f, 1.0f);
    float roughness = glm::clamp(mat.roughness, 0.0f, 1.0f);

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
            pathSegment.throughput = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
            return;
        }

        if (u01(rng) < pSpecular)
        {
            light = glm::normalize(glm::reflect(-view, normal));
            glm::vec3 F = surfaceFresnel(NdotV, mat, eta);
            pathSegment.throughput *= F / pSpecular;
        }
        else
        {
            light = sampleCosineWeightedHemisphere(normal, rng);

            float NdotL = glm::dot(normal, light);

            if (NdotL <= 0.0f)
            {
                pathSegment.throughput = glm::vec3(0.0f);
                pathSegment.remainingBounces = 0;
                return;
            }

            glm::vec3 half = glm::normalize(view + light);
            glm::vec3 F = surfaceFresnel(glm::dot(view, half), mat, eta);

            pathSegment.throughput *= (1.0f - metallic) * (glm::vec3(1.0f) - F) * mat.albedo / (1.0f - pSpecular);
        }

        pathSegment.ray.direction = light;
        pathSegment.ray.origin = intersect + 1e-3f * normal;
        return;
    }

    if (u01(rng) < pSpecular) {
        glm::vec3 half = sampleGGXNormal(normal, alpha, rng);

        if (glm::dot(view, half) <= 0.0f) {
            pathSegment.throughput = glm::vec3(0.0f);
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
        pathSegment.throughput = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
        return;
    }

    float D = ggxD(NdotH, alpha);
    glm::vec3 F = surfaceFresnel(VdotH, mat, eta);
    float G = smithG1(NdotV, alpha) * smithG1(NdotL, alpha);

    glm::vec3 diffuse = (1.0f - metallic) * (glm::vec3(1.0f) - F) * mat.albedo / PI;
    glm::vec3 specular = D * G * F / (4.0f * NdotV * NdotL);

    float pdfDiffuse = NdotL / PI;
    float pdfSpecular = D * NdotH / (4.0f * VdotH);

    float pdf = (1.0f - pSpecular) * pdfDiffuse + pSpecular * pdfSpecular;

    if (pdf > 0)
    {
        pathSegment.throughput *= (diffuse + specular) * NdotL / pdf;

        pathSegment.ray.direction = light;
        pathSegment.ray.origin = intersect + 1e-3f * normal;
    }
    else
    {
        pathSegment.throughput = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
    }
}

__host__ __device__ void scatterRay(
    PathSegment& path,
    glm::vec3 point,
    glm::vec3 normal,
    glm::vec3 geometricNormal,
    const Material& mat,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    glm::vec3 outDir = -glm::normalize(path.ray.direction);
    bool entering = glm::dot(outDir, geometricNormal) > 0.0f;

    if (glm::dot(normal, geometricNormal) < 0.0f)
        normal = -normal;
    if (!entering)
        normal = -normal;

    if (glm::dot(outDir, normal) <= 0.0f) {
        path.remainingBounces = 0;
        return;
    }

    float eta = mat.ior == 0.0f ? 0.0f
        : (mat.thinWalled || entering ? mat.ior : 1.0f / mat.ior);

    Material part = mat;
    part.metalness = u01(rng) < mat.metalness ? 1.0f : 0.0f;
    bool transmitted = false;

    if (part.metalness == 0.0f && u01(rng) < mat.transmission) {
        DielectricSample result = sampleDielectric(outDir, normal, mat, eta, rng);

        if (result.pdf <= 0.0f) {
            path.remainingBounces = 0;
            return;
        }

        path.throughput *= result.bsdf * fabsf(glm::dot(normal, result.dir)) / result.pdf;
        path.ray.direction = result.dir;
        transmitted = result.isTransmit;
    }
    else {
        scatterOpaque(path, point, normal, part, eta, rng);
    }

    if (path.remainingBounces <= 0) return;

    float side = glm::dot(outDir, geometricNormal)
        * glm::dot(path.ray.direction, geometricNormal);

    if (transmitted ? side >= 0.0f : side <= 0.0f) {
        path.remainingBounces = 0;
        return;
    }
    
    if (transmitted && !mat.thinWalled) {
        glm::vec3 sigmaA = -glm::log(mat.attenuationColor) / mat.attenuationDistance;
        path.sigmaA = entering ? sigmaA : glm::vec3(0.0f);
    }

    float offset = glm::dot(path.ray.direction, geometricNormal) > 0.0f
        ? 1e-3f : -1e-3f;

    path.ray.origin = point + offset * geometricNormal;
}