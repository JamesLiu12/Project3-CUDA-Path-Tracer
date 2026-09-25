#include "tonemap.h"

#include <cuda_runtime.h>
#include <cmath>

__host__ __device__ float acesFilm(float x)
{
    x = fmaxf(x, 0.0f);
    float y = (x * (2.51f * x + 0.03f))
        / (x * (2.43f * x + 0.59f) + 0.14f);
    return fminf(fmaxf(y, 0.0f), 1.0f);
}

__host__ __device__ float linearToSrgb(float x)
{
    return x <= 0.0031308f
        ? 12.92f * x
        : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}

__host__ __device__ glm::vec3 toneMap(glm::vec3 color, float exposure, bool enabled)
{
    color *= exp2f(exposure);

    if (enabled)
    {
        color = glm::vec3(
            acesFilm(color.x),
            acesFilm(color.y),
            acesFilm(color.z)
        );
    }

    color = glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f));

    return glm::vec3(
        linearToSrgb(color.x),
        linearToSrgb(color.y),
        linearToSrgb(color.z)
    );
}