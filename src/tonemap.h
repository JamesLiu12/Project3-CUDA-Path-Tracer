#pragma once

#include <glm/glm.hpp>

__host__ __device__ float acesFilm(float x);

__host__ __device__ float linearToSrgb(float x);

__host__ __device__ glm::vec3 toneMap(glm::vec3 color, float exposure, bool enabled);