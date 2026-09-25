#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "shading.h"
#include "bvh.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

#define MATERIAL_SORTING 0
#define STOCHASTIC_SAMPLING 1
#define USE_BVH 1

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static MeshPrimitive* dev_primitives = NULL;
static Vertex* dev_vertices = NULL;
static Triangle* dev_triangles = NULL;
static glm::vec2* dev_texcoords = NULL;
static Texture* dev_textures = NULL;
static TextureImage* dev_textureImages = NULL;
static glm::vec4* dev_texels = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static BVHNode* dev_bvhNodes = nullptr;
static PrimitiveRef* dev_bvhRefs = nullptr;
static int bvhNodeCount = 0;
static float* dev_envPixels = nullptr;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    if (!scene->primitives.empty()) {
        cudaMalloc(&dev_primitives, scene->primitives.size() * sizeof(MeshPrimitive));
        cudaMemcpy(dev_primitives, scene->primitives.data(), scene->primitives.size() * sizeof(MeshPrimitive), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(Vertex));
        cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_texcoords, scene->texcoords.size() * sizeof(glm::vec2));
        cudaMemcpy(dev_texcoords, scene->texcoords.data(), scene->texcoords.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
        cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_textureImages, scene->textureImages.size() * sizeof(TextureImage));
        cudaMemcpy(dev_textureImages, scene->textureImages.data(), scene->textureImages.size() * sizeof(TextureImage), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_texels, scene->texels.size() * sizeof(glm::vec4));
        cudaMemcpy(dev_texels, scene->texels.data(), scene->texels.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);
    }

#if USE_BVH
    BVH bvh;
    bvh.build(*scene);
    bvhNodeCount = int(bvh.nodes.size());

    if (bvhNodeCount > 0) {
        size_t nodeBytes = bvh.nodes.size() * sizeof(BVHNode);
        size_t refBytes = bvh.primitiveRefs.size() * sizeof(PrimitiveRef);

        cudaMalloc(&dev_bvhNodes, nodeBytes);
        cudaMemcpy(dev_bvhNodes, bvh.nodes.data(), nodeBytes, cudaMemcpyHostToDevice);

        cudaMalloc(&dev_bvhRefs, refBytes);
        cudaMemcpy(dev_bvhRefs, bvh.primitiveRefs.data(), refBytes, cudaMemcpyHostToDevice);
    }
#endif

    if (!scene->environment.pixels.empty()) {
        cudaMalloc(&dev_envPixels, scene->environment.pixels.size() * sizeof(float));
        cudaMemcpy(dev_envPixels, scene->environment.pixels.data(), bytes, cudaMemcpyHostToDevice);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_primitives);
    cudaFree(dev_vertices);
    cudaFree(dev_triangles);
    cudaFree(dev_texcoords);
    cudaFree(dev_textures);
    cudaFree(dev_textureImages);
    cudaFree(dev_texels);
#if USE_BVH
    cudaFree(dev_bvhNodes);
    cudaFree(dev_bvhRefs);
    bvhNodeCount = 0;
#endif

    cudaFree(dev_envPixels);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.radiance = glm::vec3(0.0f);
        segment.throughput = glm::vec3(1.0f);
        segment.sigmaA = glm::vec3(0.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

        float sampleX = static_cast<float>(x) + 0.5f;
        float sampleY = static_cast<float>(y) + 0.5f;

#if STOCHASTIC_SAMPLING
        constexpr int gridSize = 4;
        constexpr int numberOfCells = gridSize * gridSize;

        int cell = (iter - 1) % numberOfCells;

        sampleX = x + (cell % gridSize + u01(rng)) / static_cast<float>(gridSize);
        sampleY = y + (cell / gridSize + u01(rng)) / static_cast<float>(gridSize);
#endif

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sampleX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (sampleY - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void computeIntersectionsBVH(
    int num_paths,
    const PathSegment* pathSegments,
    const BVHNode* nodes,
    const PrimitiveRef* refs,
    int nodeCount,
    const Geom* geoms,
    const MeshPrimitive* primitives,
    const Vertex* vertices,
    const Triangle* triangles,
    ShadeableIntersection* intersections)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_paths) {
        intersections[index] = bvhIntersectionTest(
            pathSegments[index].ray,
            nodes, refs, nodeCount,
            geoms, primitives, vertices, triangles);
    }
}

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    const MeshPrimitive* primitives,
    const Vertex* vertices,
    const Triangle* triangles,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        intersections[path_index] = ShadeableIntersection{};

        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];
            ShadeableIntersection hitInfo{};

            if (geom.type == GeomType::CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == GeomType::SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == GeomType::Mesh)
            {
                t = meshIntersectionTest(geom, pathSegment.ray, primitives, vertices, triangles,
                    tmp_intersect, tmp_normal, outside, hitInfo);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                if (geom.type != GeomType::Mesh) 
                    hitInfo.geometricNormal = tmp_normal;

                intersections[path_index] = hitInfo;
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].geomId = hit_geom_index;
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

__device__ glm::vec3 sampleEnvironment(
    const float* pixels,
    glm::ivec2 size,
    glm::vec3 direction,
    float rotation)
{
    direction = glm::normalize(direction);

    float u = atan2f(direction.z, direction.x) / (2.0f * PI)
        + 0.5f + rotation / 360.0f;
    float v = acosf(glm::clamp(direction.y, -1.0f, 1.0f)) / PI;

    u -= floorf(u);

    int x = glm::clamp(int(u * size.x), 0, size.x - 1);
    int y = glm::clamp(int(v * size.y), 0, size.y - 1);
    int index = (y * size.x + x) * 4;

    return glm::vec3(
        pixels[index],
        pixels[index + 1],
        pixels[index + 2]);
}

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int depth,
    const Geom* geoms,
    const MeshPrimitive* primitives,
    const Vertex* vertices,
    const Triangle* triangles,
    const glm::vec2* texcoords,
    const Texture* textures,
    const TextureImage* images,
    const glm::vec4* texels,
    const float* envPixels,
    glm::ivec2 envSize,
    float envIntensity,
    float envRotation)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_paths) {
        return;
    }

    PathSegment& pathSegment = pathSegments[idx];

    if (pathSegment.remainingBounces <= 0) {
        return;
    }

    const ShadeableIntersection& intersection = shadeableIntersections[idx];

    if (intersection.t <= 0.0f) {
        if (envPixels) {
            glm::vec3 light = sampleEnvironment(envPixels, envSize,
                pathSegment.ray.direction, envRotation);

            pathSegment.radiance +=
                pathSegment.throughput * light * envIntensity;
        }

        pathSegment.remainingBounces = 0;
        return;
    }

    Material material = materials[intersection.materialId];
    glm::vec3 normal = intersection.surfaceNormal;
    glm::vec3 intersectPoint = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;

    bool volumeBoundary = !material.thinWalled && material.transmission > 0.0f;

    evaluateMaterial(material, normal, intersection, geoms, primitives, vertices, triangles,
                     texcoords, textures, images, texels);

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathSegment.pixelIndex, depth);
    thrust::uniform_real_distribution<float> u01(0, 1);

    bool skip = intersection.triangleId >= 0
        && !material.doubleSided
        && !volumeBoundary
        && glm::dot(pathSegment.ray.direction,
            intersection.geometricNormal) >= 0.0f;
    skip |= material.alphaMode == AlphaMode::Mask && material.alpha < material.alphaCutoff;
    skip |= material.alphaMode == AlphaMode::Blend && u01(rng) >= material.alpha;

    if (skip) {
        pathSegment.ray.origin = intersectPoint + 1e-3f * pathSegment.ray.direction;
        return;
    }

    pathSegment.throughput *= glm::exp(-pathSegment.sigmaA * intersection.t);

    pathSegment.radiance += pathSegment.throughput * material.emittance;

    if (--pathSegment.remainingBounces == 0)
        return;

    scatterRay(
        pathSegment,
        intersectPoint,
        normal,
        intersection.geometricNormal,
        material,
        rng);
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.radiance;
    }
}

struct IsPathAlive
{
    __host__ __device__ bool operator()(const PathSegment& path) const
    {
        return path.remainingBounces > 0;
    }
};

struct IntersectionComparer
{
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const
    {
        return (a.t > 0.0f ? a.materialId : -1) < (b.t > 0.0f ? b.materialId : -1);
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if USE_BVH
        computeIntersectionsBVH<<<numblocksPathSegmentTracing, blockSize1d>>>(
            num_paths,
            dev_paths,
            dev_bvhNodes,
            dev_bvhRefs,
            bvhNodeCount,
            dev_geoms,
            dev_primitives,
            dev_vertices,
            dev_triangles,
            dev_intersections
            );
#else
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_primitives,
            dev_vertices,
            dev_triangles,
            dev_intersections
            );
#endif

        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.

#if MATERIAL_SORTING
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, IntersectionComparer{});
        checkCUDAError("sort by material");
#endif

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            depth,
            dev_geoms,
            dev_primitives,
            dev_vertices,
            dev_triangles,
            dev_texcoords,
            dev_textures,
            dev_textureImages,
            dev_texels,
            dev_envPixels,
            hst_scene->environment.size,
            hst_scene->environment.intensity,
            hst_scene->environment.rotation
        );

        PathSegment* pathEnd = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, IsPathAlive{});
        checkCUDAError("path partition");

        num_paths = static_cast<int>(pathEnd - dev_paths);
        iterationComplete = num_paths == 0;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
