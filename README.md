# CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

- Sizhe Liu
- Tested on: Windows 11 Pro, Intel Core i9-12900H @ 2.50 GHz, 32 GB RAM, NVIDIA RTX 3090 24 GB (personal computer)
- CUDA 13.3; Visual Studio 2022; Release build

## Gallery

![Warrior vs Dragon](img/chess-scene.2026-09-27_22-28-24z.5000samp.png)
![Chess](img/chess-outdoor.2026-09-28_00-29-22z.1592samp.png)

## Path Tracer

This CUDA path tracer follows rays from the camera through a scene, sampling how light reflects and refracts at each bounce. Paths collect radiance from emissive surfaces or an HDR environment, and repeated samples are averaged into the final image. More samples per pixel (SPP) mean less noise!

Each iteration generates one path per pixel, then alternates intersection and shading kernels until the paths terminate or reach the bounce limit. The renderer supports spheres, cubes, and glTF meshes, with physically based materials and a few camera effects.

## Visual Features

### BSDF Model

The BSDF combines reflection (BRDF) and transmission (BTDF). I use Lambert diffuse and GGX microfacet specular, with Smith masking-shadowing. Microfacets model a rough surface as tiny mirrors with different orientations, so roughness controls how widely reflections spread. Metals use Schlick Fresnel; dielectrics use `fresnelDielectric` to evaluate angle-dependent reflection.

Monte Carlo sampling estimates the light contribution at each bounce, weighted by the BSDF, cosine term, and sampling probability. Cosine-weighted diffuse sampling and GGX visible-normal sampling put more samples in useful directions. See PBRT's [BSDF representation](https://pbr-book.org/4ed/Reflection_Models/BSDF_Representation) and [microfacet theory](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory).

#### Metallic-Roughness

The metallic-roughness workflow controls materials with base color, metallic, and roughness. Metallic blends between dielectric and conductor behavior; increasing roughness turns sharp reflections into broader highlights.

![Metallic-roughness sphere grid](img/metallic-roughness-shpere-grid.2026-09-26_19-26-40z.5000samp.png)

Model: [MetalRoughSpheres](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/MetalRoughSpheres).

#### Dielectric Surface

Glass can both reflect and transmit light. Fresnel determines the reflection/refraction split, Snell's law sets the refraction direction, and total internal reflection keeps the ray inside when refraction is impossible. Roughness also applies to transmission, giving frosted glass a softer appearance.

![Cornell IOR Transmission Spheres](img/cornell-ior-transmission.2026-09-26_21-44-17z.5000samp.png)

Back row, left to right: IOR **1.1 → 1.5 → 2.0**. Front row: transmission **0 → 0.5 → 1.0**. IOR changes refraction and reflectivity; transmission controls the balance between opaque and transmissive dielectric behavior.

#### NDF vs. VNDF Sampling

Sampling the GGX normal distribution (NDF) ignores the viewing direction and can select microfacets hidden from the viewer. Visible normal distribution (VNDF) sampling accounts for that visibility, reducing wasted samples and extreme weights, especially at grazing angles. Both evaluate the same BSDF with their matching PDFs.

<table align="center">
  <tr>
    <th>SPP</th>
    <th>VNDF</th>
    <th>NDF</th>
  </tr>
  <tr>
    <th>128</th>
    <td><img src="img/cornell-vndf-test.2026-09-27_00-10-46z.128samp.png" width="400" alt="VNDF, 128 spp"></td>
    <td><img src="img/cornell-ndf-test.2026-09-27_00-15-15z.128samp.png" width="400" alt="NDF, 128 spp"></td>
  </tr>
  <tr>
    <th>256</th>
    <td><img src="img/cornell-vndf-test.2026-09-27_00-11-43z.256samp.png" width="400" alt="VNDF, 256 spp"></td>
    <td><img src="img/cornell-ndf-test.2026-09-27_00-15-40z.256samp.png" width="400" alt="NDF, 256 spp"></td>
  </tr>
  <tr>
    <th>512</th>
    <td><img src="img/cornell-vndf-test.2026-09-27_00-12-27z.512samp.png" width="400" alt="VNDF, 512 spp"></td>
    <td><img src="img/cornell-ndf-test.2026-09-27_00-16-31z.512samp.png" width="400" alt="NDF, 512 spp"></td>
  </tr>
</table>

The tilted rough metal panels make the difference easier to see, with the teapot as a recognizable reflection. At 128, 256, and 512 SPP, VNDF gives cleaner panels and less noise on the surrounding walls. The scene, materials, and SPP stay fixed; only the sampling method changes.

### glTF Loading and Rendering

I use [tinygltf](https://github.com/syoyo/tinygltf) to load glTF/GLB meshes and materials. The renderer samples base-color, normal, metallic-roughness, emissive, and transmission textures on the GPU. Color textures are decoded from sRGB for linear-space shading. Occlusion textures are loaded but currently unused.

![Dragon Dispersion](img/studio-dragon.2026-09-27_00-44-23z.5000samp.png)

Model: [DragonDispersion](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/DragonDispersion), rendered with the supported material features; spectral dispersion is not implemented.

### Environment Mapping

Rays that miss the scene sample an HDR equirectangular environment map using their outgoing direction. This supplies both the background and lighting, including reflections and light transmitted through glass. Changing the environment makes a huge difference to the mood!

<table align="center">
  <tr>
    <th>Fireplace</th>
    <th>Hills</th>
  </tr>
  <tr>
    <td><img src="img/envmap-studio.2026-09-27_01-53-55z.5000samp.png" width="400" alt="Fireplace"></td>
    <td><img src="img/envmap-sky.2026-09-27_01-58-54z.5000samp.png" width="400" alt="Hills"></td>
  </tr>
</table>

### Tone Mapping

I apply exposure, the [Narkowicz ACES approximation](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/), and linear-to-sRGB conversion for display. The curve compresses bright values into the display range, keeping highlight variation that a hard clamp would lose.

<table align="center">
  <tr>
    <th>Tone mapped, exposure = −2.5 EV</th>
    <th>Clamp</th>
  </tr>
  <tr>
    <td><img src="img/wood-tunnel.2026-09-27_15-49-18z.2000samp.png" width="400" alt="Tone mapped, exposure = −2.5 EV"></td>
    <td><img src="img/wood-tunnel.2026-09-27_15-50-46z.2000samp.png" width="400" alt="Clamp"></td>
  </tr>
</table>

This is a lightweight filmic curve rather than a full ACES color-management pipeline. The example shows the combined effect of tone mapping and the indicated exposure adjustment.

### Depth of Field

The [thin-lens model](https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models) samples ray origins on a lens disk and aims them at the focus plane. Objects at the focus distance stay sharp while nearer and farther objects blur. A larger lens radius produces stronger blur, directly through ray tracing.

<table align="center">
  <tr>
    <th>DOF On</th>
    <th>DOF Off</th>
  </tr>
  <tr>
    <td><img src="img/dof-test.2026-09-27_02-35-52z.5000samp.png" width="400" alt="DOF On"></td>
    <td><img src="img/dof-test.2026-09-27_02-40-36z.5000samp.png" width="400" alt="DOF Off"></td>
  </tr>
</table>

### Stochastic Anti-aliasing

Following the idea in [Paul Bourke's ray-tracing notes](https://paulbourke.net/miscellaneous/raytracing/), I divide each pixel into a 4×4 grid, cycle through its cells over 16 iterations, and jitter within each cell. Averaging these samples smooths silhouettes and sharp reflected edges.

<table align="center">
  <tr>
    <th>AA On</th>
    <th>AA Off</th>
  </tr>
  <tr>
    <td><img src="img/cornell-sphere.2026-09-27_03-08-55z.5000samp.png" width="430" alt="AA On"></td>
    <td><img src="img/cornell-sphere.2026-09-27_03-54-14z.5000samp.png" width="430" alt="AA Off"></td>
  </tr>
</table>

The golden sphere has a smoother edge with AA enabled. Fixed pixel-center sampling repeatedly samples the same side of an edge, so more iterations alone cannot remove the jagged outline. The wall-floor seam also looks cleaner with AA, some dark pixels can be perceived without AA.

## Performance Features

### Bounding Volume Hierarchy (BVH)

Testing every primitive costs O(N) per ray. A [BVH](https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies) groups primitives into bounding boxes so a ray can skip whole subtrees. My static-scene implementation builds on the CPU and traverses on the GPU:

- Each triangle, sphere, or cube gets one primitive reference with a world-space AABB.
- The builder splits at the median along the widest centroid axis, with at most four primitives per leaf.
- Each GPU thread traces one ray using an explicit stack, visiting the nearer child first and pruning against the closest hit.

The balanced median build takes expected O(N log N) time and O(N) storage. Traversal often approaches O(log N) with well-separated bounds, but overlapping bounds can still force O(N) work.

The following captures use `studio-dragon.json`:

<table align="center">
  <tr>
    <th>BVH On</th>
    <th>BVH Off</th>
  </tr>
  <tr>
    <td><img src="img/bvh-on.png" width="400" alt="BVH On"></td>
    <td><img src="img/bvh-off.png" width="400" alt="BVH Off"></td>
  </tr>
  <tr>
    <td>115.740 ms / frame</td>
    <td>9546.722 ms / frame</td>
  </tr>
</table>

The captured averages drop from **9546.72 to 115.74 ms/frame**, about **82.5× faster**. BVH really pays off on this dense mesh!

### Stream Compaction

After each bounce, `thrust::partition` moves live paths to the front so the next intersection and shading launches process fewer paths. I use partition because final gathering still needs the terminated paths' accumulated radiance; unlike `remove_if`, it preserves both groups.

The breakdowns below use the **Nsight Systems** measurements.

![Stream compaction breakdown for the Cornell sphere scene](img/graphs/compaction-cornell.png)

In the Cornell scene, time falls from **55.48 to 54.57 ms** (**1.6% lower**). Estimated intersection time drops from 45.83 to 29.17 ms, but compaction itself costs 16.83 ms. Most of the savings disappear into that overhead, leaving only a small measured gain.

![Stream compaction breakdown for the open sphere scene](img/graphs/compaction-open.png)

In the open scene, time rises from **9.38 to 12.50 ms** (**33.3% higher**). Intersections fall from 5.07 to 1.09 ms, but compaction adds 5.17 ms. Many rays escape quickly, yet the remaining intersection work is already cheap. Compaction helps when the future work it removes exceeds the cost of rearranging paths; an open scene alone does not guarantee a win.

### Material Sorting

Before shading, `thrust::sort_by_key` reorders intersections and their paths by alpha mode, transmission category (opaque, thin-walled, or solid), and material ID. Nearby threads then take more similar branches in the same shading kernel, potentially reducing warp divergence.

![Material sorting breakdown for the IOR and transmission scene](img/graphs/material-sorting.png)

For `cornell-ior-transmission.json`, sorting increases time from **148.22 to 250.84 ms** (**69.2% higher**). Estimated shading time improves only from 11.11 to 10.25 ms, while sorting costs **84.69 ms**. Other kernel time also rises from 35.47 to 54.18 ms. Sorting every bounce is simply too expensive here; coarse material buckets or less frequent sorting would be worth trying.

## Future Improvements

**Next-event estimation (NEE) + multiple importance sampling (MIS).** Currently, light is found through BSDF-sampled paths. NEE would explicitly sample a light or the environment and trace a shadow ray toward it. MIS could combine light and BSDF sampling to reduce variance without double-counting contributions.

![Fireflies under a bright HDR sun](img/dof-test.2026-09-27_02-19-36z.5000samp.png)

The bright specks here are fireflies: a rare BSDF sample reaches the very bright sun and contributes much more than nearby samples. Environment importance sampling combined with MIS should make this converge more reliably.

**More light types.** Add explicit directional, point, and spot lights alongside the existing emissive geometry and environment. Occlusion textures are also unused; they would need a deliberate treatment to avoid double-darkening visibility already traced by the renderer.

**Texture LOD.** Textures currently use only level 0. Mip chains and a ray-footprint estimate would allow appropriate levels for distant surfaces, reducing texture aliasing and unnecessary texture detail.

## References

### Third-Party Libraries and Starter Code

- [CIS 5650 CUDA Path Tracer starter](https://github.com/CIS5650-Fall-2026/Project3-CUDA-Path-Tracer)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [GLM](https://github.com/g-truc/glm)
- [tinygltf](https://github.com/syoyo/tinygltf)

### Models and Environment Maps

- Khronos glTF Sample Assets: [A Beautiful Game](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ABeautifulGame), [DragonDispersion](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/DragonDispersion), [MetalRoughSpheres](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/MetalRoughSpheres)
- [McGuire Computer Graphics Archive](https://casual-effects.com/data/)
- Sketchfab: [Utah Teapot](https://sketchfab.com/3d-models/the-utah-teapot-1092c2832df14099807f66c8b792374d), [Desk Set](https://sketchfab.com/3d-models/desk-set-f26030d09d73422f8ff270425c7c63e0), [Warrior Toy](https://sketchfab.com/3d-models/warrior-toy-a01f57189100463cbd9c75a127609e79), [Coffee Cup](https://sketchfab.com/3d-models/coffee-cup-on-plate-with-spoon-c34fdaee38dd444fac587ad16235cc31)
- Poly Haven: [Belfast Sunset](https://polyhaven.com/a/belfast_sunset), [Citrus Orchard Road](https://polyhaven.com/a/citrus_orchard_road_puresky), [Fireplace](https://polyhaven.com/a/fireplace)

## Bloopers

I initially forgot the perfect-specular case at zero roughness. Evaluating the rough GGX formula there can produce 0/0 and invalid values, leaving bright and dark artifacts. A separate smooth-surface branch fixed that one!

![Artifacts from the missing perfect-specular case](img/cornell.2026-09-19_17-03-41z.5000samp.png)

A fixed intersection epsilon also caused tiny triangles to be rejected, leaving holes in the chess scene. Small geometry is a good reminder that one tolerance does not fit every scale.

![Missing tiny triangles in the chess scene](img/chess-outdoor-flaw.png)
