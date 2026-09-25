#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& filename, const glm::mat4& rootTransform = glm::mat4(1.0f));
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
    std::vector<MeshPrimitive> primitives;

    std::vector<glm::vec2> texcoords;

    std::vector<TextureImage> textureImages;
    std::vector<Texture> textures;
    std::vector<glm::vec4> texels;

    Environment environment;
};
