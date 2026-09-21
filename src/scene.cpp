#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

static glm::vec3 readVec3(const json& p, const char* name, glm::vec3 fallback)
{
    if (!p.contains(name))
    {
        return fallback;
    }

    const auto& value = p.at(name);

    return glm::vec3(
        value.at(0).get<float>(),
        value.at(1).get<float>(),
        value.at(2).get<float>());
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};

        if (p.contains("ALBEDO")) {
            const auto& col = p["ALBEDO"];
            newMaterial.albedo = glm::vec3(col[0], col[1], col[2]);
        }

        if (p.contains("METALNESS")) {
            newMaterial.metalness = p["METALNESS"];
        }

        if (p.contains("ROUGHNESS")) {
            newMaterial.roughness = p["ROUGHNESS"];
        }

        if (p.contains("EMITTANCE")) {
            const auto& e = p["EMITTANCE"];
            newMaterial.emittance = glm::vec3(e[0], e[1], e[2]);
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = GeomType::CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = GeomType::SPHERE;
        }
        else if (type == "mesh")
        {
            if (!p.contains("FILENAME"))
            {
                throw std::runtime_error("Mesh object is missing FILENAME.");
            }

            const glm::vec3 translation = readVec3(p, "TRANS", glm::vec3(0.0f));
            const glm::vec3 rotation = readVec3(p, "ROTAT", glm::vec3(0.0f));
            const glm::vec3 scale = readVec3(p, "SCALE", glm::vec3(1.0f));

            const glm::mat4 rootTransform =
                utilityCore::buildTransformationMatrix(
                    translation, rotation, scale);

            loadFromGLTF(p["FILENAME"], rootTransform);

            continue;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        newGeom.translation = readVec3(p, "TRANS", glm::vec3(0.0f));
        newGeom.rotation = readVec3(p, "ROTAT", glm::vec3(0.0f));
        newGeom.scale = readVec3(p, "SCALE", glm::vec3(1.0f));
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadFromGLTF(const std::string& filename, const glm::mat4& rootTransform)
{

}