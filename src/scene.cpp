#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <tiny_gltf.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <filesystem>
#include <stb_image.h>

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

static std::string toJsonRelativePath(const std::string& jsonPath, const std::string& path)
{
    return (std::filesystem::path(jsonPath).parent_path() / path).string();
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

        newMaterial.transmission = p.value("TRANSMISSION", 0.0f);
        newMaterial.ior = p.value("IOR", 1.5f);
        newMaterial.thinWalled = p.value("THIN_WALLED", false);

        if (p.contains("ATTENUATION_DISTANCE")) {
            newMaterial.attenuationColor = readVec3(p, "ATTENUATION_COLOR", glm::vec3(1.0f));
            newMaterial.attenuationDistance = p["ATTENUATION_DISTANCE"].get<float>();
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
        else if (type == "model")
        {
            if (!p.contains("FILEPATH"))
            {
                throw std::runtime_error("Mesh object is missing FILEPATH.");
            }

            const glm::vec3 translation = readVec3(p, "TRANS", glm::vec3(0.0f));
            const glm::vec3 rotation = readVec3(p, "ROTAT", glm::vec3(0.0f));
            const glm::vec3 scale = readVec3(p, "SCALE", glm::vec3(1.0f));

            const glm::mat4 rootTransform =
                utilityCore::buildTransformationMatrix(
                    translation, rotation, scale);

            loadFromGLTF(toJsonRelativePath(jsonName, p["FILEPATH"]), rootTransform);

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

namespace {
    template<typename T>
    T readNumber(const unsigned char* data)
    {
        T value;
        std::memcpy(&value, data, sizeof(T));
        return value;
    }

    double readComponent(const unsigned char* data, int type, bool normalized)
    {
        switch (type) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
            return normalized ? std::max(-1.0, readNumber<int8_t>(data) / 127.0) : readNumber<int8_t>(data);
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return readNumber<uint8_t>(data) / (normalized ? 255.0 : 1.0);
        case TINYGLTF_COMPONENT_TYPE_SHORT:
            return normalized ? std::max(-1.0, readNumber<int16_t>(data) / 32767.0) : readNumber<int16_t>(data);
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            return readNumber<uint16_t>(data) / (normalized ? 65535.0 : 1.0);
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            return readNumber<uint32_t>(data);
        default:
            return readNumber<float>(data);
        }
    }

    std::vector<double> readAccessor(const tinygltf::Model& model, int id)
    {
        const auto& a = model.accessors[id];
        int components = tinygltf::GetNumComponentsInType(a.type);
        int bytes = tinygltf::GetComponentSizeInBytes(a.componentType);
        std::vector<double> values(a.count * components, 0.0);
        auto dataAt = [&](int viewId, size_t offset) {
            const auto& view = model.bufferViews[viewId];
            return model.buffers[view.buffer].data.data() + view.byteOffset + offset;
            };
        auto readElement = [&](size_t index, const unsigned char* data) {
            for (int c = 0; c < components; ++c)
                values[index * components + c] = readComponent(data + c * bytes, a.componentType, a.normalized);
            };
        if (a.bufferView >= 0) {
            const auto* data = dataAt(a.bufferView, a.byteOffset);
            int stride = a.ByteStride(model.bufferViews[a.bufferView]);
            for (size_t i = 0; i < a.count; ++i)
                readElement(i, data + i * stride);
        }
        if (a.sparse.isSparse) {
            const auto& s = a.sparse;
            const auto* indices = dataAt(s.indices.bufferView, s.indices.byteOffset);
            const auto* data = dataAt(s.values.bufferView, s.values.byteOffset);
            int indexBytes = tinygltf::GetComponentSizeInBytes(s.indices.componentType);
            for (int i = 0; i < s.count; ++i)
                readElement(size_t(readComponent(indices + i * indexBytes, s.indices.componentType, false)),
                    data + i * components * bytes);
        }
        return values;
    }

    WrapMode readWrap(int mode)
    {
        if (mode == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) return WrapMode::ClampToEdge;
        if (mode == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) return WrapMode::MirroredRepeat;
        return WrapMode::Repeat;
    }

    FilterMode readFilter(int mode)
    {
        switch (mode) {
        case TINYGLTF_TEXTURE_FILTER_NEAREST: return FilterMode::Nearest;
        case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST: return FilterMode::NearestMipmapNearest;
        case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST: return FilterMode::LinearMipmapNearest;
        case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR: return FilterMode::NearestMipmapLinear;
        case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR: return FilterMode::LinearMipmapLinear;
        default: return FilterMode::Linear;
        }
    }

    template<typename T>
    TextureRef readTexture(const T& info, int textureOffset)
    {
        TextureRef ref;
        if (info.index < 0) return ref;
        ref.textureId = textureOffset + info.index;
        ref.texCoord = info.texCoord;
        auto it = info.extensions.find("KHR_texture_transform");
        if (it != info.extensions.end()) {
            const auto& ext = it->second;
            glm::vec2 offset(0.0f), scale(1.0f);
            float rotation = 0.0f;
            for (int i = 0; i < 2; ++i) {
                if (ext.Has("offset")) offset[i] = float(ext.Get("offset").Get(i).GetNumberAsDouble());
                if (ext.Has("scale")) scale[i] = float(ext.Get("scale").Get(i).GetNumberAsDouble());
            }
            if (ext.Has("rotation")) rotation = float(ext.Get("rotation").GetNumberAsDouble());
            if (ext.Has("texCoord")) ref.texCoord = ext.Get("texCoord").GetNumberAsInt();
            float c = std::cos(rotation), s = std::sin(rotation);
            ref.uvTransform = glm::mat3(c * scale.x, s * scale.x, 0,
                -s * scale.y, c * scale.y, 0, offset.x, offset.y, 1);
        }
        return ref;
    }
}

void Scene::loadFromGLTF(const std::string& filename, const glm::mat4& rootTransform)
{
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;

    std::string error, warning;
    bool loaded = filename.substr(filename.find_last_of('.')) == ".glb"
        ? loader.LoadBinaryFromFile(&model, &error, &warning, filename)
        : loader.LoadASCIIFromFile(&model, &error, &warning, filename);

    if (!loaded) {
        std::cerr << "Couldn't load " << filename << ": " << error << std::endl;
        return;
    }

    int imageOffset = int(textureImages.size());
    int textureOffset = int(textures.size());
    int materialOffset = int(materials.size());

    for (const auto& image : model.images) {
        TextureImage result;
        result.width = image.width;
        result.height = image.height;
        result.texelOffset = int(texels.size());
        int bytes = image.bits / 8;

        for (int i = 0; i < image.width * image.height; ++i) {
            glm::vec4 pixel(1.0f);
            for (int c = 0; c < image.component; ++c)
                pixel[c] = float(readComponent(image.image.data() + (i * image.component + c) * bytes,
                    image.pixel_type, true));
            if (image.component < 3) {
                pixel.a = image.component == 2 ? pixel.g : 1.0f;
                pixel.g = pixel.b = pixel.r;
            }
            texels.push_back(pixel);
        }
        textureImages.push_back(result);
    }

    for (const auto& texture : model.textures) {
        Texture result;
        result.imageId = texture.source < 0 ? -1 : imageOffset + texture.source;
        if (texture.sampler >= 0) {
            const auto& sampler = model.samplers[texture.sampler];
            result.wrapU = readWrap(sampler.wrapS);
            result.wrapV = readWrap(sampler.wrapT);
            result.minFilter = readFilter(sampler.minFilter);
            result.magFilter = readFilter(sampler.magFilter);
        }
        textures.push_back(result);
    }

    for (const auto& material : model.materials) {
        Material result;
        const auto& pbr = material.pbrMetallicRoughness;
        result.albedo = glm::vec3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2]);
        result.alpha = float(pbr.baseColorFactor[3]);
        result.metalness = float(pbr.metallicFactor);
        result.roughness = float(pbr.roughnessFactor);
        result.emittance = glm::vec3(material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2]);
        auto strength = material.extensions.find("KHR_materials_emissive_strength");
        if (strength != material.extensions.end())
            result.emittance *= float(strength->second.Get("emissiveStrength").GetNumberAsDouble());
        result.baseColorTexture = readTexture(pbr.baseColorTexture, textureOffset);
        result.metallicRoughnessTexture = readTexture(pbr.metallicRoughnessTexture, textureOffset);
        result.normalTexture = readTexture(material.normalTexture, textureOffset);
        result.occlusionTexture = readTexture(material.occlusionTexture, textureOffset);
        result.emissiveTexture = readTexture(material.emissiveTexture, textureOffset);
        result.normalScale = float(material.normalTexture.scale);
        result.occlusionStrength = float(material.occlusionTexture.strength);
        result.alphaMode = material.alphaMode == "MASK" ? AlphaMode::Mask
            : material.alphaMode == "BLEND" ? AlphaMode::Blend : AlphaMode::Opaque;
        result.alphaCutoff = float(material.alphaCutoff);
        result.doubleSided = material.doubleSided;

        result.thinWalled = true;

        auto transmission = material.extensions.find("KHR_materials_transmission");
        if (transmission != material.extensions.end()) {
            const auto& ext = transmission->second;

            if (ext.Has("transmissionFactor"))
                result.transmission =
                float(ext.Get("transmissionFactor").GetNumberAsDouble());

            if (ext.Has("transmissionTexture")) {
                const auto& texture = ext.Get("transmissionTexture");
                tinygltf::TextureInfo info;

                info.index = texture.Get("index").GetNumberAsInt();

                if (texture.Has("texCoord"))
                    info.texCoord = texture.Get("texCoord").GetNumberAsInt();

                if (texture.Has("extensions"))
                    info.extensions =
                    texture.Get("extensions").Get<tinygltf::Value::Object>();

                result.transmissionTexture = readTexture(info, textureOffset);
            }
        }

        auto ior = material.extensions.find("KHR_materials_ior");
        if (ior != material.extensions.end() && ior->second.Has("ior"))
            result.ior = float(ior->second.Get("ior").GetNumberAsDouble());

        auto volume = material.extensions.find("KHR_materials_volume");
        if (volume != material.extensions.end()) {
            const auto& ext = volume->second;

            if (ext.Has("thicknessFactor"))
                result.thinWalled =
                ext.Get("thicknessFactor").GetNumberAsDouble() == 0.0;

            if (ext.Has("attenuationDistance")) {
                glm::vec3 color(1.0f);

                if (ext.Has("attenuationColor"))
                    for (int i = 0; i < 3; ++i)
                        result.attenuationColor[i] = float(
                            ext.Get("attenuationColor").Get(i).GetNumberAsDouble());

                result.attenuationDistance = float(ext.Get("attenuationDistance").GetNumberAsDouble());
            }
        }

        materials.push_back(result);
    }

    int defaultMaterial = -1;
    std::vector<std::vector<std::pair<int, int>>> meshes(model.meshes.size());

    for (size_t m = 0; m < model.meshes.size(); ++m) {
        for (const auto& source : model.meshes[m].primitives) {
            int mode = source.mode < 0 ? TINYGLTF_MODE_TRIANGLES : source.mode;
            if (mode != TINYGLTF_MODE_TRIANGLES && mode != TINYGLTF_MODE_TRIANGLE_STRIP &&
                mode != TINYGLTF_MODE_TRIANGLE_FAN) continue;

            MeshPrimitive primitive;
            primitive.vertexOffset = int(vertices.size());
            primitive.vertexCount = int(model.accessors[source.attributes.at("POSITION")].count);
            primitive.triangleOffset = int(triangles.size());
            primitive.texcoordOffset = int(texcoords.size());
            vertices.resize(vertices.size() + primitive.vertexCount);

            for (const auto& attribute : source.attributes) {
                const auto& name = attribute.first;
                const auto& accessor = model.accessors[attribute.second];
                int components = tinygltf::GetNumComponentsInType(accessor.type);
                auto data = readAccessor(model, attribute.second);

                if (name == "NORMAL") primitive.hasNormals = true;
                if (name == "TANGENT") primitive.hasTangents = true;
                if (name.compare(0, 9, "TEXCOORD_") == 0) {
                    int set = std::stoi(name.substr(9));
                    primitive.texcoordSetCount = std::max(primitive.texcoordSetCount, set + 1);
                    texcoords.resize(primitive.texcoordOffset + primitive.texcoordSetCount * primitive.vertexCount, glm::vec2(0));
                    for (int i = 0; i < primitive.vertexCount; ++i)
                        texcoords[primitive.texcoordOffset + set * primitive.vertexCount + i] = glm::vec2(data[i * 2], data[i * 2 + 1]);
                }
                else {
                    for (int i = 0; i < primitive.vertexCount; ++i) {
                        auto& vertex = vertices[primitive.vertexOffset + i];
                        for (int c = 0; c < components; ++c) {
                            float value = float(data[i * components + c]);
                            if (name == "POSITION") vertex.position[c] = value;
                            else if (name == "NORMAL") vertex.normal[c] = value;
                            else if (name == "TANGENT") vertex.tangent[c] = value;
                            else if (name == "COLOR_0") vertex.color[c] = value;
                        }
                    }
                }
            }
            std::vector<double> indices(primitive.vertexCount);
            if (source.indices >= 0) indices = readAccessor(model, source.indices);
            else for (int i = 0; i < primitive.vertexCount; ++i) indices[i] = i;

            auto addTriangle = [&](size_t a, size_t b, size_t c) {
                triangles.push_back(Triangle{ {
                        uint32_t(primitive.vertexOffset + indices[a]),
                        uint32_t(primitive.vertexOffset + indices[b]),
                        uint32_t(primitive.vertexOffset + indices[c])} });
                };

            if (mode == TINYGLTF_MODE_TRIANGLES) {
                for (size_t i = 0; i + 2 < indices.size(); i += 3) addTriangle(i, i + 1, i + 2);
            }
            else {
                for (size_t i = 2; i < indices.size(); ++i) {
                    if (mode == TINYGLTF_MODE_TRIANGLE_FAN) addTriangle(0, i - 1, i);
                    else if (i % 2) addTriangle(i - 1, i - 2, i);
                    else addTriangle(i - 2, i - 1, i);
                }
            }
            primitive.triangleCount = int(triangles.size()) - primitive.triangleOffset;

            if (source.material < 0 && defaultMaterial < 0) {
                defaultMaterial = int(materials.size());
                materials.emplace_back();
            }

            int materialId = source.material < 0 ? defaultMaterial : materialOffset + source.material;
            meshes[m].push_back({ int(primitives.size()), materialId });
            primitives.push_back(primitive);
        }
    }

    std::function<void(int, const glm::mat4&)> visit = [&](int id, const glm::mat4& parent) {
        const auto& node = model.nodes[id];
        glm::mat4 local(1.0f);
        if (!node.matrix.empty()) {
            for (int i = 0; i < 16; ++i) local[i / 4][i % 4] = float(node.matrix[i]);
        }
        else {
            glm::vec3 translation(0), scale(1);
            glm::quat rotation(1, 0, 0, 0);

            for (int i = 0; i < 3; ++i) {
                if (!node.translation.empty()) translation[i] = float(node.translation[i]);
                if (!node.scale.empty()) scale[i] = float(node.scale[i]);
            }

            if (!node.rotation.empty())
                rotation = glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
            local = glm::translate(glm::mat4(1), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1), scale);
        }
        glm::mat4 world = parent * local;
        if (node.mesh >= 0) {
            for (const auto& entry : meshes[node.mesh]) {
                Geom geom{};
                geom.type = GeomType::Mesh;
                geom.primitiveId = entry.first;
                geom.materialid = entry.second;
                geom.scale = glm::vec3(1);
                geom.transform = world;
                geom.inverseTransform = glm::inverse(world);
                geom.invTranspose = glm::transpose(geom.inverseTransform);
                geoms.push_back(geom);
            }
        }
        for (int child : node.children) visit(child, world);
        };

    if (!model.scenes.empty()) {
        const auto& scene = model.scenes[model.defaultScene < 0 ? 0 : model.defaultScene];
        for (int node : scene.nodes) visit(node, rootTransform);
    }
    else {
        std::vector<bool> child(model.nodes.size(), false);

        for (const auto& node : model.nodes)
            for (int id : node.children) child[id] = true;

        for (size_t i = 0; i < model.nodes.size(); ++i)
            if (!child[i]) visit(int(i), rootTransform);
    }
}
