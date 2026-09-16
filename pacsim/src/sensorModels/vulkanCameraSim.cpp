#include "sensorModels/vulkanCameraSim.hpp"
#include "logger.hpp"

#include <Eigen/Geometry>
#include <assimp/Importer.hpp>
#include <assimp/config.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>
#include <stb_image.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace
{
constexpr float kPi = 3.14159265358979323846f;
constexpr float kCarHeadingOffsetRad = -1.57079632679f;
constexpr float kCarOriginForwardOffset = 0.9f;
constexpr float kShadowDistance = 20.0f;
constexpr float kShadowNear = 3.0f;
constexpr float kShadowFar = 40.0f;
constexpr uint32_t kShadowMapSize = 4096U;
constexpr float kConeGroundClearance = 0.01f;
constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr uint32_t kMaxDrawConstants = 16384U;
const int kModuleAddressMarker = 0;

struct TextureImage
{
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> data;

    bool empty() const { return data.empty() || width <= 0 || height <= 0 || channels <= 0; }
};

bool fileExists(const std::string& path)
{
    std::ifstream file(path);
    return file.good();
}

void requireExistingFile(const std::string& name, const std::string& path)
{
    if (path.empty() || !fileExists(path))
    {
        throw std::runtime_error("VulkanCameraSim requires existing '" + name + "': " + path);
    }
}

std::string parentDir(const std::string& path)
{
    const size_t slash = path.find_last_of("/\\");
    return (slash == std::string::npos) ? std::string() : path.substr(0, slash);
}

char nativePathSeparator()
{
#if defined(_WIN32)
    return '\\';
#else
    return '/';
#endif
}

bool isAbsolutePath(const std::string& path)
{
    if (path.empty())
    {
        return false;
    }
    if (path[0] == '/' || path[0] == '\\')
    {
        return true;
    }
    return path.size() >= 2 && std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':';
}

std::string joinPath(const std::string& lhs, const std::string& rhs)
{
    if (lhs.empty() || isAbsolutePath(rhs))
    {
        return rhs;
    }
    if (!lhs.empty() && (lhs.back() == '/' || lhs.back() == '\\'))
    {
        return lhs + rhs;
    }
    return lhs + nativePathSeparator() + rhs;
}

std::string getEnvVar(const char* name)
{
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

bool setEnvVar(const char* name, const std::string& value)
{
#if defined(_WIN32)
    return _putenv_s(name, value.c_str()) == 0;
#else
    return setenv(name, value.c_str(), 1) == 0;
#endif
}

bool unsetEnvVar(const char* name)
{
#if defined(_WIN32)
    return _putenv_s(name, "") == 0;
#else
    return unsetenv(name) == 0;
#endif
}

class ScopedEnvOverride
{
public:
    ScopedEnvOverride(const char* nameIn, const std::string& value, bool enable)
        : name(nameIn)
        , oldValue(getEnvVar(nameIn))
        , hadValue(std::getenv(nameIn) != nullptr)
        , active(enable)
    {
        if (active)
        {
            setEnvVar(name.c_str(), value);
        }
    }

    ~ScopedEnvOverride()
    {
        if (!active)
        {
            return;
        }
        if (hadValue)
        {
            setEnvVar(name.c_str(), oldValue);
        }
        else
        {
            unsetEnvVar(name.c_str());
        }
    }

    ScopedEnvOverride(const ScopedEnvOverride&) = delete;
    ScopedEnvOverride& operator=(const ScopedEnvOverride&) = delete;

private:
    std::string name;
    std::string oldValue;
    bool hadValue;
    bool active;
};

class ScopedEnvRemoval
{
public:
    explicit ScopedEnvRemoval(const char* nameIn)
        : name(nameIn)
        , oldValue(getEnvVar(nameIn))
        , hadValue(std::getenv(nameIn) != nullptr)
    {
        unsetEnvVar(name.c_str());
    }

    ~ScopedEnvRemoval()
    {
        if (hadValue)
        {
            setEnvVar(name.c_str(), oldValue);
        }
    }

    ScopedEnvRemoval(const ScopedEnvRemoval&) = delete;
    ScopedEnvRemoval& operator=(const ScopedEnvRemoval&) = delete;

private:
    std::string name;
    std::string oldValue;
    bool hadValue;
};

std::string currentModulePath()
{
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&kModuleAddressMarker), &module) == 0)
    {
        return {};
    }
    std::vector<char> buffer(1024);
    for (;;)
    {
        const DWORD length = GetModuleFileNameA(module, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0)
        {
            return {};
        }
        if (length < buffer.size() - 1)
        {
            return std::string(buffer.data(), buffer.data() + length);
        }
        buffer.resize(buffer.size() * 2U);
    }
#else
    Dl_info info {};
    if (dladdr(static_cast<const void*>(&kModuleAddressMarker), &info) != 0 && info.dli_fname != nullptr)
    {
        return std::string(info.dli_fname);
    }
    return {};
#endif
}

std::string currentWorkingDirectory()
{
#if defined(_WIN32)
    DWORD required = GetCurrentDirectoryA(0, nullptr);
    if (required == 0)
    {
        return {};
    }
    std::vector<char> buffer(required);
    DWORD length = GetCurrentDirectoryA(required, buffer.data());
    return length == 0 ? std::string() : std::string(buffer.data(), buffer.data() + length);
#else
    std::vector<char> buffer(1024);
    for (;;)
    {
        if (getcwd(buffer.data(), buffer.size()) != nullptr)
        {
            return std::string(buffer.data());
        }
        if (errno != ERANGE)
        {
            return {};
        }
        buffer.resize(buffer.size() * 2U);
    }
#endif
}

std::string toLowerAscii(const std::string& value)
{
    std::string ret;
    ret.reserve(value.size());
    for (char c : value)
    {
        ret.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return ret;
}

bool containsCaseInsensitive(const std::string& value, const std::string& needle)
{
    return toLowerAscii(value).find(toLowerAscii(needle)) != std::string::npos;
}

bool isTruthyEnvVar(const char* name)
{
    const std::string value = toLowerAscii(getEnvVar(name));
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

void appendUnique(std::vector<std::string>& values, std::unordered_set<std::string>& seen, const std::string& value,
    bool requireFile = true)
{
    if (value.empty())
    {
        return;
    }
    if (requireFile && !fileExists(value))
    {
        return;
    }
    if (seen.insert(value).second)
    {
        values.push_back(value);
    }
}

std::vector<std::string> swiftShaderIcdCandidates()
{
    std::vector<std::string> candidates;
    std::unordered_set<std::string> seen;

    appendUnique(candidates, seen, getEnvVar("PACSIM_SWIFTSHADER_ICD"), false);
    appendUnique(candidates, seen, getEnvVar("SWIFTSHADER_ICD_FILENAMES"), false);

    const std::string moduleDir = parentDir(currentModulePath());
    if (!moduleDir.empty())
    {
        appendUnique(candidates, seen, joinPath(joinPath(moduleDir, "swiftshader"), "vk_swiftshader_icd.json"));
        appendUnique(candidates, seen, joinPath(moduleDir, "vk_swiftshader_icd.json"));
        const std::string parent = parentDir(moduleDir);
        if (!parent.empty())
        {
            appendUnique(candidates, seen, joinPath(joinPath(parent, "swiftshader"), "vk_swiftshader_icd.json"));
        }
    }

    const std::string cwd = currentWorkingDirectory();
    if (!cwd.empty())
    {
        appendUnique(candidates, seen, joinPath(joinPath(cwd, "swiftshader"), "vk_swiftshader_icd.json"));
        appendUnique(candidates, seen, joinPath(cwd, "vk_swiftshader_icd.json"));
    }

#ifdef PACSIM_SWIFTSHADER_ICD_JSON
    appendUnique(candidates, seen, PACSIM_SWIFTSHADER_ICD_JSON);
#endif

    const std::string vulkanSdk = getEnvVar("VULKAN_SDK");
    if (!vulkanSdk.empty())
    {
        appendUnique(candidates, seen, joinPath(joinPath(vulkanSdk, "Bin"), "vk_swiftshader_icd.json"));
        appendUnique(candidates, seen, joinPath(joinPath(vulkanSdk, "bin"), "vk_swiftshader_icd.json"));
        appendUnique(candidates, seen, joinPath(joinPath(joinPath(vulkanSdk, "share"), "vulkan"), "icd.d/vk_swiftshader_icd.json"));
        appendUnique(candidates, seen, joinPath(joinPath(joinPath(vulkanSdk, "etc"), "vulkan"), "icd.d/vk_swiftshader_icd.json"));
    }

#if defined(_WIN32)
    appendUnique(candidates, seen, "C:\\SwiftShader\\vk_swiftshader_icd.json");
#else
    appendUnique(candidates, seen, "/usr/share/vulkan/icd.d/vk_swiftshader_icd.json");
    appendUnique(candidates, seen, "/usr/local/share/vulkan/icd.d/vk_swiftshader_icd.json");
    appendUnique(candidates, seen, "/etc/vulkan/icd.d/vk_swiftshader_icd.json");
#endif

    return candidates;
}

bool forceSwiftShader()
{
#ifdef PACSIM_FORCE_SWIFTSHADER
    return true;
#else
    return isTruthyEnvVar("PACSIM_FORCE_SWIFTSHADER");
#endif
}

void warnSwiftShaderSoftwareRendering(const std::string& deviceName)
{
    static std::once_flag warningOnce;
    std::call_once(warningOnce, [&]() {
        Logger logger;
        logger.logWarning("VulkanCameraSim is using SwiftShader software rendering device '" + deviceName
            + "'. Rendering is CPU-backed and camera performance will be degraded compared with hardware Vulkan.");
    });
}

VkSampleCountFlagBits chooseMsaaSampleCount(const VkPhysicalDeviceProperties& properties)
{
    const VkSampleCountFlags supportedCounts =
        properties.limits.framebufferColorSampleCounts & properties.limits.framebufferDepthSampleCounts;
    if ((supportedCounts & VK_SAMPLE_COUNT_4_BIT) != 0)
    {
        return VK_SAMPLE_COUNT_4_BIT;
    }
    if ((supportedCounts & VK_SAMPLE_COUNT_2_BIT) != 0)
    {
        return VK_SAMPLE_COUNT_2_BIT;
    }
    return VK_SAMPLE_COUNT_1_BIT;
}

TextureImage makeTextureImage(unsigned char* pixels, int width, int height, int channels)
{
    TextureImage image;
    if (pixels == nullptr || width <= 0 || height <= 0 || channels <= 0)
    {
        if (pixels != nullptr)
        {
            stbi_image_free(pixels);
        }
        return image;
    }
    image.width = width;
    image.height = height;
    image.channels = channels;
    const size_t byteCount = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels);
    image.data.assign(pixels, pixels + byteCount);
    stbi_image_free(pixels);
    return image;
}

TextureImage decodeTextureImage(const unsigned char* bytes, size_t byteCount)
{
    if (bytes == nullptr || byteCount == 0 || byteCount > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return {};
    }
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* pixels = stbi_load_from_memory(bytes, static_cast<int>(byteCount), &width, &height, &channels, 0);
    return makeTextureImage(pixels, width, height, channels);
}

TextureImage loadTextureImage(const std::string& path)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 0);
    return makeTextureImage(pixels, width, height, channels);
}

TextureImage decodeEmbeddedTexture(const aiTexture* texture)
{
    if (texture == nullptr)
    {
        return {};
    }
    if (texture->mHeight == 0)
    {
        return decodeTextureImage(reinterpret_cast<const unsigned char*>(texture->pcData), texture->mWidth);
    }
    TextureImage image;
    image.width = static_cast<int>(texture->mWidth);
    image.height = static_cast<int>(texture->mHeight);
    image.channels = 4;
    image.data.resize(static_cast<size_t>(image.width) * static_cast<size_t>(image.height) * 4U);
    for (size_t i = 0; i < static_cast<size_t>(image.width) * static_cast<size_t>(image.height); ++i)
    {
        image.data[i * 4U + 0U] = texture->pcData[i].r;
        image.data[i * 4U + 1U] = texture->pcData[i].g;
        image.data[i * 4U + 2U] = texture->pcData[i].b;
        image.data[i * 4U + 3U] = texture->pcData[i].a;
    }
    return image;
}

float wrap01(float v)
{
    v = std::fmod(v, 1.0f);
    return v < 0.0f ? v + 1.0f : v;
}

Eigen::Vector3f sampleTextureRgb(const TextureImage& image, const Eigen::Vector2f& uv, const Eigen::Vector3f& fallback)
{
    if (image.empty())
    {
        return fallback;
    }
    const float u = wrap01(uv.x());
    const float v = wrap01(uv.y());
    const int x = std::clamp(static_cast<int>(u * static_cast<float>(image.width)), 0, image.width - 1);
    const int y = std::clamp(static_cast<int>((1.0f - v) * static_cast<float>(image.height)), 0, image.height - 1);
    const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(image.width) + static_cast<size_t>(x))
        * static_cast<size_t>(image.channels);
    if (image.channels >= 3)
    {
        return Eigen::Vector3f(
            static_cast<float>(image.data[idx + 0U]) / 255.0f,
            static_cast<float>(image.data[idx + 1U]) / 255.0f,
            static_cast<float>(image.data[idx + 2U]) / 255.0f);
    }
    if (image.channels == 1)
    {
        const float g = static_cast<float>(image.data[idx]) / 255.0f;
        return Eigen::Vector3f(g, g, g);
    }
    return fallback;
}

std::vector<unsigned char> textureToRgba(const TextureImage& image)
{
    if (image.empty())
    {
        return {};
    }
    std::vector<unsigned char> rgba(static_cast<size_t>(image.width) * static_cast<size_t>(image.height) * 4U, 255);
    for (size_t i = 0; i < static_cast<size_t>(image.width) * static_cast<size_t>(image.height); ++i)
    {
        if (image.channels >= 3)
        {
            rgba[i * 4U + 0U] = image.data[i * static_cast<size_t>(image.channels) + 0U];
            rgba[i * 4U + 1U] = image.data[i * static_cast<size_t>(image.channels) + 1U];
            rgba[i * 4U + 2U] = image.data[i * static_cast<size_t>(image.channels) + 2U];
            rgba[i * 4U + 3U] = (image.channels >= 4) ? image.data[i * static_cast<size_t>(image.channels) + 3U] : 255;
        }
        else if (image.channels == 1)
        {
            const unsigned char g = image.data[i];
            rgba[i * 4U + 0U] = g;
            rgba[i * 4U + 1U] = g;
            rgba[i * 4U + 2U] = g;
        }
    }
    return rgba;
}

Eigen::Vector3f parseVec3FromString(const char* value, const Eigen::Vector3f& fallback)
{
    if (value == nullptr)
    {
        return fallback;
    }
    Eigen::Vector3f parsed = fallback;
    std::istringstream stream(value);
    if (!(stream >> parsed.x() >> parsed.y() >> parsed.z()))
    {
        return fallback;
    }
    return parsed;
}

Eigen::Matrix4f makeUrdfTransform(const Eigen::Vector3f& xyz, const Eigen::Vector3f& rpy)
{
    const Eigen::Matrix3f rotation =
        (Eigen::AngleAxisf(rpy.z(), Eigen::Vector3f::UnitZ())
            * Eigen::AngleAxisf(rpy.y(), Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf(rpy.x(), Eigen::Vector3f::UnitX()))
            .toRotationMatrix();
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = rotation;
    transform.block<3, 1>(0, 3) = xyz;
    return transform;
}

Eigen::Matrix4f makeScaleMatrix(const Eigen::Vector3f& scale)
{
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 0) = scale.x();
    transform(1, 1) = scale.y();
    transform(2, 2) = scale.z();
    return transform;
}

std::string resolveUrdfMeshPath(const std::string& meshFilename, const std::string& xacroPath)
{
    auto fileNameOnly = [](const std::string& path) -> std::string {
        const size_t slash = path.find_last_of('/');
        return slash == std::string::npos ? path : path.substr(slash + 1);
    };
    auto selectExistingCandidate = [](const std::vector<std::string>& candidates) -> std::string {
        for (const auto& candidate : candidates)
        {
            if (fileExists(candidate))
            {
                return candidate;
            }
        }
        return std::string();
    };

    std::string normalized = meshFilename;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](char c) { return c == '\\' ? '/' : c; });
    if (normalized.empty())
    {
        return std::string();
    }
    if (normalized.rfind("package://", 0) == 0)
    {
        const size_t packageStart = std::strlen("package://");
        const size_t packageEnd = normalized.find('/', packageStart);
        if (packageEnd == std::string::npos)
        {
            return std::string();
        }
        const std::string packageRelativePath = normalized.substr(packageEnd + 1);
        const std::string meshBaseName = fileNameOnly(packageRelativePath);
        const std::string packageRoot = parentDir(parentDir(xacroPath));
        const std::string directPath = joinPath(packageRoot, packageRelativePath);
        const std::string tiresFallbackPath = joinPath(packageRoot, "urdf/Tires/" + meshBaseName);
        return selectExistingCandidate({ directPath, tiresFallbackPath, directPath });
    }
    if (normalized[0] == '/')
    {
        return normalized;
    }
    const std::string relativePath = joinPath(parentDir(xacroPath), normalized);
    const std::string relativeTiresFallback = joinPath(parentDir(xacroPath), "Tires/" + fileNameOnly(normalized));
    return selectExistingCandidate({ relativePath, relativeTiresFallback, relativePath });
}

std::string normalizeSeparators(std::string in)
{
    std::transform(in.begin(), in.end(), in.begin(), [](char c) { return c == '\\' ? '/' : c; });
    return in;
}

std::string toLowerCopy(std::string in)
{
    std::transform(in.begin(), in.end(), in.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return in;
}

std::string canonicalizeCarPartName(std::string name)
{
    const std::string lowered = toLowerCopy(name);
    if (lowered == "steering_wheel")
    {
        return "Steering_Wheel";
    }

    const std::string insideToken = "_inside";
    const std::string outsideToken = "_outside";
    size_t insidePos = lowered.find(insideToken);
    if (insidePos != std::string::npos)
    {
        name.replace(insidePos, insideToken.size(), "_Inside");
        return name;
    }
    size_t outsidePos = lowered.find(outsideToken);
    if (outsidePos != std::string::npos)
    {
        name.replace(outsidePos, outsideToken.size(), "_Outside");
        return name;
    }
    return name;
}

void expandBoundsWithTransformedAabb(const Eigen::Vector3f& inputMin, const Eigen::Vector3f& inputMax,
    const Eigen::Matrix4f& transform, Eigen::Vector3f& outputMin, Eigen::Vector3f& outputMax, bool& hasBounds)
{
    for (int ix = 0; ix < 2; ++ix)
    {
        for (int iy = 0; iy < 2; ++iy)
        {
            for (int iz = 0; iz < 2; ++iz)
            {
                const float x = (ix == 0) ? inputMin.x() : inputMax.x();
                const float y = (iy == 0) ? inputMin.y() : inputMax.y();
                const float z = (iz == 0) ? inputMin.z() : inputMax.z();
                const Eigen::Vector3f p = (transform * Eigen::Vector4f(x, y, z, 1.0f)).head<3>();
                if (!hasBounds)
                {
                    outputMin = p;
                    outputMax = p;
                    hasBounds = true;
                }
                else
                {
                    outputMin = outputMin.cwiseMin(p);
                    outputMax = outputMax.cwiseMax(p);
                }
            }
        }
    }
}

YAML::Node mergeSensorNode(const YAML::Node& item)
{
    YAML::Node merged(YAML::NodeType::Map);
    if (item["sensor"] && item["sensor"].IsMap())
    {
        const YAML::Node sensor = item["sensor"];
        for (auto it = sensor.begin(); it != sensor.end(); ++it)
        {
            merged[it->first.Scalar()] = it->second;
        }
    }
    for (auto it = item.begin(); it != item.end(); ++it)
    {
        const std::string key = it->first.Scalar();
        if (key != "sensor" && !merged[key])
        {
            merged[key] = it->second;
        }
    }
    return merged;
}

std::string vkResultToString(VkResult result)
{
    switch (result)
    {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    default: return "VkResult(" + std::to_string(static_cast<int>(result)) + ")";
    }
}

void checkVk(VkResult result, const std::string& context)
{
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("VulkanCameraSim " + context + ": " + vkResultToString(result));
    }
}

enum class VulkanDevicePreference
{
    HardwareOnly,
    SwiftShaderOnly,
};

struct VulkanDeviceChoice
{
    VkPhysicalDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkPhysicalDeviceProperties properties {};
};

const char* deviceTypeName(VkPhysicalDeviceType type)
{
    switch (type)
    {
    case VK_PHYSICAL_DEVICE_TYPE_OTHER: return "other";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
    default: return "unknown";
    }
}

bool findGraphicsQueueFamily(VkPhysicalDevice device, uint32_t& outQueueFamily)
{
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, families.data());
    for (uint32_t i = 0; i < queueFamilyCount; ++i)
    {
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
        {
            outQueueFamily = i;
            return true;
        }
    }
    return false;
}

int scorePhysicalDevice(const VkPhysicalDeviceProperties& properties)
{
    switch (properties.deviceType)
    {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 500;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 400;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 300;
    case VK_PHYSICAL_DEVICE_TYPE_OTHER: return 200;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return 100;
    default: return 0;
    }
}

bool selectPhysicalDevice(
    VkInstance selectedInstance, VulkanDevicePreference preference, VulkanDeviceChoice& outChoice, std::string& error)
{
    uint32_t deviceCount = 0;
    VkResult result = vkEnumeratePhysicalDevices(selectedInstance, &deviceCount, nullptr);
    if (result != VK_SUCCESS)
    {
        error = "enumerate physical device count failed: " + vkResultToString(result);
        return false;
    }
    if (deviceCount == 0)
    {
        error = "no Vulkan physical devices were reported";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    result = vkEnumeratePhysicalDevices(selectedInstance, &deviceCount, devices.data());
    if (result != VK_SUCCESS)
    {
        error = "enumerate physical devices failed: " + vkResultToString(result);
        return false;
    }

    int bestScore = std::numeric_limits<int>::min();
    VulkanDeviceChoice bestChoice;
    std::vector<std::string> skipped;
    for (VkPhysicalDevice candidate : devices)
    {
        VkPhysicalDeviceProperties properties {};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        const std::string deviceName(properties.deviceName);
        const bool isSwiftShader = containsCaseInsensitive(deviceName, "swiftshader");
        const bool isCpu = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;

        if (preference == VulkanDevicePreference::HardwareOnly && (isCpu || isSwiftShader))
        {
            skipped.push_back(deviceName + " (" + deviceTypeName(properties.deviceType) + ")");
            continue;
        }
        if (preference == VulkanDevicePreference::SwiftShaderOnly && !isSwiftShader)
        {
            skipped.push_back(deviceName + " (" + deviceTypeName(properties.deviceType) + ")");
            continue;
        }

        uint32_t queueFamily = 0;
        if (!findGraphicsQueueFamily(candidate, queueFamily))
        {
            skipped.push_back(deviceName + " (no graphics queue)");
            continue;
        }

        int score = scorePhysicalDevice(properties);
        if (bestChoice.device == VK_NULL_HANDLE || score > bestScore)
        {
            bestScore = score;
            bestChoice.device = candidate;
            bestChoice.queueFamily = queueFamily;
            bestChoice.properties = properties;
        }
    }

    if (bestChoice.device == VK_NULL_HANDLE)
    {
        error = preference == VulkanDevicePreference::HardwareOnly
            ? "no hardware Vulkan GPU with a graphics queue was found"
            : "no SwiftShader Vulkan device with a graphics queue was found";
        if (!skipped.empty())
        {
            error += "; skipped devices:";
            for (const std::string& name : skipped)
            {
                error += " " + name + ";";
            }
        }
        return false;
    }

    outChoice = bestChoice;
    return true;
}

bool createInstanceAndSelectDevice(VulkanDevicePreference preference, const std::string& swiftShaderIcd,
    VkInstance& outInstance, VkPhysicalDevice& outPhysicalDevice, uint32_t& outGraphicsQueueFamily,
    std::string& outDeviceName, std::string& error)
{
    ScopedEnvOverride icdOverride("VK_ICD_FILENAMES", swiftShaderIcd, !swiftShaderIcd.empty());

    VkApplicationInfo appInfo {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "pacsim VulkanCameraSim";
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instanceInfo {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    VkInstance candidateInstance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&instanceInfo, nullptr, &candidateInstance);
    if (result != VK_SUCCESS)
    {
        error = "create instance failed: " + vkResultToString(result);
        return false;
    }

    VulkanDeviceChoice choice;
    if (!selectPhysicalDevice(candidateInstance, preference, choice, error))
    {
        vkDestroyInstance(candidateInstance, nullptr);
        return false;
    }

    outInstance = candidateInstance;
    outPhysicalDevice = choice.device;
    outGraphicsQueueFamily = choice.queueFamily;
    outDeviceName = choice.properties.deviceName;
    return true;
}
}

VulkanCameraSim::CameraComponent::CameraComponent()
    : projectionFovYRad(2.0f * std::atan((5.61e-3f) / (2.0f * 4.5e-3f)))
    , projectionNearClip(0.1f)
    , projectionFarClip(140.0f)
    , sensorEnabled(true)
    , sensorRateHzValue(10.0f)
    , sensorDelayMeanSecValue(0.0f)
    , sensorNameValue("camera")
{
}

void VulkanCameraSim::CameraComponent::setMount(const CameraMount& mount) { mountConfig = mount; }
const VulkanCameraSim::CameraMount& VulkanCameraSim::CameraComponent::mount() const { return mountConfig; }
void VulkanCameraSim::CameraComponent::setPerspective(float fovYRadIn, float nearClipIn, float farClipIn)
{
    if (!(nearClipIn > 0.0f && farClipIn > nearClipIn))
    {
        throw std::invalid_argument("VulkanCameraSim camera perspective must satisfy 0 < nearClip < farClip");
    }
    projectionFovYRad = fovYRadIn;
    projectionNearClip = nearClipIn;
    projectionFarClip = farClipIn;
}
float VulkanCameraSim::CameraComponent::fovYRad() const { return projectionFovYRad; }
float VulkanCameraSim::CameraComponent::nearClip() const { return projectionNearClip; }
float VulkanCameraSim::CameraComponent::farClip() const { return projectionFarClip; }
void VulkanCameraSim::CameraComponent::setEnabled(bool enabled) { sensorEnabled = enabled; }
bool VulkanCameraSim::CameraComponent::enabled() const { return sensorEnabled; }
void VulkanCameraSim::CameraComponent::setSensorRateHz(float rateHz) { sensorRateHzValue = rateHz; }
float VulkanCameraSim::CameraComponent::sensorRateHz() const { return sensorRateHzValue; }
void VulkanCameraSim::CameraComponent::setSensorDelayMean(float delayMeanSec) { sensorDelayMeanSecValue = delayMeanSec; }
float VulkanCameraSim::CameraComponent::sensorDelayMean() const { return sensorDelayMeanSecValue; }
void VulkanCameraSim::CameraComponent::setSensorName(const std::string& name) { sensorNameValue = name; }
const std::string& VulkanCameraSim::CameraComponent::sensorName() const { return sensorNameValue; }
void VulkanCameraSim::CameraComponent::setIntrinsics(const CameraIntrinsics& intrinsics) { intrinsicsConfig = intrinsics; }
const VulkanCameraSim::CameraIntrinsics& VulkanCameraSim::CameraComponent::intrinsics() const { return intrinsicsConfig; }

VulkanCameraSim::VulkanCameraSim(
    int width, int height, const std::string& modelRoot, const std::string& cameraConfigPath,
    const std::string& carXacroPath)
    : widthPx(width)
    , heightPx(height)
    , nearClip(0.1f)
    , farClip(140.0f)
    , fovYRad(2.0f * std::atan((5.61e-3f) / (2.0f * 4.5e-3f)))
    , modelRootPath(modelRoot)
    , cameraConfigPathValue(cameraConfigPath)
    , carXacroPathOverride(carXacroPath)
    , shadowsEnabled(true)
{
    if (!cameraConfigPath.empty())
    {
        loadCameraConfig(cameraConfigPath);
    }
    if (widthPx <= 0 || heightPx <= 0)
    {
        throw std::runtime_error("VulkanCameraSim: width/height must be > 0 unless camera config provides them");
    }
    initializeVulkan();
    initializeFramebuffer();
    initializePipeline();
    updateTrackMesh();
}

VulkanCameraSim::~VulkanCameraSim()
{
    destroyVulkan();
}

void VulkanCameraSim::initializeVulkan()
{
    // This renderer never creates a window-system surface. Some vendor ICDs,
    // particularly NVIDIA's GLX-backed Linux manifest, still probe an inherited
    // DISPLAY while vkCreateInstance loads the driver. Temporarily hiding the
    // window-server variables prevents authentication warnings and delays on
    // headless machines without changing hardware Vulkan device selection.
    ScopedEnvRemoval x11Display("DISPLAY");
    ScopedEnvRemoval waylandDisplay("WAYLAND_DISPLAY");

    std::vector<std::string> attemptErrors;
    std::string selectedDeviceName;
    const bool swiftShaderForced = forceSwiftShader();

    auto trySelect = [&](VulkanDevicePreference preference, const std::string& icdPath,
                         const std::string& label) -> bool {
        std::string error;
        if (createInstanceAndSelectDevice(
                preference, icdPath, instance, physicalDevice, graphicsQueueFamily, selectedDeviceName, error))
        {
            return true;
        }
        attemptErrors.push_back(label + ": " + error);
        instance = VK_NULL_HANDLE;
        physicalDevice = VK_NULL_HANDLE;
        graphicsQueueFamily = 0;
        return false;
    };

    bool selectedDevice = false;
#if !defined(__APPLE__) && !defined(PACSIM_FORCE_SWIFTSHADER)
    if (!swiftShaderForced)
    {
        selectedDevice = trySelect(VulkanDevicePreference::HardwareOnly, std::string(), "hardware Vulkan");
    }
#endif
    if (!selectedDevice)
    {
        trySelect(VulkanDevicePreference::SwiftShaderOnly, std::string(), "registered SwiftShader Vulkan");
        if (physicalDevice == VK_NULL_HANDLE)
        {
            for (const std::string& icdPath : swiftShaderIcdCandidates())
            {
                if (trySelect(VulkanDevicePreference::SwiftShaderOnly, icdPath, "SwiftShader ICD '" + icdPath + "'"))
                {
                    break;
                }
            }
        }
    }

    if (physicalDevice == VK_NULL_HANDLE)
    {
        std::string message;
        if (swiftShaderForced)
        {
            message = "VulkanCameraSim could not initialize SwiftShader. SwiftShader is configured as required.";
        }
#if defined(__APPLE__)
        else
        {
            message = "VulkanCameraSim could not initialize SwiftShader. macOS builds require SwiftShader for Vulkan rendering.";
        }
#else
        else
        {
            message = "VulkanCameraSim could not initialize a hardware Vulkan GPU or SwiftShader software fallback.";
        }
#endif
        message +=
            " Install/bundle SwiftShader and set PACSIM_SWIFTSHADER_ICD or configure CMake with "
            "-DPACSIM_SWIFTSHADER_ICD_JSON=/path/to/vk_swiftshader_icd.json.";
        if (!attemptErrors.empty())
        {
            message += " Attempts:";
            for (const std::string& error : attemptErrors)
            {
                message += " [" + error + "]";
            }
        }
        throw std::runtime_error(message);
    }

    VkPhysicalDeviceFeatures supportedFeatures {};
    vkGetPhysicalDeviceFeatures(physicalDevice, &supportedFeatures);
    VkPhysicalDeviceProperties deviceProperties {};
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    if (containsCaseInsensitive(deviceProperties.deviceName, "swiftshader"))
    {
        warnSwiftShaderSoftwareRendering(deviceProperties.deviceName);
    }
    if (deviceProperties.limits.maxPushConstantsSize < sizeof(uint32_t))
    {
        throw std::runtime_error("VulkanCameraSim requires at least " + std::to_string(sizeof(uint32_t))
            + " bytes of push constants, device supports "
            + std::to_string(deviceProperties.limits.maxPushConstantsSize));
    }
    samplerAnisotropyEnabled = supportedFeatures.samplerAnisotropy == VK_TRUE;
    samplerMaxAnisotropy = samplerAnisotropyEnabled ? std::min(8.0f, deviceProperties.limits.maxSamplerAnisotropy) : 1.0f;
    msaaSamples = chooseMsaaSampleCount(deviceProperties);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo {};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = graphicsQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo deviceInfo {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    VkPhysicalDeviceFeatures enabledFeatures {};
    enabledFeatures.samplerAnisotropy = samplerAnisotropyEnabled ? VK_TRUE : VK_FALSE;
    deviceInfo.pEnabledFeatures = &enabledFeatures;
    checkVk(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device), "create logical device");
    vkGetDeviceQueue(device, graphicsQueueFamily, 0, &graphicsQueue);

    VkCommandPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    checkVk(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), "create command pool");

    VkCommandBufferAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    checkVk(vkAllocateCommandBuffers(device, &allocInfo, &renderCommandBuffer), "allocate render command buffer");

    VkFenceCreateInfo fenceInfo {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    checkVk(vkCreateFence(device, &fenceInfo, nullptr, &renderFence), "create render fence");
}

void VulkanCameraSim::initializeFramebuffer()
{
    const bool msaaEnabled = msaaSamples != VK_SAMPLE_COUNT_1_BIT;
    createImage(widthPx, heightPx, kColorFormat,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, colorImage, colorImageMemory);
    colorImageView = createImageView(colorImage, kColorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    if (msaaEnabled)
    {
        createImage(widthPx, heightPx, kColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            msaaColorImage, msaaColorImageMemory, 1, msaaSamples);
        msaaColorImageView = createImageView(msaaColorImage, kColorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    VkSamplerCreateInfo colorSamplerInfo {};
    colorSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    colorSamplerInfo.magFilter = VK_FILTER_NEAREST;
    colorSamplerInfo.minFilter = VK_FILTER_NEAREST;
    colorSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    colorSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    colorSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    checkVk(vkCreateSampler(device, &colorSamplerInfo, nullptr, &colorSampler), "create color sampler");
    createImage(widthPx, heightPx, kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        depthImage, depthImageMemory, 1, msaaSamples);
    depthImageView = createImageView(depthImage, kDepthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    createImage(kShadowMapSize, kShadowMapSize, kDepthFormat,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        shadowDepthImage, shadowDepthImageMemory);
    shadowDepthImageView = createImageView(shadowDepthImage, kDepthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

    VkSamplerCreateInfo shadowSamplerInfo {};
    shadowSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    shadowSamplerInfo.magFilter = VK_FILTER_LINEAR;
    shadowSamplerInfo.minFilter = VK_FILTER_LINEAR;
    shadowSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadowSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadowSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadowSamplerInfo.compareEnable = VK_TRUE;
    shadowSamplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    shadowSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    shadowSamplerInfo.maxLod = 0.0f;
    checkVk(vkCreateSampler(device, &shadowSamplerInfo, nullptr, &shadowSampler), "create shadow sampler");

    const VkDeviceSize rgbPackBytesPerImage =
        ((static_cast<VkDeviceSize>(widthPx) * static_cast<VkDeviceSize>(heightPx) + 3U) / 4U) * 12U;
    const VkDeviceSize cameraSlots = std::max<VkDeviceSize>(1U, static_cast<VkDeviceSize>(cameraComponents.size()));
    const VkDeviceSize rgbBufferBytes = rgbPackBytesPerImage * cameraSlots;
    rgbStorageBuffer = createBuffer(rgbBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    readbackBuffer = createBuffer(rgbBufferBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    checkVk(vkMapMemory(device, readbackBuffer.memory, 0, readbackBuffer.size, 0, &readbackMapped),
        "map batched readback buffer");
    drawConstantsBuffer = createBuffer(static_cast<VkDeviceSize>(kMaxDrawConstants) * sizeof(PushConstants),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    checkVk(vkMapMemory(device, drawConstantsBuffer.memory, 0, drawConstantsBuffer.size, 0, &drawConstantsMapped),
        "map draw constants buffer");
    const InstanceData identityInstance;
    identityInstanceBuffer = createDeviceLocalBuffer(&identityInstance, sizeof(InstanceData),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    VkAttachmentDescription colorAttachment {};
    colorAttachment.format = kColorFormat;
    colorAttachment.samples = msaaSamples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = msaaEnabled ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment {};
    depthAttachment.format = kDepthFormat;
    depthAttachment.samples = msaaSamples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::vector<VkAttachmentDescription> attachments { colorAttachment, depthAttachment };
    VkAttachmentReference colorRef { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    VkAttachmentReference resolveRef { VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    if (msaaEnabled)
    {
        VkAttachmentDescription resolveAttachment {};
        resolveAttachment.format = kColorFormat;
        resolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        resolveAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        resolveAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        resolveAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        resolveAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachments.push_back(resolveAttachment);
        resolveRef.attachment = 2;
    }

    VkSubpassDescription subpass {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;
    subpass.pResolveAttachments = msaaEnabled ? &resolveRef : nullptr;

    VkRenderPassCreateInfo renderPassInfo {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    checkVk(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass), "create render pass");

    const std::vector<VkImageView> views =
        msaaEnabled ? std::vector<VkImageView> { msaaColorImageView, depthImageView, colorImageView }
                    : std::vector<VkImageView> { colorImageView, depthImageView };
    VkFramebufferCreateInfo framebufferInfo {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(views.size());
    framebufferInfo.pAttachments = views.data();
    framebufferInfo.width = static_cast<uint32_t>(widthPx);
    framebufferInfo.height = static_cast<uint32_t>(heightPx);
    framebufferInfo.layers = 1;
    checkVk(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer), "create framebuffer");

    VkAttachmentDescription shadowDepthAttachment {};
    shadowDepthAttachment.format = kDepthFormat;
    shadowDepthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    shadowDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    shadowDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    shadowDepthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    shadowDepthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference shadowDepthRef { 0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    VkSubpassDescription shadowSubpass {};
    shadowSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    shadowSubpass.pDepthStencilAttachment = &shadowDepthRef;

    VkRenderPassCreateInfo shadowRenderPassInfo {};
    shadowRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    shadowRenderPassInfo.attachmentCount = 1;
    shadowRenderPassInfo.pAttachments = &shadowDepthAttachment;
    shadowRenderPassInfo.subpassCount = 1;
    shadowRenderPassInfo.pSubpasses = &shadowSubpass;
    checkVk(vkCreateRenderPass(device, &shadowRenderPassInfo, nullptr, &shadowRenderPass), "create shadow render pass");

    VkFramebufferCreateInfo shadowFramebufferInfo {};
    shadowFramebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    shadowFramebufferInfo.renderPass = shadowRenderPass;
    shadowFramebufferInfo.attachmentCount = 1;
    shadowFramebufferInfo.pAttachments = &shadowDepthImageView;
    shadowFramebufferInfo.width = kShadowMapSize;
    shadowFramebufferInfo.height = kShadowMapSize;
    shadowFramebufferInfo.layers = 1;
    checkVk(vkCreateFramebuffer(device, &shadowFramebufferInfo, nullptr, &shadowFramebuffer), "create shadow framebuffer");
}

std::vector<uint32_t> VulkanCameraSim::compileShader(
    const std::string& source, shaderc_shader_kind kind, const std::string& name) const
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(source, kind, name.c_str(), options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        throw std::runtime_error("VulkanCameraSim shader compile failed for " + name + ": " + result.GetErrorMessage());
    }
    return { result.cbegin(), result.cend() };
}

VkShaderModule VulkanCameraSim::createShaderModule(const std::vector<uint32_t>& code) const
{
    VkShaderModuleCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size() * sizeof(uint32_t);
    info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    checkVk(vkCreateShaderModule(device, &info, nullptr, &module), "create shader module");
    return module;
}

void VulkanCameraSim::initializePipeline()
{
    const std::string vert = R"GLSL(
        #version 450
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNormal;
        layout(location = 2) in vec2 aUv;
        layout(location = 3) in vec3 aColor;
        layout(location = 4) in mat4 aInstanceModel;
        struct DrawConstants {
            mat4 mvp;
            mat4 model;
            mat4 lightViewProj;
            vec4 color;
            vec4 params;
            vec4 material;
            vec4 viewParams;
        };
        layout(set = 0, binding = 2, std430) readonly buffer DrawConstantsBuffer {
            DrawConstants draws[];
        } drawConstants;
        layout(push_constant) uniform DrawPush {
            uint drawIndex;
        } drawPush;
        layout(location = 0) out vec3 vNormal;
        layout(location = 1) out vec4 vColor;
        layout(location = 2) out vec2 vUv;
        layout(location = 3) out vec3 vWorldPos;
        layout(location = 4) out vec4 vLightSpacePos;
        void main() {
            DrawConstants pc = drawConstants.draws[drawPush.drawIndex];
            mat4 model = aInstanceModel * pc.model;
            vec4 worldPos = model * vec4(aPos, 1.0);
            vNormal = mat3(transpose(inverse(model))) * aNormal;
            vColor = vec4(aColor * pc.color.rgb, pc.color.a);
            vUv = aUv;
            vWorldPos = worldPos.xyz;
            vLightSpacePos = pc.lightViewProj * worldPos;
            gl_Position = pc.mvp * worldPos;
        }
    )GLSL";

    const std::string frag = R"GLSL(
        #version 450
        layout(location = 0) in vec3 vNormal;
        layout(location = 1) in vec4 vColor;
        layout(location = 2) in vec2 vUv;
        layout(location = 3) in vec3 vWorldPos;
        layout(location = 4) in vec4 vLightSpacePos;
        layout(set = 0, binding = 0) uniform sampler2D uTex;
        layout(set = 0, binding = 1) uniform sampler2DShadow uShadowMap;
        layout(location = 0) out vec4 outColor;
        struct DrawConstants {
            mat4 mvp;
            mat4 model;
            mat4 lightViewProj;
            vec4 color;
            vec4 params;
            vec4 material;
            vec4 viewParams;
        };
        layout(set = 0, binding = 2, std430) readonly buffer DrawConstantsBuffer {
            DrawConstants draws[];
        } drawConstants;
        layout(push_constant) uniform DrawPush {
            uint drawIndex;
        } drawPush;
        const float PI = 3.141592653589793;
        const vec3 F0 = vec3(0.04);
        vec3 fresnel(float u, vec3 f0) {
            return f0 + (vec3(1.0) - f0) * pow(2.0, (-5.55473 * u - 6.98316) * u);
        }
        float visibility(float nDotL, float nDotV, float alphaRoughness) {
            float r2 = alphaRoughness * alphaRoughness;
            float ggxv = nDotL * sqrt(nDotV * nDotV * (1.0 - r2) + r2);
            float ggxl = nDotV * sqrt(nDotL * nDotL * (1.0 - r2) + r2);
            float ggx = ggxv + ggxl;
            return (ggx > 0.0) ? (0.5 / ggx) : 0.0;
        }
        float distribution(float nDotH, float alphaRoughness) {
            float roughness2 = alphaRoughness * alphaRoughness;
            float f = (nDotH * nDotH) * (roughness2 - 1.0) + 1.0;
            return roughness2 / (PI * f * f);
        }
        vec3 fresnel_roughness(float u, vec3 f0, float roughness) {
            return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - u, 0.0, 1.0), 5.0);
        }
        vec3 irradiance_from_sh(vec3 normal) {
            vec3 c0 = vec3(1.10, 1.15, 1.20);
            vec3 c2 = vec3(0.14, 0.20, 0.26);
            vec3 c7 = vec3(-0.02, -0.02, -0.02);
            return c0 * 0.282095
                + c2 * 0.488603 * normal.z
                + c7 * (0.946176 * normal.z * normal.z - 0.315392);
        }
        vec3 env_color(vec3 dir) {
            float blend = clamp(0.5 + 0.5 * dir.z, 0.0, 1.0);
            float shaped = pow(blend, 0.7);
            return mix(vec3(0.29, 0.29, 0.30), vec3(0.55, 0.66, 0.82), shaped);
        }
        vec2 brdf_approx(float nDotV, float roughness) {
            vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
            vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
            vec4 r = roughness * c0 + c1;
            float a004 = min(r.x * r.x, exp2(-9.28 * nDotV)) * r.x + r.y;
            return vec2(clamp(-1.04 * a004 + r.z, 0.0, 1.0),
                clamp(1.04 * a004 + r.w, 0.0, 1.0));
        }
        float shadow_visibility(vec3 n, vec3 l, float shadowEnabled) {
            if (shadowEnabled < 0.5) {
                return 1.0;
            }

            vec3 projCoords = vLightSpacePos.xyz / max(vLightSpacePos.w, 1e-5);
            projCoords.xy = projCoords.xy * 0.5 + 0.5;
            if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
                projCoords.y < 0.0 || projCoords.y > 1.0 ||
                projCoords.z > 1.0) {
                return 1.0;
            }

            float ndotl = max(dot(n, l), 0.0);
            const float shadowBias = 0.0025;
            float bias = max(shadowBias * (1.0 - ndotl), shadowBias * 0.25);
            vec2 texelSize = 1.0 / vec2(textureSize(uShadowMap, 0));
            float vis = 0.0;
            const int pcfRadius = 2;
            for (int y = -pcfRadius; y <= pcfRadius; ++y) {
                for (int x = -pcfRadius; x <= pcfRadius; ++x) {
                    vec2 offset = vec2(float(x), float(y)) * texelSize;
                    vis += texture(uShadowMap, vec3(projCoords.xy + offset, projCoords.z - bias));
                }
            }
            return vis / 25.0;
        }
        void main() {
            DrawConstants pc = drawConstants.draws[drawPush.drawIndex];
            vec3 n = normalize(vNormal);
            vec3 l = normalize(vec3(-100.0, 0.0, 10.0));
            vec3 v = normalize(pc.viewParams.xyz - vWorldPos);
            vec3 h = normalize(l + v);
            vec3 albedo = vColor.rgb;
            if (pc.params.x > 0.5) {
                albedo = texture(uTex, vUv).rgb * pc.color.rgb;
            }
            if (pc.params.y > 0.5) {
                outColor = vec4(albedo, 1.0);
                return;
            }
            float metallic = clamp(pc.material.x, 0.0, 1.0);
            float perceptualRoughness = clamp(pc.material.y, 0.0, 1.0);
            float alphaRoughness = max(perceptualRoughness * perceptualRoughness, 0.002);
            vec3 diffuseColor = (albedo * (vec3(1.0) - F0)) * (1.0 - metallic);
            vec3 specColor = mix(F0, albedo, metallic);
            float nDotL = clamp(dot(n, l), 0.0, 1.0);
            float nDotV = clamp(abs(dot(n, v)), 0.0, 1.0);
            float nDotH = clamp(dot(n, h), 0.0, 1.0);
            float vDotH = clamp(dot(v, h), 0.0, 1.0);
            vec3 direct = vec3(0.0);
            if (nDotL > 0.0) {
                vec3 F = fresnel(vDotH, specColor);
                float V = visibility(nDotL, nDotV, alphaRoughness);
                float D = distribution(nDotH, alphaRoughness);
                float shadow = shadow_visibility(n, l, pc.params.z);
                direct = nDotL * shadow * vec3(1.35) * (diffuseColor * (1.0 / PI) + F * V * D);
            }
            vec3 iblF = fresnel_roughness(nDotV, specColor, perceptualRoughness);
            vec3 iblKd = (vec3(1.0) - iblF) * (1.0 - metallic);
            vec3 irradiance = max(irradiance_from_sh(n), vec3(0.0));
            vec3 iblDiff = albedo * irradiance * (1.0 / PI);
            vec3 iblR = reflect(-v, n);
            vec3 iblSpecColor = env_color(iblR);
            vec2 envBrdf = brdf_approx(nDotV, perceptualRoughness);
            vec3 iblSpec = iblSpecColor * (iblF * envBrdf.x + envBrdf.y);
            vec3 color = direct + iblKd * iblDiff + iblSpec;
            color = max(vec3(0.0), color - vec3(0.004));
            color = (color * (vec3(6.2) * color + vec3(0.5))) / (color * (vec3(6.2) * color + vec3(1.7)) + vec3(0.06));
            outColor = vec4(color, 1.0);
        }
    )GLSL";

    VkShaderModule vertModule = createShaderModule(compileShader(vert, shaderc_vertex_shader, "vulkanCamera.vert"));
    VkShaderModule fragModule = createShaderModule(compileShader(frag, shaderc_fragment_shader, "vulkanCamera.frag"));

    VkPipelineShaderStageCreateInfo stages[2] {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    VkVertexInputBindingDescription bindings[2] {};
    bindings[0].binding = 0;
    bindings[0].stride = sizeof(Vertex);
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindings[1].binding = 1;
    bindings[1].stride = sizeof(InstanceData);
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    VkVertexInputAttributeDescription attrs[8] {};
    attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, position)) };
    attrs[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, normal)) };
    attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, uv)) };
    attrs[3] = { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, color)) };
    attrs[4] = { 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
    attrs[5] = { 5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16 };
    attrs[6] = { 6, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 32 };
    attrs[7] = { 7, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 48 };

    VkPipelineVertexInputStateCreateInfo vertexInput {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 2;
    vertexInput.pVertexBindingDescriptions = bindings;
    vertexInput.vertexAttributeDescriptionCount = 8;
    vertexInput.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(widthPx);
    viewport.height = static_cast<float>(heightPx);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor { { 0, 0 }, { static_cast<uint32_t>(widthPx), static_cast<uint32_t>(heightPx) } };
    VkPipelineViewportStateCreateInfo viewportState {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo raster {};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample {};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = msaaSamples;

    VkPipelineDepthStencilStateCreateInfo depth {};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
        | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlend {};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &colorBlendAttachment;

    VkPushConstantRange pushRange {};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(uint32_t);

    VkDescriptorSetLayoutBinding descriptorBindings[3] {};
    descriptorBindings[0].binding = 0;
    descriptorBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorBindings[0].descriptorCount = 1;
    descriptorBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    descriptorBindings[1].binding = 1;
    descriptorBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorBindings[1].descriptorCount = 1;
    descriptorBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    descriptorBindings[2].binding = 2;
    descriptorBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorBindings[2].descriptorCount = 1;
    descriptorBindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo {};
    descriptorLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutInfo.bindingCount = 3;
    descriptorLayoutInfo.pBindings = descriptorBindings;
    checkVk(vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &textureDescriptorSetLayout),
        "create texture descriptor set layout");

    VkDescriptorSetLayoutBinding computeBindings[2] {};
    computeBindings[0].binding = 0;
    computeBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    computeBindings[0].descriptorCount = 1;
    computeBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    computeBindings[1].binding = 1;
    computeBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    computeBindings[1].descriptorCount = 1;
    computeBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo computeLayoutInfo {};
    computeLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    computeLayoutInfo.bindingCount = 2;
    computeLayoutInfo.pBindings = computeBindings;
    checkVk(vkCreateDescriptorSetLayout(device, &computeLayoutInfo, nullptr, &rgbComputeDescriptorSetLayout),
        "create rgb compute descriptor set layout");

    VkDescriptorPoolSize poolSizes[2] {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 8193;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 4097;
    VkDescriptorPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 4097;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    checkVk(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool), "create descriptor pool");

    const unsigned char whitePixel[4] = { 255, 255, 255, 255 };
    whiteTexture = createTextureResource(whitePixel, 1, 1);

    VkPipelineLayoutCreateInfo layoutInfo {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &textureDescriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    checkVk(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout), "create pipeline layout");

    VkPipelineLayoutCreateInfo computePipelineLayoutInfo {};
    VkPushConstantRange computePushRange {};
    computePushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    computePushRange.offset = 0;
    computePushRange.size = sizeof(uint32_t);
    computePipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computePipelineLayoutInfo.setLayoutCount = 1;
    computePipelineLayoutInfo.pSetLayouts = &rgbComputeDescriptorSetLayout;
    computePipelineLayoutInfo.pushConstantRangeCount = 1;
    computePipelineLayoutInfo.pPushConstantRanges = &computePushRange;
    checkVk(vkCreatePipelineLayout(device, &computePipelineLayoutInfo, nullptr, &rgbComputePipelineLayout),
        "create rgb compute pipeline layout");

    VkDescriptorSetAllocateInfo computeSetAlloc {};
    computeSetAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    computeSetAlloc.descriptorPool = descriptorPool;
    computeSetAlloc.descriptorSetCount = 1;
    computeSetAlloc.pSetLayouts = &rgbComputeDescriptorSetLayout;
    checkVk(vkAllocateDescriptorSets(device, &computeSetAlloc, &rgbComputeDescriptorSet),
        "allocate rgb compute descriptor set");

    VkDescriptorImageInfo colorImageInfo {};
    colorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorImageInfo.imageView = colorImageView;
    colorImageInfo.sampler = colorSampler;
    VkDescriptorBufferInfo rgbBufferInfo {};
    rgbBufferInfo.buffer = rgbStorageBuffer.buffer;
    rgbBufferInfo.offset = 0;
    rgbBufferInfo.range = rgbStorageBuffer.size;
    VkWriteDescriptorSet computeWrites[2] {};
    computeWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    computeWrites[0].dstSet = rgbComputeDescriptorSet;
    computeWrites[0].dstBinding = 0;
    computeWrites[0].descriptorCount = 1;
    computeWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    computeWrites[0].pImageInfo = &colorImageInfo;
    computeWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    computeWrites[1].dstSet = rgbComputeDescriptorSet;
    computeWrites[1].dstBinding = 1;
    computeWrites[1].descriptorCount = 1;
    computeWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    computeWrites[1].pBufferInfo = &rgbBufferInfo;
    vkUpdateDescriptorSets(device, 2, computeWrites, 0, nullptr);

    VkGraphicsPipelineCreateInfo pipelineInfo {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depth;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    checkVk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline),
        "create graphics pipeline");
    depth.depthTestEnable = VK_FALSE;
    depth.depthWriteEnable = VK_FALSE;
    checkVk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skyPipeline),
        "create sky graphics pipeline");

    const std::string shadowVert = R"GLSL(
        #version 450
        layout(location = 0) in vec3 aPos;
        struct DrawConstants {
            mat4 mvp;
            mat4 model;
            mat4 lightViewProj;
            vec4 color;
            vec4 params;
            vec4 material;
            vec4 viewParams;
        };
        layout(set = 0, binding = 2, std430) readonly buffer DrawConstantsBuffer {
            DrawConstants draws[];
        } drawConstants;
        layout(push_constant) uniform DrawPush {
            uint drawIndex;
        } drawPush;
        layout(location = 4) in mat4 aInstanceModel;
        void main() {
            DrawConstants pc = drawConstants.draws[drawPush.drawIndex];
            mat4 model = aInstanceModel * pc.model;
            gl_Position = pc.mvp * model * vec4(aPos, 1.0);
        }
    )GLSL";
    VkShaderModule shadowVertModule =
        createShaderModule(compileShader(shadowVert, shaderc_vertex_shader, "vulkanShadow.vert"));
    VkPipelineShaderStageCreateInfo shadowStage {};
    shadowStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    shadowStage.module = shadowVertModule;
    shadowStage.pName = "main";

    VkViewport shadowViewport {};
    shadowViewport.x = 0.0f;
    shadowViewport.y = 0.0f;
    shadowViewport.width = static_cast<float>(kShadowMapSize);
    shadowViewport.height = static_cast<float>(kShadowMapSize);
    shadowViewport.minDepth = 0.0f;
    shadowViewport.maxDepth = 1.0f;
    VkRect2D shadowScissor { { 0, 0 }, { kShadowMapSize, kShadowMapSize } };
    VkPipelineViewportStateCreateInfo shadowViewportState {};
    shadowViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    shadowViewportState.viewportCount = 1;
    shadowViewportState.pViewports = &shadowViewport;
    shadowViewportState.scissorCount = 1;
    shadowViewportState.pScissors = &shadowScissor;

    VkPipelineRasterizationStateCreateInfo shadowRaster = raster;
    shadowRaster.depthBiasEnable = VK_TRUE;
    shadowRaster.depthBiasSlopeFactor = 2.0f;
    shadowRaster.depthBiasConstantFactor = 4.0f;

    VkPipelineColorBlendStateCreateInfo shadowColorBlend {};
    shadowColorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    pipelineInfo.stageCount = 1;
    pipelineInfo.pStages = &shadowStage;
    pipelineInfo.pViewportState = &shadowViewportState;
    pipelineInfo.pRasterizationState = &shadowRaster;
    pipelineInfo.pColorBlendState = &shadowColorBlend;
    pipelineInfo.renderPass = shadowRenderPass;
    checkVk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowPipeline),
        "create shadow pipeline");

    const std::string rgbCompute = std::string(R"GLSL(
        #version 450
        layout(local_size_x = 256) in;
        layout(set = 0, binding = 0) uniform sampler2D uColor;
        layout(set = 0, binding = 1, std430) buffer RgbOut {
            uint words[];
        } outBuffer;
        layout(push_constant) uniform RgbPush {
            uint baseWord;
        } pc;
        const uint WIDTH = )GLSL") + std::to_string(widthPx) + std::string(R"GLSL(U;
        const uint PIXELS = )GLSL") + std::to_string(static_cast<uint64_t>(widthPx) * static_cast<uint64_t>(heightPx))
        + std::string(R"GLSL(U;
        const uint PACK_GROUPS = (PIXELS + 3U) / 4U;
        uint byteFromChannel(float v) {
            return uint(round(clamp(v, 0.0, 1.0) * 255.0));
        }
        uvec3 rgbByte(uint pixel) {
            if (pixel >= PIXELS) {
                return uvec3(0);
            }
            uint x = pixel % WIDTH;
            uint y = pixel / WIDTH;
            vec3 color = texelFetch(uColor, ivec2(int(x), int(y)), 0).rgb;
            return uvec3(byteFromChannel(color.b), byteFromChannel(color.g), byteFromChannel(color.r));
        }
        void main() {
            uint group = gl_GlobalInvocationID.x;
            if (group >= PACK_GROUPS) {
                return;
            }
            uint pixel = group * 4U;
            uint word = pc.baseWord + group * 3U;
            uvec3 p0 = rgbByte(pixel + 0U);
            uvec3 p1 = rgbByte(pixel + 1U);
            uvec3 p2 = rgbByte(pixel + 2U);
            uvec3 p3 = rgbByte(pixel + 3U);
            outBuffer.words[word + 0U] = p0.r | (p0.g << 8U) | (p0.b << 16U) | (p1.r << 24U);
            outBuffer.words[word + 1U] = p1.g | (p1.b << 8U) | (p2.r << 16U) | (p2.g << 24U);
            outBuffer.words[word + 2U] = p2.b | (p3.r << 8U) | (p3.g << 16U) | (p3.b << 24U);
        }
    )GLSL");
    VkShaderModule rgbComputeModule =
        createShaderModule(compileShader(rgbCompute, shaderc_compute_shader, "vulkanRgbPack.comp"));
    VkPipelineShaderStageCreateInfo computeStage {};
    computeStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeStage.module = rgbComputeModule;
    computeStage.pName = "main";
    VkComputePipelineCreateInfo computePipelineInfo {};
    computePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineInfo.stage = computeStage;
    computePipelineInfo.layout = rgbComputePipelineLayout;
    checkVk(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &rgbComputePipeline),
        "create rgb compute pipeline");

    vkDestroyShaderModule(device, rgbComputeModule, nullptr);
    vkDestroyShaderModule(device, shadowVertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);
    vkDestroyShaderModule(device, vertModule, nullptr);
}

void VulkanCameraSim::destroyVulkan()
{
    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }
    auto destroyModel = [&](ModelRenderData& model) {
        for (auto& mesh : model.meshes)
        {
            destroyBuffer(mesh.vertexBuffer);
            destroyBuffer(mesh.indexBuffer);
        }
        model.meshes.clear();
        model.valid = false;
    };
    destroyModel(groundPlaneModel);
    destroyModel(skyModel);
    destroyModel(carModel);
    destroyModel(blueConeModel);
    destroyModel(yellowConeModel);
    destroyModel(trackMeshModel);
    for (auto& texture : textures)
    {
        destroyTexture(texture);
    }
    textures.clear();
    destroyTexture(whiteTexture);
    destroyBuffer(yellowConeInstanceBuffer);
    destroyBuffer(blueConeInstanceBuffer);
    destroyBuffer(identityInstanceBuffer);
    if (drawConstantsMapped != nullptr)
    {
        vkUnmapMemory(device, drawConstantsBuffer.memory);
        drawConstantsMapped = nullptr;
    }
    destroyBuffer(drawConstantsBuffer);
    if (readbackMapped != nullptr)
    {
        vkUnmapMemory(device, readbackBuffer.memory);
        readbackMapped = nullptr;
    }
    destroyBuffer(readbackBuffer);
    destroyBuffer(rgbStorageBuffer);
    if (rgbComputePipeline) vkDestroyPipeline(device, rgbComputePipeline, nullptr);
    rgbComputePipeline = VK_NULL_HANDLE;
    if (shadowPipeline) vkDestroyPipeline(device, shadowPipeline, nullptr);
    shadowPipeline = VK_NULL_HANDLE;
    if (skyPipeline) vkDestroyPipeline(device, skyPipeline, nullptr);
    skyPipeline = VK_NULL_HANDLE;
    if (graphicsPipeline) vkDestroyPipeline(device, graphicsPipeline, nullptr);
    graphicsPipeline = VK_NULL_HANDLE;
    if (rgbComputePipelineLayout) vkDestroyPipelineLayout(device, rgbComputePipelineLayout, nullptr);
    rgbComputePipelineLayout = VK_NULL_HANDLE;
    if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    pipelineLayout = VK_NULL_HANDLE;
    if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    descriptorPool = VK_NULL_HANDLE;
    rgbComputeDescriptorSet = VK_NULL_HANDLE;
    if (rgbComputeDescriptorSetLayout) vkDestroyDescriptorSetLayout(device, rgbComputeDescriptorSetLayout, nullptr);
    rgbComputeDescriptorSetLayout = VK_NULL_HANDLE;
    if (textureDescriptorSetLayout) vkDestroyDescriptorSetLayout(device, textureDescriptorSetLayout, nullptr);
    textureDescriptorSetLayout = VK_NULL_HANDLE;
    if (shadowFramebuffer) vkDestroyFramebuffer(device, shadowFramebuffer, nullptr);
    shadowFramebuffer = VK_NULL_HANDLE;
    if (framebuffer) vkDestroyFramebuffer(device, framebuffer, nullptr);
    framebuffer = VK_NULL_HANDLE;
    if (shadowRenderPass) vkDestroyRenderPass(device, shadowRenderPass, nullptr);
    shadowRenderPass = VK_NULL_HANDLE;
    if (renderPass) vkDestroyRenderPass(device, renderPass, nullptr);
    renderPass = VK_NULL_HANDLE;
    if (shadowSampler) vkDestroySampler(device, shadowSampler, nullptr);
    shadowSampler = VK_NULL_HANDLE;
    if (colorSampler) vkDestroySampler(device, colorSampler, nullptr);
    colorSampler = VK_NULL_HANDLE;
    if (msaaColorImageView) vkDestroyImageView(device, msaaColorImageView, nullptr);
    msaaColorImageView = VK_NULL_HANDLE;
    if (msaaColorImage) vkDestroyImage(device, msaaColorImage, nullptr);
    msaaColorImage = VK_NULL_HANDLE;
    if (msaaColorImageMemory) vkFreeMemory(device, msaaColorImageMemory, nullptr);
    msaaColorImageMemory = VK_NULL_HANDLE;
    if (shadowDepthImageView) vkDestroyImageView(device, shadowDepthImageView, nullptr);
    shadowDepthImageView = VK_NULL_HANDLE;
    if (shadowDepthImage) vkDestroyImage(device, shadowDepthImage, nullptr);
    shadowDepthImage = VK_NULL_HANDLE;
    if (shadowDepthImageMemory) vkFreeMemory(device, shadowDepthImageMemory, nullptr);
    shadowDepthImageMemory = VK_NULL_HANDLE;
    if (depthImageView) vkDestroyImageView(device, depthImageView, nullptr);
    depthImageView = VK_NULL_HANDLE;
    if (depthImage) vkDestroyImage(device, depthImage, nullptr);
    depthImage = VK_NULL_HANDLE;
    if (depthImageMemory) vkFreeMemory(device, depthImageMemory, nullptr);
    depthImageMemory = VK_NULL_HANDLE;
    if (colorImageView) vkDestroyImageView(device, colorImageView, nullptr);
    colorImageView = VK_NULL_HANDLE;
    if (colorImage) vkDestroyImage(device, colorImage, nullptr);
    colorImage = VK_NULL_HANDLE;
    if (colorImageMemory) vkFreeMemory(device, colorImageMemory, nullptr);
    colorImageMemory = VK_NULL_HANDLE;
    if (renderFence) vkDestroyFence(device, renderFence, nullptr);
    renderFence = VK_NULL_HANDLE;
    renderCommandBuffer = VK_NULL_HANDLE;
    if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
    commandPool = VK_NULL_HANDLE;
    if (device) vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
    graphicsQueue = VK_NULL_HANDLE;
    physicalDevice = VK_NULL_HANDLE;
    graphicsQueueFamily = 0;
    if (instance) vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
}

uint32_t VulkanCameraSim::findMemoryType(
    uint32_t typeFilter, VkMemoryPropertyFlags requiredProperties, VkMemoryPropertyFlags preferredProperties) const
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    if (preferredProperties != 0)
    {
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
        {
            const VkMemoryPropertyFlags flags = memProperties.memoryTypes[i].propertyFlags;
            if ((typeFilter & (1U << i)) && (flags & requiredProperties) == requiredProperties
                && (flags & preferredProperties) == preferredProperties)
            {
                return i;
            }
        }
    }
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
    {
        const VkMemoryPropertyFlags flags = memProperties.memoryTypes[i].propertyFlags;
        if ((typeFilter & (1U << i)) && (flags & requiredProperties) == requiredProperties)
        {
            return i;
        }
    }
    throw std::runtime_error("VulkanCameraSim failed to find suitable memory type");
}

VulkanCameraSim::GpuBuffer VulkanCameraSim::createBuffer(
    VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags requiredProperties,
    VkMemoryPropertyFlags preferredProperties)
{
    GpuBuffer out;
    out.size = size;
    VkBufferCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    checkVk(vkCreateBuffer(device, &info, nullptr, &out.buffer), "create buffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, out.buffer, &req);
    VkMemoryAllocateInfo alloc {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = findMemoryType(req.memoryTypeBits, requiredProperties, preferredProperties);
    checkVk(vkAllocateMemory(device, &alloc, nullptr, &out.memory), "allocate buffer memory");
    checkVk(vkBindBufferMemory(device, out.buffer, out.memory, 0), "bind buffer memory");
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    out.memoryProperties = memProperties.memoryTypes[alloc.memoryTypeIndex].propertyFlags;
    return out;
}

VulkanCameraSim::GpuBuffer VulkanCameraSim::createDeviceLocalBuffer(
    const void* data, VkDeviceSize size, VkBufferUsageFlags usage)
{
    GpuBuffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uploadToBuffer(staging, data, size);

    GpuBuffer out = createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    copyBuffer(staging, out, size);
    destroyBuffer(staging);
    return out;
}

void VulkanCameraSim::uploadToBuffer(const GpuBuffer& buffer, const void* data, VkDeviceSize size)
{
    void* mapped = nullptr;
    checkVk(vkMapMemory(device, buffer.memory, 0, size, 0, &mapped), "map buffer");
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(device, buffer.memory);
}

void VulkanCameraSim::copyBuffer(const GpuBuffer& src, const GpuBuffer& dst, VkDeviceSize size)
{
    VkCommandBuffer cmd = beginOneTimeCommands();
    VkBufferCopy copy {};
    copy.size = size;
    vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &copy);
    endOneTimeCommands(cmd);
}

void VulkanCameraSim::destroyBuffer(GpuBuffer& buffer)
{
    if (buffer.buffer) vkDestroyBuffer(device, buffer.buffer, nullptr);
    if (buffer.memory) vkFreeMemory(device, buffer.memory, nullptr);
    buffer = {};
}

uint32_t VulkanCameraSim::appendDrawConstants(const PushConstants& constants) const
{
    if (drawConstantsMapped == nullptr)
    {
        throw std::runtime_error("VulkanCameraSim draw constants buffer is not mapped");
    }
    if (drawConstantsCount >= kMaxDrawConstants)
    {
        throw std::runtime_error("VulkanCameraSim exceeded draw constants capacity");
    }
    const uint32_t index = drawConstantsCount++;
    auto* constantsOut = static_cast<PushConstants*>(drawConstantsMapped);
    constantsOut[index] = constants;
    return index;
}

VkCommandBuffer VulkanCameraSim::beginOneTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    checkVk(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer), "allocate one-time command buffer");

    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checkVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "begin one-time command buffer");
    return commandBuffer;
}

void VulkanCameraSim::endOneTimeCommands(VkCommandBuffer commandBuffer)
{
    checkVk(vkEndCommandBuffer(commandBuffer), "end one-time command buffer");
    VkSubmitInfo submitInfo {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    checkVk(vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE), "submit one-time command buffer");
    checkVk(vkQueueWaitIdle(graphicsQueue), "wait one-time command buffer");
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

VulkanCameraSim::TextureResource VulkanCameraSim::createTextureResource(
    const unsigned char* rgbaPixels, int textureWidth, int textureHeight)
{
    if (rgbaPixels == nullptr || textureWidth <= 0 || textureHeight <= 0)
    {
        return {};
    }

    const VkDeviceSize imageBytes = static_cast<VkDeviceSize>(textureWidth) * static_cast<VkDeviceSize>(textureHeight) * 4U;
    GpuBuffer staging = createBuffer(imageBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uploadToBuffer(staging, rgbaPixels, imageBytes);

    TextureResource texture;
    texture.mipLevels = static_cast<uint32_t>(
        std::floor(std::log2(static_cast<float>(std::max(textureWidth, textureHeight))))) + 1U;
    createImage(static_cast<uint32_t>(textureWidth), static_cast<uint32_t>(textureHeight), kColorFormat,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        texture.image, texture.memory, texture.mipLevels);

    VkCommandBuffer cmd = beginOneTimeCommands();
    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.image = texture.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = texture.mipLevels;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy copy {};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = { static_cast<uint32_t>(textureWidth), static_cast<uint32_t>(textureHeight), 1 };
    vkCmdCopyBufferToImage(cmd, staging.buffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    int32_t mipWidth = textureWidth;
    int32_t mipHeight = textureHeight;
    barrier.subresourceRange.levelCount = 1;
    for (uint32_t mip = 1; mip < texture.mipLevels; ++mip)
    {
        barrier.subresourceRange.baseMipLevel = mip - 1U;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkImageBlit blit {};
        blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = mip - 1U;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[1] = { std::max(1, mipWidth / 2), std::max(1, mipHeight / 2), 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = mip;
        blit.dstSubresource.layerCount = 1;
        vkCmdBlitImage(cmd, texture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        mipWidth = std::max(1, mipWidth / 2);
        mipHeight = std::max(1, mipHeight / 2);
    }

    barrier.subresourceRange.baseMipLevel = texture.mipLevels - 1U;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    endOneTimeCommands(cmd);
    destroyBuffer(staging);

    texture.view = createImageView(texture.image, kColorFormat, VK_IMAGE_ASPECT_COLOR_BIT, texture.mipLevels);

    VkSamplerCreateInfo samplerInfo {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.anisotropyEnable = samplerAnisotropyEnabled ? VK_TRUE : VK_FALSE;
    samplerInfo.maxAnisotropy = samplerMaxAnisotropy;
    samplerInfo.maxLod = static_cast<float>(texture.mipLevels - 1U);
    checkVk(vkCreateSampler(device, &samplerInfo, nullptr, &texture.sampler), "create texture sampler");

    VkDescriptorSetAllocateInfo setAlloc {};
    setAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAlloc.descriptorPool = descriptorPool;
    setAlloc.descriptorSetCount = 1;
    setAlloc.pSetLayouts = &textureDescriptorSetLayout;
    checkVk(vkAllocateDescriptorSets(device, &setAlloc, &texture.descriptorSet), "allocate texture descriptor set");

    VkDescriptorImageInfo imageInfo {};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = texture.view;
    imageInfo.sampler = texture.sampler;
    VkDescriptorImageInfo shadowInfo {};
    shadowInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    shadowInfo.imageView = shadowDepthImageView;
    shadowInfo.sampler = shadowSampler;
    VkDescriptorBufferInfo drawConstantsInfo {};
    drawConstantsInfo.buffer = drawConstantsBuffer.buffer;
    drawConstantsInfo.offset = 0;
    drawConstantsInfo.range = drawConstantsBuffer.size;

    VkWriteDescriptorSet writes[3] {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = texture.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &imageInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = texture.descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &shadowInfo;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = texture.descriptorSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &drawConstantsInfo;
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    texture.valid = true;
    return texture;
}

void VulkanCameraSim::destroyTexture(TextureResource& texture)
{
    if (texture.sampler) vkDestroySampler(device, texture.sampler, nullptr);
    if (texture.view) vkDestroyImageView(device, texture.view, nullptr);
    if (texture.image) vkDestroyImage(device, texture.image, nullptr);
    if (texture.memory) vkFreeMemory(device, texture.memory, nullptr);
    texture = {};
}

void VulkanCameraSim::createImage(
    uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory,
    uint32_t mipLevels, VkSampleCountFlagBits samples)
{
    VkImageCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.extent = { w, h, 1 };
    info.mipLevels = mipLevels;
    info.arrayLayers = 1;
    info.format = format;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.usage = usage;
    info.samples = samples;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    checkVk(vkCreateImage(device, &info, nullptr, &image), "create image");
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, image, &req);
    VkMemoryAllocateInfo alloc {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    checkVk(vkAllocateMemory(device, &alloc, nullptr, &memory), "allocate image memory");
    checkVk(vkBindImageMemory(device, image, memory, 0), "bind image memory");
}

VkImageView VulkanCameraSim::createImageView(
    VkImage image, VkFormat format, VkImageAspectFlags aspectMask, uint32_t mipLevels)
{
    VkImageViewCreateInfo info {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.subresourceRange.aspectMask = aspectMask;
    info.subresourceRange.levelCount = mipLevels;
    info.subresourceRange.layerCount = 1;
    VkImageView view = VK_NULL_HANDLE;
    checkVk(vkCreateImageView(device, &info, nullptr, &view), "create image view");
    return view;
}

void VulkanCameraSim::loadCameraConfig(const std::string& configPath)
{
    if (!fileExists(configPath))
    {
        throw std::runtime_error("VulkanCameraSim: camera config file not found: " + configPath);
    }
    const YAML::Node root = YAML::LoadFile(configPath);
    const YAML::Node camerasNode = root["cameras"];
    if (!root || !root.IsMap() || !camerasNode || !camerasNode.IsSequence())
    {
        throw std::runtime_error("VulkanCameraSim: camera config root must contain cameras sequence");
    }

    std::vector<CameraComponent> newCameras;
    for (size_t i = 0; i < camerasNode.size(); ++i)
    {
        const YAML::Node item = camerasNode[i];
        if (!item || !item.IsMap())
        {
            continue;
        }
        const YAML::Node cfg = mergeSensorNode(item);
        CameraComponent camera;
        camera.setPerspective(fovYRad, nearClip, farClip);
        camera.setSensorName(cfg["name"] ? cfg["name"].as<std::string>() : ("camera_" + std::to_string(newCameras.size())));
        if (cfg["enabled"]) camera.setEnabled(cfg["enabled"].as<bool>());
        if (cfg["rate"])
        {
            const float rate = cfg["rate"].as<float>();
            if (rate > 0.0f) camera.setSensorRateHz(rate);
        }
        if (cfg["delay"] && cfg["delay"].IsMap() && cfg["delay"]["mean"])
        {
            camera.setSensorDelayMean(cfg["delay"]["mean"].as<float>());
        }

        CameraMount mount = camera.mount();
        if (cfg["pose"] && cfg["pose"].IsMap())
        {
            const YAML::Node pose = cfg["pose"];
            if (pose["position"] && pose["position"].IsSequence() && pose["position"].size() == 3)
            {
                mount.localPosition = Eigen::Vector3f(
                    pose["position"][0].as<float>(), pose["position"][1].as<float>(), pose["position"][2].as<float>());
            }
            if (pose["orientation"] && pose["orientation"].IsSequence() && pose["orientation"].size() == 3)
            {
                mount.pitch = pose["orientation"][1].as<float>();
                mount.yawOffset = pose["orientation"][2].as<float>();
            }
        }
        camera.setMount(mount);

        int cfgResX = widthPx;
        int cfgResY = heightPx;
        if (cfg["resolution"] && cfg["resolution"].IsMap() && cfg["resolution"]["x"] && cfg["resolution"]["y"])
        {
            cfgResX = cfg["resolution"]["x"].as<int>();
            cfgResY = cfg["resolution"]["y"].as<int>();
            if (widthPx <= 0 || heightPx <= 0)
            {
                widthPx = cfgResX;
                heightPx = cfgResY;
            }
            else if (cfgResX != widthPx || cfgResY != heightPx)
            {
                throw std::runtime_error("VulkanCameraSim: camera config resolution must match simulator resolution");
            }
        }

        CameraIntrinsics intr;
        if (cfg["intrinsics"] && cfg["intrinsics"].IsMap())
        {
            const YAML::Node n = cfg["intrinsics"];
            if (n["fx"]) intr.fx = n["fx"].as<float>();
            if (n["fy"])
            {
                intr.fy = n["fy"].as<float>();
                if (intr.fy > 0.0f)
                {
                    camera.setPerspective(2.0f * std::atan(static_cast<float>(cfgResY) / (2.0f * intr.fy)),
                        camera.nearClip(), camera.farClip());
                }
            }
            if (n["cx"]) intr.cx = n["cx"].as<float>();
            if (n["cy"]) intr.cy = n["cy"].as<float>();
        }
        if (intr.fx <= 0.0f && intr.fy > 0.0f) intr.fx = intr.fy;
        if (intr.fy <= 0.0f && intr.fx > 0.0f) intr.fy = intr.fx;
        if (intr.cx <= 0.0f) intr.cx = (static_cast<float>(cfgResX) - 1.0f) * 0.5f;
        if (intr.cy <= 0.0f) intr.cy = (static_cast<float>(cfgResY) - 1.0f) * 0.5f;
        camera.setIntrinsics(intr);
        newCameras.push_back(std::move(camera));
    }
    if (newCameras.empty())
    {
        throw std::runtime_error("VulkanCameraSim: camera config contains no usable camera entries");
    }
    cameraComponents = std::move(newCameras);
}

void VulkanCameraSim::setAssetPaths(const std::string& leftConeAssetPath, const std::string& rightConeAssetPath,
    const std::string& groundPlaneAssetPath, const std::string& skyboxAssetPath)
{
    requireExistingFile("left_cone_asset_path", leftConeAssetPath);
    requireExistingFile("right_cone_asset_path", rightConeAssetPath);
    requireExistingFile("ground_plane_asset_path", groundPlaneAssetPath);
    requireExistingFile("skybox_asset_path", skyboxAssetPath);
    leftConeAssetPathOverride = leftConeAssetPath;
    rightConeAssetPathOverride = rightConeAssetPath;
    groundPlaneAssetPathOverride = groundPlaneAssetPath;
    skyboxAssetPathOverride = skyboxAssetPath;
    loadAssetModels();
}

void VulkanCameraSim::loadAssetModels()
{
    auto release = [&](ModelRenderData& model) {
        for (auto& mesh : model.meshes)
        {
            destroyBuffer(mesh.vertexBuffer);
            destroyBuffer(mesh.indexBuffer);
        }
        model.meshes.clear();
        model.valid = false;
    };
    release(groundPlaneModel);
    release(skyModel);
    release(carModel);
    release(blueConeModel);
    release(yellowConeModel);

    loadModel(groundPlaneAssetPathOverride, groundPlaneModel);
    loadModel(skyboxAssetPathOverride, skyModel, true);
    carModelUsesUrdfFrame = false;
    if (!loadCarModelFromUrdfXacro(carXacroPathOverride, carModel))
    {
        const std::string fallbackCar = joinPath(modelRootPath, "car.glb");
        if (fileExists(fallbackCar))
        {
            loadModel(fallbackCar, carModel);
        }
    }
    else
    {
        carModelUsesUrdfFrame = true;
    }
    loadModel(leftConeAssetPathOverride, blueConeModel);
    loadModel(rightConeAssetPathOverride, yellowConeModel);
}

bool VulkanCameraSim::loadModel(const std::string& filePath, ModelRenderData& outModel, bool forceRegenerateSmoothNormals)
{
    Assimp::Importer importer;
    unsigned int flags = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices
        | aiProcess_ImproveCacheLocality | aiProcess_SortByPType | aiProcess_FlipUVs;
    if (forceRegenerateSmoothNormals)
    {
        importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, static_cast<int>(aiComponent_NORMALS));
        importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 175.0f);
        flags |= aiProcess_RemoveComponent;
    }
    else
    {
        importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 80.0f);
    }

    const aiScene* scene = importer.ReadFile(filePath, flags);
    if (scene == nullptr || scene->mRootNode == nullptr || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) != 0)
    {
        return false;
    }
    const std::string normalizedPath = normalizeSeparators(filePath);
    const bool isTrackModel = (normalizedPath.find("/track/") != std::string::npos);
    const std::string modelDir = parentDir(filePath);
    std::unordered_map<std::string, int> textureCache;
    bool hasBounds = false;
    Eigen::Vector3f boundsMin = Eigen::Vector3f::Zero();
    Eigen::Vector3f boundsMax = Eigen::Vector3f::Zero();

    std::unordered_map<unsigned int, Eigen::Matrix4f> meshTransforms;
    std::function<void(aiNode*, const Eigen::Matrix4f&)> traverse = [&](aiNode* node, const Eigen::Matrix4f& parent) {
        Eigen::Matrix4f local;
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < 4; ++c)
            {
                local(r, c) = node->mTransformation[r][c];
            }
        }
        Eigen::Matrix4f global = parent * local;
        for (unsigned int i = 0; i < node->mNumMeshes; ++i)
        {
            meshTransforms[node->mMeshes[i]] = global;
        }
        for (unsigned int i = 0; i < node->mNumChildren; ++i)
        {
            traverse(node->mChildren[i], global);
        }
    };
    traverse(scene->mRootNode, Eigen::Matrix4f::Identity());

    for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
    {
        const aiMesh* mesh = scene->mMeshes[meshIndex];
        if (!mesh || mesh->mNumVertices == 0 || mesh->mNumFaces == 0)
        {
            continue;
        }

        Eigen::Vector3f materialColor(0.7f, 0.7f, 0.7f);
        float metallicFactor = 0.0f;
        float roughnessFactor = 1.0f;
        TextureImage baseColorTexture;
        int baseColorTextureIndex = -1;
        if (mesh->mMaterialIndex < scene->mNumMaterials)
        {
            const aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
            aiColor4D color;
            if (aiGetMaterialColor(material, AI_MATKEY_BASE_COLOR, &color) == AI_SUCCESS
                || aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &color) == AI_SUCCESS)
            {
                materialColor = Eigen::Vector3f(color.r, color.g, color.b);
            }

            if (material->Get(AI_MATKEY_METALLIC_FACTOR, metallicFactor) == AI_SUCCESS)
            {
                metallicFactor = std::clamp(metallicFactor, 0.0f, 1.0f);
            }

            if (material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughnessFactor) == AI_SUCCESS)
            {
                roughnessFactor = std::clamp(roughnessFactor, 0.0f, 1.0f);
            }

            aiString matName;
            material->Get(AI_MATKEY_NAME, matName);
            if (metallicFactor == 1.0f && roughnessFactor == 1.0f && std::string(matName.C_Str()).empty())
            {
                metallicFactor = 0.0f;
                roughnessFactor = 0.8f;
            }

            aiString texturePath;
            bool hasBaseColorTex = (material->GetTexture(aiTextureType_BASE_COLOR, 0, &texturePath) == AI_SUCCESS);
            if (!hasBaseColorTex)
            {
                hasBaseColorTex = (material->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath) == AI_SUCCESS);
            }
            if (hasBaseColorTex)
            {
                std::string key = normalizeSeparators(texturePath.C_Str());
                if (!key.empty())
                {
                    auto cached = textureCache.find(key);
                    if (cached == textureCache.end())
                    {
                        TextureImage image;
                        if (key[0] == '*')
                        {
                            image = decodeEmbeddedTexture(scene->GetEmbeddedTexture(key.c_str()));
                        }
                        else
                        {
                            image = loadTextureImage(joinPath(modelDir, key));
                        }
                        int textureIndex = -1;
                        std::vector<unsigned char> rgba = textureToRgba(image);
                        if (!rgba.empty())
                        {
                            TextureResource texture = createTextureResource(rgba.data(), image.width, image.height);
                            if (texture.valid)
                            {
                                textureIndex = static_cast<int>(textures.size());
                                textures.push_back(std::move(texture));
                            }
                        }
                        cached = textureCache.emplace(key, textureIndex).first;
                    }
                    baseColorTextureIndex = cached->second;
                }
            }
        }

        const Eigen::Matrix4f meshTransform =
            meshTransforms.count(meshIndex) ? meshTransforms[meshIndex] : Eigen::Matrix4f::Identity();
        bool hasMeshBounds = false;
        Eigen::Vector3f meshMin = Eigen::Vector3f::Zero();
        Eigen::Vector3f meshMax = Eigen::Vector3f::Zero();

        std::vector<Vertex> vertices;
        vertices.reserve(mesh->mNumVertices);
        for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
        {
            const aiVector3D p = mesh->mVertices[i];
            const aiVector3D n = mesh->HasNormals() ? mesh->mNormals[i] : aiVector3D(0.0f, 0.0f, 1.0f);
            const aiVector3D uv = mesh->HasTextureCoords(0) ? mesh->mTextureCoords[0][i] : aiVector3D(0.0f, 0.0f, 0.0f);
            const Eigen::Vector3f pos(p.x, p.y, p.z);
            Eigen::Vector2f uvOut(uv.x, uv.y);
            if (isTrackModel)
            {
                uvOut = Eigen::Vector2f(1.0f - uv.y, uv.x);
            }
            if (!hasBounds)
            {
                boundsMin = pos;
                boundsMax = pos;
                hasBounds = true;
            }
            else
            {
                boundsMin = boundsMin.cwiseMin(pos);
                boundsMax = boundsMax.cwiseMax(pos);
            }
            const Eigen::Vector3f transformedPos = (meshTransform * Eigen::Vector4f(pos.x(), pos.y(), pos.z(), 1.0f)).head<3>();
            if (!hasMeshBounds)
            {
                meshMin = transformedPos;
                meshMax = transformedPos;
                hasMeshBounds = true;
            }
            else
            {
                meshMin = meshMin.cwiseMin(transformedPos);
                meshMax = meshMax.cwiseMax(transformedPos);
            }
            Eigen::Vector3f vertexColor = Eigen::Vector3f::Ones();
            if (baseColorTextureIndex < 0)
            {
                vertexColor = sampleTextureRgb(baseColorTexture, uvOut, Eigen::Vector3f::Ones());
            }
            vertices.push_back(Vertex {
                pos,
                Eigen::Vector3f(n.x, n.y, n.z),
                uvOut,
                vertexColor,
            });
        }

        std::vector<uint32_t> indices;
        indices.reserve(mesh->mNumFaces * 3U);
        for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex)
        {
            const aiFace& face = mesh->mFaces[faceIndex];
            if (face.mNumIndices == 3)
            {
                indices.push_back(face.mIndices[0]);
                indices.push_back(face.mIndices[1]);
                indices.push_back(face.mIndices[2]);
            }
        }
        if (indices.empty())
        {
            continue;
        }

        MeshRenderData renderMesh;
        renderMesh.indexCount = static_cast<uint32_t>(indices.size());
        renderMesh.baseTransform = meshTransform;
        renderMesh.name = mesh->mName.C_Str();
        renderMesh.baseColor = (baseColorTextureIndex >= 0) ? Eigen::Vector3f::Ones() : materialColor;
        renderMesh.center = hasMeshBounds ? Eigen::Vector3f(0.5f * (meshMin + meshMax)) : Eigen::Vector3f::Zero();
        renderMesh.metallicFactor = metallicFactor;
        renderMesh.roughnessFactor = roughnessFactor;
        renderMesh.textureIndex = baseColorTextureIndex;
        renderMesh.hasTexture = baseColorTextureIndex >= 0;
        renderMesh.vertexBuffer = createDeviceLocalBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        renderMesh.indexBuffer = createDeviceLocalBuffer(indices.data(), indices.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        outModel.meshes.push_back(std::move(renderMesh));
    }
    outModel.valid = !outModel.meshes.empty();
    if (outModel.valid && hasBounds)
    {
        outModel.boundsMin = boundsMin;
        outModel.boundsMax = boundsMax;
    }
    else
    {
        outModel.boundsMin = Eigen::Vector3f::Zero();
        outModel.boundsMax = Eigen::Vector3f::Zero();
    }
    return outModel.valid;
}

bool VulkanCameraSim::loadCarModelFromUrdfXacro(const std::string& xacroPath, ModelRenderData& outModel)
{
    outModel.meshes.clear();
    outModel.valid = false;
    if (xacroPath.empty() || !fileExists(xacroPath))
    {
        return false;
    }

    tinyxml2::XMLDocument document;
    if (document.LoadFile(xacroPath.c_str()) != tinyxml2::XML_SUCCESS)
    {
        return false;
    }
    const tinyxml2::XMLElement* robot = document.FirstChildElement("robot");
    if (robot == nullptr)
    {
        return false;
    }

    struct VisualEntry
    {
        std::string linkName;
        std::string meshPath;
        Eigen::Matrix4f visualTransform = Eigen::Matrix4f::Identity();
    };
    struct JointEntry
    {
        std::string parent;
        std::string child;
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    };

    std::vector<VisualEntry> visuals;
    std::vector<JointEntry> joints;
    std::unordered_set<std::string> allLinks;
    std::unordered_set<std::string> childLinks;

    for (const tinyxml2::XMLElement* link = robot->FirstChildElement("link"); link != nullptr;
         link = link->NextSiblingElement("link"))
    {
        const char* linkNameRaw = link->Attribute("name");
        if (linkNameRaw == nullptr)
        {
            continue;
        }
        const std::string linkName(linkNameRaw);
        allLinks.insert(linkName);
        for (const tinyxml2::XMLElement* visual = link->FirstChildElement("visual"); visual != nullptr;
             visual = visual->NextSiblingElement("visual"))
        {
            Eigen::Vector3f xyz = Eigen::Vector3f::Zero();
            Eigen::Vector3f rpy = Eigen::Vector3f::Zero();
            if (const tinyxml2::XMLElement* origin = visual->FirstChildElement("origin"); origin != nullptr)
            {
                xyz = parseVec3FromString(origin->Attribute("xyz"), Eigen::Vector3f::Zero());
                rpy = parseVec3FromString(origin->Attribute("rpy"), Eigen::Vector3f::Zero());
            }
            const tinyxml2::XMLElement* geometry = visual->FirstChildElement("geometry");
            const tinyxml2::XMLElement* mesh = geometry ? geometry->FirstChildElement("mesh") : nullptr;
            const char* filenameRaw = mesh ? mesh->Attribute("filename") : nullptr;
            if (filenameRaw == nullptr)
            {
                continue;
            }
            const Eigen::Vector3f scale = parseVec3FromString(mesh->Attribute("scale"), Eigen::Vector3f::Ones());
            const std::string meshPath = resolveUrdfMeshPath(filenameRaw, xacroPath);
            if (meshPath.empty())
            {
                continue;
            }
            visuals.push_back(VisualEntry { linkName, meshPath, makeUrdfTransform(xyz, rpy) * makeScaleMatrix(scale) });
        }
    }

    for (const tinyxml2::XMLElement* joint = robot->FirstChildElement("joint"); joint != nullptr;
         joint = joint->NextSiblingElement("joint"))
    {
        const tinyxml2::XMLElement* parent = joint->FirstChildElement("parent");
        const tinyxml2::XMLElement* child = joint->FirstChildElement("child");
        const char* parentName = parent ? parent->Attribute("link") : nullptr;
        const char* childName = child ? child->Attribute("link") : nullptr;
        if (parentName == nullptr || childName == nullptr)
        {
            continue;
        }
        Eigen::Vector3f xyz = Eigen::Vector3f::Zero();
        Eigen::Vector3f rpy = Eigen::Vector3f::Zero();
        if (const tinyxml2::XMLElement* origin = joint->FirstChildElement("origin"); origin != nullptr)
        {
            xyz = parseVec3FromString(origin->Attribute("xyz"), Eigen::Vector3f::Zero());
            rpy = parseVec3FromString(origin->Attribute("rpy"), Eigen::Vector3f::Zero());
        }
        joints.push_back(JointEntry { parentName, childName, makeUrdfTransform(xyz, rpy) });
        allLinks.insert(parentName);
        allLinks.insert(childName);
        childLinks.insert(childName);
    }

    std::unordered_map<std::string, Eigen::Matrix4f> linkTransforms;
    for (const auto& link : allLinks)
    {
        if (childLinks.find(link) == childLinks.end())
        {
            linkTransforms[link] = Eigen::Matrix4f::Identity();
        }
    }
    if (linkTransforms.empty() && !allLinks.empty())
    {
        linkTransforms[*allLinks.begin()] = Eigen::Matrix4f::Identity();
    }

    bool progress = true;
    while (progress)
    {
        progress = false;
        for (const auto& joint : joints)
        {
            if (linkTransforms.find(joint.child) != linkTransforms.end())
            {
                continue;
            }
            const auto parentIt = linkTransforms.find(joint.parent);
            if (parentIt == linkTransforms.end())
            {
                continue;
            }
            linkTransforms[joint.child] = parentIt->second * joint.transform;
            progress = true;
        }
    }

    bool hasBounds = false;
    Eigen::Vector3f boundsMin = Eigen::Vector3f::Zero();
    Eigen::Vector3f boundsMax = Eigen::Vector3f::Zero();

    for (const auto& visual : visuals)
    {
        if (!fileExists(visual.meshPath))
        {
            continue;
        }
        ModelRenderData partModel;
        if (!loadModel(visual.meshPath, partModel))
        {
            continue;
        }
        const auto linkIt = linkTransforms.find(visual.linkName);
        const Eigen::Matrix4f linkTransform =
            (linkIt != linkTransforms.end()) ? linkIt->second : Eigen::Matrix4f::Identity();
        const Eigen::Matrix4f visualTransform = linkTransform * visual.visualTransform;

        for (auto& mesh : partModel.meshes)
        {
            mesh.name = canonicalizeCarPartName(visual.linkName);
            const bool isSteeringWheel = (mesh.name == "Steering_Wheel");
            mesh.baseTransform = visualTransform * mesh.baseTransform;
            mesh.center =
                (visualTransform * Eigen::Vector4f(mesh.center.x(), mesh.center.y(), mesh.center.z(), 1.0f)).head<3>();
            mesh.hasTexture = false;
            mesh.textureIndex = -1;
            mesh.baseColor = isSteeringWheel
                ? Eigen::Vector3f(0.12f, 0.12f, 0.13f)
                : Eigen::Vector3f(0.82f, 0.82f, 0.84f);
            mesh.metallicFactor = 0.0f;
            mesh.roughnessFactor = isSteeringWheel ? 0.5f : 0.65f;
            outModel.meshes.push_back(std::move(mesh));
        }

        if (partModel.valid)
        {
            expandBoundsWithTransformedAabb(
                partModel.boundsMin, partModel.boundsMax, visualTransform, boundsMin, boundsMax, hasBounds);
        }
    }

    outModel.valid = !outModel.meshes.empty();
    if (outModel.valid && hasBounds)
    {
        outModel.boundsMin = boundsMin;
        outModel.boundsMax = boundsMax;
    }
    else
    {
        outModel.boundsMin = Eigen::Vector3f::Zero();
        outModel.boundsMax = Eigen::Vector3f::Zero();
    }
    return outModel.valid;
}

void VulkanCameraSim::setTrackAndCones(const Track& track)
{
    std::vector<Eigen::Vector3d> leftBoundary;
    std::vector<Eigen::Vector3d> rightBoundary;
    for (const auto& lm : track.left_lane) leftBoundary.push_back(lm.position);
    for (const auto& lm : track.right_lane) rightBoundary.push_back(lm.position);
    setTrackBoundaries(leftBoundary, rightBoundary);
    setCones(leftBoundary, rightBoundary);
}

void VulkanCameraSim::setTrackBoundaries(
    const std::vector<Eigen::Vector3d>& leftBoundary, const std::vector<Eigen::Vector3d>& rightBoundary)
{
    trackLeft.clear();
    trackRight.clear();
    for (const auto& p : leftBoundary) trackLeft.push_back(p.cast<float>());
    for (const auto& p : rightBoundary) trackRight.push_back(p.cast<float>());
    updateTrackMesh();
}

void VulkanCameraSim::setCones(
    const std::vector<Eigen::Vector3d>& blueCones, const std::vector<Eigen::Vector3d>& yellowCones)
{
    cones.clear();
    for (const auto& p : blueCones) cones.push_back(ConeInstance { p.cast<float>(), true });
    for (const auto& p : yellowCones) cones.push_back(ConeInstance { p.cast<float>(), false });
    updateConeInstanceBuffers();
}

void VulkanCameraSim::setShadowsEnabled(bool enabled)
{
    shadowsEnabled = enabled;
}

void VulkanCameraSim::updateConeInstanceBuffers()
{
    destroyBuffer(blueConeInstanceBuffer);
    destroyBuffer(yellowConeInstanceBuffer);
    blueConeInstanceCount = 0;
    yellowConeInstanceCount = 0;

    if (device == VK_NULL_HANDLE)
    {
        return;
    }

    const Eigen::Matrix4f assetAlign = modelToWorldAlignment();
    const Eigen::Matrix4f coneScale = makeScale(1.0f);
    std::vector<InstanceData> blueInstances;
    std::vector<InstanceData> yellowInstances;
    blueInstances.reserve(cones.size());
    yellowInstances.reserve(cones.size());
    for (const auto& cone : cones)
    {
        InstanceData instance;
        const Eigen::Vector3f renderPosition = cone.position + Eigen::Vector3f(0.0f, 0.0f, kConeGroundClearance);
        instance.model = makeTranslation(renderPosition) * assetAlign * coneScale;
        if (cone.isBlue)
        {
            blueInstances.push_back(instance);
        }
        else
        {
            yellowInstances.push_back(instance);
        }
    }

    if (!blueInstances.empty())
    {
        blueConeInstanceCount = static_cast<uint32_t>(blueInstances.size());
        blueConeInstanceBuffer = createDeviceLocalBuffer(blueInstances.data(), blueInstances.size() * sizeof(InstanceData),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }
    if (!yellowInstances.empty())
    {
        yellowConeInstanceCount = static_cast<uint32_t>(yellowInstances.size());
        yellowConeInstanceBuffer = createDeviceLocalBuffer(yellowInstances.data(), yellowInstances.size() * sizeof(InstanceData),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }
}

void VulkanCameraSim::updateTrackMesh()
{
    for (auto& mesh : trackMeshModel.meshes)
    {
        destroyBuffer(mesh.vertexBuffer);
        destroyBuffer(mesh.indexBuffer);
    }
    trackMeshModel.meshes.clear();
    trackMeshModel.valid = false;

    const size_t n = std::min(trackLeft.size(), trackRight.size());
    if (n < 3 || device == VK_NULL_HANDLE)
    {
        return;
    }

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vertices.reserve(n * 2);
    indices.reserve(n * 6);
    const Eigen::Vector3f up(0.0f, 0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i)
    {
        Eigen::Vector3f l = trackLeft[i];
        Eigen::Vector3f r = trackRight[i];
        l.z() = 0.0f;
        r.z() = 0.0f;
        const Eigen::Vector3f color(0.17f, 0.17f, 0.18f);
        vertices.push_back(Vertex { l, up, Eigen::Vector2f(0.0f, static_cast<float>(i) / static_cast<float>(n)), color });
        vertices.push_back(Vertex { r, up, Eigen::Vector2f(1.0f, static_cast<float>(i) / static_cast<float>(n)), color });
    }
    for (size_t i = 0; i < n; ++i)
    {
        const size_t j = (i + 1) % n;
        const uint32_t a = static_cast<uint32_t>(2 * i);
        const uint32_t b = static_cast<uint32_t>(2 * i + 1);
        const uint32_t c = static_cast<uint32_t>(2 * j);
        const uint32_t d = static_cast<uint32_t>(2 * j + 1);
        indices.insert(indices.end(), { a, b, c, b, d, c });
    }
    MeshRenderData mesh;
    mesh.indexCount = static_cast<uint32_t>(indices.size());
    mesh.baseColor = Eigen::Vector3f(0.17f, 0.17f, 0.18f);
    mesh.vertexBuffer = createDeviceLocalBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    mesh.indexBuffer = createDeviceLocalBuffer(indices.data(), indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    trackMeshModel.meshes.push_back(std::move(mesh));
    trackMeshModel.valid = true;
}

std::vector<std::vector<uint8_t>> VulkanCameraSim::render(
    const Eigen::Vector3d& carPosition, const Eigen::Vector3d& carOrientation,
    float steeringAngle, const Wheels& wheelOrientations)
{
    if (cameraComponents.empty())
    {
        throw std::runtime_error("VulkanCameraSim: no camera instances configured");
    }
    const size_t pixelCount = static_cast<size_t>(widthPx) * static_cast<size_t>(heightPx);
    const size_t imageBytes = pixelCount * 3U;
    const Eigen::Vector4f wheelOrientationVec(static_cast<float>(wheelOrientations.FL),
        static_cast<float>(wheelOrientations.FR), static_cast<float>(wheelOrientations.RL),
        static_cast<float>(wheelOrientations.RR));
    const float carYaw = static_cast<float>(carOrientation.z());
    const Eigen::Matrix3f rCar = yawMatrix(carYaw);
    const Eigen::Vector3f carPos = carPosition.cast<float>();
    const Eigen::Matrix4f assetAlign = modelToWorldAlignment();
    Eigen::Matrix4f carModelMatrix = Eigen::Matrix4f::Identity();
    if (carModelUsesUrdfFrame)
    {
        carModelMatrix = makeTranslation(carPos) * makeRotationZ(carYaw);
    }
    else
    {
        carModelMatrix = makeTranslation(carPos + rCar * Eigen::Vector3f(kCarOriginForwardOffset, 0.0f, 0.0f))
            * makeRotationZ(carYaw + kCarHeadingOffsetRad) * assetAlign;
    }

    checkVk(vkResetCommandBuffer(renderCommandBuffer, 0), "reset render batch command buffer");
    drawConstantsCount = 0;
    VkCommandBuffer cmd = renderCommandBuffer;

    VkCommandBufferBeginInfo beginInfo {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    checkVk(vkBeginCommandBuffer(cmd, &beginInfo), "begin render batch command buffer");

    const Eigen::Matrix4f lightViewProj =
        recordShadowMap(cmd, carPos, carModelMatrix, steeringAngle, wheelOrientationVec);
    for (size_t i = 0; i < cameraComponents.size(); ++i)
    {
        if (cameraComponents[i].enabled())
        {
            recordSingleCamera(cmd, cameraComponents[i], carPosition, carOrientation, steeringAngle,
                wheelOrientationVec, lightViewProj, static_cast<uint32_t>(i));
        }
    }

    VkBufferMemoryBarrier bufferBarrier {};
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bufferBarrier.buffer = rgbStorageBuffer.buffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = rgbStorageBuffer.size;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 1, &bufferBarrier, 0, nullptr);

    VkBufferCopy copyRegion {};
    copyRegion.size = std::min(rgbStorageBuffer.size, readbackBuffer.size);
    vkCmdCopyBuffer(cmd, rgbStorageBuffer.buffer, readbackBuffer.buffer, 1, &copyRegion);

    VkBufferMemoryBarrier readbackBarrier {};
    readbackBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    readbackBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    readbackBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    readbackBarrier.buffer = readbackBuffer.buffer;
    readbackBarrier.offset = 0;
    readbackBarrier.size = readbackBuffer.size;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 0, nullptr, 1, &readbackBarrier, 0, nullptr);

    checkVk(vkEndCommandBuffer(cmd), "end render batch command buffer");
    VkSubmitInfo submit {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    checkVk(vkResetFences(device, 1, &renderFence), "reset render fence");
    checkVk(vkQueueSubmit(graphicsQueue, 1, &submit, renderFence), "submit render batch");
    checkVk(vkWaitForFences(device, 1, &renderFence, VK_TRUE, UINT64_MAX), "wait render batch");
    if ((readbackBuffer.memoryProperties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
    {
        VkMappedMemoryRange range {};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = readbackBuffer.memory;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        checkVk(vkInvalidateMappedMemoryRanges(device, 1, &range), "invalidate readback buffer");
    }

    const size_t rgbPackBytesPerImage = ((pixelCount + 3U) / 4U) * 12U;
    const auto* bytes = static_cast<const uint8_t*>(readbackMapped);
    std::vector<std::vector<uint8_t>> out;
    out.reserve(cameraComponents.size());
    for (size_t i = 0; i < cameraComponents.size(); ++i)
    {
        if (cameraComponents[i].enabled())
        {
            const uint8_t* imageStart = bytes + i * rgbPackBytesPerImage;
            out.emplace_back(imageStart, imageStart + imageBytes);
        }
        else
        {
            out.emplace_back(imageBytes, 0);
        }
    }
    return out;
}

Eigen::Matrix4f VulkanCameraSim::recordShadowMap(VkCommandBuffer cmd, const Eigen::Vector3f& carPos,
    const Eigen::Matrix4f& carModelMatrix, float steeringAngle, const Eigen::Vector4f& wheelOrientations)
{
    Eigen::Matrix4f glToVk = Eigen::Matrix4f::Identity();
    glToVk(1, 1) = -1.0f;
    glToVk(2, 2) = 0.5f;
    glToVk(2, 3) = 0.5f;

    const Eigen::Vector3f lightDir = Eigen::Vector3f(100.0f, 0.0f, -10.0f).normalized();
    const Eigen::Vector3f lightPos = carPos - lightDir * kShadowDistance;
    Eigen::Vector3f up(0.0f, 0.0f, 1.0f);
    if (std::abs(lightDir.dot(up)) > 0.99f)
    {
        up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    }
    const Eigen::Matrix4f lightView = makeLookAt(lightPos, carPos, up);
    const Eigen::Matrix4f lightProj =
        makeOrtho(-kShadowDistance, kShadowDistance, -kShadowDistance, kShadowDistance, kShadowNear, kShadowFar);
    const Eigen::Matrix4f lightViewProj = glToVk * lightProj * lightView;

    if (!shadowsEnabled || shadowFramebuffer == VK_NULL_HANDLE || shadowPipeline == VK_NULL_HANDLE)
    {
        return lightViewProj;
    }

    VkClearValue clear {};
    clear.depthStencil = { 1.0f, 0 };
    VkRenderPassBeginInfo rp {};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = shadowRenderPass;
    rp.framebuffer = shadowFramebuffer;
    rp.renderArea.extent = { kShadowMapSize, kShadowMapSize };
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);

    const Eigen::Matrix4f assetAlign = modelToWorldAlignment();
    if (groundPlaneModel.valid)
    {
        recordDrawModel(cmd, groundPlaneModel, assetAlign, lightViewProj,
            Eigen::Vector3f::Zero(), false, Eigen::Vector3f::Zero(), false, lightViewProj);
    }
    else if (trackMeshModel.valid)
    {
        recordDrawModel(cmd, trackMeshModel, Eigen::Matrix4f::Identity(), lightViewProj,
            Eigen::Vector3f(0.17f, 0.17f, 0.18f), true, Eigen::Vector3f::Zero(), false, lightViewProj);
    }
    if (carModel.valid)
    {
        recordDrawModel(cmd, carModel, carModelMatrix, lightViewProj,
            Eigen::Vector3f::Zero(), false, Eigen::Vector3f::Zero(), false,
            lightViewProj, steeringAngle, wheelOrientations);
    }

    if (blueConeModel.valid && blueConeInstanceBuffer.buffer != VK_NULL_HANDLE && blueConeInstanceCount > 0)
    {
        recordDrawInstancedModel(cmd, blueConeModel, blueConeInstanceBuffer, blueConeInstanceCount, lightViewProj,
            Eigen::Vector3f(40.0f / 255.0f, 96.0f / 255.0f, 220.0f / 255.0f),
            false, Eigen::Vector3f::Zero(), false, lightViewProj);
    }
    if (yellowConeModel.valid && yellowConeInstanceBuffer.buffer != VK_NULL_HANDLE && yellowConeInstanceCount > 0)
    {
        recordDrawInstancedModel(cmd, yellowConeModel, yellowConeInstanceBuffer, yellowConeInstanceCount, lightViewProj,
            Eigen::Vector3f(246.0f / 255.0f, 210.0f / 255.0f, 72.0f / 255.0f),
            false, Eigen::Vector3f::Zero(), false, lightViewProj);
    }

    vkCmdEndRenderPass(cmd);

    VkImageMemoryBarrier shadowBarrier {};
    shadowBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    shadowBarrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    shadowBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    shadowBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    shadowBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    shadowBarrier.image = shadowDepthImage;
    shadowBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    shadowBarrier.subresourceRange.levelCount = 1;
    shadowBarrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &shadowBarrier);
    return lightViewProj;
}

void VulkanCameraSim::recordSingleCamera(VkCommandBuffer cmd,
    const CameraComponent& camera, const Eigen::Vector3d& carPosition, const Eigen::Vector3d& carOrientation,
    float steeringAngle, const Eigen::Vector4f& wheelOrientations, const Eigen::Matrix4f& lightViewProj,
    uint32_t cameraSlot)
{
    const size_t pixelCount = static_cast<size_t>(widthPx) * static_cast<size_t>(heightPx);
    if (!camera.enabled())
    {
        return;
    }

    const float carYaw = static_cast<float>(carOrientation.z());
    const Eigen::Matrix3f rCar = yawMatrix(carYaw);
    const Eigen::Vector3f carPos = carPosition.cast<float>();
    const CameraMount& mount = camera.mount();
    const Eigen::Vector3f camPos = carPos + rCar * mount.localPosition;
    const float camYaw = carYaw + mount.yawOffset;
    const float cp = std::cos(mount.pitch);
    const float sp = std::sin(mount.pitch);
    Eigen::Vector3f forward(std::cos(camYaw) * cp, std::sin(camYaw) * cp, -sp);
    forward.normalize();
    Eigen::Vector3f up(0.0f, 0.0f, 1.0f);
    if (std::abs(forward.dot(up)) > 0.99f)
    {
        up = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    }

    const Eigen::Matrix4f view = makeLookAt(camPos, camPos + forward, up);
    const float aspect = static_cast<float>(widthPx) / static_cast<float>(heightPx);
    const CameraIntrinsics& intr = camera.intrinsics();
    const Eigen::Matrix4f projGl = (intr.fx > 0.0f && intr.fy > 0.0f)
        ? makePerspectiveFromIntrinsics(intr, widthPx, heightPx, camera.nearClip(), camera.farClip())
        : makePerspective(camera.fovYRad(), aspect, camera.nearClip(), camera.farClip());
    const Eigen::Matrix4f skyProjGl = (intr.fx > 0.0f && intr.fy > 0.0f)
        ? makePerspectiveFromIntrinsics(intr, widthPx, heightPx, camera.nearClip(), 10000.0f)
        : makePerspective(camera.fovYRad(), aspect, camera.nearClip(), 10000.0f);
    Eigen::Matrix4f glToVk = Eigen::Matrix4f::Identity();
    glToVk(1, 1) = -1.0f;
    glToVk(2, 2) = 0.5f;
    glToVk(2, 3) = 0.5f;
    const Eigen::Matrix4f viewProj = glToVk * projGl * view;
    const Eigen::Matrix4f skyViewProj = glToVk * skyProjGl * view;

    VkClearValue clears[2] {};
    clears[0].color = { { 0.24f, 0.37f, 0.55f, 1.0f } };
    clears[1].depthStencil = { 1.0f, 0 };
    VkRenderPassBeginInfo rp {};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = renderPass;
    rp.framebuffer = framebuffer;
    rp.renderArea.extent = { static_cast<uint32_t>(widthPx), static_cast<uint32_t>(heightPx) };
    rp.clearValueCount = 2;
    rp.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    const Eigen::Matrix4f assetAlign = modelToWorldAlignment();
    Eigen::Matrix4f carModelMatrix = Eigen::Matrix4f::Identity();
    if (carModelUsesUrdfFrame)
    {
        carModelMatrix = makeTranslation(carPos) * makeRotationZ(carYaw);
    }
    else
    {
        carModelMatrix = makeTranslation(carPos + rCar * Eigen::Vector3f(kCarOriginForwardOffset, 0.0f, 0.0f))
            * makeRotationZ(carYaw + kCarHeadingOffsetRad) * assetAlign;
    }
    const Eigen::Matrix4f skyModelMatrix = makeTranslation(Eigen::Vector3f(0.0f, 0.0f, -20.0f)) * assetAlign * makeScale(3000.0f);

    if (skyModel.valid)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline);
        recordDrawModel(cmd, skyModel, skyModelMatrix, skyViewProj,
            Eigen::Vector3f::Ones(), false, camPos, true, lightViewProj);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    if (carModel.valid)
    {
        recordDrawModel(cmd, carModel, carModelMatrix, viewProj,
            Eigen::Vector3f::Zero(), false, camPos, false, lightViewProj, steeringAngle, wheelOrientations);
    }

    if (blueConeModel.valid && blueConeInstanceBuffer.buffer != VK_NULL_HANDLE && blueConeInstanceCount > 0)
    {
        recordDrawInstancedModel(cmd, blueConeModel, blueConeInstanceBuffer, blueConeInstanceCount, viewProj,
            Eigen::Vector3f(40.0f / 255.0f, 96.0f / 255.0f, 220.0f / 255.0f),
            false, camPos, false, lightViewProj);
    }
    if (yellowConeModel.valid && yellowConeInstanceBuffer.buffer != VK_NULL_HANDLE && yellowConeInstanceCount > 0)
    {
        recordDrawInstancedModel(cmd, yellowConeModel, yellowConeInstanceBuffer, yellowConeInstanceCount, viewProj,
            Eigen::Vector3f(246.0f / 255.0f, 210.0f / 255.0f, 72.0f / 255.0f),
            false, camPos, false, lightViewProj);
    }

    if (groundPlaneModel.valid)
    {
        recordDrawModel(cmd, groundPlaneModel, assetAlign, viewProj,
            Eigen::Vector3f::Zero(), false, camPos, false, lightViewProj);
    }
    else if (trackMeshModel.valid)
    {
        recordDrawModel(cmd, trackMeshModel, Eigen::Matrix4f::Identity(), viewProj,
            Eigen::Vector3f(0.17f, 0.17f, 0.18f), true, camPos, false, lightViewProj);
    }

    vkCmdEndRenderPass(cmd);

    VkImageMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.image = colorImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rgbComputePipeline);
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rgbComputePipelineLayout, 0, 1, &rgbComputeDescriptorSet, 0, nullptr);
    const uint32_t packGroups = static_cast<uint32_t>((pixelCount + 3U) / 4U);
    const uint32_t baseWord = cameraSlot * packGroups * 3U;
    vkCmdPushConstants(cmd, rgbComputePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(baseWord), &baseWord);
    vkCmdDispatch(cmd, (packGroups + 255U) / 256U, 1, 1);

    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void VulkanCameraSim::recordDrawModel(VkCommandBuffer cmd, const ModelRenderData& model,
    const Eigen::Matrix4f& modelMatrix, const Eigen::Matrix4f& viewProj, const Eigen::Vector3f& colorOverride,
    bool useOverrideColor, const Eigen::Vector3f& cameraPosition, bool unlit,
    const Eigen::Matrix4f& lightViewProj, float steeringAngle, const Eigen::Vector4f& wheelOrientations) const
{
    for (const auto& mesh : model.meshes)
    {
        Eigen::Matrix4f localTransform = Eigen::Matrix4f::Identity();
        const bool isSteeringWheel = mesh.name.find("Steering_Wheel") != std::string::npos;
        const bool isInside = mesh.name.find("_Inside") != std::string::npos;
        const bool isOutside = mesh.name.find("_Outside") != std::string::npos;

        if (isSteeringWheel)
        {
            localTransform =
                makeTranslation(mesh.center) * makeRotationX(-steeringAngle) * makeTranslation(-mesh.center);
        }
        else if (isInside || isOutside)
        {
            const float midX = 0.5f * (model.boundsMin.x() + model.boundsMax.x());
            const float midZ = 0.5f * (model.boundsMin.z() + model.boundsMax.z());
            const bool nameHasFl = mesh.name.find("FL_") != std::string::npos;
            const bool nameHasFr = mesh.name.find("FR_") != std::string::npos;
            const bool nameHasRl = mesh.name.find("RL_") != std::string::npos;
            const bool nameHasRr = mesh.name.find("RR_") != std::string::npos;
            const bool hasNamedWheel = nameHasFl || nameHasFr || nameHasRl || nameHasRr;
            const bool isFront = hasNamedWheel ? (nameHasFl || nameHasFr) : (mesh.center.x() > midX);
            const bool isLeft = hasNamedWheel ? (nameHasFl || nameHasRl) : (mesh.center.z() < midZ);

            const float steer = isFront ? (steeringAngle * 0.23f) : 0.0f;
            const float wheelSpin =
                (isFront && isLeft) ? wheelOrientations[0]
                : (isFront && !isLeft) ? wheelOrientations[1]
                : (!isFront && isLeft) ? wheelOrientations[2]
                : wheelOrientations[3];
            const Eigen::Matrix4f steerRotation =
                carModelUsesUrdfFrame ? makeRotationZ(steer) : makeRotationY(steer);
            const Eigen::Matrix4f spinRotation =
                carModelUsesUrdfFrame ? makeRotationY(wheelSpin) : makeRotationZ(wheelSpin);

            if (isInside)
            {
                localTransform =
                    makeTranslation(mesh.center) * steerRotation * makeTranslation(-mesh.center);
            }
            else
            {
                localTransform =
                    makeTranslation(mesh.center) * steerRotation * spinRotation * makeTranslation(-mesh.center);
            }
        }
        const Eigen::Vector3f color = useOverrideColor ? colorOverride : mesh.baseColor;
        recordDrawMesh(cmd, mesh, modelMatrix * localTransform * mesh.baseTransform,
            viewProj, color, cameraPosition, unlit, lightViewProj);
    }
}

void VulkanCameraSim::recordDrawInstancedModel(VkCommandBuffer cmd, const ModelRenderData& model,
    const GpuBuffer& instanceBuffer, uint32_t instanceCount, const Eigen::Matrix4f& viewProj,
    const Eigen::Vector3f& colorOverride, bool useOverrideColor, const Eigen::Vector3f& cameraPosition, bool unlit,
    const Eigen::Matrix4f& lightViewProj) const
{
    if (instanceCount == 0)
    {
        return;
    }
    for (const auto& mesh : model.meshes)
    {
        const Eigen::Vector3f color = useOverrideColor ? colorOverride : mesh.baseColor;
        recordDrawMesh(cmd, mesh, mesh.baseTransform, viewProj, color, cameraPosition, unlit, lightViewProj,
            &instanceBuffer, instanceCount);
    }
}

void VulkanCameraSim::recordDrawMesh(VkCommandBuffer cmd, const MeshRenderData& mesh,
    const Eigen::Matrix4f& modelMatrix, const Eigen::Matrix4f& viewProj, const Eigen::Vector3f& color,
    const Eigen::Vector3f& cameraPosition, bool unlit, const Eigen::Matrix4f& lightViewProj,
    const GpuBuffer* instanceBuffer, uint32_t instanceCount) const
{
    if (mesh.indexCount == 0 || instanceCount == 0)
    {
        return;
    }
    PushConstants pc;
    pc.mvp = viewProj;
    pc.model = modelMatrix;
    pc.lightViewProj = lightViewProj;
    pc.color = Eigen::Vector4f(color.x(), color.y(), color.z(), 1.0f);
    pc.params = Eigen::Vector4f(mesh.hasTexture ? 1.0f : 0.0f, unlit ? 1.0f : 0.0f,
        (shadowsEnabled && shadowDepthImageView != VK_NULL_HANDLE) ? 1.0f : 0.0f, 0.0f);
    pc.material = Eigen::Vector4f(mesh.metallicFactor, mesh.roughnessFactor, 0.0f, 0.0f);
    pc.viewParams = Eigen::Vector4f(cameraPosition.x(), cameraPosition.y(), cameraPosition.z(), 0.0f);
    const uint32_t drawIndex = appendDrawConstants(pc);
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(drawIndex), &drawIndex);
    VkDescriptorSet descriptorSet = whiteTexture.descriptorSet;
    if (mesh.hasTexture && mesh.textureIndex >= 0 && mesh.textureIndex < static_cast<int>(textures.size())
        && textures[static_cast<size_t>(mesh.textureIndex)].valid)
    {
        descriptorSet = textures[static_cast<size_t>(mesh.textureIndex)].descriptorSet;
    }
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    const VkBuffer instanceVkBuffer = (instanceBuffer != nullptr) ? instanceBuffer->buffer : identityInstanceBuffer.buffer;
    VkBuffer vertexBuffers[2] = { mesh.vertexBuffer.buffer, instanceVkBuffer };
    VkDeviceSize offsets[2] = { 0, 0 };
    vkCmdBindVertexBuffers(cmd, 0, 2, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, mesh.indexCount, instanceCount, 0, 0, 0);
}

int VulkanCameraSim::width() const { return widthPx; }
int VulkanCameraSim::height() const { return heightPx; }
size_t VulkanCameraSim::cameraCount() const { return cameraComponents.size(); }

std::vector<std::string> VulkanCameraSim::cameraNames() const
{
    std::vector<std::string> names;
    for (const auto& camera : cameraComponents) names.push_back(camera.sensorName());
    return names;
}

std::vector<bool> VulkanCameraSim::cameraEnabledFlags() const
{
    std::vector<bool> flags;
    for (const auto& camera : cameraComponents) flags.push_back(camera.enabled());
    return flags;
}

std::vector<float> VulkanCameraSim::cameraRatesHz() const
{
    std::vector<float> rates;
    for (const auto& camera : cameraComponents) rates.push_back(camera.sensorRateHz());
    return rates;
}

std::vector<float> VulkanCameraSim::cameraDelayMeans() const
{
    std::vector<float> delays;
    for (const auto& camera : cameraComponents) delays.push_back(camera.sensorDelayMean());
    return delays;
}

std::vector<VulkanCameraSim::CameraIntrinsics> VulkanCameraSim::cameraIntrinsics() const
{
    std::vector<CameraIntrinsics> intrinsics;
    for (const auto& camera : cameraComponents) intrinsics.push_back(camera.intrinsics());
    return intrinsics;
}

std::vector<Eigen::Vector3d> VulkanCameraSim::cameraLocalPositions() const
{
    std::vector<Eigen::Vector3d> positions;
    for (const auto& camera : cameraComponents) positions.push_back(camera.mount().localPosition.cast<double>());
    return positions;
}

std::vector<Eigen::Vector3d> VulkanCameraSim::cameraLocalOrientations() const
{
    std::vector<Eigen::Vector3d> orientations;
    for (const auto& camera : cameraComponents)
    {
        const auto& mount = camera.mount();
        orientations.emplace_back(0.0, static_cast<double>(mount.pitch), static_cast<double>(mount.yawOffset));
    }
    return orientations;
}

Eigen::Matrix3f VulkanCameraSim::yawMatrix(float yaw)
{
    const float c = std::cos(yaw);
    const float s = std::sin(yaw);
    Eigen::Matrix3f m = Eigen::Matrix3f::Identity();
    m(0, 0) = c;
    m(0, 1) = -s;
    m(1, 0) = s;
    m(1, 1) = c;
    return m;
}

Eigen::Matrix4f VulkanCameraSim::modelToWorldAlignment()
{
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 0) = 0.0f; m(0, 1) = 0.0f; m(0, 2) = 1.0f;
    m(1, 0) = 1.0f; m(1, 1) = 0.0f; m(1, 2) = 0.0f;
    m(2, 0) = 0.0f; m(2, 1) = 1.0f; m(2, 2) = 0.0f;
    return m;
}

Eigen::Matrix4f VulkanCameraSim::makePerspective(float fovyRad, float aspect, float zNear, float zFar)
{
    const float f = 1.0f / std::tan(0.5f * fovyRad);
    Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
    m(0, 0) = f / aspect;
    m(1, 1) = f;
    m(2, 2) = (zFar + zNear) / (zNear - zFar);
    m(2, 3) = (2.0f * zFar * zNear) / (zNear - zFar);
    m(3, 2) = -1.0f;
    return m;
}

Eigen::Matrix4f VulkanCameraSim::makePerspectiveFromIntrinsics(
    const CameraIntrinsics& intrinsics, int widthPxIn, int heightPxIn, float zNear, float zFar)
{
    const float w = static_cast<float>(widthPxIn);
    const float h = static_cast<float>(heightPxIn);
    Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
    m(0, 0) = (2.0f * intrinsics.fx) / w;
    m(1, 1) = (2.0f * intrinsics.fy) / h;
    m(0, 2) = 1.0f - (2.0f * intrinsics.cx) / w;
    m(1, 2) = (2.0f * intrinsics.cy) / h - 1.0f;
    m(2, 2) = (zFar + zNear) / (zNear - zFar);
    m(2, 3) = (2.0f * zFar * zNear) / (zNear - zFar);
    m(3, 2) = -1.0f;
    return m;
}

Eigen::Matrix4f VulkanCameraSim::makeOrtho(float left, float right, float bottom, float top, float zNear, float zFar)
{
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 0) = 2.0f / (right - left);
    m(1, 1) = 2.0f / (top - bottom);
    m(2, 2) = -2.0f / (zFar - zNear);
    m(0, 3) = -(right + left) / (right - left);
    m(1, 3) = -(top + bottom) / (top - bottom);
    m(2, 3) = -(zFar + zNear) / (zFar - zNear);
    return m;
}

Eigen::Matrix4f VulkanCameraSim::makeLookAt(
    const Eigen::Vector3f& eye, const Eigen::Vector3f& center, const Eigen::Vector3f& up)
{
    const Eigen::Vector3f f = (center - eye).normalized();
    const Eigen::Vector3f s = f.cross(up).normalized();
    const Eigen::Vector3f u = s.cross(f);
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 0) = s.x(); m(0, 1) = s.y(); m(0, 2) = s.z();
    m(1, 0) = u.x(); m(1, 1) = u.y(); m(1, 2) = u.z();
    m(2, 0) = -f.x(); m(2, 1) = -f.y(); m(2, 2) = -f.z();
    m(0, 3) = -s.dot(eye);
    m(1, 3) = -u.dot(eye);
    m(2, 3) = f.dot(eye);
    return m;
}

Eigen::Matrix4f VulkanCameraSim::makeTranslation(const Eigen::Vector3f& t)
{
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 3) = t.x();
    m(1, 3) = t.y();
    m(2, 3) = t.z();
    return m;
}

Eigen::Matrix4f VulkanCameraSim::makeRotationX(float angleRad)
{
    const float c = std::cos(angleRad);
    const float s = std::sin(angleRad);
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(1, 1) = c; m(1, 2) = -s;
    m(2, 1) = s; m(2, 2) = c;
    return m;
}

Eigen::Matrix4f VulkanCameraSim::makeRotationY(float angleRad)
{
    const float c = std::cos(angleRad);
    const float s = std::sin(angleRad);
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 0) = c; m(0, 2) = s;
    m(2, 0) = -s; m(2, 2) = c;
    return m;
}

Eigen::Matrix4f VulkanCameraSim::makeRotationZ(float yawRad)
{
    const float c = std::cos(yawRad);
    const float s = std::sin(yawRad);
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 0) = c; m(0, 1) = -s;
    m(1, 0) = s; m(1, 1) = c;
    return m;
}

Eigen::Matrix4f VulkanCameraSim::makeScale(float s)
{
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 0) = s;
    m(1, 1) = s;
    m(2, 2) = s;
    return m;
}
