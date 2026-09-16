#ifndef PACSIMVULKANCAMERASIM_HPP
#define PACSIMVULKANCAMERASIM_HPP

#include "types.hpp"

#include <Eigen/Core>
#include <shaderc/shaderc.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class VulkanCameraSim
{
public:
    struct CameraIntrinsics
    {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
    };

    struct RuntimeOptions
    {
        bool shadowsEnabled = true;
        std::string cameraConfigPath;
        std::string carXacroPath;
        std::string modelRoot;
        std::string leftConeAssetPath;
        std::string rightConeAssetPath;
        std::string groundPlaneAssetPath;
        std::string skyboxAssetPath;
    };

    VulkanCameraSim(int width = 306, int height = 256,
        const std::string& modelRoot = "/root/workspace/pipeline/Models",
        const std::string& cameraConfigPath = "",
        const std::string& carXacroPath = "");
    ~VulkanCameraSim();

    VulkanCameraSim(const VulkanCameraSim&) = delete;
    VulkanCameraSim& operator=(const VulkanCameraSim&) = delete;

    void setAssetPaths(const std::string& leftConeAssetPath, const std::string& rightConeAssetPath,
        const std::string& groundPlaneAssetPath, const std::string& skyboxAssetPath);
    void setTrackAndCones(const Track& track);
    void setShadowsEnabled(bool enabled);

    std::vector<std::vector<uint8_t>> render(
        const Eigen::Vector3d& carPosition, const Eigen::Vector3d& carOrientation,
        float steeringAngle, const Wheels& wheelOrientations);

    int width() const;
    int height() const;
    size_t cameraCount() const;
    std::vector<std::string> cameraNames() const;
    std::vector<bool> cameraEnabledFlags() const;
    std::vector<float> cameraRatesHz() const;
    std::vector<float> cameraDelayMeans() const;
    std::vector<CameraIntrinsics> cameraIntrinsics() const;
    std::vector<Eigen::Vector3d> cameraLocalPositions() const;
    std::vector<Eigen::Vector3d> cameraLocalOrientations() const;

private:
    struct Vertex
    {
        Eigen::Vector3f position;
        Eigen::Vector3f normal;
        Eigen::Vector2f uv;
        Eigen::Vector3f color;
    };

    struct GpuBuffer
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        VkMemoryPropertyFlags memoryProperties = 0;
    };

    struct MeshRenderData
    {
        GpuBuffer vertexBuffer;
        GpuBuffer indexBuffer;
        uint32_t indexCount = 0;
        Eigen::Matrix4f baseTransform = Eigen::Matrix4f::Identity();
        Eigen::Vector3f baseColor = Eigen::Vector3f(0.7f, 0.7f, 0.7f);
        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        float metallicFactor = 0.0f;
        float roughnessFactor = 1.0f;
        int textureIndex = -1;
        bool hasTexture = false;
        std::string name;
    };

    struct TextureResource
    {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        uint32_t mipLevels = 1;
        bool valid = false;
    };

    struct ModelRenderData
    {
        std::vector<MeshRenderData> meshes;
        Eigen::Vector3f boundsMin = Eigen::Vector3f::Zero();
        Eigen::Vector3f boundsMax = Eigen::Vector3f::Zero();
        bool valid = false;
    };

    struct ConeInstance
    {
        Eigen::Vector3f position;
        bool isBlue;
    };

    struct InstanceData
    {
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    };

    struct CameraMount
    {
        float yawOffset = 0.0f;
        float pitch = 0.0f;
        Eigen::Vector3f localPosition = Eigen::Vector3f::Zero();
    };

    class CameraComponent
    {
    public:
        CameraComponent();
        void setMount(const CameraMount& mount);
        const CameraMount& mount() const;
        void setPerspective(float fovYRad, float nearClip, float farClip);
        float fovYRad() const;
        float nearClip() const;
        float farClip() const;
        void setEnabled(bool enabled);
        bool enabled() const;
        void setSensorRateHz(float rateHz);
        float sensorRateHz() const;
        void setSensorDelayMean(float delayMeanSec);
        float sensorDelayMean() const;
        void setSensorName(const std::string& name);
        const std::string& sensorName() const;
        void setIntrinsics(const CameraIntrinsics& intrinsics);
        const CameraIntrinsics& intrinsics() const;

    private:
        CameraMount mountConfig;
        float projectionFovYRad;
        float projectionNearClip;
        float projectionFarClip;
        bool sensorEnabled;
        float sensorRateHzValue;
        float sensorDelayMeanSecValue;
        std::string sensorNameValue;
        CameraIntrinsics intrinsicsConfig;
    };

    struct PushConstants
    {
        Eigen::Matrix4f mvp;
        Eigen::Matrix4f model;
        Eigen::Matrix4f lightViewProj;
        Eigen::Vector4f color;
        Eigen::Vector4f params;
        Eigen::Vector4f material;
        Eigen::Vector4f viewParams;
    };

    void initializeVulkan();
    void initializeFramebuffer();
    void initializePipeline();
    void destroyVulkan();
    void loadCameraConfig(const std::string& configPath);
    void loadAssetModels();
    bool loadModel(const std::string& filePath, ModelRenderData& outModel, bool forceRegenerateSmoothNormals = false);
    bool loadCarModelFromUrdfXacro(const std::string& xacroPath, ModelRenderData& outModel);
    void updateTrackMesh();
    void setTrackBoundaries(
        const std::vector<Eigen::Vector3d>& leftBoundary, const std::vector<Eigen::Vector3d>& rightBoundary);
    void setCones(const std::vector<Eigen::Vector3d>& blueCones, const std::vector<Eigen::Vector3d>& yellowCones);

    void recordSingleCamera(VkCommandBuffer cmd,
        const CameraComponent& camera, const Eigen::Vector3d& carPosition, const Eigen::Vector3d& carOrientation,
        float steeringAngle, const Eigen::Vector4f& wheelOrientations, const Eigen::Matrix4f& lightViewProj,
        uint32_t cameraSlot);
    Eigen::Matrix4f recordShadowMap(VkCommandBuffer cmd, const Eigen::Vector3f& carPos, const Eigen::Matrix4f& carModelMatrix,
        float steeringAngle, const Eigen::Vector4f& wheelOrientations);
    void updateConeInstanceBuffers();
    void recordDrawModel(VkCommandBuffer cmd, const ModelRenderData& model,
        const Eigen::Matrix4f& modelMatrix, const Eigen::Matrix4f& viewProj, const Eigen::Vector3f& colorOverride,
        bool useOverrideColor, const Eigen::Vector3f& cameraPosition, bool unlit = false,
        const Eigen::Matrix4f& lightViewProj = Eigen::Matrix4f::Identity(), float steeringAngle = 0.0f,
        const Eigen::Vector4f& wheelOrientations = Eigen::Vector4f::Zero()) const;
    void recordDrawInstancedModel(VkCommandBuffer cmd, const ModelRenderData& model, const GpuBuffer& instanceBuffer,
        uint32_t instanceCount, const Eigen::Matrix4f& viewProj, const Eigen::Vector3f& colorOverride,
        bool useOverrideColor, const Eigen::Vector3f& cameraPosition, bool unlit = false,
        const Eigen::Matrix4f& lightViewProj = Eigen::Matrix4f::Identity()) const;
    void recordDrawMesh(VkCommandBuffer cmd, const MeshRenderData& mesh,
        const Eigen::Matrix4f& modelMatrix, const Eigen::Matrix4f& viewProj, const Eigen::Vector3f& color,
        const Eigen::Vector3f& cameraPosition, bool unlit = false,
        const Eigen::Matrix4f& lightViewProj = Eigen::Matrix4f::Identity(), const GpuBuffer* instanceBuffer = nullptr,
        uint32_t instanceCount = 1) const;

    GpuBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags requiredProperties,
        VkMemoryPropertyFlags preferredProperties = 0);
    GpuBuffer createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage);
    void uploadToBuffer(const GpuBuffer& buffer, const void* data, VkDeviceSize size);
    void copyBuffer(const GpuBuffer& src, const GpuBuffer& dst, VkDeviceSize size);
    void destroyBuffer(GpuBuffer& buffer);
    uint32_t appendDrawConstants(const PushConstants& constants) const;
    TextureResource createTextureResource(
        const unsigned char* rgbaPixels, int textureWidth, int textureHeight);
    void destroyTexture(TextureResource& texture);
    VkCommandBuffer beginOneTimeCommands();
    void endOneTimeCommands(VkCommandBuffer commandBuffer);
    void createImage(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage,
        VkImage& image, VkDeviceMemory& memory, uint32_t mipLevels = 1,
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);
    VkImageView createImageView(
        VkImage image, VkFormat format, VkImageAspectFlags aspectMask, uint32_t mipLevels = 1);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags requiredProperties,
        VkMemoryPropertyFlags preferredProperties = 0) const;
    VkShaderModule createShaderModule(const std::vector<uint32_t>& code) const;
    std::vector<uint32_t> compileShader(const std::string& source, shaderc_shader_kind kind, const std::string& name) const;

    static Eigen::Matrix4f makePerspective(float fovyRad, float aspect, float zNear, float zFar);
    static Eigen::Matrix4f makePerspectiveFromIntrinsics(
        const CameraIntrinsics& intrinsics, int widthPx, int heightPx, float zNear, float zFar);
    static Eigen::Matrix4f makeOrtho(float left, float right, float bottom, float top, float zNear, float zFar);
    static Eigen::Matrix4f makeLookAt(
        const Eigen::Vector3f& eye, const Eigen::Vector3f& center, const Eigen::Vector3f& up);
    static Eigen::Matrix4f makeTranslation(const Eigen::Vector3f& t);
    static Eigen::Matrix4f makeRotationX(float angleRad);
    static Eigen::Matrix4f makeRotationY(float angleRad);
    static Eigen::Matrix4f makeRotationZ(float yawRad);
    static Eigen::Matrix4f makeScale(float s);
    static Eigen::Matrix4f modelToWorldAlignment();
    static Eigen::Matrix3f yawMatrix(float yaw);

    int widthPx;
    int heightPx;
    float nearClip;
    float farClip;
    float fovYRad;
    std::string modelRootPath;
    std::string cameraConfigPathValue;
    std::string carXacroPathOverride;
    std::string leftConeAssetPathOverride;
    std::string rightConeAssetPathOverride;
    std::string groundPlaneAssetPathOverride;
    std::string skyboxAssetPathOverride;
    bool shadowsEnabled;

    std::vector<CameraComponent> cameraComponents;
    std::vector<ConeInstance> cones;
    std::vector<Eigen::Vector3f> trackLeft;
    std::vector<Eigen::Vector3f> trackRight;

    ModelRenderData groundPlaneModel;
    ModelRenderData skyModel;
    ModelRenderData carModel;
    bool carModelUsesUrdfFrame = false;
    ModelRenderData blueConeModel;
    ModelRenderData yellowConeModel;
    ModelRenderData trackMeshModel;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily = 0;
    bool samplerAnisotropyEnabled = false;
    float samplerMaxAnisotropy = 1.0f;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer renderCommandBuffer = VK_NULL_HANDLE;
    VkFence renderFence = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout textureDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout rgbComputeDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout rgbComputePipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkPipeline skyPipeline = VK_NULL_HANDLE;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VkPipeline rgbComputePipeline = VK_NULL_HANDLE;
    VkDescriptorSet rgbComputeDescriptorSet = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    VkImage msaaColorImage = VK_NULL_HANDLE;
    VkDeviceMemory msaaColorImageMemory = VK_NULL_HANDLE;
    VkImageView msaaColorImageView = VK_NULL_HANDLE;
    VkRenderPass shadowRenderPass = VK_NULL_HANDLE;
    VkFramebuffer shadowFramebuffer = VK_NULL_HANDLE;
    VkImage colorImage = VK_NULL_HANDLE;
    VkDeviceMemory colorImageMemory = VK_NULL_HANDLE;
    VkImageView colorImageView = VK_NULL_HANDLE;
    VkSampler colorSampler = VK_NULL_HANDLE;
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;
    VkImage shadowDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory shadowDepthImageMemory = VK_NULL_HANDLE;
    VkImageView shadowDepthImageView = VK_NULL_HANDLE;
    VkSampler shadowSampler = VK_NULL_HANDLE;
    GpuBuffer drawConstantsBuffer;
    void* drawConstantsMapped = nullptr;
    mutable uint32_t drawConstantsCount = 0;
    GpuBuffer rgbStorageBuffer;
    GpuBuffer readbackBuffer;
    void* readbackMapped = nullptr;
    GpuBuffer identityInstanceBuffer;
    GpuBuffer blueConeInstanceBuffer;
    GpuBuffer yellowConeInstanceBuffer;
    uint32_t blueConeInstanceCount = 0;
    uint32_t yellowConeInstanceCount = 0;
    TextureResource whiteTexture;
    std::vector<TextureResource> textures;
};

#endif /* PACSIMVULKANCAMERASIM_HPP */
