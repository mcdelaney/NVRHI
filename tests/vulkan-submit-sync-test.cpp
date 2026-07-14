// Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <nvrhi/vulkan.h>
#include "state-tracking.h"
#include "vulkan-backend.h"
#include "vulkan-queue-utils.h"
#include "vulkan-timestamp-utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

static_assert(nvrhi::vulkan::detail::timestampDelta(0xfffffff0ull, 0x20ull, 32) == 48);
static_assert(nvrhi::vulkan::detail::timestampDelta(0xfffffffffffffff0ull, 0x20ull, 64) == 48);
static_assert(nvrhi::vulkan::detail::timestampMask(0) == 0);

namespace
{
    struct ValidationState
    {
        std::atomic<uint32_t> errors { 0 };
    };

    VKAPI_ATTR VkBool32 VKAPI_CALL validationCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
        void* userData)
    {
        auto* state = static_cast<ValidationState*>(userData);
        const char* message = callbackData && callbackData->pMessage
            ? callbackData->pMessage : "<no validation message>";
        const bool syncHazard = std::strstr(message, "SYNC-HAZARD") != nullptr
            || std::strstr(message, "hazard detected") != nullptr;
        if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0 || syncHazard)
        {
            state->errors.fetch_add(1, std::memory_order_relaxed);
            std::cerr << "[Vulkan validation] " << message << '\n';
        }
        return VK_FALSE;
    }

    VKAPI_ATTR VkResult VKAPI_CALL failQueueSubmit2(
        VkQueue,
        uint32_t,
        const VkSubmitInfo2*,
        VkFence)
    {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    PFN_vkQueueSubmit2 forwardQueueSubmit2 = nullptr;

    struct QueueSubmit2Probe
    {
        bool failSubmission = false;
        bool exact = true;
        uint32_t calls = 0;
        VkSemaphore accumulatorWait = VK_NULL_HANDLE;
        VkSemaphore extraWait = VK_NULL_HANDLE;
        VkSemaphore accumulatorSignal = VK_NULL_HANDLE;
        VkSemaphore extraSignal = VK_NULL_HANDLE;
        VkSemaphore trackingSignal = VK_NULL_HANDLE;
        uint64_t trackingValue = 0;
    } queueSubmit2Probe;

    uint32_t countSemaphoreInfo(
        const VkSemaphoreSubmitInfo* infos,
        uint32_t count,
        VkSemaphore semaphore,
        uint64_t value,
        VkPipelineStageFlags2 stageMask)
    {
        uint32_t matches = 0;
        for (uint32_t i = 0; i < count; ++i)
        {
            if (infos[i].semaphore == semaphore
                && infos[i].value == value
                && infos[i].stageMask == stageMask)
            {
                ++matches;
            }
        }
        return matches;
    }

    VKAPI_ATTR VkResult VKAPI_CALL probeQueueSubmit2(
        VkQueue queue,
        uint32_t submitCount,
        const VkSubmitInfo2* submits,
        VkFence fence)
    {
        ++queueSubmit2Probe.calls;
        bool exact = submitCount == 1 && submits != nullptr;
        if (exact)
        {
            const VkSubmitInfo2& submit = submits[0];
            exact = submit.commandBufferInfoCount == 1
                && submit.pCommandBufferInfos != nullptr
                && submit.waitSemaphoreInfoCount == 2
                && submit.pWaitSemaphoreInfos != nullptr
                && submit.signalSemaphoreInfoCount == 3
                && submit.pSignalSemaphoreInfos != nullptr
                && countSemaphoreInfo(
                    submit.pWaitSemaphoreInfos,
                    submit.waitSemaphoreInfoCount,
                    queueSubmit2Probe.accumulatorWait,
                    1,
                    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT) == 1
                && countSemaphoreInfo(
                    submit.pWaitSemaphoreInfos,
                    submit.waitSemaphoreInfoCount,
                    queueSubmit2Probe.extraWait,
                    1,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT) == 1
                && countSemaphoreInfo(
                    submit.pSignalSemaphoreInfos,
                    submit.signalSemaphoreInfoCount,
                    queueSubmit2Probe.accumulatorSignal,
                    1,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) == 1
                && countSemaphoreInfo(
                    submit.pSignalSemaphoreInfos,
                    submit.signalSemaphoreInfoCount,
                    queueSubmit2Probe.extraSignal,
                    1,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) == 1
                && countSemaphoreInfo(
                    submit.pSignalSemaphoreInfos,
                    submit.signalSemaphoreInfoCount,
                    queueSubmit2Probe.trackingSignal,
                    queueSubmit2Probe.trackingValue,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) == 1;
        }
        queueSubmit2Probe.exact &= exact;

        if (queueSubmit2Probe.failSubmission)
            return VK_ERROR_OUT_OF_HOST_MEMORY;

        return forwardQueueSubmit2(queue, submitCount, submits, fence);
    }

    PFN_vkDeviceWaitIdle realDeviceWaitIdle = nullptr;
    std::mutex* waitIdleQueueMutex = nullptr;
    bool waitIdleObservedQueueLock = false;

    PFN_vkCreateDescriptorSetLayout realCreateDescriptorSetLayout = nullptr;
    PFN_vkCreateDescriptorPool realCreateDescriptorPool = nullptr;

    struct DescriptorUpdateAfterBindProbe
    {
        bool active = false;
        bool sawLayout = false;
        bool layoutUsesUpdateAfterBindPool = false;
        bool sawBindingFlags = false;
        bool bindingsUseUpdateAfterBind = false;
        bool bindingsUseUpdateUnusedWhilePending = false;
        bool sawPool = false;
        bool poolUsesUpdateAfterBind = false;

        void reset()
        {
            sawLayout = false;
            layoutUsesUpdateAfterBindPool = false;
            sawBindingFlags = false;
            bindingsUseUpdateAfterBind = false;
            bindingsUseUpdateUnusedWhilePending = false;
            sawPool = false;
            poolUsesUpdateAfterBind = false;
        }
    } descriptorUpdateAfterBindProbe;

    VKAPI_ATTR VkResult VKAPI_CALL probeCreateDescriptorSetLayout(
        VkDevice device,
        const VkDescriptorSetLayoutCreateInfo* createInfo,
        const VkAllocationCallbacks* allocationCallbacks,
        VkDescriptorSetLayout* descriptorSetLayout)
    {
        if (descriptorUpdateAfterBindProbe.active && createInfo)
        {
            descriptorUpdateAfterBindProbe.sawLayout = true;
            descriptorUpdateAfterBindProbe.layoutUsesUpdateAfterBindPool =
                (createInfo->flags
                    & VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT) != 0;

            const VkBaseInStructure* next =
                static_cast<const VkBaseInStructure*>(createInfo->pNext);
            while (next)
            {
                if (next->sType
                    == VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO)
                {
                    const auto* bindingFlags =
                        reinterpret_cast<const VkDescriptorSetLayoutBindingFlagsCreateInfo*>(next);
                    descriptorUpdateAfterBindProbe.sawBindingFlags = true;
                    descriptorUpdateAfterBindProbe.bindingsUseUpdateAfterBind =
                        bindingFlags->bindingCount != 0;
                    descriptorUpdateAfterBindProbe.bindingsUseUpdateUnusedWhilePending =
                        bindingFlags->bindingCount != 0;
                    for (uint32_t i = 0; i < bindingFlags->bindingCount; ++i)
                    {
                        if ((bindingFlags->pBindingFlags[i]
                            & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT) == 0)
                            descriptorUpdateAfterBindProbe.bindingsUseUpdateAfterBind = false;
                        if ((bindingFlags->pBindingFlags[i]
                            & VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT) == 0)
                            descriptorUpdateAfterBindProbe.bindingsUseUpdateUnusedWhilePending = false;
                    }
                    break;
                }
                next = next->pNext;
            }
        }

        return realCreateDescriptorSetLayout(
            device, createInfo, allocationCallbacks, descriptorSetLayout);
    }

    VKAPI_ATTR VkResult VKAPI_CALL probeCreateDescriptorPool(
        VkDevice device,
        const VkDescriptorPoolCreateInfo* createInfo,
        const VkAllocationCallbacks* allocationCallbacks,
        VkDescriptorPool* descriptorPool)
    {
        if (descriptorUpdateAfterBindProbe.active && createInfo)
        {
            descriptorUpdateAfterBindProbe.sawPool = true;
            descriptorUpdateAfterBindProbe.poolUsesUpdateAfterBind =
                (createInfo->flags & VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT) != 0;
        }

        return realCreateDescriptorPool(
            device, createInfo, allocationCallbacks, descriptorPool);
    }

    VKAPI_ATTR VkResult VKAPI_CALL probeDeviceWaitIdle(VkDevice device)
    {
        bool probeAcquiredQueueLock = false;
        std::thread probe([&]() {
            if (waitIdleQueueMutex && waitIdleQueueMutex->try_lock())
            {
                probeAcquiredQueueLock = true;
                waitIdleQueueMutex->unlock();
            }
        });
        probe.join();
        waitIdleObservedQueueLock = waitIdleQueueMutex
            && !probeAcquiredQueueLock;
        return realDeviceWaitIdle(device);
    }

    class MessageCallback : public nvrhi::IMessageCallback
    {
    public:
        void message(nvrhi::MessageSeverity severity, const char* messageText) override
        {
            if (severity == nvrhi::MessageSeverity::Error || severity == nvrhi::MessageSeverity::Fatal)
            {
                ++errors;
                std::cerr << "[NVRHI] " << messageText << '\n';
            }
        }

        uint32_t errors = 0;
    };

    class ForeignTexture final : public nvrhi::ITexture
    {
    public:
        unsigned long AddRef() override { return ++references; }
        unsigned long Release() override { return references ? --references : 0; }
        unsigned long GetRefCount() override { return references; }
        const nvrhi::TextureDesc& getDesc() const override { return desc; }
        nvrhi::Object getNativeView(
            nvrhi::ObjectType,
            nvrhi::Format,
            nvrhi::TextureSubresourceSet,
            nvrhi::TextureDimension,
            bool) override
        {
            return nullptr;
        }

        nvrhi::TextureDesc desc {};
        unsigned long references = 1;
    };

    class ForeignBuffer final : public nvrhi::IBuffer
    {
    public:
        unsigned long AddRef() override { return ++references; }
        unsigned long Release() override { return references ? --references : 0; }
        unsigned long GetRefCount() override { return references; }
        const nvrhi::BufferDesc& getDesc() const override { return desc; }
        nvrhi::GpuVirtualAddress getGpuVirtualAddress() const override { return 0; }

        nvrhi::BufferDesc desc {};
        unsigned long references = 1;
    };

    bool hasLayer(const char* name)
    {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> layers(count);
        vkEnumerateInstanceLayerProperties(&count, layers.data());
        for (const auto& layer : layers)
            if (std::strcmp(layer.layerName, name) == 0)
                return true;
        return false;
    }

    bool hasInstanceExtension(const char* name, const char* layerName = nullptr)
    {
        uint32_t count = 0;
        vkEnumerateInstanceExtensionProperties(layerName, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        vkEnumerateInstanceExtensionProperties(layerName, &count, extensions.data());
        for (const auto& extension : extensions)
            if (std::strcmp(extension.extensionName, name) == 0)
                return true;
        return false;
    }

    bool hasDeviceExtension(VkPhysicalDevice physicalDevice, const char* name)
    {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(
            physicalDevice, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> extensions(count);
        vkEnumerateDeviceExtensionProperties(
            physicalDevice, nullptr, &count, extensions.data());
        for (const auto& extension : extensions)
            if (std::strcmp(extension.extensionName, name) == 0)
                return true;
        return false;
    }

    VkSemaphore createSemaphore(VkDevice device, bool timeline, uint64_t initialValue = 0)
    {
        VkSemaphoreTypeCreateInfo typeInfo { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        typeInfo.semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
        typeInfo.initialValue = initialValue;
        VkSemaphoreCreateInfo createInfo { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        createInfo.pNext = timeline ? &typeInfo : nullptr;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        return vkCreateSemaphore(device, &createInfo, nullptr, &semaphore) == VK_SUCCESS
            ? semaphore : VK_NULL_HANDLE;
    }

    bool signalTimeline(VkDevice device, VkSemaphore semaphore, uint64_t value)
    {
        VkSemaphoreSignalInfo info { VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO };
        info.semaphore = semaphore;
        info.value = value;
        return vkSignalSemaphore(device, &info) == VK_SUCCESS;
    }

    bool waitTimeline(VkDevice device, VkSemaphore semaphore, uint64_t value)
    {
        VkSemaphoreWaitInfo info { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
        info.semaphoreCount = 1;
        info.pSemaphores = &semaphore;
        info.pValues = &value;
        return vkWaitSemaphores(device, &info, 5'000'000'000ull) == VK_SUCCESS;
    }

    uint64_t timelineValue(VkDevice device, VkSemaphore semaphore)
    {
        uint64_t value = 0;
        vkGetSemaphoreCounterValue(device, semaphore, &value);
        return value;
    }

    VkBuffer findUploadChunk(nvrhi::ICommandList* commandList)
    {
        auto* vulkanCommandList = static_cast<nvrhi::vulkan::CommandList*>(commandList);
        const nvrhi::vulkan::TrackedCommandBufferPtr trackedCommandBuffer =
            vulkanCommandList->getCurrentCmdBuf();
        if (!trackedCommandBuffer)
            return VK_NULL_HANDLE;

        for (const nvrhi::RefCountPtr<nvrhi::vulkan::Buffer>& buffer
            : trackedCommandBuffer->referencedStagingBuffers)
        {
            if (buffer && buffer->getDesc().debugName == "UploadChunk")
                return buffer->getNativeObject(nvrhi::ObjectTypes::VK_Buffer);
        }

        return VK_NULL_HANDLE;
    }
}

int main()
{
    using nvrhi::vulkan::detail::isWaitStageMaskSupported;
    const vk::QueueFlags transferQueueFlags = vk::QueueFlagBits::eTransfer;
    const vk::QueueFlags computeQueueFlags = vk::QueueFlagBits::eCompute;
    const vk::QueueFlags graphicsQueueFlags = vk::QueueFlagBits::eGraphics;
    const bool stagePredicatePassed =
        !isWaitStageMaskSupported({}, transferQueueFlags)
        && isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eCopy, transferQueueFlags)
        && isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eAllCommands, transferQueueFlags)
        && !isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eFragmentShader, transferQueueFlags)
        && !isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eColorAttachmentOutput, transferQueueFlags)
        && !isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eComputeShader, transferQueueFlags)
        && !isWaitStageMaskSupported(
            vk::PipelineStageFlagBits2::eCopy | vk::PipelineStageFlagBits2::eFragmentShader,
            transferQueueFlags)
        && isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eComputeShader, computeQueueFlags)
        && isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eCopy, computeQueueFlags)
        && isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eDrawIndirect, computeQueueFlags)
        && !isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eFragmentShader, computeQueueFlags)
        && !isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eColorAttachmentOutput, computeQueueFlags)
        && isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eFragmentShader, graphicsQueueFlags)
        && isWaitStageMaskSupported(vk::PipelineStageFlagBits2::eColorAttachmentOutput, graphicsQueueFlags);
    if (!stagePredicatePassed)
    {
        std::cerr << "Queue-family wait-stage predicate tests failed\n";
        return 1;
    }

    constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    if (!hasLayer(validationLayer)
        || !hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)
        || !hasInstanceExtension(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME, validationLayer))
    {
        std::cerr << "The Khronos validation layer with synchronization validation is required\n";
        return 1;
    }

    ValidationState validationState;
    VkDebugUtilsMessengerCreateInfoEXT debugInfo { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
    debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugInfo.pfnUserCallback = validationCallback;
    debugInfo.pUserData = &validationState;

    const VkValidationFeatureEnableEXT enabledValidation[] = {
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
    };
    VkValidationFeaturesEXT validationFeatures { VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT };
    validationFeatures.pNext = &debugInfo;
    validationFeatures.enabledValidationFeatureCount = 1;
    validationFeatures.pEnabledValidationFeatures = enabledValidation;

    VkApplicationInfo appInfo { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName = "nvrhi_vk_submit_sync_test";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    const std::array<const char*, 2> instanceExtensions = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME
    };
    VkInstanceCreateInfo instanceInfo { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instanceInfo.pNext = &validationFeatures;
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledLayerCount = 1;
    instanceInfo.ppEnabledLayerNames = &validationLayer;
    instanceInfo.enabledExtensionCount = uint32_t(instanceExtensions.size());
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS)
        return 1;

    auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (!createMessenger || createMessenger(instance, &debugInfo, nullptr, &messenger) != VK_SUCCESS)
        return 1;

    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr);
    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
    vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = UINT32_MAX;
    uint32_t distinctTransferQueueFamilyIndex = UINT32_MAX;
    bool supportsDescriptorUpdateAfterBindProbe = false;
    bool supportsMaintenance8 = false;
    for (VkPhysicalDevice candidate : physicalDevices)
    {
        const bool candidateHasMaintenance8 = hasDeviceExtension(
            candidate, VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
        VkPhysicalDeviceMaintenance8FeaturesKHR supportedMaintenance8 {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR
        };
        VkPhysicalDeviceVulkan13Features supported13 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
        VkPhysicalDeviceVulkan12Features supported12 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        supported12.pNext = &supported13;
        supported13.pNext = candidateHasMaintenance8
            ? &supportedMaintenance8 : nullptr;
        VkPhysicalDeviceFeatures2 supported { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        supported.pNext = &supported12;
        vkGetPhysicalDeviceFeatures2(candidate, &supported);

        VkPhysicalDeviceProperties properties {};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_3
            || !supported12.timelineSemaphore || !supported13.synchronization2)
            continue;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, queueFamilies.data());
        uint32_t candidateGraphicsFamily = UINT32_MAX;
        for (uint32_t i = 0; i < queueFamilyCount; ++i)
        {
            if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0
                && queueFamilies[i].timestampValidBits != 0)
            {
                candidateGraphicsFamily = i;
                break;
            }
        }
        if (candidateGraphicsFamily == UINT32_MAX)
            continue;

        uint32_t candidateTransferFamily = UINT32_MAX;
        for (uint32_t i = 0; i < queueFamilyCount; ++i)
        {
            if (i == candidateGraphicsFamily
                || (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) == 0)
            {
                continue;
            }
            candidateTransferFamily = i;
            if ((queueFamilies[i].queueFlags
                    & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0)
            {
                break;
            }
        }

        physicalDevice = candidate;
        queueFamilyIndex = candidateGraphicsFamily;
        distinctTransferQueueFamilyIndex = candidateTransferFamily;
        supportsDescriptorUpdateAfterBindProbe =
            supported12.descriptorIndexing
            && supported12.descriptorBindingPartiallyBound
            && supported12.descriptorBindingSampledImageUpdateAfterBind
            && supported12.descriptorBindingUpdateUnusedWhilePending;
        supportsMaintenance8 = candidateHasMaintenance8
            && supportedMaintenance8.maintenance8;
        if (physicalDevice)
            break;
    }
    if (!physicalDevice)
        return 1;

    const float queuePriority = 1.f;
    std::array<VkDeviceQueueCreateInfo, 2> queueInfos {};
    queueInfos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfos[0].queueFamilyIndex = queueFamilyIndex;
    queueInfos[0].queueCount = 1;
    queueInfos[0].pQueuePriorities = &queuePriority;
    uint32_t queueInfoCount = 1;
    if (distinctTransferQueueFamilyIndex != UINT32_MAX)
    {
        queueInfos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfos[1].queueFamilyIndex = distinctTransferQueueFamilyIndex;
        queueInfos[1].queueCount = 1;
        queueInfos[1].pQueuePriorities = &queuePriority;
        queueInfoCount = 2;
    }
    VkPhysicalDeviceMaintenance8FeaturesKHR enabledMaintenance8 {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR
    };
    enabledMaintenance8.maintenance8 = supportsMaintenance8 ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceVulkan13Features enabled13 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    enabled13.synchronization2 = VK_TRUE;
    enabled13.pNext = supportsMaintenance8 ? &enabledMaintenance8 : nullptr;
    VkPhysicalDeviceVulkan12Features enabled12 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    enabled12.pNext = &enabled13;
    enabled12.timelineSemaphore = VK_TRUE;
    if (supportsDescriptorUpdateAfterBindProbe)
    {
        enabled12.descriptorIndexing = VK_TRUE;
        enabled12.descriptorBindingPartiallyBound = VK_TRUE;
        enabled12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
        enabled12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    }
    VkDeviceCreateInfo deviceInfo { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceInfo.pNext = &enabled12;
    deviceInfo.queueCreateInfoCount = queueInfoCount;
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    std::array<const char*, 1> deviceExtensions = {
        VK_KHR_MAINTENANCE_8_EXTENSION_NAME
    };
    deviceInfo.enabledExtensionCount = supportsMaintenance8 ? 1u : 0u;
    deviceInfo.ppEnabledExtensionNames = supportsMaintenance8
        ? deviceExtensions.data() : nullptr;

    VkDevice vkDevice = VK_NULL_HANDLE;
    if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &vkDevice) != VK_SUCCESS)
        return 1;
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(vkDevice, queueFamilyIndex, 0, &queue);
    VkQueue distinctTransferQueue = VK_NULL_HANDLE;
    if (distinctTransferQueueFamilyIndex != UINT32_MAX)
    {
        vkGetDeviceQueue(
            vkDevice, distinctTransferQueueFamilyIndex, 0,
            &distinctTransferQueue);
    }
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance, vkGetInstanceProcAddr, vkDevice);

    MessageCallback messageCallback;
    nvrhi::vulkan::DeviceDesc nvrhiDesc {};
    nvrhiDesc.errorCB = &messageCallback;
    nvrhiDesc.instance = instance;
    nvrhiDesc.physicalDevice = physicalDevice;
    nvrhiDesc.device = vkDevice;
    nvrhiDesc.deviceExtensions = supportsMaintenance8
        ? deviceExtensions.data() : nullptr;
    nvrhiDesc.numDeviceExtensions = supportsMaintenance8 ? 1u : 0u;
    nvrhiDesc.maintenance8Supported = supportsMaintenance8;
    nvrhiDesc.graphicsQueue = queue;
    nvrhiDesc.graphicsQueueIndex = int(queueFamilyIndex);
    // Bind all three logical queues to one physical VkQueue. Besides keeping
    // the test portable to devices with only one graphics-family queue, this
    // verifies that NVRHI aliases the physical queue's synchronization and
    // lifetime state instead of constructing three independent wrappers.
    nvrhiDesc.computeQueue = queue;
    nvrhiDesc.computeQueueIndex = int(queueFamilyIndex);
    nvrhiDesc.transferQueue = queue;
    nvrhiDesc.transferQueueIndex = int(queueFamilyIndex);
    nvrhi::vulkan::DeviceHandle device = nvrhi::vulkan::createDevice(nvrhiDesc);
    if (!device)
        return 1;

    bool passed = true;

    // Stage-qualified resource-state tracking must narrow only shader-visible
    // portions of a dependency. Fixed-function stages remain state-derived,
    // and the legacy unqualified mapping remains conservative.
    {
        const auto hasFlags = [](auto value, auto expected) {
            return (value & expected) == expected;
        };
        const auto stageUnion = [](nvrhi::ShaderType left,
                                   nvrhi::ShaderType right) {
            return left | right;
        };

        const auto legacySrv = nvrhi::vulkan::convertResourceState(
            nvrhi::ResourceStates::ShaderResource, false);
        const auto computeSrv = nvrhi::vulkan::convertResourceState(
            nvrhi::ResourceStates::ShaderResource,
            false, false, false, nvrhi::ShaderType::Compute);
        const auto fragmentSrv = nvrhi::vulkan::convertResourceState(
            nvrhi::ResourceStates::ShaderResource,
            false, false, false, nvrhi::ShaderType::Pixel);
        const auto computeUav = nvrhi::vulkan::convertResourceState(
            nvrhi::ResourceStates::UnorderedAccess,
            false, false, false, nvrhi::ShaderType::Compute);
        const auto copyDest = nvrhi::vulkan::convertResourceState(
            nvrhi::ResourceStates::CopyDest, false);
        const auto indirect = nvrhi::vulkan::convertResourceState(
            nvrhi::ResourceStates::IndirectArgument,
            false, false, false, nvrhi::ShaderType::Compute);
        const auto computeUavIndirect = nvrhi::vulkan::convertResourceState(
            nvrhi::ResourceStates::UnorderedAccess
                | nvrhi::ResourceStates::IndirectArgument,
            false, false, false, nvrhi::ShaderType::Compute);
        const auto meshFragmentSrv = nvrhi::vulkan::convertResourceState(
            nvrhi::ResourceStates::ShaderResource,
            false, false, false,
            nvrhi::ShaderType::Mesh | nvrhi::ShaderType::Pixel);
        const auto allGraphicsSrv = nvrhi::vulkan::convertResourceState(
            nvrhi::ResourceStates::ShaderResource,
            false, false, false, nvrhi::ShaderType::AllGraphics);
        const auto fragmentAsRead = nvrhi::vulkan::convertResourceState(
            nvrhi::ResourceStates::AccelStructRead,
            false, false, false, nvrhi::ShaderType::Pixel);

        bool mappingPassed =
            !nvrhi::CommandListParameters {}.enableStageQualifiedBindingBarriers
            && legacySrv.stageFlags == vk::PipelineStageFlagBits2::eAllCommands
            && computeSrv.stageFlags == vk::PipelineStageFlagBits2::eComputeShader
            && fragmentSrv.stageFlags == vk::PipelineStageFlagBits2::eFragmentShader
            && computeUav.stageFlags == vk::PipelineStageFlagBits2::eComputeShader
            && hasFlags(computeUav.accessMask,
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite)
            && copyDest.stageFlags == vk::PipelineStageFlagBits2::eTransfer
            && copyDest.accessMask == vk::AccessFlagBits2::eTransferWrite
            && indirect.stageFlags == vk::PipelineStageFlagBits2::eDrawIndirect
            && computeUavIndirect.stageFlags
                == (vk::PipelineStageFlagBits2::eComputeShader
                    | vk::PipelineStageFlagBits2::eDrawIndirect)
            && meshFragmentSrv.stageFlags
                == (vk::PipelineStageFlagBits2::eMeshShaderNV
                    | vk::PipelineStageFlagBits2::eFragmentShader)
            && allGraphicsSrv.stageFlags
                == vk::PipelineStageFlagBits2::eAllGraphics
            && fragmentAsRead.stageFlags
                == vk::PipelineStageFlagBits2::eFragmentShader;

        const nvrhi::ShaderType meshPipelineStages =
            nvrhi::ShaderType::Amplification
            | nvrhi::ShaderType::Mesh
            | nvrhi::ShaderType::Pixel;
        mappingPassed &= nvrhi::vulkan::resolveBindingBarrierShaderStages(
                nvrhi::ShaderType::Pixel,
                nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel)
                == nvrhi::ShaderType::Pixel
            && nvrhi::vulkan::resolveBindingBarrierShaderStages(
                nvrhi::ShaderType::All, nvrhi::ShaderType::Compute)
                == nvrhi::ShaderType::Compute
            && nvrhi::vulkan::resolveBindingBarrierShaderStages(
                nvrhi::ShaderType::AllGraphics, meshPipelineStages)
                == meshPipelineStages
            && nvrhi::vulkan::resolveBindingBarrierShaderStages(
                nvrhi::ShaderType::All,
                nvrhi::ShaderType::AllRayTracing)
                == nvrhi::ShaderType::AllRayTracing
            && nvrhi::vulkan::resolveBindingBarrierShaderStages(
                nvrhi::ShaderType::Compute, nvrhi::ShaderType::All)
                == nvrhi::ShaderType::Compute
            && nvrhi::vulkan::resolveBindingBarrierShaderStages(
                nvrhi::ShaderType::Pixel, nvrhi::ShaderType::Compute)
                == nvrhi::ShaderType::All
            && nvrhi::vulkan::resolveBindingBarrierShaderStages(
                nvrhi::ShaderType::None, nvrhi::ShaderType::Compute)
                == nvrhi::ShaderType::All;

        nvrhi::TextureDesc sampledDepthDesc {};
        sampledDepthDesc.format = nvrhi::Format::D32;
        const auto sampledDepth = nvrhi::vulkan::convertTextureState(
            nvrhi::ResourceStates::ShaderResource
                | nvrhi::ResourceStates::DepthRead,
            sampledDepthDesc,
            nvrhi::ShaderType::Pixel);
        mappingPassed &= sampledDepth.imageLayout
                == vk::ImageLayout::eDepthStencilReadOnlyOptimal
            && sampledDepth.stageFlags
                == (vk::PipelineStageFlagBits2::eFragmentShader
                    | vk::PipelineStageFlagBits2::eEarlyFragmentTests
                    | vk::PipelineStageFlagBits2::eLateFragmentTests)
            && hasFlags(sampledDepth.accessMask,
                vk::AccessFlagBits2::eShaderRead
                    | vk::AccessFlagBits2::eDepthStencilAttachmentRead);

        // Queue-family ownership transfers are deliberately split into an
        // immediate release barrier and an immediate acquire barrier. A
        // same-family transfer collapses to one ordinary transition.
        {
            const nvrhi::vulkan::QueueOwnershipTransferDesc defaultTransfer {};
            const nvrhi::vulkan::MemoryDependencyDesc defaultDependency {};
            const nvrhi::vulkan::GraphResourceState defaultGraphState {};
            const nvrhi::vulkan::GraphResourceStateTransition
                defaultGraphTransition {};
            nvrhi::TextureDesc ownershipTextureDesc {};
            ownershipTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
            const auto ownershipBefore = nvrhi::vulkan::convertTextureState(
                nvrhi::ResourceStates::CopyDest, ownershipTextureDesc);
            const auto ownershipAfter = nvrhi::vulkan::convertTextureState(
                nvrhi::ResourceStates::ShaderResource,
                ownershipTextureDesc, nvrhi::ShaderType::Pixel);
            const auto imageRange = vk::ImageSubresourceRange()
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setBaseMipLevel(0)
                .setLevelCount(1)
                .setBaseArrayLayer(0)
                .setLayerCount(1);

            const auto imageRelease =
                nvrhi::vulkan::detail::buildQueueOwnershipImageBarrier(
                    {}, imageRange, ownershipBefore, ownershipAfter,
                    3, 5, true, false);
            const auto imageAcquire =
                nvrhi::vulkan::detail::buildQueueOwnershipImageBarrier(
                    {}, imageRange, ownershipBefore, ownershipAfter,
                    3, 5, false, false);
            const auto imageSameFamily =
                nvrhi::vulkan::detail::buildQueueOwnershipImageBarrier(
                    {}, imageRange, ownershipBefore, ownershipAfter,
                    3, 3, true, true);

            const auto bufferRelease =
                nvrhi::vulkan::detail::buildQueueOwnershipBufferBarrier(
                    {}, 64, copyDest, fragmentSrv, 3, 5, true, false);
            const auto bufferAcquire =
                nvrhi::vulkan::detail::buildQueueOwnershipBufferBarrier(
                    {}, 64, copyDest, fragmentSrv, 3, 5, false, false);
            const auto bufferSameFamily =
                nvrhi::vulkan::detail::buildQueueOwnershipBufferBarrier(
                    {}, 64, copyDest, fragmentSrv, 3, 3, true, true);
            const vk::DependencyFlags maintenance8OwnershipFlag =
                vk::DependencyFlagBits::eQueueFamilyOwnershipTransferUseAllStagesKHR;

            mappingPassed &= nvrhi::c_HeaderVersion == 30
                && defaultTransfer.sourceQueue == nvrhi::CommandQueue::Count
                && defaultTransfer.destinationQueue == nvrhi::CommandQueue::Count
                && defaultTransfer.stateBefore == nvrhi::ResourceStates::Unknown
                && defaultTransfer.stateAfter == nvrhi::ResourceStates::Unknown
                && defaultDependency.state == nvrhi::ResourceStates::Unknown
                && defaultDependency.shaderStagesBefore == nvrhi::ShaderType::All
                && defaultDependency.shaderStagesAfter == nvrhi::ShaderType::All
                && defaultGraphState.state == nvrhi::ResourceStates::Unknown
                && defaultGraphState.shaderStages == nvrhi::ShaderType::None
                && defaultGraphTransition.stateBefore
                    == nvrhi::ResourceStates::Unknown
                && defaultGraphTransition.stateAfter
                    == nvrhi::ResourceStates::Unknown
                && defaultGraphTransition.shaderStagesBefore
                    == nvrhi::ShaderType::None
                && defaultGraphTransition.shaderStagesAfter
                    == nvrhi::ShaderType::None
                && imageRelease.srcStageMask
                    == vk::PipelineStageFlagBits2::eTransfer
                && imageRelease.srcAccessMask
                    == vk::AccessFlagBits2::eTransferWrite
                && imageRelease.dstStageMask
                    == vk::PipelineStageFlagBits2::eTransfer
                && !imageRelease.dstAccessMask
                && imageRelease.oldLayout
                    == vk::ImageLayout::eTransferDstOptimal
                && imageRelease.newLayout
                    == vk::ImageLayout::eShaderReadOnlyOptimal
                && imageRelease.srcQueueFamilyIndex == 3
                && imageRelease.dstQueueFamilyIndex == 5
                && imageAcquire.srcStageMask
                    == vk::PipelineStageFlagBits2::eFragmentShader
                && !imageAcquire.srcAccessMask
                && imageAcquire.dstStageMask
                    == vk::PipelineStageFlagBits2::eFragmentShader
                && imageAcquire.dstAccessMask
                    == vk::AccessFlagBits2::eShaderRead
                && imageAcquire.oldLayout
                    == vk::ImageLayout::eTransferDstOptimal
                && imageAcquire.newLayout
                    == vk::ImageLayout::eShaderReadOnlyOptimal
                && imageAcquire.srcQueueFamilyIndex == 3
                && imageAcquire.dstQueueFamilyIndex == 5
                && imageSameFamily.srcStageMask
                    == vk::PipelineStageFlagBits2::eTransfer
                && imageSameFamily.srcAccessMask
                    == vk::AccessFlagBits2::eTransferWrite
                && imageSameFamily.dstStageMask
                    == vk::PipelineStageFlagBits2::eFragmentShader
                && imageSameFamily.dstAccessMask
                    == vk::AccessFlagBits2::eShaderRead
                && imageSameFamily.srcQueueFamilyIndex
                    == VK_QUEUE_FAMILY_IGNORED
                && imageSameFamily.dstQueueFamilyIndex
                    == VK_QUEUE_FAMILY_IGNORED
                && bufferRelease.srcStageMask
                    == vk::PipelineStageFlagBits2::eTransfer
                && bufferRelease.srcAccessMask
                    == vk::AccessFlagBits2::eTransferWrite
                && bufferRelease.dstStageMask
                    == vk::PipelineStageFlagBits2::eTransfer
                && !bufferRelease.dstAccessMask
                && bufferRelease.srcQueueFamilyIndex == 3
                && bufferRelease.dstQueueFamilyIndex == 5
                && bufferRelease.offset == 0
                && bufferRelease.size == 64
                && bufferAcquire.srcStageMask
                    == vk::PipelineStageFlagBits2::eFragmentShader
                && !bufferAcquire.srcAccessMask
                && bufferAcquire.dstStageMask
                    == vk::PipelineStageFlagBits2::eFragmentShader
                && bufferAcquire.dstAccessMask
                    == vk::AccessFlagBits2::eShaderRead
                && bufferAcquire.srcQueueFamilyIndex == 3
                && bufferAcquire.dstQueueFamilyIndex == 5
                && bufferSameFamily.srcStageMask
                    == vk::PipelineStageFlagBits2::eTransfer
                && bufferSameFamily.srcAccessMask
                    == vk::AccessFlagBits2::eTransferWrite
                && bufferSameFamily.dstStageMask
                    == vk::PipelineStageFlagBits2::eFragmentShader
                && bufferSameFamily.dstAccessMask
                    == vk::AccessFlagBits2::eShaderRead
                && bufferSameFamily.srcQueueFamilyIndex
                    == VK_QUEUE_FAMILY_IGNORED
                && bufferSameFamily.dstQueueFamilyIndex
                    == VK_QUEUE_FAMILY_IGNORED
                && nvrhi::vulkan::detail::queueOwnershipDependencyFlags(
                    false, true) == maintenance8OwnershipFlag
                && !nvrhi::vulkan::detail::queueOwnershipDependencyFlags(
                    false, false)
                && !nvrhi::vulkan::detail::queueOwnershipDependencyFlags(
                    true, true);
        }

        passed &= mappingPassed;
        if (!mappingPassed)
            std::cerr << "Stage-qualified Vulkan state mapping test failed\n";

        nvrhi::BufferDesc trackerBufferDesc {};
        trackerBufferDesc.byteSize = 64;
        trackerBufferDesc.canHaveUAVs = true;
        trackerBufferDesc.isDrawIndirectArgs = true;
        trackerBufferDesc.debugName = "StageQualifiedTrackerBuffer";
        nvrhi::BufferStateExtension trackerBuffer(trackerBufferDesc);

        const auto checkBufferTransition = [&](nvrhi::ResourceStates beforeState,
                                                nvrhi::ShaderType beforeStages,
                                                nvrhi::ResourceStates afterState,
                                                nvrhi::ShaderType afterStages) {
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingBufferState(
                &trackerBuffer, beforeState, beforeStages);
            tracker.requireBufferState(
                &trackerBuffer, afterState, afterStages);
            const auto& barriers = tracker.getBufferBarriers();
            return barriers.size() == 1
                && barriers[0].stateBefore == beforeState
                && barriers[0].stateAfter == afterState
                && barriers[0].shaderStagesBefore == beforeStages
                && barriers[0].shaderStagesAfter == afterStages;
        };

        bool trackerPassed = checkBufferTransition(
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Compute,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Pixel)
            && checkBufferTransition(
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Compute)
            && checkBufferTransition(
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Compute,
                nvrhi::ResourceStates::IndirectArgument,
                nvrhi::ShaderType::None);

        // A read moving between shader stages does not need a barrier. The
        // outstanding stage union must still become the source of a later WAR.
        {
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Compute);
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Pixel);
            trackerPassed &= tracker.getBufferBarriers().empty();
            const nvrhi::ShaderType outstandingStages =
                tracker.getBufferShaderStages(&trackerBuffer);
            const auto ownershipReleaseSource =
                nvrhi::vulkan::convertResourceState(
                    nvrhi::ResourceStates::ShaderResource,
                    false, false, false,
                    nvrhi::ShaderType::Compute | outstandingStages);
            const auto ownershipReleaseDestination =
                nvrhi::vulkan::convertResourceState(
                    nvrhi::ResourceStates::ShaderResource,
                    false, false, false, nvrhi::ShaderType::Compute);
            const auto ownershipRelease =
                nvrhi::vulkan::detail::buildQueueOwnershipBufferBarrier(
                    {}, trackerBufferDesc.byteSize,
                    ownershipReleaseSource, ownershipReleaseDestination,
                    3, 5, true, false);
            trackerPassed &= outstandingStages
                    == stageUnion(nvrhi::ShaderType::Compute,
                        nvrhi::ShaderType::Pixel)
                && ownershipRelease.srcStageMask
                    == (vk::PipelineStageFlagBits2::eComputeShader
                        | vk::PipelineStageFlagBits2::eFragmentShader);
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Compute);
            const auto& barriers = tracker.getBufferBarriers();
            trackerPassed &= barriers.size() == 1
                && barriers[0].shaderStagesBefore
                    == stageUnion(nvrhi::ShaderType::Compute,
                        nvrhi::ShaderType::Pixel)
                && barriers[0].shaderStagesAfter == nvrhi::ShaderType::Compute;
        }

        // If the transition into a read state has not been committed yet, a
        // second read stage widens that pending barrier's destination scope.
        {
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None);
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Compute);
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Pixel);
            const auto& barriers = tracker.getBufferBarriers();
            trackerPassed &= barriers.size() == 1
                && barriers[0].shaderStagesAfter
                    == stageUnion(nvrhi::ShaderType::Compute,
                        nvrhi::ShaderType::Pixel);
        }

        // A repeated UAV declaration before barrier submission must widen the
        // pending transition, not append a second dependency to the same batch.
        // The widened stage set also remains the source of the next transition.
        {
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None);
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Compute);
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Pixel);
            const nvrhi::ShaderType uavStages =
                stageUnion(nvrhi::ShaderType::Compute,
                    nvrhi::ShaderType::Pixel);
            const auto& pendingBarriers = tracker.getBufferBarriers();
            trackerPassed &= pendingBarriers.size() == 1
                && pendingBarriers[0].stateBefore == nvrhi::ResourceStates::CopyDest
                && pendingBarriers[0].stateAfter == nvrhi::ResourceStates::UnorderedAccess
                && pendingBarriers[0].shaderStagesAfter == uavStages;

            tracker.clearBarriers();
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::CopySource,
                nvrhi::ShaderType::None);
            const auto& laterBarriers = tracker.getBufferBarriers();
            trackerPassed &= laterBarriers.size() == 1
                && laterBarriers[0].stateBefore == nvrhi::ResourceStates::UnorderedAccess
                && laterBarriers[0].stateAfter == nvrhi::ResourceStates::CopySource
                && laterBarriers[0].shaderStagesBefore == uavStages;
        }

        // Repeated UAV access remains a RAW/WAW dependency with exact stages.
        {
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Compute);
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Pixel);
            const auto& barriers = tracker.getBufferBarriers();
            trackerPassed &= barriers.size() == 1
                && barriers[0].stateBefore == nvrhi::ResourceStates::UnorderedAccess
                && barriers[0].stateAfter == nvrhi::ResourceStates::UnorderedAccess
                && barriers[0].shaderStagesBefore == nvrhi::ShaderType::Compute
                && barriers[0].shaderStagesAfter == nvrhi::ShaderType::Pixel;
        }

        // Legacy AS reads cover compute + RT. An inline fragment ray query on
        // the same logical state adds Fragment without losing those stages.
        {
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::AccelStructRead,
                nvrhi::ShaderType::All);
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::AccelStructRead,
                nvrhi::ShaderType::Pixel);
            trackerPassed &= tracker.getBufferBarriers().empty();
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::AccelStructWrite,
                nvrhi::ShaderType::None);
            const auto& barriers = tracker.getBufferBarriers();
            const nvrhi::ShaderType legacyAsStages =
                nvrhi::ShaderType::Compute | nvrhi::ShaderType::AllRayTracing;
            trackerPassed &= barriers.size() == 1
                && barriers[0].shaderStagesBefore
                    == (legacyAsStages | nvrhi::ShaderType::Pixel)
                && barriers[0].shaderStagesAfter == nvrhi::ShaderType::None;
        }

        // Stage visibility is tracked independently per texture subresource.
        {
            nvrhi::TextureDesc trackerTextureDesc {};
            trackerTextureDesc.width = 4;
            trackerTextureDesc.height = 4;
            trackerTextureDesc.mipLevels = 2;
            trackerTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
            trackerTextureDesc.debugName = "StageQualifiedTrackerTexture";
            nvrhi::TextureStateExtension trackerTexture(trackerTextureDesc);
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingTextureState(
                &trackerTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Compute);
            const nvrhi::TextureSubresourceSet mip0(0, 1, 0, 1);
            const nvrhi::TextureSubresourceSet mip1(1, 1, 0, 1);
            tracker.requireTextureState(
                &trackerTexture, mip0,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Pixel);
            trackerPassed &= tracker.getTextureBarriers().empty()
                && tracker.getTextureSubresourceShaderStages(
                    &trackerTexture, 0, 0)
                    == stageUnion(nvrhi::ShaderType::Compute,
                        nvrhi::ShaderType::Pixel)
                && tracker.getTextureSubresourceShaderStages(
                    &trackerTexture, 0, 1)
                    == nvrhi::ShaderType::Compute;
            tracker.requireTextureState(
                &trackerTexture, mip0,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Compute);
            tracker.requireTextureState(
                &trackerTexture, mip1,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Pixel);
            const auto& barriers = tracker.getTextureBarriers();
            trackerPassed &= barriers.size() == 2
                && barriers[0].mipLevel == 0
                && barriers[0].shaderStagesBefore
                    == stageUnion(nvrhi::ShaderType::Compute,
                        nvrhi::ShaderType::Pixel)
                && barriers[0].shaderStagesAfter == nvrhi::ShaderType::Compute
                && barriers[1].mipLevel == 1
                && barriers[1].shaderStagesBefore == nvrhi::ShaderType::Compute
                && barriers[1].shaderStagesAfter == nvrhi::ShaderType::Pixel;
        }

        // A permanent transition is pending from recording until successful
        // submission. QFOT validation must be able to reject that state before
        // the resource-level permanentState field is committed.
        {
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingBufferState(
                &trackerBuffer, nvrhi::ResourceStates::Common);
            tracker.setPermanentBufferState(
                &trackerBuffer, nvrhi::ResourceStates::CopyDest);
            trackerPassed &= tracker.hasPendingPermanentBufferState(
                &trackerBuffer);

            nvrhi::TextureDesc permanentTextureDesc {};
            permanentTextureDesc.width = 4;
            permanentTextureDesc.height = 4;
            permanentTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
            nvrhi::TextureStateExtension permanentTexture(
                permanentTextureDesc);
            tracker.beginTrackingTextureState(
                &permanentTexture, nvrhi::AllSubresources,
                nvrhi::ResourceStates::Common);
            tracker.setPermanentTextureState(
                &permanentTexture, nvrhi::AllSubresources,
                nvrhi::ResourceStates::CopyDest);
            trackerPassed &= tracker.hasPendingPermanentTextureState(
                &permanentTexture);
        }

        // Pending per-subresource transitions widen only the subresource that
        // receives the additional shader-stage declaration.
        {
            nvrhi::TextureDesc trackerTextureDesc {};
            trackerTextureDesc.width = 4;
            trackerTextureDesc.height = 4;
            trackerTextureDesc.mipLevels = 2;
            trackerTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
            trackerTextureDesc.debugName = "PendingStageQualifiedTrackerTexture";
            nvrhi::TextureStateExtension trackerTexture(trackerTextureDesc);
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingTextureState(
                &trackerTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None);
            const nvrhi::TextureSubresourceSet mip0(0, 1, 0, 1);
            const nvrhi::TextureSubresourceSet mip1(1, 1, 0, 1);
            tracker.requireTextureState(
                &trackerTexture, mip0,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Compute);
            tracker.requireTextureState(
                &trackerTexture, mip1,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Compute);
            tracker.requireTextureState(
                &trackerTexture, mip0,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Pixel);
            const auto& barriers = tracker.getTextureBarriers();
            trackerPassed &= barriers.size() == 2
                && barriers[0].mipLevel == 0
                && barriers[0].shaderStagesAfter
                    == stageUnion(nvrhi::ShaderType::Compute,
                        nvrhi::ShaderType::Pixel)
                && barriers[1].mipLevel == 1
                && barriers[1].shaderStagesAfter == nvrhi::ShaderType::Compute;
        }

        // Whole-texture transitions use the same pending-destination widening.
        {
            nvrhi::TextureDesc trackerTextureDesc {};
            trackerTextureDesc.width = 4;
            trackerTextureDesc.height = 4;
            trackerTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
            trackerTextureDesc.debugName = "PendingWholeStageTrackerTexture";
            nvrhi::TextureStateExtension trackerTexture(trackerTextureDesc);
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingTextureState(
                &trackerTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None);
            tracker.requireTextureState(
                &trackerTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Compute);
            tracker.requireTextureState(
                &trackerTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Pixel);
            const auto& barriers = tracker.getTextureBarriers();
            trackerPassed &= barriers.size() == 1
                && barriers[0].entireTexture
                && barriers[0].shaderStagesAfter
                    == stageUnion(nvrhi::ShaderType::Compute,
                        nvrhi::ShaderType::Pixel);
        }

        // Pending whole-texture UAV transitions coalesce in the same way as
        // buffers, retaining the stage union for the following dependency.
        {
            nvrhi::TextureDesc trackerTextureDesc {};
            trackerTextureDesc.width = 4;
            trackerTextureDesc.height = 4;
            trackerTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
            trackerTextureDesc.isUAV = true;
            trackerTextureDesc.debugName = "PendingWholeUavTrackerTexture";
            nvrhi::TextureStateExtension trackerTexture(trackerTextureDesc);
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingTextureState(
                &trackerTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None);
            tracker.requireTextureState(
                &trackerTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Compute);
            tracker.requireTextureState(
                &trackerTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Pixel);
            const nvrhi::ShaderType uavStages =
                stageUnion(nvrhi::ShaderType::Compute,
                    nvrhi::ShaderType::Pixel);
            const auto& pendingBarriers = tracker.getTextureBarriers();
            trackerPassed &= pendingBarriers.size() == 1
                && pendingBarriers[0].entireTexture
                && pendingBarriers[0].stateBefore == nvrhi::ResourceStates::CopyDest
                && pendingBarriers[0].stateAfter == nvrhi::ResourceStates::UnorderedAccess
                && pendingBarriers[0].shaderStagesAfter == uavStages;

            tracker.clearBarriers();
            tracker.requireTextureState(
                &trackerTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::CopySource,
                nvrhi::ShaderType::None);
            const auto& laterBarriers = tracker.getTextureBarriers();
            trackerPassed &= laterBarriers.size() == 1
                && laterBarriers[0].entireTexture
                && laterBarriers[0].stateBefore == nvrhi::ResourceStates::UnorderedAccess
                && laterBarriers[0].stateAfter == nvrhi::ResourceStates::CopySource
                && laterBarriers[0].shaderStagesBefore == uavStages;
        }

        // Partial texture tracking uses a separate subresource path, including
        // its own UAV-barrier suppression state. It must preserve the same union.
        {
            nvrhi::TextureDesc trackerTextureDesc {};
            trackerTextureDesc.width = 4;
            trackerTextureDesc.height = 4;
            trackerTextureDesc.mipLevels = 2;
            trackerTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
            trackerTextureDesc.isUAV = true;
            trackerTextureDesc.debugName = "PendingPartialUavTrackerTexture";
            nvrhi::TextureStateExtension trackerTexture(trackerTextureDesc);
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            const nvrhi::TextureSubresourceSet mip0(0, 1, 0, 1);
            tracker.beginTrackingTextureState(
                &trackerTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None);
            tracker.requireTextureState(
                &trackerTexture,
                mip0,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Compute);
            tracker.requireTextureState(
                &trackerTexture,
                mip0,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Pixel);
            const nvrhi::ShaderType uavStages =
                stageUnion(nvrhi::ShaderType::Compute,
                    nvrhi::ShaderType::Pixel);
            const auto& pendingBarriers = tracker.getTextureBarriers();
            trackerPassed &= pendingBarriers.size() == 1
                && !pendingBarriers[0].entireTexture
                && pendingBarriers[0].mipLevel == 0
                && pendingBarriers[0].stateBefore == nvrhi::ResourceStates::CopyDest
                && pendingBarriers[0].stateAfter == nvrhi::ResourceStates::UnorderedAccess
                && pendingBarriers[0].shaderStagesAfter == uavStages;

            tracker.clearBarriers();
            tracker.requireTextureState(
                &trackerTexture,
                mip0,
                nvrhi::ResourceStates::CopySource,
                nvrhi::ShaderType::None);
            const auto& laterBarriers = tracker.getTextureBarriers();
            trackerPassed &= laterBarriers.size() == 1
                && !laterBarriers[0].entireTexture
                && laterBarriers[0].mipLevel == 0
                && laterBarriers[0].stateBefore == nvrhi::ResourceStates::UnorderedAccess
                && laterBarriers[0].stateAfter == nvrhi::ResourceStates::CopySource
                && laterBarriers[0].shaderStagesBefore == uavStages;
        }

        // Explicit dependencies preserve resource state while resetting the
        // outstanding stage scope to the declared destination.
        {
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None);
            trackerPassed &= tracker.addBufferMemoryDependency(
                &trackerBuffer,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None,
                nvrhi::ShaderType::None);
            const auto& barriers = tracker.getBufferBarriers();
            trackerPassed &= barriers.size() == 1
                && barriers[0].stateBefore == nvrhi::ResourceStates::CopyDest
                && barriers[0].stateAfter == nvrhi::ResourceStates::CopyDest
                && barriers[0].shaderStagesBefore == nvrhi::ShaderType::None
                && barriers[0].shaderStagesAfter == nvrhi::ShaderType::None;
        }

        {
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Compute);
            trackerPassed &= tracker.addBufferMemoryDependency(
                &trackerBuffer,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Pixel,
                nvrhi::ShaderType::Pixel);
            const auto& dependencyBarriers = tracker.getBufferBarriers();
            trackerPassed &= dependencyBarriers.size() == 1
                && dependencyBarriers[0].shaderStagesBefore
                    == stageUnion(
                        nvrhi::ShaderType::Compute,
                        nvrhi::ShaderType::Pixel)
                && dependencyBarriers[0].shaderStagesAfter
                    == nvrhi::ShaderType::Pixel;

            tracker.clearBarriers();
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::UnorderedAccess,
                nvrhi::ShaderType::Compute);
            const auto& laterBarriers = tracker.getBufferBarriers();
            trackerPassed &= laterBarriers.size() == 1
                && laterBarriers[0].shaderStagesBefore
                    == nvrhi::ShaderType::Pixel
                && laterBarriers[0].shaderStagesAfter
                    == nvrhi::ShaderType::Compute;
        }

        // A dependency on one mip resets only that mip's stage frontier.
        {
            nvrhi::TextureDesc dependencyTextureDesc {};
            dependencyTextureDesc.width = 4;
            dependencyTextureDesc.height = 4;
            dependencyTextureDesc.mipLevels = 2;
            dependencyTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
            dependencyTextureDesc.debugName = "MemoryDependencyTrackerTexture";
            nvrhi::TextureStateExtension dependencyTexture(
                dependencyTextureDesc);
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingTextureState(
                &dependencyTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Compute);
            const nvrhi::TextureSubresourceSet mip0(0, 1, 0, 1);
            trackerPassed &= tracker.addTextureMemoryDependency(
                &dependencyTexture,
                mip0,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Pixel,
                nvrhi::ShaderType::Pixel);
            const auto& barriers = tracker.getTextureBarriers();
            trackerPassed &= barriers.size() == 1
                && !barriers[0].entireTexture
                && barriers[0].mipLevel == 0
                && barriers[0].shaderStagesBefore
                    == stageUnion(
                        nvrhi::ShaderType::Compute,
                        nvrhi::ShaderType::Pixel)
                && barriers[0].shaderStagesAfter
                    == nvrhi::ShaderType::Pixel
                && tracker.getTextureSubresourceShaderStages(
                    &dependencyTexture, 0, 0)
                    == nvrhi::ShaderType::Pixel
                && tracker.getTextureSubresourceShaderStages(
                    &dependencyTexture, 0, 1)
                    == nvrhi::ShaderType::Compute;
        }

        // A same-state declaration following a pending dependency widens the
        // dependency destination instead of appending a duplicate barrier.
        {
            nvrhi::CommandListResourceStateTracker tracker(&messageCallback);
            tracker.beginTrackingBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Compute);
            trackerPassed &= tracker.addBufferMemoryDependency(
                &trackerBuffer,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Compute,
                nvrhi::ShaderType::Compute);
            tracker.requireBufferState(
                &trackerBuffer,
                nvrhi::ResourceStates::ShaderResource,
                nvrhi::ShaderType::Pixel);
            const auto& barriers = tracker.getBufferBarriers();
            trackerPassed &= barriers.size() == 1
                && barriers[0].shaderStagesAfter
                    == stageUnion(
                        nvrhi::ShaderType::Compute,
                        nvrhi::ShaderType::Pixel);
        }

        // Tracker-level validation must reject missing or divergent state,
        // permanent lifetimes, pending permanent transitions, and bad ranges.
        {
            const uint32_t errorsBefore = messageCallback.errors;

            nvrhi::BufferStateExtension untrackedBuffer(trackerBufferDesc);
            nvrhi::CommandListResourceStateTracker untrackedTracker(
                &messageCallback);
            trackerPassed &= !untrackedTracker.addBufferMemoryDependency(
                &untrackedBuffer,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None,
                nvrhi::ShaderType::None);

            nvrhi::BufferStateExtension mismatchBuffer(trackerBufferDesc);
            nvrhi::CommandListResourceStateTracker mismatchTracker(
                &messageCallback);
            mismatchTracker.beginTrackingBufferState(
                &mismatchBuffer, nvrhi::ResourceStates::CopyDest);
            trackerPassed &= !mismatchTracker.addBufferMemoryDependency(
                &mismatchBuffer,
                nvrhi::ResourceStates::CopySource,
                nvrhi::ShaderType::None,
                nvrhi::ShaderType::None);

            nvrhi::BufferStateExtension permanentBuffer(trackerBufferDesc);
            permanentBuffer.permanentState = nvrhi::ResourceStates::CopyDest;
            nvrhi::CommandListResourceStateTracker permanentTracker(
                &messageCallback);
            trackerPassed &= !permanentTracker.addBufferMemoryDependency(
                &permanentBuffer,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None,
                nvrhi::ShaderType::None);

            nvrhi::BufferStateExtension pendingBuffer(trackerBufferDesc);
            nvrhi::CommandListResourceStateTracker pendingTracker(
                &messageCallback);
            pendingTracker.beginTrackingBufferState(
                &pendingBuffer, nvrhi::ResourceStates::Common);
            pendingTracker.setPermanentBufferState(
                &pendingBuffer, nvrhi::ResourceStates::CopyDest);
            trackerPassed &= !pendingTracker.addBufferMemoryDependency(
                &pendingBuffer,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None,
                nvrhi::ShaderType::None);

            nvrhi::TextureDesc rangeTextureDesc {};
            rangeTextureDesc.width = 4;
            rangeTextureDesc.height = 4;
            rangeTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
            nvrhi::TextureStateExtension rangeTexture(rangeTextureDesc);
            nvrhi::CommandListResourceStateTracker rangeTracker(
                &messageCallback);
            rangeTracker.beginTrackingTextureState(
                &rangeTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::CopyDest);
            const nvrhi::TextureSubresourceSet outOfRange(
                rangeTextureDesc.mipLevels, 1, 0, 1);
            trackerPassed &= !rangeTracker.addTextureMemoryDependency(
                &rangeTexture,
                outOfRange,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None,
                nvrhi::ShaderType::None);

            trackerPassed &= messageCallback.errors == errorsBefore + 5;
            messageCallback.errors = errorsBefore;
        }

        passed &= trackerPassed;
        if (!trackerPassed)
            std::cerr << "Stage-qualified resource tracker test failed\n";
    }

    // Bindless update-after-bind is opt-in. Probe Vulkan-Hpp's dispatched
    // creation calls so the test covers all three required pieces: binding,
    // descriptor-set layout, and descriptor pool flags. The default layout
    // must retain the old behavior.
    passed &= !nvrhi::BindlessLayoutDesc {}.enableUpdateAfterBind;
    passed &= !nvrhi::BindlessLayoutDesc {}.enableUpdateUnusedWhilePending;
    if (supportsDescriptorUpdateAfterBindProbe)
    {
        realCreateDescriptorSetLayout =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorSetLayout;
        realCreateDescriptorPool =
            VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorPool;
        VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorSetLayout =
            probeCreateDescriptorSetLayout;
        VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorPool =
            probeCreateDescriptorPool;
        descriptorUpdateAfterBindProbe.active = true;

        const auto createBindlessTable = [&](bool enableUpdateAfterBind,
                                             bool enableUpdateUnusedWhilePending) {
            descriptorUpdateAfterBindProbe.reset();
            nvrhi::BindlessLayoutDesc desc {};
            desc.visibility = nvrhi::ShaderType::All;
            desc.maxCapacity = 8;
            desc.registerSpaces.push_back(
                nvrhi::BindingLayoutItem::Texture_SRV(0));
            desc.enableUpdateAfterBind = enableUpdateAfterBind;
            desc.enableUpdateUnusedWhilePending = enableUpdateUnusedWhilePending;

            nvrhi::BindingLayoutHandle layout =
                device->createBindlessLayout(desc);
            nvrhi::DescriptorTableHandle table;
            if (layout)
                table = device->createDescriptorTable(layout);

            const bool created = layout && table;
            const bool flagsMatch =
                descriptorUpdateAfterBindProbe.sawLayout
                && descriptorUpdateAfterBindProbe.sawBindingFlags
                && descriptorUpdateAfterBindProbe.sawPool
                && (descriptorUpdateAfterBindProbe.layoutUsesUpdateAfterBindPool
                    == enableUpdateAfterBind)
                && (descriptorUpdateAfterBindProbe.bindingsUseUpdateAfterBind
                    == enableUpdateAfterBind)
                && (descriptorUpdateAfterBindProbe.bindingsUseUpdateUnusedWhilePending
                    == enableUpdateUnusedWhilePending)
                && (descriptorUpdateAfterBindProbe.poolUsesUpdateAfterBind
                    == enableUpdateAfterBind);
            return created && flagsMatch;
        };

        passed &= createBindlessTable(false, false);
        passed &= createBindlessTable(true, false);
        passed &= createBindlessTable(true, true);

        descriptorUpdateAfterBindProbe.active = false;
        VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorSetLayout =
            realCreateDescriptorSetLayout;
        VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorPool =
            realCreateDescriptorPool;
    }

    // Binding-set cache identity must include both descriptor-array placement
    // and the per-item implicit-transition policy.
    {
        const nvrhi::BindingSetItem defaultItem =
            nvrhi::BindingSetItem::Texture_SRV(3, nullptr);
        nvrhi::BindingSetItem optedOutItem = defaultItem;
        optedOutItem.setEnableAutomaticTransitions(false);
        nvrhi::BindingSetItem arrayItem = defaultItem;
        arrayItem.setArrayElement(1);
        nvrhi::BindingSetItem depthReadOnlyItem = defaultItem;
        depthReadOnlyItem.setDepthReadOnlyAttachment(true);

        const std::hash<nvrhi::BindingSetItem> itemHash {};
        const bool itemIdentityPassed = defaultItem.enableAutomaticTransitions
            && !defaultItem.depthReadOnlyAttachment
            && !optedOutItem.enableAutomaticTransitions
            && depthReadOnlyItem.depthReadOnlyAttachment
            && defaultItem != optedOutItem
            && itemHash(defaultItem) != itemHash(optedOutItem)
            && defaultItem != arrayItem
            && itemHash(defaultItem) != itemHash(arrayItem)
            && defaultItem != depthReadOnlyItem
            && itemHash(defaultItem) != itemHash(depthReadOnlyItem);
        passed &= itemIdentityPassed;
        if (!itemIdentityPassed)
            std::cerr << "BindingSetItem transition/array/depth identity test failed\n";
    }

    // Opting a UAV out of implicit transitions also opts it out of the
    // repeated UAV-barrier fast path, without removing its transition index
    // from the explicit setResourceStatesForBindingSet path.
    {
        nvrhi::BufferDesc bufferDesc {};
        bufferDesc.byteSize = 16;
        bufferDesc.structStride = sizeof(uint32_t);
        bufferDesc.canHaveUAVs = true;
        bufferDesc.debugName = "BindingTransitionPolicyTest";
        nvrhi::BufferHandle buffer = device->createBuffer(bufferDesc);

        nvrhi::BindingLayoutDesc layoutDesc {};
        layoutDesc.visibility = nvrhi::ShaderType::Compute;
        layoutDesc.bindings = {
            nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0)
        };
        nvrhi::BindingLayoutHandle layout = device->createBindingLayout(layoutDesc);

        const nvrhi::BindingSetItem automaticItem =
            nvrhi::BindingSetItem::StructuredBuffer_UAV(0, buffer);
        nvrhi::BindingSetItem optedOutItem = automaticItem;
        optedOutItem.setEnableAutomaticTransitions(false);

        nvrhi::BindingSetDesc automaticDesc {};
        automaticDesc.bindings = { automaticItem };
        nvrhi::BindingSetDesc optedOutDesc {};
        optedOutDesc.bindings = { optedOutItem };
        nvrhi::BindingSetHandle automaticSet;
        nvrhi::BindingSetHandle optedOutSet;
        if (buffer && layout)
        {
            automaticSet = device->createBindingSet(automaticDesc, layout);
            optedOutSet = device->createBindingSet(optedOutDesc, layout);
        }

        bool uavPolicyPassed = bool(buffer) && bool(layout)
            && bool(automaticSet) && bool(optedOutSet);
        if (uavPolicyPassed)
        {
            const auto* automaticVkSet =
                static_cast<const nvrhi::vulkan::BindingSet*>(automaticSet.Get());
            const auto* optedOutVkSet =
                static_cast<const nvrhi::vulkan::BindingSet*>(optedOutSet.Get());
            uavPolicyPassed = automaticVkSet->hasUavBindings
                && !optedOutVkSet->hasUavBindings
                && automaticVkSet->bindingsThatNeedTransitions.size() == 1
                && optedOutVkSet->bindingsThatNeedTransitions.size() == 1;
        }
        passed &= uavPolicyPassed;
        if (!uavPolicyPassed)
            std::cerr << "BindingSetItem implicit UAV-transition policy test failed\n";
    }

    // TextureDesc::useGeneralLayout changes only the Vulkan image layout. State-derived
    // stage and access masks are still supplied by convertResourceState and
    // are covered by the validation-backed transfer test below.
    nvrhi::TextureDesc defaultLayoutDesc {};
    nvrhi::TextureDesc generalLayoutDesc {};
    generalLayoutDesc.useGeneralLayout = true;
    struct LayoutCase
    {
        nvrhi::ResourceStates state;
        vk::ImageLayout defaultLayout;
        vk::ImageLayout generalLayout;
    };
    const std::array<LayoutCase, 11> layoutCases {{
        { nvrhi::ResourceStates::ShaderResource,
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral },
        { nvrhi::ResourceStates::UnorderedAccess,
            vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral },
        { nvrhi::ResourceStates::CopySource,
            vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral },
        { nvrhi::ResourceStates::CopyDest,
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral },
        { nvrhi::ResourceStates::ResolveSource,
            vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral },
        { nvrhi::ResourceStates::ResolveDest,
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral },
        { nvrhi::ResourceStates::RenderTarget,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eGeneral },
        { nvrhi::ResourceStates::DepthWrite,
            vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eGeneral },
        { nvrhi::ResourceStates::DepthRead,
            vk::ImageLayout::eDepthStencilReadOnlyOptimal, vk::ImageLayout::eGeneral },
        { nvrhi::ResourceStates::ShadingRateSurface,
            vk::ImageLayout::eFragmentShadingRateAttachmentOptimalKHR, vk::ImageLayout::eGeneral },
        { nvrhi::ResourceStates::Present,
            vk::ImageLayout::ePresentSrcKHR, vk::ImageLayout::ePresentSrcKHR },
    }};
    for (const LayoutCase& layoutCase : layoutCases)
    {
        const vk::ImageLayout defaultLayout =
            nvrhi::vulkan::convertTextureLayout(layoutCase.state, defaultLayoutDesc);
        const vk::ImageLayout generalLayout =
            nvrhi::vulkan::convertTextureLayout(layoutCase.state, generalLayoutDesc);
        const bool casePassed = defaultLayout == layoutCase.defaultLayout
            && generalLayout == layoutCase.generalLayout;
        passed &= casePassed;
        if (!casePassed)
            std::cerr << "GENERAL-layout state mapping test failed\n";
    }
    passed &= nvrhi::vulkan::convertTextureLayout(
        nvrhi::ResourceStates::Common, generalLayoutDesc) == vk::ImageLayout::eUndefined;
    passed &= nvrhi::vulkan::convertTextureLayout(
        nvrhi::ResourceStates::ShaderResource | nvrhi::ResourceStates::UnorderedAccess,
        generalLayoutDesc) == vk::ImageLayout::eGeneral;

    // A sampled read-only depth attachment keeps both usages in its access and
    // stage masks while using the one Vulkan layout compatible with both.
    const nvrhi::ResourceStates sampledDepthState =
        nvrhi::ResourceStates::ShaderResource | nvrhi::ResourceStates::DepthRead;
    nvrhi::TextureDesc depthLayoutDesc {};
    depthLayoutDesc.format = nvrhi::Format::D32;
    nvrhi::TextureDesc generalDepthLayoutDesc = depthLayoutDesc;
    generalDepthLayoutDesc.useGeneralLayout = true;
    const nvrhi::vulkan::ResourceStateMapping sampledDepthMapping =
        nvrhi::vulkan::convertTextureState(sampledDepthState, depthLayoutDesc);
    const nvrhi::vulkan::ResourceStateMapping generalSampledDepthMapping =
        nvrhi::vulkan::convertTextureState(sampledDepthState, generalDepthLayoutDesc);
    const vk::AccessFlags2 sampledDepthAccess =
        vk::AccessFlagBits2::eShaderRead
        | vk::AccessFlagBits2::eDepthStencilAttachmentRead;
    const vk::PipelineStageFlags2 sampledDepthStages =
        vk::PipelineStageFlagBits2::eAllCommands
        | vk::PipelineStageFlagBits2::eEarlyFragmentTests
        | vk::PipelineStageFlagBits2::eLateFragmentTests;
    const bool sampledDepthMappingPassed =
        sampledDepthMapping.nvrhiState == sampledDepthState
        && sampledDepthMapping.imageLayout == vk::ImageLayout::eDepthStencilReadOnlyOptimal
        && (sampledDepthMapping.accessMask & sampledDepthAccess) == sampledDepthAccess
        && (sampledDepthMapping.stageFlags & sampledDepthStages) == sampledDepthStages
        && generalSampledDepthMapping.nvrhiState == sampledDepthState
        && generalSampledDepthMapping.imageLayout == vk::ImageLayout::eGeneral
        && generalSampledDepthMapping.accessMask == sampledDepthMapping.accessMask
        && generalSampledDepthMapping.stageFlags == sampledDepthMapping.stageFlags;
    passed &= sampledDepthMappingPassed;
    if (!sampledDepthMappingPassed)
        std::cerr << "Sampled read-only depth state mapping test failed\n";

    // Exercise the actual binding-set -> read-only framebuffer ordering used by
    // CardEffects. The framebuffer's DepthRead request is a subset and must not
    // narrow the combined ShaderResource | DepthRead state, for either whole-
    // resource or expanded subresource tracking.
    {
        struct DepthFormatCandidate
        {
            VkFormat vkFormat;
            nvrhi::Format nvrhiFormat;
        };
        constexpr std::array<DepthFormatCandidate, 4> depthFormatCandidates {{
            { VK_FORMAT_D32_SFLOAT_S8_UINT, nvrhi::Format::D32S8 },
            { VK_FORMAT_D24_UNORM_S8_UINT, nvrhi::Format::D24S8 },
            { VK_FORMAT_D32_SFLOAT, nvrhi::Format::D32 },
            { VK_FORMAT_D16_UNORM, nvrhi::Format::D16 },
        }};

        nvrhi::Format depthFormat = nvrhi::Format::UNKNOWN;
        for (const DepthFormatCandidate& candidate : depthFormatCandidates)
        {
            VkFormatProperties properties {};
            vkGetPhysicalDeviceFormatProperties(
                physicalDevice, candidate.vkFormat, &properties);
            constexpr VkFormatFeatureFlags requiredFeatures =
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
                | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
            if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures)
            {
                depthFormat = candidate.nvrhiFormat;
                break;
            }
        }

        nvrhi::TextureDesc wholeDepthDesc {};
        wholeDepthDesc.width = 4;
        wholeDepthDesc.height = 4;
        wholeDepthDesc.format = depthFormat;
        wholeDepthDesc.isRenderTarget = true;
        wholeDepthDesc.debugName = "SampledReadOnlyDepthWhole";
        nvrhi::TextureHandle wholeDepthTexture;
        nvrhi::TextureHandle mipDepthTexture;
        if (depthFormat != nvrhi::Format::UNKNOWN)
        {
            wholeDepthTexture = device->createTexture(wholeDepthDesc);
            nvrhi::TextureDesc mipDepthDesc = wholeDepthDesc;
            mipDepthDesc.mipLevels = 2;
            mipDepthDesc.debugName = "SampledReadOnlyDepthSubresource";
            mipDepthTexture = device->createTexture(mipDepthDesc);
        }

        nvrhi::BindingLayoutDesc depthLayoutBindingDesc {};
        depthLayoutBindingDesc.visibility = nvrhi::ShaderType::Pixel;
        depthLayoutBindingDesc.bindings = {
            nvrhi::BindingLayoutItem::Texture_SRV(0)
        };
        nvrhi::BindingLayoutHandle depthBindingLayout =
            device->createBindingLayout(depthLayoutBindingDesc);

        const auto createSampledDepthSet = [&](nvrhi::ITexture* texture,
                                                bool automaticTransitions) {
            nvrhi::BindingSetItem item =
                nvrhi::BindingSetItem::Texture_SRV(0, texture);
            item.setDepthReadOnlyAttachment(true)
                .setEnableAutomaticTransitions(automaticTransitions);
            nvrhi::BindingSetDesc desc {};
            desc.bindings = { item };
            return texture && depthBindingLayout
                ? device->createBindingSet(desc, depthBindingLayout)
                : nvrhi::BindingSetHandle {};
        };

        nvrhi::BindingSetHandle wholeDepthSet =
            createSampledDepthSet(wholeDepthTexture.Get(), true);
        nvrhi::BindingSetHandle optedOutDepthSet =
            createSampledDepthSet(wholeDepthTexture.Get(), false);
        nvrhi::BindingSetHandle mipDepthSet =
            createSampledDepthSet(mipDepthTexture.Get(), true);

        nvrhi::FramebufferAttachment wholeDepthAttachment {};
        wholeDepthAttachment.setTexture(wholeDepthTexture.Get()).setReadOnly(true);
        nvrhi::FramebufferHandle wholeDepthFramebuffer = wholeDepthTexture
            ? device->createFramebuffer(
                nvrhi::FramebufferDesc().setDepthAttachment(wholeDepthAttachment))
            : nvrhi::FramebufferHandle {};
        nvrhi::FramebufferHandle writableDepthFramebuffer = wholeDepthTexture
            ? device->createFramebuffer(
                nvrhi::FramebufferDesc().setDepthAttachment(wholeDepthTexture))
            : nvrhi::FramebufferHandle {};

        const nvrhi::TextureSubresourceSet secondMip(1, 1, 0, 1);
        nvrhi::FramebufferAttachment mipDepthAttachment {};
        mipDepthAttachment.setTexture(mipDepthTexture.Get())
            .setSubresources(secondMip)
            .setReadOnly(true);
        nvrhi::FramebufferHandle mipDepthFramebuffer = mipDepthTexture
            ? device->createFramebuffer(
                nvrhi::FramebufferDesc().setDepthAttachment(mipDepthAttachment))
            : nvrhi::FramebufferHandle {};

        nvrhi::CommandListHandle depthCommandList = device->createCommandList();
        bool sampledDepthTrackingPassed = depthFormat != nvrhi::Format::UNKNOWN
            && wholeDepthTexture && mipDepthTexture && depthBindingLayout
            && wholeDepthSet && optedOutDepthSet && mipDepthSet
            && wholeDepthFramebuffer && writableDepthFramebuffer
            && mipDepthFramebuffer && depthCommandList;
        if (sampledDepthTrackingPassed)
        {
            const auto* wholeVkSet = static_cast<const nvrhi::vulkan::BindingSet*>(
                wholeDepthSet.Get());
            const auto* optedOutVkSet = static_cast<const nvrhi::vulkan::BindingSet*>(
                optedOutDepthSet.Get());
            const auto* wholeVkFramebuffer =
                static_cast<const nvrhi::vulkan::Framebuffer*>(
                    wholeDepthFramebuffer.Get());
            const auto* writableVkFramebuffer =
                static_cast<const nvrhi::vulkan::Framebuffer*>(
                    writableDepthFramebuffer.Get());
            sampledDepthTrackingPassed =
                wholeVkSet->hasDepthReadOnlyAttachmentBindings
                && !optedOutVkSet->hasDepthReadOnlyAttachmentBindings
                && wholeVkFramebuffer->depthAttachment.storeOp
                    == vk::AttachmentStoreOp::eNone
                && writableVkFramebuffer->depthAttachment.storeOp
                    == vk::AttachmentStoreOp::eStore;
            if (nvrhi::getFormatInfo(depthFormat).hasStencil)
            {
                sampledDepthTrackingPassed &=
                    wholeVkFramebuffer->stencilAttachment.storeOp
                        == vk::AttachmentStoreOp::eNone
                    && writableVkFramebuffer->stencilAttachment.storeOp
                        == vk::AttachmentStoreOp::eStore;
            }

            depthCommandList->open();
            depthCommandList->beginTrackingTextureState(
                wholeDepthTexture, nvrhi::AllSubresources,
                nvrhi::ResourceStates::Common);
            depthCommandList->setResourceStatesForBindingSet(wholeDepthSet);
            depthCommandList->setResourceStatesForFramebuffer(wholeDepthFramebuffer);
            sampledDepthTrackingPassed &=
                depthCommandList->getTextureSubresourceState(
                    wholeDepthTexture, 0, 0) == sampledDepthState;

            depthCommandList->beginTrackingTextureState(
                mipDepthTexture, nvrhi::AllSubresources,
                nvrhi::ResourceStates::Common);
            depthCommandList->setResourceStatesForBindingSet(mipDepthSet);
            depthCommandList->setResourceStatesForFramebuffer(mipDepthFramebuffer);
            sampledDepthTrackingPassed &=
                depthCommandList->getTextureSubresourceState(
                    mipDepthTexture, 0, 0) == sampledDepthState
                && depthCommandList->getTextureSubresourceState(
                    mipDepthTexture, 0, 1) == sampledDepthState;
            depthCommandList->close();
            const uint64_t depthSubmitId =
                device->executeCommandList(depthCommandList);
            sampledDepthTrackingPassed &= depthSubmitId != 0
                && waitTimeline(
                    vkDevice,
                    device->getQueueSemaphore(nvrhi::CommandQueue::Graphics),
                    depthSubmitId);
        }
        passed &= sampledDepthTrackingPassed;
        if (!sampledDepthTrackingPassed)
            std::cerr << "Sampled read-only depth binding/framebuffer state test failed\n";
    }

    std::vector<VkSemaphore> semaphores;
    auto timeline = [&](uint64_t initialValue = 0) {
        VkSemaphore semaphore = createSemaphore(vkDevice, true, initialValue);
        if (!semaphore)
        {
            std::cerr << "Failed to create a timeline semaphore\n";
            std::exit(1);
        }
        semaphores.push_back(semaphore);
        return semaphore;
    };
    auto binary = [&]() {
        VkSemaphore semaphore = createSemaphore(vkDevice, false);
        if (!semaphore)
        {
            std::cerr << "Failed to create a binary semaphore\n";
            std::exit(1);
        }
        semaphores.push_back(semaphore);
        return semaphore;
    };

    const VkSemaphore graphicsTracking = device->getQueueSemaphore(nvrhi::CommandQueue::Graphics);
    const VkSemaphore computeTracking = device->getQueueSemaphore(nvrhi::CommandQueue::Compute);
    const VkSemaphore copyTracking = device->getQueueSemaphore(nvrhi::CommandQueue::Copy);
    passed &= graphicsTracking != VK_NULL_HANDLE
        && graphicsTracking == computeTracking
        && graphicsTracking == copyTracking;
    passed &= &device->getQueueMutex(nvrhi::CommandQueue::Graphics)
            == &device->getQueueMutex(nvrhi::CommandQueue::Compute)
        && &device->getQueueMutex(nvrhi::CommandQueue::Graphics)
            == &device->getQueueMutex(nvrhi::CommandQueue::Copy);

    // An isolated worker submit reports failure as ID 0 without advancing the
    // queue timeline or consuming the command list. The same closed list can
    // then be retried after the transient failure is removed.
    nvrhi::CommandListHandle failureProbe = device->createCommandList(
        nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Graphics));
    failureProbe->open();
    failureProbe->close();
    auto* failureProbeVk =
        static_cast<nvrhi::vulkan::CommandList*>(failureProbe.Get());
    const nvrhi::vulkan::TrackedCommandBuffer* failureProbeBuffer =
        failureProbeVk->getCurrentCmdBuf().get();
    auto* concreteDevice = static_cast<nvrhi::vulkan::Device*>(device.Get());
    nvrhi::vulkan::Queue* graphicsQueue =
        concreteDevice->getQueue(nvrhi::CommandQueue::Graphics);
    const uint64_t beforeFailedSubmit = graphicsQueue->getLastSubmittedID();
    const uint32_t errorsBeforeFailedSubmit = messageCallback.errors;
    const auto realQueueSubmit2 = VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit2;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit2 = failQueueSubmit2;
    nvrhi::ICommandList* failureProbePtr = failureProbe.Get();
    nvrhi::vulkan::SubmitSyncExtras failureExtras {};
    const uint64_t failedSubmitId = device->tryExecuteCommandListsWithSyncIsolated(
        &failureProbePtr, 1, nvrhi::CommandQueue::Graphics, failureExtras);
    VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit2 = realQueueSubmit2;
    passed &= failedSubmitId == 0;
    passed &= graphicsQueue->getLastSubmittedID() == beforeFailedSubmit;
    passed &= failureProbeVk->getCurrentCmdBuf().get() == failureProbeBuffer;
    passed &= messageCallback.errors == errorsBeforeFailedSubmit + 1;
    messageCallback.errors = errorsBeforeFailedSubmit;

    const uint64_t retrySubmitId = device->tryExecuteCommandListsWithSyncIsolated(
        &failureProbePtr, 1, nvrhi::CommandQueue::Graphics, failureExtras);
    passed &= retrySubmitId == beforeFailedSubmit + 1;
    passed &= failureProbeVk->getCurrentCmdBuf() == nullptr;
    passed &= waitTimeline(
        vkDevice,
        device->getQueueSemaphore(nvrhi::CommandQueue::Graphics),
        retrySubmitId);

    // A retry-aware draining submit must preserve the pre-call accumulator on
    // a guaranteed-not-submitted failure while removing the per-call extras
    // and tracking signal it appended. Probe both attempts at vkQueueSubmit2:
    // each semaphore must appear exactly once on the exact retry.
    VkSemaphore drainingAccumulatorGate = timeline(1);
    VkSemaphore drainingExtraGate = timeline(1);
    VkSemaphore drainingAccumulatorSignal = timeline();
    VkSemaphore drainingExtraSignal = timeline();
    device->queueWaitForSemaphoreAtStage(
        nvrhi::CommandQueue::Graphics,
        drainingAccumulatorGate,
        1,
        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
    device->queueSignalSemaphore(
        nvrhi::CommandQueue::Graphics,
        drainingAccumulatorSignal,
        1);

    nvrhi::vulkan::SubmitSyncExtras drainingFailureExtras {};
    const uint64_t drainingExtraWaitValue = 1;
    const uint64_t drainingExtraSignalValue = 1;
    const VkPipelineStageFlags2 drainingExtraWaitStage =
        VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    drainingFailureExtras.waitSemaphores = &drainingExtraGate;
    drainingFailureExtras.waitValues = &drainingExtraWaitValue;
    drainingFailureExtras.waitStageMasks = &drainingExtraWaitStage;
    drainingFailureExtras.numWaits = 1;
    drainingFailureExtras.signalSemaphores = &drainingExtraSignal;
    drainingFailureExtras.signalValues = &drainingExtraSignalValue;
    drainingFailureExtras.numSignals = 1;

    nvrhi::CommandListHandle drainingFailureProbe = device->createCommandList(
        nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Graphics));
    drainingFailureProbe->open();
    drainingFailureProbe->close();
    auto* drainingFailureProbeVk =
        static_cast<nvrhi::vulkan::CommandList*>(drainingFailureProbe.Get());
    const nvrhi::vulkan::TrackedCommandBuffer* drainingFailureProbeBuffer =
        drainingFailureProbeVk->getCurrentCmdBuf().get();
    nvrhi::ICommandList* drainingFailureProbePtr = drainingFailureProbe.Get();

    const uint64_t beforeFailedDrainingSubmit =
        graphicsQueue->getLastSubmittedID();
    const uint32_t errorsBeforeFailedDrainingSubmit = messageCallback.errors;
    queueSubmit2Probe = {};
    queueSubmit2Probe.failSubmission = true;
    queueSubmit2Probe.accumulatorWait = drainingAccumulatorGate;
    queueSubmit2Probe.extraWait = drainingExtraGate;
    queueSubmit2Probe.accumulatorSignal = drainingAccumulatorSignal;
    queueSubmit2Probe.extraSignal = drainingExtraSignal;
    queueSubmit2Probe.trackingSignal = graphicsTracking;
    queueSubmit2Probe.trackingValue = beforeFailedDrainingSubmit + 1;
    forwardQueueSubmit2 = realQueueSubmit2;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit2 = probeQueueSubmit2;

    const uint64_t failedDrainingSubmitId =
        device->tryExecuteCommandListsWithSyncDraining(
            &drainingFailureProbePtr,
            1,
            nvrhi::CommandQueue::Graphics,
            drainingFailureExtras);
    const bool drainingSafeFailurePassed = failedDrainingSubmitId == 0
        && graphicsQueue->getLastSubmittedID() == beforeFailedDrainingSubmit
        && drainingFailureProbeVk->getCurrentCmdBuf().get()
            == drainingFailureProbeBuffer
        && timelineValue(vkDevice, drainingAccumulatorSignal) == 0
        && timelineValue(vkDevice, drainingExtraSignal) == 0
        && queueSubmit2Probe.calls == 1
        && queueSubmit2Probe.exact
        && messageCallback.errors == errorsBeforeFailedDrainingSubmit + 1;
    passed &= drainingSafeFailurePassed;
    if (!drainingSafeFailurePassed)
        std::cerr << "Retry-aware draining safe-failure retention test failed\n";
    messageCallback.errors = errorsBeforeFailedDrainingSubmit;

    queueSubmit2Probe.failSubmission = false;
    const uint64_t retryDrainingSubmitId =
        device->tryExecuteCommandListsWithSyncDraining(
            &drainingFailureProbePtr,
            1,
            nvrhi::CommandQueue::Graphics,
            drainingFailureExtras);
    VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit2 = realQueueSubmit2;
    forwardQueueSubmit2 = nullptr;

    const bool drainingRetryPassed =
        retryDrainingSubmitId == beforeFailedDrainingSubmit + 1
        && drainingFailureProbeVk->getCurrentCmdBuf() == nullptr
        && queueSubmit2Probe.calls == 2
        && queueSubmit2Probe.exact
        && waitTimeline(vkDevice, drainingAccumulatorSignal, 1)
        && waitTimeline(vkDevice, drainingExtraSignal, 1)
        && waitTimeline(vkDevice, graphicsTracking, retryDrainingSubmitId)
        && messageCallback.errors == errorsBeforeFailedDrainingSubmit;
    passed &= drainingRetryPassed;
    if (!drainingRetryPassed)
        std::cerr << "Retry-aware draining exact-retry test failed\n";

    VkSemaphore aliasCompletion = timeline();
    auto signalOnQueue = [&](nvrhi::CommandQueue queueType, uint64_t value) {
        nvrhi::vulkan::SubmitSyncExtras extras {};
        extras.signalSemaphores = &aliasCompletion;
        extras.signalValues = &value;
        extras.numSignals = 1;
        return device->executeCommandListsWithSyncIsolated(nullptr, 0, queueType, extras);
    };
    const uint64_t graphicsId = signalOnQueue(nvrhi::CommandQueue::Graphics, 1);
    const uint64_t computeId = signalOnQueue(nvrhi::CommandQueue::Compute, 2);
    const uint64_t aliasCopyId = signalOnQueue(nvrhi::CommandQueue::Copy, 3);
    passed &= computeId == graphicsId + 1 && aliasCopyId == computeId + 1;
    passed &= waitTimeline(vkDevice, aliasCompletion, 3);
    passed &= waitTimeline(vkDevice, graphicsTracking, aliasCopyId);

    VkSemaphore completion = timeline();
    auto isolatedSignal = [&](VkSemaphore semaphore, uint64_t value) {
        nvrhi::vulkan::SubmitSyncExtras extras {};
        extras.signalSemaphores = &semaphore;
        extras.signalValues = &value;
        extras.numSignals = 1;
        return device->executeCommandListsWithSyncIsolated(
            nullptr, 0, nvrhi::CommandQueue::Graphics, extras);
    };

    uint64_t previousId = aliasCopyId;
    uint64_t id = isolatedSignal(completion, 1);
    passed &= id > previousId && waitTimeline(vkDevice, completion, 1);
    previousId = id;

    VkSemaphore accumulated = timeline();
    device->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, accumulated, 1);
    id = isolatedSignal(completion, 2);
    passed &= id > previousId && waitTimeline(vkDevice, completion, 2);
    previousId = id;
    passed &= timelineValue(vkDevice, accumulated) == 0;
    nvrhi::vulkan::SubmitSyncExtras emptyExtras {};
    id = device->executeCommandListsWithSyncDraining(
        nullptr, 0, nvrhi::CommandQueue::Graphics, emptyExtras);
    passed &= id > previousId && waitTimeline(vkDevice, accumulated, 1);
    previousId = id;
    id = device->executeCommandListsWithSyncDraining(
        nullptr, 0, nvrhi::CommandQueue::Graphics, emptyExtras);
    passed &= id > previousId;
    previousId = id;

    VkSemaphore binarySemaphore = binary();
    std::array<VkSemaphore, 2> binarySignals { binarySemaphore, completion };
    std::array<uint64_t, 2> binarySignalValues { 0, 3 };
    nvrhi::vulkan::SubmitSyncExtras binarySignalExtras {};
    binarySignalExtras.signalSemaphores = binarySignals.data();
    binarySignalExtras.signalValues = binarySignalValues.data();
    binarySignalExtras.numSignals = uint32_t(binarySignals.size());
    id = device->executeCommandListsWithSyncIsolated(
        nullptr, 0, nvrhi::CommandQueue::Graphics, binarySignalExtras);
    passed &= id > previousId && waitTimeline(vkDevice, completion, 3);
    previousId = id;

    const VkPipelineStageFlags2 binaryWaitStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    const uint64_t binaryWaitValue = 0;
    const uint64_t completion4 = 4;
    nvrhi::vulkan::SubmitSyncExtras binaryWaitExtras {};
    binaryWaitExtras.waitSemaphores = &binarySemaphore;
    binaryWaitExtras.waitValues = &binaryWaitValue;
    binaryWaitExtras.numWaits = 1;
    binaryWaitExtras.waitStageMasks = &binaryWaitStage;
    binaryWaitExtras.signalSemaphores = &completion;
    binaryWaitExtras.signalValues = &completion4;
    binaryWaitExtras.numSignals = 1;
    id = device->executeCommandListsWithSyncIsolated(
        nullptr, 0, nvrhi::CommandQueue::Graphics, binaryWaitExtras);
    passed &= id > previousId && waitTimeline(vkDevice, completion, completion4);
    previousId = id;

    std::array<VkSemaphore, 2> stagedWaits { timeline(), timeline() };
    passed &= signalTimeline(vkDevice, stagedWaits[0], 1);
    passed &= signalTimeline(vkDevice, stagedWaits[1], 1);
    const std::array<uint64_t, 2> stagedValues { 1, 1 };
    const std::array<VkPipelineStageFlags2, 2> stagedMasks {
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
    };
    const uint64_t completion5 = 5;
    nvrhi::vulkan::SubmitSyncExtras stagedExtras {};
    stagedExtras.waitSemaphores = stagedWaits.data();
    stagedExtras.waitValues = stagedValues.data();
    stagedExtras.numWaits = uint32_t(stagedWaits.size());
    stagedExtras.signalSemaphores = &completion;
    stagedExtras.signalValues = &completion5;
    stagedExtras.numSignals = 1;
    stagedExtras.waitStageMasks = stagedMasks.data();
    id = device->executeCommandListsWithSyncIsolated(
        nullptr, 0, nvrhi::CommandQueue::Graphics, stagedExtras);
    passed &= id > previousId && waitTimeline(vkDevice, completion, completion5);
    previousId = id;

    VkSemaphore accumulatorWait = timeline();
    passed &= signalTimeline(vkDevice, accumulatorWait, 1);
    device->queueWaitForSemaphoreAtStage(
        nvrhi::CommandQueue::Graphics, accumulatorWait, 1,
        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
    device->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, completion, 6);
    id = device->executeCommandListsWithSyncDraining(
        nullptr, 0, nvrhi::CommandQueue::Graphics, emptyExtras);
    passed &= id > previousId && waitTimeline(vkDevice, completion, 6);
    previousId = id;

    // An isolated submit must not consume an unsatisfied accumulator wait.
    // If it did, the isolated completion cannot advance until the host opens
    // the gate, and the test records the timeout before cleaning it up.
    VkSemaphore accumulatorGate = timeline();
    VkSemaphore isolatedCompletion = timeline();
    device->queueWaitForSemaphoreAtStage(
        nvrhi::CommandQueue::Graphics, accumulatorGate, 1,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
    id = isolatedSignal(isolatedCompletion, 1);
    const bool isolatedWaitBypassedAccumulator = waitTimeline(vkDevice, isolatedCompletion, 1);
    passed &= id > previousId && isolatedWaitBypassedAccumulator;
    previousId = id;
    passed &= signalTimeline(vkDevice, accumulatorGate, 1);
    device->queueSignalSemaphore(nvrhi::CommandQueue::Graphics, completion, 7);
    id = device->executeCommandListsWithSyncDraining(
        nullptr, 0, nvrhi::CommandQueue::Graphics, emptyExtras);
    passed &= id > previousId && waitTimeline(vkDevice, completion, 7);
    previousId = id;

    // Real command-buffer and memory-visibility chain through all three NVRHI
    // queue types. The Compute logical queue deliberately performs a transfer
    // operation so this compact test needs no shader compiler or embedded
    // device-specific SPIR-V; stage-scoped shader consumers are covered by the
    // engine SyncVal acceptance suite.
    constexpr std::array<uint32_t, 4> expectedWords {
        0x10203040u, 0x55667788u, 0x90abcdefu, 0xfedcba09u
    };
    nvrhi::BufferDesc gpuBufferDesc {};
    gpuBufferDesc.byteSize = sizeof(expectedWords);
    gpuBufferDesc.initialState = nvrhi::ResourceStates::Common;
    gpuBufferDesc.keepInitialState = true;
    gpuBufferDesc.sharedAcrossQueues = true;
    gpuBufferDesc.debugName = "SubmitSyncTestA";
    nvrhi::BufferHandle bufferA = device->createBuffer(gpuBufferDesc);
    gpuBufferDesc.debugName = "SubmitSyncTestB";
    nvrhi::BufferHandle bufferB = device->createBuffer(gpuBufferDesc);
    nvrhi::BufferDesc readbackDesc {};
    readbackDesc.byteSize = sizeof(expectedWords);
    readbackDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
    readbackDesc.initialState = nvrhi::ResourceStates::CopyDest;
    readbackDesc.keepInitialState = true;
    readbackDesc.sharedAcrossQueues = true;
    readbackDesc.debugName = "SubmitSyncTestReadback";
    nvrhi::BufferHandle readback = device->createBuffer(readbackDesc);
    constexpr size_t uploadProbeSize = 64 * 1024 + 4;
    std::vector<uint8_t> uploadProbeData(uploadProbeSize, 0x5a);
    nvrhi::BufferDesc uploadProbeDesc {};
    uploadProbeDesc.byteSize = uploadProbeSize;
    uploadProbeDesc.initialState = nvrhi::ResourceStates::Common;
    uploadProbeDesc.keepInitialState = true;
    uploadProbeDesc.sharedAcrossQueues = true;
    uploadProbeDesc.debugName = "CopyUploadRetirementProbe";
    nvrhi::BufferHandle copyUploadProbe = device->createBuffer(uploadProbeDesc);
    uploadProbeDesc.debugName = "ComputeUploadRetirementProbe";
    nvrhi::BufferHandle computeUploadProbe = device->createBuffer(uploadProbeDesc);
    passed &= bool(bufferA) && bool(bufferB) && bool(readback)
        && bool(copyUploadProbe) && bool(computeUploadProbe);

    nvrhi::CommandListHandle copyCommandList = device->createCommandList(
        nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Copy));
    nvrhi::CommandListHandle computeCommandList = device->createCommandList(
        nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Compute));
    nvrhi::CommandListHandle graphicsCommandList = device->createCommandList(
        nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Graphics));
    passed &= bool(copyCommandList) && bool(computeCommandList) && bool(graphicsCommandList);

    VkSemaphore contentTimeline = timeline();
    if (bufferA && bufferB && readback && copyUploadProbe && computeUploadProbe
        && copyCommandList && computeCommandList && graphicsCommandList)
    {
        copyCommandList->open();
        copyCommandList->writeBuffer(bufferA, expectedWords.data(), sizeof(expectedWords));
        copyCommandList->writeBuffer(
            copyUploadProbe, uploadProbeData.data(), uploadProbeData.size());
        copyCommandList->close();
        const VkBuffer firstCopyUploadChunk = findUploadChunk(copyCommandList);
        passed &= firstCopyUploadChunk != VK_NULL_HANDLE;
        if (firstCopyUploadChunk == VK_NULL_HANDLE)
            std::cerr << "Copy UploadManager did not allocate an upload chunk\n";
        nvrhi::ICommandList* copyPtr = copyCommandList.Get();
        const uint64_t contentValue1 = 1;
        nvrhi::vulkan::SubmitSyncExtras copyExtras {};
        copyExtras.signalSemaphores = &contentTimeline;
        copyExtras.signalValues = &contentValue1;
        copyExtras.numSignals = 1;
        const uint64_t copyId = device->executeCommandListsWithSyncIsolated(
            &copyPtr, 1, nvrhi::CommandQueue::Copy, copyExtras);

        computeCommandList->open();
        computeCommandList->copyBuffer(
            bufferB, 0, bufferA, 0, sizeof(expectedWords));
        computeCommandList->writeBuffer(
            computeUploadProbe, uploadProbeData.data(), uploadProbeData.size());
        computeCommandList->close();
        const VkBuffer firstComputeUploadChunk = findUploadChunk(computeCommandList);
        passed &= firstComputeUploadChunk != VK_NULL_HANDLE;
        if (firstComputeUploadChunk == VK_NULL_HANDLE)
            std::cerr << "Compute UploadManager did not allocate an upload chunk\n";
        device->queueWaitForCommandListAtStage(
            nvrhi::CommandQueue::Compute, nvrhi::CommandQueue::Copy, copyId,
            VK_PIPELINE_STAGE_2_COPY_BIT);
        nvrhi::ICommandList* computePtr = computeCommandList.Get();
        const uint64_t contentValue2 = 2;
        nvrhi::vulkan::SubmitSyncExtras computeExtras {};
        computeExtras.waitSemaphores = &contentTimeline;
        computeExtras.waitValues = &contentValue1;
        computeExtras.numWaits = 1;
        // Null waitStageMasks intentionally proves the ALL_COMMANDS source-
        // compatibility path while the accumulator wait above is stage-scoped.
        computeExtras.signalSemaphores = &contentTimeline;
        computeExtras.signalValues = &contentValue2;
        computeExtras.numSignals = 1;
        device->executeCommandListsWithSyncDraining(
            &computePtr, 1, nvrhi::CommandQueue::Compute, computeExtras);

        graphicsCommandList->open();
        graphicsCommandList->copyBuffer(
            readback, 0, bufferB, 0, sizeof(expectedWords));
        graphicsCommandList->close();
        nvrhi::ICommandList* graphicsPtr = graphicsCommandList.Get();
        const uint64_t contentValue3 = 3;
        const VkPipelineStageFlags2 copyWaitStage = VK_PIPELINE_STAGE_2_COPY_BIT;
        nvrhi::vulkan::SubmitSyncExtras graphicsExtras {};
        graphicsExtras.waitSemaphores = &contentTimeline;
        graphicsExtras.waitValues = &contentValue2;
        graphicsExtras.numWaits = 1;
        graphicsExtras.waitStageMasks = &copyWaitStage;
        graphicsExtras.signalSemaphores = &contentTimeline;
        graphicsExtras.signalValues = &contentValue3;
        graphicsExtras.numSignals = 1;
        device->executeCommandListsWithSyncIsolated(
            &graphicsPtr, 1, nvrhi::CommandQueue::Graphics, graphicsExtras);

        passed &= waitTimeline(vkDevice, contentTimeline, contentValue3);
        void* mapped = device->mapBuffer(readback, nvrhi::CpuAccessMode::Read);
        passed &= mapped != nullptr;
        if (mapped)
        {
            passed &= std::memcmp(mapped, expectedWords.data(), sizeof(expectedWords)) == 0;
            device->unmapBuffer(readback);
        }

        // A write whose byte count is not a multiple of four must not modify
        // the following byte. The fourth source byte is deliberately poison:
        // rounding the write up for vkCmdUpdateBuffer would expose it.
        constexpr size_t partialWriteSize = 3;
        constexpr std::array<uint8_t, 4> partialWriteData {
            0xa1, 0xb2, 0xc3, 0xee
        };
        std::array<uint8_t, sizeof(expectedWords)> partialWriteExpected {};
        std::memcpy(
            partialWriteExpected.data(), expectedWords.data(), sizeof(expectedWords));
        std::memcpy(
            partialWriteExpected.data() + 4, partialWriteData.data(), partialWriteSize);

        graphicsCommandList->open();
        graphicsCommandList->writeBuffer(
            bufferA, partialWriteData.data(), partialWriteSize, 4);
        graphicsCommandList->copyBuffer(
            readback, 0, bufferA, 0, sizeof(expectedWords));
        graphicsCommandList->close();

        const uint64_t partialWriteValue = 4;
        nvrhi::vulkan::SubmitSyncExtras partialWriteExtras {};
        partialWriteExtras.signalSemaphores = &contentTimeline;
        partialWriteExtras.signalValues = &partialWriteValue;
        partialWriteExtras.numSignals = 1;
        device->executeCommandListsWithSyncIsolated(
            &graphicsPtr, 1, nvrhi::CommandQueue::Graphics, partialWriteExtras);

        passed &= waitTimeline(vkDevice, contentTimeline, partialWriteValue);
        mapped = device->mapBuffer(readback, nvrhi::CpuAccessMode::Read);
        passed &= mapped != nullptr;
        if (mapped)
        {
            const bool partialWritePreservedNeighbor =
                std::memcmp(
                    mapped, partialWriteExpected.data(), partialWriteExpected.size()) == 0;
            passed &= partialWritePreservedNeighbor;
            if (!partialWritePreservedNeighbor)
                std::cerr << "Partial writeBuffer modified bytes outside its destination range\n";
            device->unmapBuffer(readback);
        }

        // The writes exceed vkCmdUpdateBuffer's 64 KiB limit and therefore
        // force UploadManager chunks. Once the first submissions complete,
        // both logical queues must retire and reuse those exact chunks even
        // though they alias the Graphics physical Queue object.
        VkSemaphore uploadReuseTimeline = timeline();
        copyCommandList->open();
        copyCommandList->writeBuffer(
            copyUploadProbe, uploadProbeData.data(), uploadProbeData.size());
        copyCommandList->close();
        const VkBuffer secondCopyUploadChunk = findUploadChunk(copyCommandList);
        passed &= secondCopyUploadChunk == firstCopyUploadChunk;
        if (secondCopyUploadChunk != firstCopyUploadChunk)
            std::cerr << "Copy UploadManager did not retire and reuse its upload chunk\n";

        const uint64_t uploadReuseValue1 = 1;
        nvrhi::vulkan::SubmitSyncExtras copyReuseExtras {};
        copyReuseExtras.signalSemaphores = &uploadReuseTimeline;
        copyReuseExtras.signalValues = &uploadReuseValue1;
        copyReuseExtras.numSignals = 1;
        const uint64_t copyReuseId = device->executeCommandListsWithSyncIsolated(
            &copyPtr, 1, nvrhi::CommandQueue::Copy, copyReuseExtras);

        computeCommandList->open();
        computeCommandList->writeBuffer(
            computeUploadProbe, uploadProbeData.data(), uploadProbeData.size());
        computeCommandList->close();
        const VkBuffer secondComputeUploadChunk = findUploadChunk(computeCommandList);
        passed &= secondComputeUploadChunk == firstComputeUploadChunk;
        if (secondComputeUploadChunk != firstComputeUploadChunk)
            std::cerr << "Compute UploadManager did not retire and reuse its upload chunk\n";

        const uint64_t uploadReuseValue2 = 2;
        nvrhi::vulkan::SubmitSyncExtras computeReuseExtras {};
        computeReuseExtras.signalSemaphores = &uploadReuseTimeline;
        computeReuseExtras.signalValues = &uploadReuseValue2;
        computeReuseExtras.numSignals = 1;
        const uint64_t computeReuseId = device->executeCommandListsWithSyncIsolated(
            &computePtr, 1, nvrhi::CommandQueue::Compute, computeReuseExtras);
        passed &= computeReuseId > copyReuseId;
        passed &= waitTimeline(vkDevice, uploadReuseTimeline, uploadReuseValue2);
        previousId = computeReuseId;
    }

    // Real same-state dependencies: two transfer writes target each resource
    // without a state transition between them. SyncVal observes the WAW
    // hazards, and the readbacks prove that the second writes won.
    {
        constexpr std::array<uint32_t, 4> firstDependencyWords {{
            0x01010101u, 0x02020202u, 0x03030303u, 0x04040404u
        }};
        constexpr std::array<uint32_t, 4> secondDependencyWords {{
            0xa1a1a1a1u, 0xb2b2b2b2u, 0xc3c3c3c3u, 0xd4d4d4d4u
        }};
        constexpr uint32_t dependencyTextureWidth = 4;
        constexpr uint32_t dependencyTextureHeight = 4;
        constexpr std::array<uint32_t,
            dependencyTextureWidth * dependencyTextureHeight>
            firstDependencyTexels {{
                0x00000001u, 0x00000002u, 0x00000003u, 0x00000004u,
                0x00000005u, 0x00000006u, 0x00000007u, 0x00000008u,
                0x00000009u, 0x0000000au, 0x0000000bu, 0x0000000cu,
                0x0000000du, 0x0000000eu, 0x0000000fu, 0x00000010u,
            }};
        constexpr std::array<uint32_t,
            dependencyTextureWidth * dependencyTextureHeight>
            secondDependencyTexels {{
                0x10000001u, 0x10000002u, 0x10000003u, 0x10000004u,
                0x10000005u, 0x10000006u, 0x10000007u, 0x10000008u,
                0x10000009u, 0x1000000au, 0x1000000bu, 0x1000000cu,
                0x1000000du, 0x1000000eu, 0x1000000fu, 0x10000010u,
            }};

        nvrhi::BufferDesc dependencyBufferDesc {};
        dependencyBufferDesc.byteSize = sizeof(secondDependencyWords);
        dependencyBufferDesc.initialState = nvrhi::ResourceStates::Common;
        dependencyBufferDesc.keepInitialState = false;
        dependencyBufferDesc.debugName = "SameStateDependencyBuffer";
        nvrhi::BufferHandle dependencyBuffer =
            device->createBuffer(dependencyBufferDesc);

        nvrhi::BufferDesc dependencyReadbackDesc {};
        dependencyReadbackDesc.byteSize = sizeof(secondDependencyWords);
        dependencyReadbackDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
        dependencyReadbackDesc.initialState = nvrhi::ResourceStates::CopyDest;
        dependencyReadbackDesc.keepInitialState = true;
        dependencyReadbackDesc.debugName =
            "SameStateDependencyBufferReadback";
        nvrhi::BufferHandle dependencyReadback =
            device->createBuffer(dependencyReadbackDesc);

        nvrhi::TextureDesc dependencyTextureDesc {};
        dependencyTextureDesc.width = dependencyTextureWidth;
        dependencyTextureDesc.height = dependencyTextureHeight;
        dependencyTextureDesc.format = nvrhi::Format::R32_UINT;
        dependencyTextureDesc.initialState = nvrhi::ResourceStates::Common;
        dependencyTextureDesc.keepInitialState = false;
        dependencyTextureDesc.debugName = "SameStateDependencyTexture";
        nvrhi::TextureHandle dependencyTexture =
            device->createTexture(dependencyTextureDesc);
        dependencyTextureDesc.debugName =
            "SameStateDependencyTextureReadback";
        nvrhi::StagingTextureHandle dependencyTextureReadback =
            device->createStagingTexture(
                dependencyTextureDesc, nvrhi::CpuAccessMode::Read);

        nvrhi::CommandListHandle dependencyCommandList =
            device->createCommandList(
                nvrhi::CommandListParameters().setQueueType(
                    nvrhi::CommandQueue::Copy));
        const bool dependencyResourcesCreated = dependencyBuffer
            && dependencyReadback
            && dependencyTexture
            && dependencyTextureReadback
            && dependencyCommandList;
        passed &= dependencyResourcesCreated;
        if (dependencyResourcesCreated)
        {
            nvrhi::vulkan::MemoryDependencyDesc dependency {};
            dependency.setState(nvrhi::ResourceStates::CopyDest)
                .setShaderStagesBefore(nvrhi::ShaderType::None)
                .setShaderStagesAfter(nvrhi::ShaderType::None);
            const nvrhi::TextureSubresourceSet mip0(0, 1, 0, 1);
            const nvrhi::TextureSlice textureSlice {};
            constexpr size_t textureRowPitch =
                dependencyTextureWidth * sizeof(uint32_t);
            constexpr size_t textureDepthPitch =
                textureRowPitch * dependencyTextureHeight;

            dependencyCommandList->open();
            dependencyCommandList->beginTrackingBufferState(
                dependencyBuffer,
                nvrhi::ResourceStates::Common,
                nvrhi::ShaderType::None);
            dependencyCommandList->beginTrackingTextureState(
                dependencyTexture,
                mip0,
                nvrhi::ResourceStates::Common,
                nvrhi::ShaderType::None);
            dependencyCommandList->writeBuffer(
                dependencyBuffer,
                firstDependencyWords.data(),
                sizeof(firstDependencyWords));
            dependencyCommandList->writeTexture(
                dependencyTexture,
                0,
                0,
                firstDependencyTexels.data(),
                textureRowPitch,
                textureDepthPitch);

            // Keep both dependencies pending together, then emit the bounded
            // pre-node batch explicitly before either second write.
            passed &= device->addBufferMemoryDependency(
                dependencyCommandList, dependencyBuffer, dependency);
            passed &= device->addTextureMemoryDependency(
                dependencyCommandList, dependencyTexture, mip0, dependency);
            dependencyCommandList->commitBarriers();

            dependencyCommandList->writeBuffer(
                dependencyBuffer,
                secondDependencyWords.data(),
                sizeof(secondDependencyWords));
            dependencyCommandList->writeTexture(
                dependencyTexture,
                0,
                0,
                secondDependencyTexels.data(),
                textureRowPitch,
                textureDepthPitch);
            dependencyCommandList->copyBuffer(
                dependencyReadback,
                0,
                dependencyBuffer,
                0,
                sizeof(secondDependencyWords));
            dependencyCommandList->copyTexture(
                dependencyTextureReadback,
                textureSlice,
                dependencyTexture,
                textureSlice);
            dependencyCommandList->close();

            nvrhi::ICommandList* dependencyCommandListPtr =
                dependencyCommandList.Get();
            const uint64_t dependencySubmitId =
                device->executeCommandListsWithSyncIsolated(
                    &dependencyCommandListPtr,
                    1,
                    nvrhi::CommandQueue::Copy,
                    emptyExtras);
            passed &= dependencySubmitId > previousId;
            previousId = dependencySubmitId;
            passed &= waitTimeline(
                vkDevice,
                device->getQueueSemaphore(nvrhi::CommandQueue::Copy),
                dependencySubmitId);

            void* dependencyBufferData =
                device->mapBuffer(
                    dependencyReadback, nvrhi::CpuAccessMode::Read);
            passed &= dependencyBufferData != nullptr;
            if (dependencyBufferData)
            {
                passed &= std::memcmp(
                    dependencyBufferData,
                    secondDependencyWords.data(),
                    sizeof(secondDependencyWords)) == 0;
                device->unmapBuffer(dependencyReadback);
            }

            size_t dependencyReadbackRowPitch = 0;
            void* dependencyTextureData = device->mapStagingTexture(
                dependencyTextureReadback,
                textureSlice,
                nvrhi::CpuAccessMode::Read,
                &dependencyReadbackRowPitch);
            passed &= dependencyTextureData != nullptr
                && dependencyReadbackRowPitch >= textureRowPitch;
            if (dependencyTextureData)
            {
                for (uint32_t row = 0;
                     row < dependencyTextureHeight;
                     ++row)
                {
                    passed &= std::memcmp(
                        static_cast<const uint8_t*>(dependencyTextureData)
                            + row * dependencyReadbackRowPitch,
                        secondDependencyTexels.data()
                            + row * dependencyTextureWidth,
                        textureRowPitch) == 0;
                }
                device->unmapStagingTexture(dependencyTextureReadback);
            }
        }
    }

    // Strict graph tracking initializes fresh command-list state exactly,
    // accepts exact idempotent ensures, retains transitioned resources,
    // orders sequential overlapping transitions, and leaves the final
    // transitions pending for one caller-controlled barrier commit.
    {
        nvrhi::BufferDesc graphBufferDesc {};
        graphBufferDesc.byteSize = 64;
        graphBufferDesc.initialState = nvrhi::ResourceStates::Common;
        graphBufferDesc.keepInitialState = false;
        graphBufferDesc.debugName = "StrictGraphBuffer";
        nvrhi::BufferHandle graphBuffer = device->createBuffer(graphBufferDesc);
        graphBufferDesc.debugName = "StrictGraphAllStagesBuffer";
        nvrhi::BufferHandle graphAllStagesBuffer =
            device->createBuffer(graphBufferDesc);
        graphBufferDesc.debugName = "StrictGraphEnsureOnlyBuffer";
        nvrhi::BufferHandle graphEnsureOnlyBuffer =
            device->createBuffer(graphBufferDesc);

        nvrhi::TextureDesc graphTextureDesc {};
        graphTextureDesc.width = 4;
        graphTextureDesc.height = 4;
        graphTextureDesc.mipLevels = 2;
        graphTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
        graphTextureDesc.initialState = nvrhi::ResourceStates::Common;
        graphTextureDesc.keepInitialState = false;
        graphTextureDesc.debugName = "StrictGraphTexture";
        nvrhi::TextureHandle graphTexture =
            device->createTexture(graphTextureDesc);
        graphTextureDesc.debugName = "StrictGraphEnsureOnlyTexture";
        nvrhi::TextureHandle graphEnsureOnlyTexture =
            device->createTexture(graphTextureDesc);

        nvrhi::CommandListHandle graphList = device->createCommandList(
            nvrhi::CommandListParameters().setQueueType(
                nvrhi::CommandQueue::Graphics));
        passed &= graphBuffer && graphAllStagesBuffer && graphEnsureOnlyBuffer
            && graphTexture && graphEnsureOnlyTexture && graphList;
        if (graphBuffer && graphAllStagesBuffer && graphEnsureOnlyBuffer
            && graphTexture && graphEnsureOnlyTexture && graphList)
        {
            nvrhi::vulkan::GraphResourceState common {};
            common.setState(nvrhi::ResourceStates::Common)
                .setShaderStages(nvrhi::ShaderType::None);
            nvrhi::vulkan::GraphResourceState shaderRead {};
            shaderRead.setState(nvrhi::ResourceStates::ShaderResource)
                .setShaderStages(nvrhi::ShaderType::Pixel);
            nvrhi::vulkan::GraphResourceState shaderAll {};
            shaderAll.setState(nvrhi::ResourceStates::ShaderResource)
                .setShaderStages(nvrhi::ShaderType::All);
            nvrhi::vulkan::GraphResourceState copyDest {};
            copyDest.setState(nvrhi::ResourceStates::CopyDest)
                .setShaderStages(nvrhi::ShaderType::None);
            nvrhi::vulkan::GraphResourceStateTransition toShaderRead {};
            toShaderRead.setStateBefore(nvrhi::ResourceStates::Common)
                .setStateAfter(nvrhi::ResourceStates::ShaderResource)
                .setShaderStagesBefore(nvrhi::ShaderType::None)
                .setShaderStagesAfter(nvrhi::ShaderType::Pixel);
            nvrhi::vulkan::GraphResourceStateTransition shaderReadToCopy {};
            shaderReadToCopy.setStateBefore(
                    nvrhi::ResourceStates::ShaderResource)
                .setStateAfter(nvrhi::ResourceStates::CopyDest)
                .setShaderStagesBefore(nvrhi::ShaderType::Pixel)
                .setShaderStagesAfter(nvrhi::ShaderType::None);
            nvrhi::vulkan::GraphResourceStateTransition allToCopy {};
            allToCopy.setStateBefore(nvrhi::ResourceStates::ShaderResource)
                .setStateAfter(nvrhi::ResourceStates::CopyDest)
                .setShaderStagesBefore(nvrhi::ShaderType::All)
                .setShaderStagesAfter(nvrhi::ShaderType::None);

            graphList->open();
            passed &= device->ensureBufferStateTracked(
                graphList, graphBuffer, common);
            passed &= device->ensureTextureStateTracked(
                graphList, graphTexture, nvrhi::AllSubresources, common);
            passed &= device->ensureBufferStateTracked(
                graphList, graphAllStagesBuffer, shaderAll);
            passed &= device->ensureBufferStateTracked(
                graphList, graphEnsureOnlyBuffer, common);
            passed &= device->ensureTextureStateTracked(
                graphList, graphEnsureOnlyTexture,
                nvrhi::AllSubresources, common);
            passed &= device->ensureBufferStateTracked(
                graphList, graphAllStagesBuffer, shaderAll);
            passed &= device->ensureBufferStateTracked(
                graphList, graphBuffer, common);
            passed &= device->ensureTextureStateTracked(
                graphList, graphTexture, nvrhi::AllSubresources, common);
            passed &= device->transitionBufferState(
                graphList, graphBuffer, toShaderRead);
            passed &= device->ensureBufferStateTracked(
                graphList, graphBuffer, shaderRead);
            passed &= device->transitionBufferState(
                graphList, graphBuffer, shaderReadToCopy);
            passed &= device->ensureBufferStateTracked(
                graphList, graphBuffer, copyDest);
            passed &= device->transitionTextureState(
                graphList, graphTexture, nvrhi::AllSubresources,
                toShaderRead);
            passed &= device->ensureTextureStateTracked(
                graphList, graphTexture, nvrhi::AllSubresources,
                shaderRead);
            passed &= device->transitionTextureState(
                graphList, graphTexture, nvrhi::AllSubresources,
                shaderReadToCopy);
            passed &= device->ensureTextureStateTracked(
                graphList, graphTexture, nvrhi::AllSubresources,
                copyDest);
            passed &= device->transitionBufferState(
                graphList, graphAllStagesBuffer, allToCopy);
            passed &= device->ensureBufferStateTracked(
                graphList, graphAllStagesBuffer, copyDest);

            auto* vulkanGraphList =
                dynamic_cast<nvrhi::vulkan::CommandList*>(graphList.Get());
            passed &= vulkanGraphList != nullptr;
            if (vulkanGraphList)
            {
                const auto commandBuffer = vulkanGraphList->getCurrentCmdBuf();
                const auto hasReference = [&](nvrhi::IResource* resource) {
                    return commandBuffer && std::any_of(
                        commandBuffer->referencedResources.begin(),
                        commandBuffer->referencedResources.end(),
                        [resource](const nvrhi::RefCountPtr<nvrhi::IResource>& item) {
                            return item.Get() == resource;
                        });
                };
                passed &= hasReference(graphBuffer.Get());
                passed &= hasReference(graphAllStagesBuffer.Get());
                passed &= hasReference(graphEnsureOnlyBuffer.Get());
                passed &= hasReference(graphTexture.Get());
                passed &= hasReference(graphEnsureOnlyTexture.Get());
            }

            // Drop every caller-owned handle before barrier emission and
            // close. Strict graph tracking must retain even ensure-only
            // resources while its local tracker still contains raw pointers.
            graphBuffer = nullptr;
            graphAllStagesBuffer = nullptr;
            graphEnsureOnlyBuffer = nullptr;
            graphTexture = nullptr;
            graphEnsureOnlyTexture = nullptr;
            graphList->commitBarriers();
            graphList->close();

            nvrhi::ICommandList* graphListPtr = graphList.Get();
            const uint64_t graphSubmitId =
                device->executeCommandListsWithSyncIsolated(
                    &graphListPtr,
                    1,
                    nvrhi::CommandQueue::Graphics,
                    emptyExtras);
            passed &= graphSubmitId > previousId;
            previousId = graphSubmitId;
            passed &= waitTimeline(
                vkDevice,
                device->getQueueSemaphore(nvrhi::CommandQueue::Graphics),
                graphSubmitId);
        }
    }

    // Strict graph operations reject untrusted object provenance, closed
    // command lists, non-graph-managed resources, ambiguous texture ranges,
    // inexact seed state/stages, and inexact transition sources.
    {
        const uint32_t errorsBeforeGraphRejections = messageCallback.errors;

        nvrhi::BufferDesc trackedBufferDesc {};
        trackedBufferDesc.byteSize = 64;
        trackedBufferDesc.initialState = nvrhi::ResourceStates::Common;
        trackedBufferDesc.keepInitialState = false;
        trackedBufferDesc.debugName = "StrictGraphTrackedBuffer";
        nvrhi::BufferHandle trackedBuffer =
            device->createBuffer(trackedBufferDesc);
        trackedBufferDesc.debugName = "StrictGraphStageBuffer";
        nvrhi::BufferHandle stageBuffer =
            device->createBuffer(trackedBufferDesc);
        trackedBufferDesc.debugName = "StrictGraphPermanentBuffer";
        nvrhi::BufferHandle permanentBuffer =
            device->createBuffer(trackedBufferDesc);

        nvrhi::TextureDesc rangeTextureDesc {};
        rangeTextureDesc.width = 4;
        rangeTextureDesc.height = 4;
        rangeTextureDesc.mipLevels = 2;
        rangeTextureDesc.arraySize = 2;
        rangeTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
        rangeTextureDesc.initialState = nvrhi::ResourceStates::Common;
        rangeTextureDesc.keepInitialState = false;
        rangeTextureDesc.debugName = "StrictGraphRangeTexture";
        nvrhi::TextureHandle rangeTexture =
            device->createTexture(rangeTextureDesc);
        rangeTextureDesc.mipLevels = 1;
        rangeTextureDesc.debugName = "StrictGraphPermanentTexture";
        nvrhi::TextureHandle permanentTexture =
            device->createTexture(rangeTextureDesc);

        nvrhi::BufferDesc keepBufferDesc = trackedBufferDesc;
        keepBufferDesc.keepInitialState = true;
        keepBufferDesc.debugName = "StrictGraphKeepInitialBuffer";
        nvrhi::BufferHandle keepBuffer = device->createBuffer(keepBufferDesc);
        nvrhi::TextureDesc keepTextureDesc = rangeTextureDesc;
        keepTextureDesc.keepInitialState = true;
        keepTextureDesc.debugName = "StrictGraphKeepInitialTexture";
        nvrhi::TextureHandle keepTexture =
            device->createTexture(keepTextureDesc);

        nvrhi::BufferDesc cpuBufferDesc = trackedBufferDesc;
        cpuBufferDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
        cpuBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
        cpuBufferDesc.debugName = "StrictGraphCpuBuffer";
        nvrhi::BufferHandle cpuBuffer = device->createBuffer(cpuBufferDesc);
        nvrhi::BufferDesc volatileBufferDesc = trackedBufferDesc;
        volatileBufferDesc.isConstantBuffer = true;
        volatileBufferDesc.isVolatile = true;
        volatileBufferDesc.maxVersions = 2;
        volatileBufferDesc.debugName = "StrictGraphVolatileBuffer";
        nvrhi::BufferHandle volatileBuffer =
            device->createBuffer(volatileBufferDesc);

        nvrhi::CommandListHandle graphRejectionList =
            device->createCommandList(
                nvrhi::CommandListParameters().setQueueType(
                    nvrhi::CommandQueue::Graphics));

        nvrhi::vulkan::DeviceHandle foreignDevice =
            nvrhi::vulkan::createDevice(nvrhiDesc);
        nvrhi::BufferHandle foreignBuffer;
        nvrhi::CommandListHandle foreignList;
        if (foreignDevice)
        {
            trackedBufferDesc.debugName = "StrictGraphForeignBuffer";
            foreignBuffer = foreignDevice->createBuffer(trackedBufferDesc);
            foreignList = foreignDevice->createCommandList(
                nvrhi::CommandListParameters().setQueueType(
                    nvrhi::CommandQueue::Graphics));
        }

        passed &= trackedBuffer && stageBuffer && permanentBuffer
            && rangeTexture && permanentTexture && keepBuffer && keepTexture
            && cpuBuffer && volatileBuffer && graphRejectionList
            && foreignDevice && foreignBuffer && foreignList;
        if (trackedBuffer && stageBuffer && permanentBuffer
            && rangeTexture && permanentTexture && keepBuffer && keepTexture
            && cpuBuffer && volatileBuffer && graphRejectionList
            && foreignDevice && foreignBuffer && foreignList)
        {
            nvrhi::vulkan::GraphResourceState common {};
            common.setState(nvrhi::ResourceStates::Common)
                .setShaderStages(nvrhi::ShaderType::None);
            nvrhi::vulkan::GraphResourceState copyDest {};
            copyDest.setState(nvrhi::ResourceStates::CopyDest)
                .setShaderStages(nvrhi::ShaderType::None);
            nvrhi::vulkan::GraphResourceState shaderPixel {};
            shaderPixel.setState(nvrhi::ResourceStates::ShaderResource)
                .setShaderStages(nvrhi::ShaderType::Pixel);
            nvrhi::vulkan::GraphResourceState shaderCompute {};
            shaderCompute.setState(nvrhi::ResourceStates::ShaderResource)
                .setShaderStages(nvrhi::ShaderType::Compute);

            nvrhi::vulkan::GraphResourceStateTransition commonToCopy {};
            commonToCopy.setStateBefore(nvrhi::ResourceStates::Common)
                .setStateAfter(nvrhi::ResourceStates::CopyDest)
                .setShaderStagesBefore(nvrhi::ShaderType::None)
                .setShaderStagesAfter(nvrhi::ShaderType::None);

            // Closed command list.
            passed &= !device->ensureBufferStateTracked(
                graphRejectionList, trackedBuffer, common);
            passed &= !device->transitionBufferState(
                graphRejectionList, trackedBuffer, commonToCopy);

            graphRejectionList->open();

            // Non-Vulkan object types and wrong device ownership.
            ForeignTexture foreignTextureType;
            ForeignBuffer foreignBufferType;
            passed &= !device->ensureTextureStateTracked(
                graphRejectionList, &foreignTextureType,
                nvrhi::AllSubresources, common);
            passed &= !device->ensureBufferStateTracked(
                graphRejectionList, &foreignBufferType, common);
            passed &= !device->ensureBufferStateTracked(
                foreignList, trackedBuffer, common);
            passed &= !device->ensureBufferStateTracked(
                graphRejectionList, foreignBuffer, common);

            // Automatic initial-state restoration and CPU/volatile storage.
            passed &= !device->ensureBufferStateTracked(
                graphRejectionList, keepBuffer, common);
            passed &= !device->ensureTextureStateTracked(
                graphRejectionList, keepTexture,
                nvrhi::AllSubresources, common);
            passed &= !device->ensureBufferStateTracked(
                graphRejectionList, cpuBuffer, copyDest);
            passed &= !device->ensureBufferStateTracked(
                graphRejectionList, volatileBuffer, common);

            // Unknown state, exact-state mismatch, and exact-stage mismatch.
            passed &= !device->ensureBufferStateTracked(
                graphRejectionList, trackedBuffer,
                nvrhi::vulkan::GraphResourceState {});
            passed &= device->ensureBufferStateTracked(
                graphRejectionList, trackedBuffer, common);
            passed &= !device->ensureBufferStateTracked(
                graphRejectionList, trackedBuffer, copyDest);
            passed &= device->ensureBufferStateTracked(
                graphRejectionList, stageBuffer, shaderPixel);
            passed &= !device->ensureBufferStateTracked(
                graphRejectionList, stageBuffer, shaderCompute);

            // Transition source state and stage must both match exactly.
            nvrhi::vulkan::GraphResourceStateTransition wrongStateSource {};
            wrongStateSource.setStateBefore(nvrhi::ResourceStates::CopySource)
                .setStateAfter(nvrhi::ResourceStates::ShaderResource)
                .setShaderStagesBefore(nvrhi::ShaderType::None)
                .setShaderStagesAfter(nvrhi::ShaderType::Pixel);
            passed &= !device->transitionBufferState(
                graphRejectionList, trackedBuffer, wrongStateSource);
            nvrhi::vulkan::GraphResourceStateTransition wrongStageSource {};
            wrongStageSource.setStateBefore(nvrhi::ResourceStates::ShaderResource)
                .setStateAfter(nvrhi::ResourceStates::CopyDest)
                .setShaderStagesBefore(nvrhi::ShaderType::Compute)
                .setShaderStagesAfter(nvrhi::ShaderType::None);
            passed &= !device->transitionBufferState(
                graphRejectionList, stageBuffer, wrongStageSource);

            // Texture ranges are all-unknown or all-known-exact; mixed and
            // out-of-range declarations fail rather than overwrite tracking.
            const nvrhi::TextureSubresourceSet mip0(0, 1, 0, 1);
            const nvrhi::TextureSubresourceSet outOfRange(
                2, 1, 0, 1);
            const nvrhi::TextureSubresourceSet overflowingRange(
                1,
                nvrhi::TextureSubresourceSet::AllMipLevels - 1,
                0,
                1);
            const nvrhi::TextureSubresourceSet outOfRangeArray(
                0, 1, 2, 1);
            const nvrhi::TextureSubresourceSet overflowingArray(
                0,
                1,
                1,
                nvrhi::TextureSubresourceSet::AllArraySlices - 1);
            passed &= device->ensureTextureStateTracked(
                graphRejectionList, rangeTexture, mip0, common);
            passed &= !device->ensureTextureStateTracked(
                graphRejectionList, rangeTexture,
                nvrhi::AllSubresources, common);
            passed &= !device->ensureTextureStateTracked(
                graphRejectionList, rangeTexture, outOfRange, common);
            passed &= !device->ensureTextureStateTracked(
                graphRejectionList, rangeTexture, overflowingRange, common);
            passed &= !device->ensureTextureStateTracked(
                graphRejectionList, rangeTexture, outOfRangeArray, common);
            passed &= !device->ensureTextureStateTracked(
                graphRejectionList, rangeTexture, overflowingArray, common);
            passed &= !device->transitionTextureState(
                graphRejectionList, rangeTexture, outOfRangeArray,
                commonToCopy);
            passed &= !device->ensureTextureStateTracked(
                graphRejectionList, rangeTexture, mip0, copyDest);

            // Pending permanent transitions are already outside graph-owned
            // state lifetime and must reject both resource kinds.
            passed &= device->ensureBufferStateTracked(
                graphRejectionList, permanentBuffer, common);
            graphRejectionList->setPermanentBufferState(
                permanentBuffer, nvrhi::ResourceStates::CopyDest);
            passed &= !device->ensureBufferStateTracked(
                graphRejectionList, permanentBuffer, common);
            passed &= device->ensureTextureStateTracked(
                graphRejectionList, permanentTexture,
                nvrhi::AllSubresources, common);
            graphRejectionList->setPermanentTextureState(
                permanentTexture, nvrhi::ResourceStates::CopyDest);
            passed &= !device->ensureTextureStateTracked(
                graphRejectionList, permanentTexture,
                nvrhi::AllSubresources, common);

            graphRejectionList->close();
            nvrhi::ICommandList* graphRejectionListPtr =
                graphRejectionList.Get();
            const uint64_t graphRejectionSubmitId =
                device->executeCommandListsWithSyncIsolated(
                    &graphRejectionListPtr,
                    1,
                    nvrhi::CommandQueue::Graphics,
                    emptyExtras);
            passed &= graphRejectionSubmitId > previousId;
            previousId = graphRejectionSubmitId;
            passed &= waitTimeline(
                vkDevice,
                device->getQueueSemaphore(nvrhi::CommandQueue::Graphics),
                graphRejectionSubmitId);

            // Submission promotes the pending permanent transitions to the
            // resource-global state. A fresh command list must reject those
            // resources independently of pending-transition tracking.
            nvrhi::CommandListHandle committedPermanentList =
                device->createCommandList(
                    nvrhi::CommandListParameters().setQueueType(
                        nvrhi::CommandQueue::Graphics));
            passed &= bool(committedPermanentList);
            if (committedPermanentList)
            {
                committedPermanentList->open();
                passed &= !device->ensureBufferStateTracked(
                    committedPermanentList, permanentBuffer, common);
                passed &= !device->ensureTextureStateTracked(
                    committedPermanentList, permanentTexture,
                    nvrhi::AllSubresources, common);
                committedPermanentList->close();
                nvrhi::ICommandList* committedPermanentListPtr =
                    committedPermanentList.Get();
                const uint64_t committedPermanentSubmitId =
                    device->executeCommandListsWithSyncIsolated(
                        &committedPermanentListPtr,
                        1,
                        nvrhi::CommandQueue::Graphics,
                        emptyExtras);
                passed &= committedPermanentSubmitId > previousId;
                previousId = committedPermanentSubmitId;
                passed &= waitTimeline(
                    vkDevice,
                    device->getQueueSemaphore(
                        nvrhi::CommandQueue::Graphics),
                    committedPermanentSubmitId);
            }
        }

        passed &= messageCallback.errors
            == errorsBeforeGraphRejections + 26;
        messageCallback.errors = errorsBeforeGraphRejections;

        foreignBuffer = nullptr;
        foreignList = nullptr;
        if (foreignDevice)
        {
            passed &= foreignDevice->waitForIdle();
            foreignDevice->runGarbageCollection();
        }
        foreignDevice = nullptr;
    }

    // Public validation failures must be rejected before any Vulkan command is
    // recorded. Restore the expected diagnostic count afterward so the final
    // test epilogue still detects only unexpected NVRHI failures.
    {
        const uint32_t errorsBeforeRejections = messageCallback.errors;

        nvrhi::BufferDesc rejectionBufferDesc {};
        rejectionBufferDesc.byteSize = 16;
        rejectionBufferDesc.initialState = nvrhi::ResourceStates::Common;
        rejectionBufferDesc.keepInitialState = false;
        rejectionBufferDesc.debugName = "MemoryDependencyRejectionBuffer";
        nvrhi::BufferHandle rejectionBuffer =
            device->createBuffer(rejectionBufferDesc);

        nvrhi::TextureDesc rejectionTextureDesc {};
        rejectionTextureDesc.width = 4;
        rejectionTextureDesc.height = 4;
        rejectionTextureDesc.mipLevels = 2;
        rejectionTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
        rejectionTextureDesc.initialState = nvrhi::ResourceStates::Common;
        rejectionTextureDesc.keepInitialState = false;
        rejectionTextureDesc.debugName = "MemoryDependencyRejectionTexture";
        nvrhi::TextureHandle rejectionTexture =
            device->createTexture(rejectionTextureDesc);

        nvrhi::BufferDesc cpuBufferDesc {};
        cpuBufferDesc.byteSize = 16;
        cpuBufferDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
        cpuBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
        cpuBufferDesc.keepInitialState = false;
        cpuBufferDesc.debugName = "MemoryDependencyCpuBuffer";
        nvrhi::BufferHandle cpuBuffer = device->createBuffer(cpuBufferDesc);

        nvrhi::CommandListHandle rejectionList =
            device->createCommandList(
                nvrhi::CommandListParameters().setQueueType(
                    nvrhi::CommandQueue::Copy));
        passed &= rejectionBuffer
            && rejectionTexture
            && cpuBuffer
            && rejectionList;
        if (rejectionBuffer && rejectionTexture && cpuBuffer && rejectionList)
        {
            nvrhi::vulkan::MemoryDependencyDesc copyDependency {};
            copyDependency.setState(nvrhi::ResourceStates::CopyDest)
                .setShaderStagesBefore(nvrhi::ShaderType::None)
                .setShaderStagesAfter(nvrhi::ShaderType::None);

            // Closed command list.
            passed &= !device->addBufferMemoryDependency(
                rejectionList, rejectionBuffer, copyDependency);

            rejectionList->open();
            rejectionList->beginTrackingBufferState(
                rejectionBuffer,
                nvrhi::ResourceStates::Common,
                nvrhi::ShaderType::None);
            rejectionList->beginTrackingTextureState(
                rejectionTexture,
                nvrhi::AllSubresources,
                nvrhi::ResourceStates::Common,
                nvrhi::ShaderType::None);

            // Unknown state and tracked-state mismatch.
            passed &= !device->addBufferMemoryDependency(
                rejectionList,
                rejectionBuffer,
                nvrhi::vulkan::MemoryDependencyDesc {});
            passed &= !device->addBufferMemoryDependency(
                rejectionList, rejectionBuffer, copyDependency);

            // Empty/out-of-range texture range.
            const nvrhi::TextureSubresourceSet outOfRange(
                rejectionTextureDesc.mipLevels, 1, 0, 1);
            passed &= !device->addTextureMemoryDependency(
                rejectionList,
                rejectionTexture,
                outOfRange,
                copyDependency);

            // The selected range is tracked, but only mip 0 matches. The API
            // must validate every subresource rather than accepting the
            // legacy tracker's any-matching-subresource predicate.
            const nvrhi::TextureSubresourceSet mip0(0, 1, 0, 1);
            rejectionList->setTextureState(
                rejectionTexture,
                mip0,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None);
            passed &= !device->addTextureMemoryDependency(
                rejectionList,
                rejectionTexture,
                nvrhi::AllSubresources,
                copyDependency);

            // Pending permanent transition and CPU-visible buffer.
            rejectionList->setPermanentBufferState(
                rejectionBuffer, nvrhi::ResourceStates::CopyDest);
            passed &= !device->addBufferMemoryDependency(
                rejectionList, rejectionBuffer, copyDependency);
            rejectionList->beginTrackingBufferState(
                cpuBuffer,
                nvrhi::ResourceStates::CopyDest,
                nvrhi::ShaderType::None);
            passed &= !device->addBufferMemoryDependency(
                rejectionList, cpuBuffer, copyDependency);

            rejectionList->close();
            nvrhi::ICommandList* rejectionListPtr = rejectionList.Get();
            const uint64_t rejectionSubmitId =
                device->executeCommandListsWithSyncIsolated(
                    &rejectionListPtr,
                    1,
                    nvrhi::CommandQueue::Copy,
                    emptyExtras);
            passed &= rejectionSubmitId > previousId;
            previousId = rejectionSubmitId;
            passed &= waitTimeline(
                vkDevice,
                device->getQueueSemaphore(nvrhi::CommandQueue::Copy),
                rejectionSubmitId);
        }

        passed &= messageCallback.errors == errorsBeforeRejections + 7;
        messageCallback.errors = errorsBeforeRejections;
    }

    // Public QFOT smoke test. This test device aliases every logical queue to
    // one Vulkan family, so release performs the ordinary state transition
    // and acquire only seeds the destination command list. The timeline wait
    // remains explicit because submission ordering is outside the QFOT API.
    {
        nvrhi::BufferDesc ownershipBufferDesc {};
        ownershipBufferDesc.byteSize = sizeof(expectedWords);
        ownershipBufferDesc.initialState = nvrhi::ResourceStates::Common;
        ownershipBufferDesc.keepInitialState = false;
        ownershipBufferDesc.sharedAcrossQueues = false;
        ownershipBufferDesc.debugName = "QueueOwnershipTransferBuffer";
        nvrhi::BufferHandle ownershipBuffer =
            device->createBuffer(ownershipBufferDesc);
        VkSemaphore ownershipTimeline = timeline();

        nvrhi::vulkan::QueueOwnershipTransferDesc transfer {};
        transfer.setSourceQueue(nvrhi::CommandQueue::Copy)
            .setDestinationQueue(nvrhi::CommandQueue::Graphics)
            .setStateBefore(nvrhi::ResourceStates::CopyDest)
            .setStateAfter(nvrhi::ResourceStates::CopySource)
            .setShaderStagesBefore(nvrhi::ShaderType::None)
            .setShaderStagesAfter(nvrhi::ShaderType::None);

        passed &= bool(ownershipBuffer) && ownershipTimeline != VK_NULL_HANDLE;
        passed &= device->supportsEfficientQueueOwnershipTransfer(
                nvrhi::CommandQueue::Copy, nvrhi::CommandQueue::Graphics)
            && !device->supportsEfficientQueueOwnershipTransfer(
                nvrhi::CommandQueue::Copy, nvrhi::CommandQueue::Copy)
            && !device->supportsEfficientQueueOwnershipTransfer(
                nvrhi::CommandQueue::Count, nvrhi::CommandQueue::Graphics);
        if (ownershipBuffer && ownershipTimeline
            && copyCommandList && graphicsCommandList && readback)
        {
            copyCommandList->open();
            copyCommandList->beginTrackingBufferState(
                ownershipBuffer, nvrhi::ResourceStates::Common);
            copyCommandList->writeBuffer(
                ownershipBuffer, expectedWords.data(), sizeof(expectedWords));
            const bool released = device->releaseBufferQueueOwnership(
                copyCommandList, ownershipBuffer, transfer);
            copyCommandList->close();
            passed &= released;

            nvrhi::ICommandList* copyPtr = copyCommandList.Get();
            const uint64_t ownershipReleaseValue = 1;
            nvrhi::vulkan::SubmitSyncExtras releaseExtras {};
            releaseExtras.signalSemaphores = &ownershipTimeline;
            releaseExtras.signalValues = &ownershipReleaseValue;
            releaseExtras.numSignals = 1;
            device->executeCommandListsWithSyncIsolated(
                &copyPtr, 1, nvrhi::CommandQueue::Copy, releaseExtras);

            graphicsCommandList->open();
            const bool acquired = device->acquireBufferQueueOwnership(
                graphicsCommandList, ownershipBuffer, transfer);
            graphicsCommandList->copyBuffer(
                readback, 0, ownershipBuffer, 0, sizeof(expectedWords));
            graphicsCommandList->close();
            passed &= acquired;

            nvrhi::ICommandList* graphicsPtr = graphicsCommandList.Get();
            const uint64_t ownershipAcquireValue = 2;
            const VkPipelineStageFlags2 ownershipWaitStage =
                VK_PIPELINE_STAGE_2_COPY_BIT;
            nvrhi::vulkan::SubmitSyncExtras acquireExtras {};
            acquireExtras.waitSemaphores = &ownershipTimeline;
            acquireExtras.waitValues = &ownershipReleaseValue;
            acquireExtras.numWaits = 1;
            acquireExtras.waitStageMasks = &ownershipWaitStage;
            acquireExtras.signalSemaphores = &ownershipTimeline;
            acquireExtras.signalValues = &ownershipAcquireValue;
            acquireExtras.numSignals = 1;
            device->executeCommandListsWithSyncIsolated(
                &graphicsPtr, 1, nvrhi::CommandQueue::Graphics,
                acquireExtras);

            passed &= waitTimeline(
                vkDevice, ownershipTimeline, ownershipAcquireValue);
            void* mapped = device->mapBuffer(
                readback, nvrhi::CpuAccessMode::Read);
            passed &= mapped != nullptr;
            if (mapped)
            {
                passed &= std::memcmp(
                    mapped, expectedWords.data(), sizeof(expectedWords)) == 0;
                device->unmapBuffer(readback);
            }
        }
    }

    // Invalid QFOT requests must fail before recording a Vulkan ownership
    // barrier. Cover the two state-lifetime directions and resolved range
    // validation that are easy to regress while the public API evolves.
    {
        const uint32_t errorsBeforeRejections = messageCallback.errors;
        nvrhi::vulkan::QueueOwnershipTransferDesc transfer {};
        transfer.setSourceQueue(nvrhi::CommandQueue::Copy)
            .setDestinationQueue(nvrhi::CommandQueue::Graphics)
            .setStateBefore(nvrhi::ResourceStates::CopyDest)
            .setStateAfter(nvrhi::ResourceStates::CopySource)
            .setShaderStagesBefore(nvrhi::ShaderType::None)
            .setShaderStagesAfter(nvrhi::ShaderType::None);

        nvrhi::BufferDesc permanentBufferDesc {};
        permanentBufferDesc.byteSize = 64;
        permanentBufferDesc.keepInitialState = false;
        permanentBufferDesc.sharedAcrossQueues = false;
        permanentBufferDesc.debugName = "PendingPermanentQfotBuffer";
        nvrhi::BufferHandle permanentBuffer =
            device->createBuffer(permanentBufferDesc);
        nvrhi::CommandListHandle permanentBufferList =
            device->createCommandList(
                nvrhi::CommandListParameters().setQueueType(
                    nvrhi::CommandQueue::Copy));
        passed &= bool(permanentBuffer) && bool(permanentBufferList);
        if (permanentBuffer && permanentBufferList)
        {
            permanentBufferList->open();
            permanentBufferList->beginTrackingBufferState(
                permanentBuffer, nvrhi::ResourceStates::Common);
            permanentBufferList->setPermanentBufferState(
                permanentBuffer, nvrhi::ResourceStates::CopyDest);
            passed &= !device->releaseBufferQueueOwnership(
                permanentBufferList, permanentBuffer, transfer);
            permanentBufferList->close();
            nvrhi::ICommandList* permanentBufferListPtr =
                permanentBufferList.Get();
            passed &= device->executeCommandListsWithSyncIsolated(
                &permanentBufferListPtr, 1, nvrhi::CommandQueue::Copy,
                emptyExtras) != 0;
        }

        nvrhi::TextureDesc permanentTextureDesc {};
        permanentTextureDesc.width = 4;
        permanentTextureDesc.height = 4;
        permanentTextureDesc.format = nvrhi::Format::RGBA8_UNORM;
        permanentTextureDesc.keepInitialState = false;
        permanentTextureDesc.sharedAcrossQueues = false;
        permanentTextureDesc.debugName = "PendingPermanentQfotTexture";
        nvrhi::TextureHandle permanentTexture =
            device->createTexture(permanentTextureDesc);
        nvrhi::CommandListHandle permanentTextureList =
            device->createCommandList(
                nvrhi::CommandListParameters().setQueueType(
                    nvrhi::CommandQueue::Copy));
        passed &= bool(permanentTexture) && bool(permanentTextureList);
        if (permanentTexture && permanentTextureList)
        {
            permanentTextureList->open();
            permanentTextureList->beginTrackingTextureState(
                permanentTexture, nvrhi::AllSubresources,
                nvrhi::ResourceStates::Common);
            permanentTextureList->setPermanentTextureState(
                permanentTexture, nvrhi::ResourceStates::CopyDest);
            passed &= !device->releaseTextureQueueOwnership(
                permanentTextureList, permanentTexture,
                nvrhi::AllSubresources, transfer);
            permanentTextureList->close();
            nvrhi::ICommandList* permanentTextureListPtr =
                permanentTextureList.Get();
            passed &= device->executeCommandListsWithSyncIsolated(
                &permanentTextureListPtr, 1, nvrhi::CommandQueue::Copy,
                emptyExtras) != 0;
        }

        nvrhi::TextureDesc invalidRangeTextureDesc = permanentTextureDesc;
        invalidRangeTextureDesc.debugName = "InvalidRangeQfotTexture";
        nvrhi::TextureHandle invalidRangeTexture =
            device->createTexture(invalidRangeTextureDesc);
        nvrhi::CommandListHandle invalidRangeList =
            device->createCommandList(
                nvrhi::CommandListParameters().setQueueType(
                    nvrhi::CommandQueue::Graphics));
        passed &= bool(invalidRangeTexture) && bool(invalidRangeList);
        if (invalidRangeTexture && invalidRangeList)
        {
            invalidRangeList->open();
            const nvrhi::TextureSubresourceSet outOfRange(
                invalidRangeTextureDesc.mipLevels, 1, 0, 1);
            passed &= !device->acquireTextureQueueOwnership(
                invalidRangeList, invalidRangeTexture,
                outOfRange, transfer);
            invalidRangeList->close();
            nvrhi::ICommandList* invalidRangeListPtr =
                invalidRangeList.Get();
            passed &= device->executeCommandListsWithSyncIsolated(
                &invalidRangeListPtr, 1, nvrhi::CommandQueue::Graphics,
                emptyExtras) != 0;
        }

        passed &= messageCallback.errors == errorsBeforeRejections + 3;
        messageCallback.errors = errorsBeforeRejections;
    }

    // When the driver exposes both maintenance8 and a distinct transfer
    // family, run the real release/semaphore/acquire sequence under SyncVal.
    // Hardware with only one usable family retains the mandatory same-family
    // smoke test above and skips this optional portability-dependent case.
    if (supportsMaintenance8 && distinctTransferQueue != VK_NULL_HANDLE)
    {
        nvrhi::vulkan::DeviceDesc ownershipDeviceDesc {};
        ownershipDeviceDesc.errorCB = &messageCallback;
        ownershipDeviceDesc.instance = instance;
        ownershipDeviceDesc.physicalDevice = physicalDevice;
        ownershipDeviceDesc.device = vkDevice;
        ownershipDeviceDesc.graphicsQueue = queue;
        ownershipDeviceDesc.graphicsQueueIndex = int(queueFamilyIndex);
        ownershipDeviceDesc.computeQueue = queue;
        ownershipDeviceDesc.computeQueueIndex = int(queueFamilyIndex);
        ownershipDeviceDesc.transferQueue = distinctTransferQueue;
        ownershipDeviceDesc.transferQueueIndex =
            int(distinctTransferQueueFamilyIndex);
        ownershipDeviceDesc.deviceExtensions = deviceExtensions.data();
        ownershipDeviceDesc.numDeviceExtensions = 1;
        ownershipDeviceDesc.maintenance8Supported = true;
        nvrhi::vulkan::DeviceHandle ownershipDevice =
            nvrhi::vulkan::createDevice(ownershipDeviceDesc);
        passed &= bool(ownershipDevice);

        if (ownershipDevice)
        {
            nvrhi::BufferDesc sourceDesc {};
            sourceDesc.byteSize = sizeof(expectedWords);
            sourceDesc.initialState = nvrhi::ResourceStates::Common;
            sourceDesc.keepInitialState = false;
            sourceDesc.sharedAcrossQueues = false;
            sourceDesc.debugName = "DistinctFamilyOwnershipSource";
            nvrhi::BufferHandle source =
                ownershipDevice->createBuffer(sourceDesc);

            nvrhi::BufferDesc ownershipReadbackDesc {};
            ownershipReadbackDesc.byteSize = sizeof(expectedWords);
            ownershipReadbackDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
            ownershipReadbackDesc.initialState =
                nvrhi::ResourceStates::CopyDest;
            ownershipReadbackDesc.keepInitialState = true;
            ownershipReadbackDesc.sharedAcrossQueues = false;
            ownershipReadbackDesc.debugName =
                "DistinctFamilyOwnershipReadback";
            nvrhi::BufferHandle ownershipReadback =
                ownershipDevice->createBuffer(ownershipReadbackDesc);

            nvrhi::CommandListHandle ownershipReleaseList =
                ownershipDevice->createCommandList(
                    nvrhi::CommandListParameters().setQueueType(
                        nvrhi::CommandQueue::Copy));
            nvrhi::CommandListHandle ownershipAcquireList =
                ownershipDevice->createCommandList(
                    nvrhi::CommandListParameters().setQueueType(
                        nvrhi::CommandQueue::Graphics));
            VkSemaphore distinctOwnershipTimeline = timeline();

            passed &= bool(source) && bool(ownershipReadback)
                && bool(ownershipReleaseList) && bool(ownershipAcquireList)
                && distinctOwnershipTimeline != VK_NULL_HANDLE
                && ownershipDevice->getQueueFamilyIndex(
                    nvrhi::CommandQueue::Copy)
                    == distinctTransferQueueFamilyIndex
                && ownershipDevice->getQueueFamilyIndex(
                    nvrhi::CommandQueue::Graphics) == queueFamilyIndex
                && ownershipDevice->supportsEfficientQueueOwnershipTransfer(
                    nvrhi::CommandQueue::Copy,
                    nvrhi::CommandQueue::Graphics);

            if (source && ownershipReadback && ownershipReleaseList
                && ownershipAcquireList && distinctOwnershipTimeline)
            {
                nvrhi::vulkan::QueueOwnershipTransferDesc transfer {};
                transfer.setSourceQueue(nvrhi::CommandQueue::Copy)
                    .setDestinationQueue(nvrhi::CommandQueue::Graphics)
                    .setStateBefore(nvrhi::ResourceStates::CopyDest)
                    .setStateAfter(nvrhi::ResourceStates::CopySource)
                    .setShaderStagesBefore(nvrhi::ShaderType::None)
                    .setShaderStagesAfter(nvrhi::ShaderType::None);

                ownershipReleaseList->open();
                ownershipReleaseList->beginTrackingBufferState(
                    source, nvrhi::ResourceStates::Common);
                ownershipReleaseList->writeBuffer(
                    source, expectedWords.data(), sizeof(expectedWords));
                const bool released =
                    ownershipDevice->releaseBufferQueueOwnership(
                        ownershipReleaseList, source, transfer);
                ownershipReleaseList->close();
                passed &= released;

                nvrhi::ICommandList* releasePtr = ownershipReleaseList.Get();
                const uint64_t releaseValue = 1;
                nvrhi::vulkan::SubmitSyncExtras releaseExtras {};
                releaseExtras.signalSemaphores =
                    &distinctOwnershipTimeline;
                releaseExtras.signalValues = &releaseValue;
                releaseExtras.numSignals = 1;
                ownershipDevice->executeCommandListsWithSyncIsolated(
                    &releasePtr, 1, nvrhi::CommandQueue::Copy,
                    releaseExtras);

                ownershipAcquireList->open();
                const bool acquired =
                    ownershipDevice->acquireBufferQueueOwnership(
                        ownershipAcquireList, source, transfer);
                ownershipAcquireList->copyBuffer(
                    ownershipReadback, 0, source, 0,
                    sizeof(expectedWords));
                ownershipAcquireList->close();
                passed &= acquired;

                nvrhi::ICommandList* acquirePtr = ownershipAcquireList.Get();
                const uint64_t acquireValue = 2;
                const VkPipelineStageFlags2 waitStage =
                    VK_PIPELINE_STAGE_2_COPY_BIT;
                nvrhi::vulkan::SubmitSyncExtras acquireExtras {};
                acquireExtras.waitSemaphores =
                    &distinctOwnershipTimeline;
                acquireExtras.waitValues = &releaseValue;
                acquireExtras.numWaits = 1;
                acquireExtras.waitStageMasks = &waitStage;
                acquireExtras.signalSemaphores =
                    &distinctOwnershipTimeline;
                acquireExtras.signalValues = &acquireValue;
                acquireExtras.numSignals = 1;
                ownershipDevice->executeCommandListsWithSyncIsolated(
                    &acquirePtr, 1, nvrhi::CommandQueue::Graphics,
                    acquireExtras);

                passed &= waitTimeline(
                    vkDevice, distinctOwnershipTimeline, acquireValue);
                void* mapped = ownershipDevice->mapBuffer(
                    ownershipReadback, nvrhi::CpuAccessMode::Read);
                passed &= mapped != nullptr;
                if (mapped)
                {
                    passed &= std::memcmp(
                        mapped, expectedWords.data(),
                        sizeof(expectedWords)) == 0;
                    ownershipDevice->unmapBuffer(ownershipReadback);
                }
            }

            // Repeat the handoff for an image so SyncVal also observes the
            // matched queue-family indices and TransferDst -> TransferSrc
            // layout transition on both halves of the ownership operation.
            {
                nvrhi::TextureDesc ownershipTextureDesc {};
                ownershipTextureDesc.width = uint32_t(expectedWords.size());
                ownershipTextureDesc.height = 1;
                ownershipTextureDesc.format = nvrhi::Format::R32_UINT;
                ownershipTextureDesc.initialState =
                    nvrhi::ResourceStates::Common;
                ownershipTextureDesc.keepInitialState = false;
                ownershipTextureDesc.sharedAcrossQueues = false;
                ownershipTextureDesc.debugName =
                    "DistinctFamilyOwnershipTexture";
                nvrhi::TextureHandle ownershipTexture =
                    ownershipDevice->createTexture(ownershipTextureDesc);

                ownershipTextureDesc.debugName =
                    "DistinctFamilyOwnershipTextureUpload";
                nvrhi::StagingTextureHandle ownershipTextureUpload =
                    ownershipDevice->createStagingTexture(
                        ownershipTextureDesc,
                        nvrhi::CpuAccessMode::Write);
                ownershipTextureDesc.debugName =
                    "DistinctFamilyOwnershipTextureReadback";
                nvrhi::StagingTextureHandle ownershipTextureReadback =
                    ownershipDevice->createStagingTexture(
                        ownershipTextureDesc,
                        nvrhi::CpuAccessMode::Read);
                const nvrhi::TextureSlice wholeTexture {};
                size_t uploadRowPitch = 0;
                void* uploadData = ownershipDevice->mapStagingTexture(
                    ownershipTextureUpload, wholeTexture,
                    nvrhi::CpuAccessMode::Write, &uploadRowPitch);

                passed &= bool(ownershipTexture)
                    && bool(ownershipTextureUpload)
                    && bool(ownershipTextureReadback)
                    && uploadData != nullptr
                    && uploadRowPitch >= sizeof(expectedWords);
                if (uploadData)
                {
                    std::memcpy(
                        uploadData, expectedWords.data(),
                        sizeof(expectedWords));
                    ownershipDevice->unmapStagingTexture(
                        ownershipTextureUpload);
                }

                if (ownershipTexture && ownershipTextureUpload
                    && ownershipTextureReadback && uploadData
                    && ownershipReleaseList && ownershipAcquireList
                    && distinctOwnershipTimeline)
                {
                    nvrhi::vulkan::QueueOwnershipTransferDesc transfer {};
                    transfer.setSourceQueue(nvrhi::CommandQueue::Copy)
                        .setDestinationQueue(nvrhi::CommandQueue::Graphics)
                        .setStateBefore(nvrhi::ResourceStates::CopyDest)
                        .setStateAfter(nvrhi::ResourceStates::CopySource)
                        .setShaderStagesBefore(nvrhi::ShaderType::None)
                        .setShaderStagesAfter(nvrhi::ShaderType::None);

                    ownershipReleaseList->open();
                    ownershipReleaseList->beginTrackingTextureState(
                        ownershipTexture, nvrhi::AllSubresources,
                        nvrhi::ResourceStates::Common);
                    ownershipReleaseList->copyTexture(
                        ownershipTexture, wholeTexture,
                        ownershipTextureUpload, wholeTexture);
                    const bool released =
                        ownershipDevice->releaseTextureQueueOwnership(
                            ownershipReleaseList, ownershipTexture,
                            nvrhi::AllSubresources, transfer);
                    ownershipReleaseList->close();
                    passed &= released;

                    nvrhi::ICommandList* releasePtr =
                        ownershipReleaseList.Get();
                    const uint64_t releaseValue = 3;
                    nvrhi::vulkan::SubmitSyncExtras releaseExtras {};
                    releaseExtras.signalSemaphores =
                        &distinctOwnershipTimeline;
                    releaseExtras.signalValues = &releaseValue;
                    releaseExtras.numSignals = 1;
                    ownershipDevice->executeCommandListsWithSyncIsolated(
                        &releasePtr, 1, nvrhi::CommandQueue::Copy,
                        releaseExtras);

                    ownershipAcquireList->open();
                    const bool acquired =
                        ownershipDevice->acquireTextureQueueOwnership(
                            ownershipAcquireList, ownershipTexture,
                            nvrhi::AllSubresources, transfer);
                    ownershipAcquireList->copyTexture(
                        ownershipTextureReadback, wholeTexture,
                        ownershipTexture, wholeTexture);
                    ownershipAcquireList->close();
                    passed &= acquired;

                    nvrhi::ICommandList* acquirePtr =
                        ownershipAcquireList.Get();
                    const uint64_t acquireValue = 4;
                    const VkPipelineStageFlags2 waitStage =
                        VK_PIPELINE_STAGE_2_COPY_BIT;
                    nvrhi::vulkan::SubmitSyncExtras acquireExtras {};
                    acquireExtras.waitSemaphores =
                        &distinctOwnershipTimeline;
                    acquireExtras.waitValues = &releaseValue;
                    acquireExtras.numWaits = 1;
                    acquireExtras.waitStageMasks = &waitStage;
                    acquireExtras.signalSemaphores =
                        &distinctOwnershipTimeline;
                    acquireExtras.signalValues = &acquireValue;
                    acquireExtras.numSignals = 1;
                    ownershipDevice->executeCommandListsWithSyncIsolated(
                        &acquirePtr, 1, nvrhi::CommandQueue::Graphics,
                        acquireExtras);

                    passed &= waitTimeline(
                        vkDevice, distinctOwnershipTimeline, acquireValue);
                    size_t readbackRowPitch = 0;
                    void* readbackData =
                        ownershipDevice->mapStagingTexture(
                            ownershipTextureReadback, wholeTexture,
                            nvrhi::CpuAccessMode::Read,
                            &readbackRowPitch);
                    passed &= readbackData != nullptr
                        && readbackRowPitch >= sizeof(expectedWords);
                    if (readbackData)
                    {
                        passed &= std::memcmp(
                            readbackData, expectedWords.data(),
                            sizeof(expectedWords)) == 0;
                        ownershipDevice->unmapStagingTexture(
                            ownershipTextureReadback);
                    }
                }
            }

            ownershipReleaseList = nullptr;
            ownershipAcquireList = nullptr;
            source = nullptr;
            ownershipReadback = nullptr;
            passed &= ownershipDevice->waitForIdle();
            ownershipDevice->runGarbageCollection();
            ownershipDevice->runGarbageCollection();
        }
        ownershipDevice = nullptr;
    }

    // Exercise every transfer direction used by streaming textures while the
    // images remain in GENERAL: staging upload -> image -> image -> staging
    // readback. Synchronization validation catches any command/layout mismatch,
    // and the texel comparison proves that the commands executed successfully.
    {
        constexpr uint32_t textureWidth = 4;
        constexpr uint32_t textureHeight = 4;
        constexpr std::array<uint32_t, textureWidth * textureHeight> expectedTexels {{
            0x01020304u, 0x11121314u, 0x21222324u, 0x31323334u,
            0x41424344u, 0x51525354u, 0x61626364u, 0x71727374u,
            0x81828384u, 0x91929394u, 0xa1a2a3a4u, 0xb1b2b3b4u,
            0xc1c2c3c4u, 0xd1d2d3d4u, 0xe1e2e3e4u, 0xf1f2f3f4u,
        }};

        nvrhi::TextureDesc textureDesc {};
        textureDesc.width = textureWidth;
        textureDesc.height = textureHeight;
        textureDesc.format = nvrhi::Format::R32_UINT;
        textureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        textureDesc.keepInitialState = true;
        textureDesc.sharedAcrossQueues = true;
        textureDesc.useGeneralLayout = true;
        textureDesc.debugName = "GeneralLayoutTransferSource";
        nvrhi::TextureHandle sourceTexture = device->createTexture(textureDesc);
        textureDesc.debugName = "GeneralLayoutTransferDestination";
        nvrhi::TextureHandle destinationTexture = device->createTexture(textureDesc);
        textureDesc.debugName = "GeneralLayoutUploadStaging";
        nvrhi::StagingTextureHandle uploadTexture = device->createStagingTexture(
            textureDesc, nvrhi::CpuAccessMode::Write);
        textureDesc.debugName = "GeneralLayoutReadbackStaging";
        nvrhi::StagingTextureHandle readbackTexture = device->createStagingTexture(
            textureDesc, nvrhi::CpuAccessMode::Read);
        nvrhi::CommandListHandle layoutCommandList = device->createCommandList(
            nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Copy));

        const bool resourcesCreated = sourceTexture && destinationTexture
            && uploadTexture && readbackTexture && layoutCommandList;
        passed &= resourcesCreated;
        if (resourcesCreated)
        {
            const nvrhi::TextureSlice wholeTexture {};
            size_t uploadRowPitch = 0;
            void* uploadData = device->mapStagingTexture(
                uploadTexture, wholeTexture, nvrhi::CpuAccessMode::Write, &uploadRowPitch);
            passed &= uploadData != nullptr && uploadRowPitch >= textureWidth * sizeof(uint32_t);
            if (uploadData)
            {
                for (uint32_t row = 0; row < textureHeight; ++row)
                {
                    std::memcpy(
                        static_cast<uint8_t*>(uploadData) + row * uploadRowPitch,
                        expectedTexels.data() + row * textureWidth,
                        textureWidth * sizeof(uint32_t));
                }
                device->unmapStagingTexture(uploadTexture);
            }

            if (uploadData)
            {
                layoutCommandList->open();
                layoutCommandList->copyTexture(
                    sourceTexture, wholeTexture, uploadTexture, wholeTexture);
                layoutCommandList->copyTexture(
                    destinationTexture, wholeTexture, sourceTexture, wholeTexture);
                layoutCommandList->copyTexture(
                    readbackTexture, wholeTexture, destinationTexture, wholeTexture);
                layoutCommandList->close();

                nvrhi::ICommandList* layoutCommandListPtr = layoutCommandList.Get();
                const uint64_t layoutSubmitId = device->executeCommandListsWithSyncIsolated(
                    &layoutCommandListPtr, 1, nvrhi::CommandQueue::Copy, emptyExtras);
                passed &= layoutSubmitId != 0;
                passed &= waitTimeline(
                    vkDevice,
                    device->getQueueSemaphore(nvrhi::CommandQueue::Copy),
                    layoutSubmitId);

                size_t readbackRowPitch = 0;
                void* readbackData = device->mapStagingTexture(
                    readbackTexture, wholeTexture, nvrhi::CpuAccessMode::Read, &readbackRowPitch);
                passed &= readbackData != nullptr
                    && readbackRowPitch >= textureWidth * sizeof(uint32_t);
                if (readbackData)
                {
                    for (uint32_t row = 0; row < textureHeight; ++row)
                    {
                        passed &= std::memcmp(
                            static_cast<const uint8_t*>(readbackData) + row * readbackRowPitch,
                            expectedTexels.data() + row * textureWidth,
                            textureWidth * sizeof(uint32_t)) == 0;
                    }
                    device->unmapStagingTexture(readbackTexture);
                }
            }
        }
    }

    // Exercise the raw timer range API against an actual Vulkan timestamp
    // query. Waiting for the queue tracking semaphore guarantees that the
    // nonblocking poll can resolve without a host-side busy wait.
    nvrhi::TimerQueryHandle timerQuery = device->createTimerQuery();
    nvrhi::CommandListHandle timerCommandList = device->createCommandList(
        nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Graphics));
    passed &= bool(timerQuery) && bool(timerCommandList);
    if (timerQuery && timerCommandList)
    {
        nvrhi::vulkan::TimerQueryTimestampRange unresolvedRange {};
        passed &= !device->getTimerQueryTimestampRange(timerQuery, unresolvedRange);

        device->resetTimerQuery(timerQuery);
        timerCommandList->open();
        timerCommandList->beginTimerQuery(timerQuery);
        timerCommandList->endTimerQuery(timerQuery);
        timerCommandList->close();

        nvrhi::ICommandList* timerPtr = timerCommandList.Get();
        const uint64_t timerId = device->executeCommandListsWithSyncIsolated(
            &timerPtr, 1, nvrhi::CommandQueue::Graphics, emptyExtras);
        passed &= timerId > previousId;
        previousId = timerId;
        passed &= waitTimeline(vkDevice, graphicsTracking, timerId);

        const bool timerResolved = device->pollTimerQuery(timerQuery);
        passed &= timerResolved;
        if (timerResolved)
        {
            nvrhi::vulkan::TimerQueryTimestampRange range {};
            passed &= device->getTimerQueryTimestampRange(timerQuery, range);
            passed &= range.timestampValidBits > 0 && range.timestampValidBits <= 64;
            passed &= range.beginTimestamp
                == nvrhi::vulkan::detail::maskTimestamp(range.beginTimestamp, range.timestampValidBits);
            passed &= range.endTimestamp
                == nvrhi::vulkan::detail::maskTimestamp(range.endTimestamp, range.timestampValidBits);

            VkPhysicalDeviceProperties properties {};
            vkGetPhysicalDeviceProperties(physicalDevice, &properties);
            const uint64_t elapsedTicks = nvrhi::vulkan::detail::timestampDelta(
                range.beginTimestamp, range.endTimestamp, range.timestampValidBits);
            const double rawSeconds = double(elapsedTicks)
                * double(properties.limits.timestampPeriod) * 1e-9;
            const double nvrhiSeconds = double(device->getTimerQueryTime(timerQuery));
            const double tolerance = std::max(
                double(properties.limits.timestampPeriod) * 1e-9,
                std::abs(rawSeconds) * double(std::numeric_limits<float>::epsilon()) * 4.0);
            passed &= std::abs(nvrhiSeconds - rawSeconds) <= tolerance;

            nvrhi::vulkan::TimerQueryTimestampRange retainedRange {};
            passed &= device->getTimerQueryTimestampRange(timerQuery, retainedRange);
            passed &= retainedRange.beginTimestamp == range.beginTimestamp
                && retainedRange.endTimestamp == range.endTimestamp
                && retainedRange.timestampValidBits == range.timestampValidBits;
        }
    }

    VkSemaphore ordered = timeline();
    id = isolatedSignal(ordered, 1);
    passed &= id > previousId;
    previousId = id;
    id = isolatedSignal(ordered, 2);
    passed &= id > previousId && waitTimeline(vkDevice, ordered, 2);
    previousId = id;

    VkSemaphore tracking = device->getQueueSemaphore(nvrhi::CommandQueue::Graphics);
    passed &= waitTimeline(vkDevice, tracking, previousId);
    // vkDeviceWaitIdle is externally synchronized against every VkQueue.
    // Intercept the driver call and probe from another thread: the aliased
    // physical queue mutex must already be held when NVRHI enters the driver.
    realDeviceWaitIdle = VULKAN_HPP_DEFAULT_DISPATCHER.vkDeviceWaitIdle;
    waitIdleQueueMutex = &device->getQueueMutex(nvrhi::CommandQueue::Graphics);
    waitIdleObservedQueueLock = false;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDeviceWaitIdle = probeDeviceWaitIdle;
    const bool waitForIdlePassed = device->waitForIdle();
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDeviceWaitIdle = realDeviceWaitIdle;
    passed &= waitForIdlePassed && waitIdleObservedQueueLock;

    copyCommandList = nullptr;
    computeCommandList = nullptr;
    graphicsCommandList = nullptr;
    timerCommandList = nullptr;
    failureProbe = nullptr;
    timerQuery = nullptr;
    bufferA = nullptr;
    bufferB = nullptr;
    readback = nullptr;
    copyUploadProbe = nullptr;
    computeUploadProbe = nullptr;
    device->runGarbageCollection();
    device->runGarbageCollection();
    device = nullptr;
    for (VkSemaphore semaphore : semaphores)
        if (semaphore)
            vkDestroySemaphore(vkDevice, semaphore, nullptr);
    vkDestroyDevice(vkDevice, nullptr);
    destroyMessenger(instance, messenger, nullptr);
    vkDestroyInstance(instance, nullptr);

    if (validationState.errors.load(std::memory_order_relaxed) != 0 || messageCallback.errors != 0)
        passed = false;

    if (!passed)
    {
        std::cerr << "NVRHI Vulkan synchronization integration test failed\n";
        return 1;
    }

    std::cout << "NVRHI Vulkan synchronization integration test passed\n";
    return 0;
}
