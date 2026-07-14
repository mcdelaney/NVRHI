// Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <nvrhi/vulkan.h>
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
    bool supportsDescriptorUpdateAfterBindProbe = false;
    for (VkPhysicalDevice candidate : physicalDevices)
    {
        VkPhysicalDeviceVulkan13Features supported13 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
        VkPhysicalDeviceVulkan12Features supported12 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        supported12.pNext = &supported13;
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
        for (uint32_t i = 0; i < queueFamilyCount; ++i)
        {
            if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0
                && queueFamilies[i].timestampValidBits != 0)
            {
                physicalDevice = candidate;
                queueFamilyIndex = i;
                supportsDescriptorUpdateAfterBindProbe =
                    supported12.descriptorIndexing
                    && supported12.descriptorBindingPartiallyBound
                    && supported12.descriptorBindingSampledImageUpdateAfterBind
                    && supported12.descriptorBindingUpdateUnusedWhilePending;
                break;
            }
        }
        if (physicalDevice)
            break;
    }
    if (!physicalDevice)
        return 1;

    const float queuePriority = 1.f;
    VkDeviceQueueCreateInfo queueInfo { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queueInfo.queueFamilyIndex = queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;
    VkPhysicalDeviceVulkan13Features enabled13 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    enabled13.synchronization2 = VK_TRUE;
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
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;

    VkDevice vkDevice = VK_NULL_HANDLE;
    if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &vkDevice) != VK_SUCCESS)
        return 1;
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(vkDevice, queueFamilyIndex, 0, &queue);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance, vkGetInstanceProcAddr, vkDevice);

    MessageCallback messageCallback;
    nvrhi::vulkan::DeviceDesc nvrhiDesc {};
    nvrhiDesc.errorCB = &messageCallback;
    nvrhiDesc.instance = instance;
    nvrhiDesc.physicalDevice = physicalDevice;
    nvrhiDesc.device = vkDevice;
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
