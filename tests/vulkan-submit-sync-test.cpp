// Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <nvrhi/vulkan.h>
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
    passed &= bool(bufferA) && bool(bufferB) && bool(readback);

    nvrhi::CommandListHandle copyCommandList = device->createCommandList(
        nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Copy));
    nvrhi::CommandListHandle computeCommandList = device->createCommandList(
        nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Compute));
    nvrhi::CommandListHandle graphicsCommandList = device->createCommandList(
        nvrhi::CommandListParameters().setQueueType(nvrhi::CommandQueue::Graphics));
    passed &= bool(copyCommandList) && bool(computeCommandList) && bool(graphicsCommandList);

    VkSemaphore contentTimeline = timeline();
    if (bufferA && bufferB && readback
        && copyCommandList && computeCommandList && graphicsCommandList)
    {
        copyCommandList->open();
        copyCommandList->writeBuffer(bufferA, expectedWords.data(), sizeof(expectedWords));
        copyCommandList->close();
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
        computeCommandList->close();
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
    passed &= device->waitForIdle();

    copyCommandList = nullptr;
    computeCommandList = nullptr;
    graphicsCommandList = nullptr;
    timerCommandList = nullptr;
    timerQuery = nullptr;
    bufferA = nullptr;
    bufferB = nullptr;
    readback = nullptr;
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
