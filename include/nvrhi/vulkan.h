/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <vulkan/vulkan.h>
#include <nvrhi/nvrhi.h>

#include <mutex>

namespace nvrhi 
{
    namespace ObjectTypes
    {
        constexpr ObjectType Nvrhi_VK_Device = 0x00030101;
    };
}

namespace nvrhi::vulkan
{
    // Per-submit wait/signal extras for IDevice::executeCommandListsWithSyncIsolated.
    // The semaphores are appended to this submit's wait/signal lists ONLY —
    // the queue's wait/signal accumulator (queueWaitForSemaphore /
    // queueSignalSemaphore / queueWaitForCommandList) is left untouched, so
    // entries queued there for "the next submit" are NOT consumed by this
    // submit. Use this for worker submits that share a queue with another
    // submitter that has critical syncs queued via the accumulator (e.g.
    // the swapchain-tail's acquire wait + present_sem signal). The plain
    // executeCommandLists() drains the accumulator as before.
    struct SubmitSyncExtras
    {
        const VkSemaphore* waitSemaphores = nullptr;
        const uint64_t* waitValues = nullptr;     // 0 for binary, monotonic value for timeline
        uint32_t numWaits = 0;
        const VkSemaphore* signalSemaphores = nullptr;
        const uint64_t* signalValues = nullptr;   // 0 for binary, monotonic value for timeline
        uint32_t numSignals = 0;
        // Optional destination stage for each wait. Null preserves the
        // conservative behavior and waits at ALL_COMMANDS. Every supplied
        // mask must be nonzero and supported by the destination queue family.
        // The caller is also responsible for enabling any Vulkan feature or
        // extension required by feature-gated stage bits.
        const VkPipelineStageFlags2* waitStageMasks = nullptr;
    };

    // Raw timestamps captured by a timer query. The timestamp values are
    // masked to timestampValidBits, as reported by the Vulkan queue family
    // that recorded the query. Consumers must use modular subtraction when
    // computing a duration because the counter may wrap between endpoints.
    struct TimerQueryTimestampRange
    {
        uint64_t beginTimestamp = 0;
        uint64_t endTimestamp = 0;
        uint32_t timestampValidBits = 0;
    };

    class IDevice : public nvrhi::IDevice
    {
    public:
        // Additional Vulkan-specific public methods
        virtual VkSemaphore getQueueSemaphore(CommandQueue queue) = 0;
        virtual void queueWaitForSemaphore(CommandQueue waitQueue, VkSemaphore semaphore, uint64_t value) = 0;
        virtual void queueSignalSemaphore(CommandQueue executionQueue, VkSemaphore semaphore, uint64_t value) = 0;
        virtual uint64_t queueGetCompletedInstance(CommandQueue queue) = 0;

        // Vulkan requires external synchronization on a VkQueue across ALL
        // queue ops (vkQueueSubmit, vkQueuePresentKHR, vkQueueBindSparse).
        // NVRHI's submit and bindSparse take this mutex internally; the
        // application MUST hold this mutex when calling vkQueuePresentKHR
        // (or any raw VkQueue operation) on the same queue from another
        // thread. Without this, multi-threaded submit + present produces
        // VK validator THREADING_ERROR plus undefined behavior up to and
        // including vk::Queue::*: ErrorDeviceLost.
        virtual std::mutex& getQueueMutex(CommandQueue queue) = 0;

        // Sparse-residency variant that signals signalSemaphore at signalValue
        // from vkQueueBindSparse, so the caller can wait on the bind from a
        // subsequent submit without a host stall. Pass VK_NULL_HANDLE for
        // signalSemaphore to skip signaling (equivalent to updateTextureTileMappings).
        virtual void updateTextureTileMappingsSignal(
            ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings,
            CommandQueue executionQueue,
            VkSemaphore signalSemaphore, uint64_t signalValue) = 0;

        // Submit ppCmd with the per-submit wait/signal extras attached to
        // THIS submit only, BYPASSING the queue accumulator. See SubmitSyncExtras
        // doc above.
        virtual uint64_t executeCommandListsWithSyncIsolated(
            ICommandList* const* pCommandLists, size_t numCommandLists,
            CommandQueue executionQueue,
            const SubmitSyncExtras& extras) = 0;

        // Submit ppCmd with the per-submit wait/signal extras AND drain
        // the queue's accumulator atomically (under the queue mutex).
        // Use for the swapchain-tail submit when both extras (e.g.
        // acquire wait + present_sem signal) AND queued accumulator
        // items (e.g. a worker-submit-id wait queued earlier in the
        // frame) need to attach to this specific submit.
        virtual uint64_t executeCommandListsWithSyncDraining(
            ICommandList* const* pCommandLists, size_t numCommandLists,
            CommandQueue executionQueue,
            const SubmitSyncExtras& extras) = 0;

        // Stage-aware variants for waits stored in the queue accumulator.
        // The existing methods remain conservative and use ALL_COMMANDS.
        virtual void queueWaitForSemaphoreAtStage(
            CommandQueue waitQueue, VkSemaphore semaphore, uint64_t value,
            VkPipelineStageFlags2 waitStageMask) = 0;
        virtual void queueWaitForCommandListAtStage(
            CommandQueue waitQueue, CommandQueue executionQueue, uint64_t instance,
            VkPipelineStageFlags2 waitStageMask) = 0;

        // Returns the resolved raw timestamp endpoints without waiting,
        // resetting, or consuming the timer query. Returns false until
        // pollTimerQuery has resolved the query.
        virtual bool getTimerQueryTimestampRange(
            ITimerQuery* query, TimerQueryTimestampRange& range) = 0;
    };

    typedef RefCountPtr<IDevice> DeviceHandle;

    struct DeviceDesc
    {
        IMessageCallback* errorCB = nullptr;

        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkDevice device;

        // any of the queues can be null if this context doesn't intend to use them
        VkQueue graphicsQueue;
        int graphicsQueueIndex = -1;
        VkQueue transferQueue;
        int transferQueueIndex = -1;
        VkQueue computeQueue;
        int computeQueueIndex = -1;

        VkAllocationCallbacks *allocationCallbacks = nullptr;

        const char **instanceExtensions = nullptr;
        size_t numInstanceExtensions = 0;
        
        const char **deviceExtensions = nullptr;
        size_t numDeviceExtensions = 0;

        uint32_t maxTimerQueries = 256;

        // Indicates if VkPhysicalDeviceVulkan12Features::bufferDeviceAddress was set to 'true' at device creation time
        bool bufferDeviceAddressSupported = false;
        bool aftermathEnabled = false;
        bool logBufferLifetime = false;

        std::string vulkanLibraryName; // if empty, use default
    };

    NVRHI_API DeviceHandle createDevice(const DeviceDesc& desc);
   
    NVRHI_API VkFormat convertFormat(nvrhi::Format format);

    NVRHI_API const char* resultToString(VkResult result);
}
