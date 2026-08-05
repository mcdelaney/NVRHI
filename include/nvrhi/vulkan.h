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
        // Makes the exact successfully-submitted queue frontier explicit in
        // this submit by waiting on NVRHI's tracking timeline semaphore while
        // the queue submission mutex is held. The wait is inserted after
        // accumulator waits and before the caller's waits. Vulkan queue order
        // already guarantees this dependency; this option exists for tooling
        // that reconstructs queue history only from explicit semaphore waits.
        bool waitForCurrentQueueFrontier = false;
    };

    // Describes one explicit queue-family ownership handoff for an exclusive
    // Vulkan resource. The same descriptor must be supplied to the release
    // recorded on sourceQueue and the acquire recorded on destinationQueue.
    // Submission ordering (normally a semaphore signal/wait pair) remains the
    // caller's responsibility and is intentionally not part of this API.
    struct QueueOwnershipTransferDesc
    {
        CommandQueue sourceQueue = CommandQueue::Count;
        CommandQueue destinationQueue = CommandQueue::Count;
        ResourceStates stateBefore = ResourceStates::Unknown;
        ResourceStates stateAfter = ResourceStates::Unknown;
        ShaderType shaderStagesBefore = ShaderType::All;
        ShaderType shaderStagesAfter = ShaderType::All;

        QueueOwnershipTransferDesc& setSourceQueue(CommandQueue value) { sourceQueue = value; return *this; }
        QueueOwnershipTransferDesc& setDestinationQueue(CommandQueue value) { destinationQueue = value; return *this; }
        QueueOwnershipTransferDesc& setStateBefore(ResourceStates value) { stateBefore = value; return *this; }
        QueueOwnershipTransferDesc& setStateAfter(ResourceStates value) { stateAfter = value; return *this; }
        QueueOwnershipTransferDesc& setShaderStagesBefore(ShaderType value) { shaderStagesBefore = value; return *this; }
        QueueOwnershipTransferDesc& setShaderStagesAfter(ShaderType value) { shaderStagesAfter = value; return *this; }
    };

    // Describes an explicit Vulkan memory dependency that preserves the
    // resource state and, for textures, the image layout. The resource must
    // already be explicitly tracked in exactly state. shaderStagesBefore is
    // unioned with every outstanding local access recorded by the state
    // tracker; shaderStagesAfter becomes the new outstanding local scope.
    struct MemoryDependencyDesc
    {
        ResourceStates state = ResourceStates::Unknown;
        ShaderType shaderStagesBefore = ShaderType::All;
        ShaderType shaderStagesAfter = ShaderType::All;

        MemoryDependencyDesc& setState(ResourceStates value) { state = value; return *this; }
        MemoryDependencyDesc& setShaderStagesBefore(ShaderType value) { shaderStagesBefore = value; return *this; }
        MemoryDependencyDesc& setShaderStagesAfter(ShaderType value) { shaderStagesAfter = value; return *this; }
    };

    // Exact graph-owned state used to initialize an open command list's local
    // tracker. Shader-visible states require a non-None shader-stage mask;
    // ShaderType::All is accepted here as the exact logical union of every
    // concrete stage (not as a declaration-time unknown-scope sentinel).
    // Vulkan lowering remains conservative for aggregate masks and may use
    // ALL_COMMANDS or ALL_GRAPHICS. Fixed-function states require
    // ShaderType::None.
    struct GraphResourceState
    {
        ResourceStates state = ResourceStates::Unknown;
        ShaderType shaderStages = ShaderType::None;

        GraphResourceState& setState(ResourceStates value) { state = value; return *this; }
        GraphResourceState& setShaderStages(ShaderType value) { shaderStages = value; return *this; }
    };

    // Exact logical transition authored by a graph. stateBefore and
    // shaderStagesBefore must exactly match every addressed tracked
    // subresource. shaderStagesAfter qualifies the exact destination state.
    struct GraphResourceStateTransition
    {
        ResourceStates stateBefore = ResourceStates::Unknown;
        ResourceStates stateAfter = ResourceStates::Unknown;
        ShaderType shaderStagesBefore = ShaderType::None;
        ShaderType shaderStagesAfter = ShaderType::None;

        GraphResourceStateTransition& setStateBefore(ResourceStates value) { stateBefore = value; return *this; }
        GraphResourceStateTransition& setStateAfter(ResourceStates value) { stateAfter = value; return *this; }
        GraphResourceStateTransition& setShaderStagesBefore(ShaderType value) { shaderStagesBefore = value; return *this; }
        GraphResourceStateTransition& setShaderStagesAfter(ShaderType value) { shaderStagesAfter = value; return *this; }
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

    // Byte accounting for the device-local and host-visible pools NVRHI owns
    // internally, which no public getter otherwise reaches. An application that
    // budgets VRAM cannot see any of this through the resource handles it holds,
    // so it presents as untracked driver overhead.
    struct InternalMemoryStats
    {
        // Per-CommandList UploadManager (host-visible) and ScratchManager
        // (device-local) chunk pools, summed over every live command list.
        // A pool never releases memory before its command list is destroyed --
        // submitChunks only recycles chunk versions and there is no trim path --
        // so each value is a high-water mark rather than a live usage.
        uint64_t uploadChunkBytes = 0;
        uint64_t scratchChunkBytes = 0;
        uint32_t uploadChunkCount = 0;
        uint32_t scratchChunkCount = 0;

        // RTXMU block suballocators, device-local. Zero unless the library was
        // built with NVRHI_WITH_RTXMU. These are the block reservations, so
        // they include the slack inside each block -- which summing the
        // per-AccelStruct logical sizes does not see.
        //
        // NOT thread-safe against concurrent acceleration-structure work.
        // rtxmu::Suballocator guards allocate/free with an internal mutex but
        // its getStats/getSize/getFragmentation walk m_blocks and each block's
        // freeSubBlocks WITHOUT taking it, so a query racing a BLAS build can
        // walk a vector mid-reallocation rather than merely read a stale total.
        // Query these from a thread that is not recording or executing RT work.
        uint64_t rtxmuResultBytes = 0;
        uint64_t rtxmuTransientResultBytes = 0;
        uint64_t rtxmuCompactionBytes = 0;
        // Unused bytes within the three pools above; already counted in them.
        uint64_t rtxmuUnusedBytes = 0;

        // Live totals of what vkAllocateMemory was ACTUALLY asked for, split by
        // heap kind, plus the live allocation count. NVRHI performs one
        // vkAllocateMemory per resource, so these are the sum of each resource's
        // VkMemoryRequirements::size -- which exceeds the logical size implied by
        // its TextureDesc/BufferDesc by alignment and tiling padding.
        //
        // Two things an app cannot otherwise measure:
        //  - deviceLocalAllocatedBytes minus the app's own descriptor-derived sum
        //    is the padding, which no per-subsystem byte getter can see.
        //  - VK_EXT_memory_budget's device usage minus deviceLocalAllocatedBytes
        //    is everything allocated OUTSIDE NVRHI: driver overhead, pipeline and
        //    shader objects, descriptor pools, and vendor SDKs (DLSS/NGX).
        //
        // These are live (decremented on free), unlike the chunk pools above,
        // which are high-water marks. Includes the chunk pools themselves, since
        // those also allocate through this path.
        uint64_t deviceLocalAllocatedBytes = 0;
        uint64_t hostVisibleAllocatedBytes = 0;
        uint32_t liveAllocationCount = 0;

        uint64_t chunkTotal() const { return uploadChunkBytes + scratchChunkBytes; }
        uint64_t rtxmuTotal() const { return rtxmuResultBytes + rtxmuTransientResultBytes + rtxmuCompactionBytes; }
        uint64_t allocatedTotal() const { return deviceLocalAllocatedBytes + hostVisibleAllocatedBytes; }
    };

    class IDevice : public nvrhi::IDevice
    {
    public:
        // Additional Vulkan-specific public methods
        virtual VkSemaphore getQueueSemaphore(CommandQueue queue) = 0;
        virtual void queueWaitForSemaphore(CommandQueue waitQueue, VkSemaphore semaphore, uint64_t value) = 0;
        virtual void queueSignalSemaphore(CommandQueue executionQueue, VkSemaphore semaphore, uint64_t value) = 0;
        virtual uint64_t queueGetCompletedInstance(CommandQueue queue) = 0;

        // Returns the concrete Vulkan queue-family index backing a logical
        // NVRHI queue, or UINT32_MAX when that queue is unavailable.
        virtual uint32_t getQueueFamilyIndex(CommandQueue queue) const = 0;

        // Returns true when the queue pair can use the explicit ownership
        // transfer API without falling back to Vulkan's legacy ALL_COMMANDS
        // ownership-transfer synchronization. Same-family pairs are always
        // supported; distinct families require enabled VK_KHR_maintenance8.
        virtual bool supportsEfficientQueueOwnershipTransfer(
            CommandQueue sourceQueue, CommandQueue destinationQueue) const = 0;

        // Vulkan requires external synchronization on a VkQueue across ALL
        // queue ops (vkQueueSubmit, vkQueuePresentKHR, vkQueueBindSparse).
        // NVRHI's submit and bindSparse take this mutex internally; the
        // application MUST hold this mutex when calling vkQueuePresentKHR
        // (or any raw VkQueue operation) on the same queue from another
        // thread. Without this, multi-threaded submit + present produces
        // VK validator THREADING_ERROR plus undefined behavior up to and
        // including vk::Queue::*: ErrorDeviceLost.
        virtual std::mutex& getQueueMutex(CommandQueue queue) = 0;

        // Creates and caches the exact VkImageView that a later
        // IDevice::writeDescriptorTable call would use for this Texture_SRV
        // item. Slot and array placement are ignored. This method never
        // creates or updates a descriptor set/table and is safe to call
        // before a descriptor-publication commit point. Returns false for an
        // invalid/non-Vulkan texture or when vkCreateImageView fails.
        virtual bool precreateTextureSrvView(const BindingSetItem& binding) = 0;

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
        // doc above. Submission errors retain the legacy fatal/exception behavior.
        virtual uint64_t executeCommandListsWithSyncIsolated(
            ICommandList* const* pCommandLists, size_t numCommandLists,
            CommandQueue executionQueue,
            const SubmitSyncExtras& extras) = 0;

        // Retry-aware form of executeCommandListsWithSyncIsolated. Returns 0
        // only when Vulkan guarantees that no command buffer, semaphore, or
        // fence operation was submitted (currently OUT_OF_HOST_MEMORY and
        // OUT_OF_DEVICE_MEMORY). In that case every command list remains closed
        // and executable and MUST be retried exactly; do not call open() on it.
        // Device loss is ambiguous and is conservatively treated as submitted.
        virtual uint64_t tryExecuteCommandListsWithSyncIsolated(
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

        // Retry-aware form of executeCommandListsWithSyncDraining. Returns 0
        // only when Vulkan guarantees that nothing was submitted. In that
        // case every command list remains closed and executable, and the
        // queue accumulator is restored exactly to its pre-call contents;
        // per-call extras and the internal tracking signal are not retained.
        // Retry the same closed command lists and extras without re-queuing
        // accumulator items. Device loss is conservatively treated as submitted.
        virtual uint64_t tryExecuteCommandListsWithSyncDraining(
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

        // Adds a mandatory same-state dependency to an open Vulkan command
        // list. These calls are intended for graph-managed resources
        // (keepInitialState == false and no permanent state). Older pending
        // transitions are emitted first so the same-state dependency cannot
        // accidentally share their pipelineBarrier2 batch. The dependency
        // itself remains pending: call ICommandList::commitBarriers before
        // invoking code that records the dependent commands.
        virtual bool addTextureMemoryDependency(
            ICommandList* commandList, ITexture* texture,
            TextureSubresourceSet subresources,
            const MemoryDependencyDesc& dependency) = 0;
        virtual bool addBufferMemoryDependency(
            ICommandList* commandList, IBuffer* buffer,
            const MemoryDependencyDesc& dependency) = 0;

        // Fail-closed graph tracker initialization. If every addressed
        // subresource is unknown, these methods initialize it to exactState.
        // If every addressed subresource is already tracked in the exact same
        // logical state and shader-stage scope, they are idempotent. Mixed
        // known/unknown ranges and state or stage mismatches fail.
        // Resources must be Vulkan-managed, graph-managed
        // (keepInitialState == false, no permanent state), and belong to this
        // device. commandList must be open and belong to this device.
        // Because NVRHI wraps an externally created VkDevice, these calls do
        // not validate which optional features were enabled at device creation.
        // Callers requiring fail-closed feature validation must preflight every
        // declared shader-stage bit and optional fixed-function resource state
        // before invoking them. Aggregate shader masks require every
        // constituent stage to be enabled.
        // A graph-owned resource range must not also be seeded or mutated by
        // legacy unqualified state APIs on the same command list. The tracker
        // stores ShaderType::All as a mask without provenance and therefore
        // cannot distinguish an exact full-stage graph union from a legacy
        // conservative None/All request; exclusive graph state ownership is a
        // caller invariant.
        virtual bool ensureTextureStateTracked(
            ICommandList* commandList, ITexture* texture,
            TextureSubresourceSet subresources,
            const GraphResourceState& exactState) = 0;
        virtual bool ensureBufferStateTracked(
            ICommandList* commandList, IBuffer* buffer,
            const GraphResourceState& exactState) = 0;

        // Fail-closed, verify-only graph tracker queries. These apply the
        // same ownership, command-list, resource, range, stage-scope, and
        // lifetime eligibility rules as the ensure methods, but every
        // addressed state must already be known and must match exactState's
        // logical state and shader-stage scope exactly. They never seed or
        // otherwise mutate tracked state. Successful verification retains the
        // resource for the command buffer's lifetime.
        virtual bool verifyTextureStateTracked(
            ICommandList* commandList, ITexture* texture,
            TextureSubresourceSet subresources,
            const GraphResourceState& exactState) = 0;
        virtual bool verifyBufferStateTracked(
            ICommandList* commandList, IBuffer* buffer,
            const GraphResourceState& exactState) = 0;

        // Fail-closed graph-authored logical transitions. Every addressed
        // subresource must already be tracked in transition.stateBefore and
        // transition.shaderStagesBefore exactly.
        // Successful transitions remain pending until the caller invokes
        // ICommandList::commitBarriers, allowing independent graph transitions
        // to batch. If an earlier pending transition overlaps the same resource
        // range, it is emitted first to preserve sequential ordering; the newly
        // requested transition remains pending. The same command-list/resource
        // ownership and graph-managed restrictions as the ensure methods apply.
        virtual bool transitionTextureState(
            ICommandList* commandList, ITexture* texture,
            TextureSubresourceSet subresources,
            const GraphResourceStateTransition& transition) = 0;
        virtual bool transitionBufferState(
            ICommandList* commandList, IBuffer* buffer,
            const GraphResourceStateTransition& transition) = 0;

        // Records one half of a queue-family ownership transfer. These calls
        // are valid only for exclusive, NVRHI-managed, graph-tracked resources
        // (keepInitialState == false and no permanent state). A release is the
        // final local use on its command list; an acquire must precede the first
        // local use on its command list. Same-family handoffs collapse to one
        // ordinary transition on release and tracker initialization on acquire.
        // A use recorded after release is invalid caller behavior. The backend
        // diagnoses it when resource-state tracking observes it, but does not
        // suppress commands recorded through paths with automatic barriers
        // disabled.
        // Distinct-family handoffs require enabled VK_KHR_maintenance8 and use
        // its stage-aware ownership-transfer dependency flag; query
        // supportsEfficientQueueOwnershipTransfer before scheduling them.
        virtual bool releaseTextureQueueOwnership(
            ICommandList* commandList, ITexture* texture,
            TextureSubresourceSet subresources,
            const QueueOwnershipTransferDesc& transfer) = 0;
        virtual bool acquireTextureQueueOwnership(
            ICommandList* commandList, ITexture* texture,
            TextureSubresourceSet subresources,
            const QueueOwnershipTransferDesc& transfer) = 0;
        virtual bool releaseBufferQueueOwnership(
            ICommandList* commandList, IBuffer* buffer,
            const QueueOwnershipTransferDesc& transfer) = 0;
        virtual bool acquireBufferQueueOwnership(
            ICommandList* commandList, IBuffer* buffer,
            const QueueOwnershipTransferDesc& transfer) = 0;

        // Returns the resolved raw timestamp endpoints without waiting,
        // resetting, or consuming the timer query. Returns false until
        // pollTimerQuery has resolved the query.
        virtual bool getTimerQueryTimestampRange(
            ITimerQuery* query, TimerQueryTimestampRange& range) = 0;

        // Bytes held in NVRHI-internal pools that no other query reports. The
        // chunk counters are lock-free atomics; the RTXMU pools walk their
        // block lists, so treat this as a diagnostic call rather than a
        // per-frame one.
        virtual InternalMemoryStats getInternalMemoryStats() = 0;
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
        // Indicates that VK_KHR_maintenance8 is enabled and
        // VkPhysicalDeviceMaintenance8FeaturesKHR::maintenance8 was set to true
        // when the externally-owned VkDevice was created. The extension name
        // must also be present in deviceExtensions.
        bool maintenance8Supported = false;
        // Creates acceleration-structure storage buffers with
        // VK_SHARING_MODE_CONCURRENT across every distinct queue family supplied
        // above. This is an immutable device-creation policy: it covers native
        // TLAS/non-RTXMU BLAS backing and RTXMU result/compaction pools, while
        // transient scratch/update/query resources remain queue-exclusive.
        bool accelStructStorageSharedAcrossQueues = false;
        bool aftermathEnabled = false;
        bool logBufferLifetime = false;

        std::string vulkanLibraryName; // if empty, use default
    };

    NVRHI_API DeviceHandle createDevice(const DeviceDesc& desc);

    NVRHI_API VkFormat convertFormat(nvrhi::Format format);

    NVRHI_API const char* resultToString(VkResult result);

    // Thread-local barrier emission counters (f111-pig): monotonically
    // increasing totals of what THIS thread's command lists emitted via
    // vkCmdPipelineBarrier2. Thread-local so a render thread sampling deltas
    // around its recording regions attributes its own batches exactly, without
    // contamination from worker-thread uploads. `full_drain_barriers` counts
    // barriers whose source or destination stage scope is ALL_COMMANDS — the
    // full pipeline drains that empty the machine.
    struct BarrierStatsSnapshot
    {
        uint64_t commits = 0;            // pipelineBarrier2 calls
        uint64_t image_barriers = 0;
        uint64_t buffer_barriers = 0;
        uint64_t full_drain_barriers = 0;
    };
    NVRHI_API BarrierStatsSnapshot getThreadBarrierStats();

    // Arms the per-commit barrier dump (PIG_BARRIER_STATS=2 logging) for the
    // NEXT `commit_count` pipelineBarrier2 emissions on the CALLING thread.
    // The app arms this at a known steady-state frame; count-based windows
    // cannot find steady state because loading burns an unknown number of
    // commits before the profiler's region labels start printing.
    NVRHI_API void armThreadBarrierDump(uint64_t commit_count);

    // f111-pig: the shader-state stage substitution. ConstantBuffer /
    // ShaderResource / UnorderedAccess carry shader-only access masks; the
    // conservative map nonetheless scoped them at ALL_COMMANDS, so every
    // unqualified barrier on those states ordered against transfers,
    // indirect fetches, attachment output and presents — pure
    // over-synchronization. The application calls this ONCE after device
    // creation with the union of the pipeline stages its ENABLED features
    // can actually shade from (stage bits whose feature is off violate the
    // synchronization2 VUs). Until called (or if called with 0) the legacy
    // ALL_COMMANDS behavior is preserved bit-for-bit.
    NVRHI_API void setShaderStateStageUnion(VkPipelineStageFlags2 stage_union);
    NVRHI_API VkPipelineStageFlags2 getShaderStateStageUnion();
}
