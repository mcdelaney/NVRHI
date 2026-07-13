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

#include "vulkan-backend.h"
#include "vulkan-queue-utils.h"
#include "nvrhi/common/misc.h"

namespace nvrhi::vulkan
{
    extern vk::ImageAspectFlags guessImageAspectFlags(vk::Format format);

    TrackedCommandBuffer::~TrackedCommandBuffer()
    {
        m_Context.device.destroyCommandPool(cmdPool, m_Context.allocationCallbacks);
    }

    Queue::Queue(const VulkanContext& context, CommandQueue queueID, vk::Queue queue, uint32_t queueFamilyIndex)
        : m_Context(context)
        , m_Queue(queue)
        , m_QueueID(queueID)
        , m_QueueFamilyIndex(queueFamilyIndex)
    {
        const std::vector<vk::QueueFamilyProperties> queueFamilies =
            context.physicalDevice.getQueueFamilyProperties();
        if (queueFamilyIndex >= queueFamilies.size())
        {
            context.error("The Vulkan queue family index is out of range");
            assert(false && "Vulkan queue family index is out of range");
        }
        else
        {
            m_QueueFlags = queueFamilies[queueFamilyIndex].queueFlags;
            m_TimestampValidBits = queueFamilies[queueFamilyIndex].timestampValidBits;
        }

        auto semaphoreTypeInfo = vk::SemaphoreTypeCreateInfo()
            .setSemaphoreType(vk::SemaphoreType::eTimeline);

        auto semaphoreInfo = vk::SemaphoreCreateInfo()
            .setPNext(&semaphoreTypeInfo);

        trackingSemaphore = context.device.createSemaphore(semaphoreInfo, context.allocationCallbacks);
    }

    Queue::~Queue()
    {
        m_Context.device.destroySemaphore(trackingSemaphore, m_Context.allocationCallbacks);
        trackingSemaphore = vk::Semaphore();
    }

    TrackedCommandBufferPtr Queue::createCommandBuffer()
    {
        vk::Result res;

        TrackedCommandBufferPtr ret = std::make_shared<TrackedCommandBuffer>(m_Context);

        auto cmdPoolInfo = vk::CommandPoolCreateInfo()
                            .setQueueFamilyIndex(m_QueueFamilyIndex)
                            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
                                        vk::CommandPoolCreateFlagBits::eTransient);

        res = m_Context.device.createCommandPool(&cmdPoolInfo, m_Context.allocationCallbacks, &ret->cmdPool);
        CHECK_VK_FAIL(res)
        
        // allocate command buffer
        auto allocInfo = vk::CommandBufferAllocateInfo()
                            .setLevel(vk::CommandBufferLevel::ePrimary)
                            .setCommandPool(ret->cmdPool)
                            .setCommandBufferCount(1);

        res = m_Context.device.allocateCommandBuffers(&allocInfo, &ret->cmdBuf);
        CHECK_VK_FAIL(res)

        return ret;
    }

    TrackedCommandBufferPtr Queue::getOrCreateCommandBuffer()
    {
        std::lock_guard lockGuard(m_Mutex); // this is called from CommandList::open, so free-threaded

        uint64_t recordingID = m_LastRecordingID.fetch_add(1, std::memory_order_relaxed) + 1;

        TrackedCommandBufferPtr cmdBuf;
        if (m_CommandBuffersPool.empty())
        {
            cmdBuf = createCommandBuffer();
        }
        else
        {
            cmdBuf = m_CommandBuffersPool.front();
            m_CommandBuffersPool.pop_front();
        }

        cmdBuf->recordingID = recordingID;
        return cmdBuf;
    }

    vk::PipelineStageFlags2 Queue::normalizeWaitStageMask(vk::PipelineStageFlags2 stageMask) const
    {
        if (detail::isWaitStageMaskSupported(stageMask, m_QueueFlags))
            return stageMask;

        m_Context.error("A Vulkan semaphore wait stage mask is zero or unsupported by the destination queue family");
        assert(false && "Invalid Vulkan semaphore wait stage mask");
        return vk::PipelineStageFlagBits2::eAllCommands;
    }

    void Queue::addWaitSemaphore(vk::Semaphore semaphore, uint64_t value, vk::PipelineStageFlags2 stageMask)
    {
        if (!semaphore)
            return;

        stageMask = normalizeWaitStageMask(stageMask);

        // Lock so concurrent submitters from different threads don't race
        // on the wait/signal accumulator vectors. The mutex is the same
        // one submit() holds for its entire body — addWaitSemaphore and
        // the subsequent submit are logically a single transaction for
        // any given submitter.
        std::lock_guard lockGuard(m_Mutex);
        m_WaitSemaphores.push_back(PendingSemaphoreWait{ semaphore, value, stageMask });
    }

    void Queue::addSignalSemaphore(vk::Semaphore semaphore, uint64_t value)
    {
        if (!semaphore)
            return;

        std::lock_guard lockGuard(m_Mutex);
        m_SignalSemaphores.push_back(semaphore);
        m_SignalSemaphoreValues.push_back(value);
    }

    uint64_t Queue::submit(ICommandList* const* ppCmd, size_t numCmd)
    {
        return submitImpl(ppCmd, numCmd, /*extras=*/nullptr, /*drainAccumulator=*/true);
    }

    uint64_t Queue::submitWithSyncIsolated(ICommandList* const* ppCmd, size_t numCmd, const SubmitSyncExtras& extras)
    {
        return submitImpl(ppCmd, numCmd, &extras, /*drainAccumulator=*/false);
    }

    uint64_t Queue::submitWithSyncDraining(ICommandList* const* ppCmd, size_t numCmd, const SubmitSyncExtras& extras)
    {
        return submitImpl(ppCmd, numCmd, &extras, /*drainAccumulator=*/true);
    }

    uint64_t Queue::submitImpl(ICommandList* const* ppCmd, size_t numCmd, const SubmitSyncExtras* extras, bool drainAccumulator)
    {
        // Hold the queue mutex for the entire body. This serves three
        // purposes:
        //   1. Serializes vk::Queue.submit calls — Vulkan requires external
        //      synchronization on a VkQueue; concurrent submits from
        //      different threads otherwise produce validator THREADING
        //      ERROR plus undefined behavior.
        //   2. Protects the wait/signal semaphore accumulator vectors and
        //      m_LastSubmittedID against concurrent addWait/addSignal/submit.
        //   3. Guarantees tracking-semaphore signal-value monotonicity:
        //      without the lock, two submitters could both observe the
        //      same m_LastSubmittedID and signal the trackingSemaphore at
        //      duplicate values (validator VUID-VkSubmitInfo-pSignalSemaphores-03242).
        std::lock_guard lockGuard(m_Mutex);

        // Decide which wait/signal sources to use:
        //   drainAccumulator = true (default submit): use the queue's
        //     accumulator (queueWaitForSemaphore / queueSignalSemaphore /
        //     queueWaitForCommandList queued before this submit). This is
        //     the original behavior — every submit on this queue consumes
        //     anything that's been queued for "the next submit".
        //   drainAccumulator = false (submitWithSyncIsolated): bypass the
        //     accumulator entirely. Use ONLY the per-submit `extras`. This
        //     is what worker threads should use when they don't want to
        //     consume waits/signals queued by other code for a different
        //     intended submit (e.g., the swapchain-tail's acquire wait +
        //     present_sem signal, or terrain-front-use signals queued by
        //     the render thread for its own tail submit). Solves the
        //     "accumulator stealing" race.
        std::vector<PendingSemaphoreWait> localWaitSemaphores;
        std::vector<vk::Semaphore> localSignalSemaphores;
        std::vector<uint64_t> localSignalSemaphoreValues;

        std::vector<PendingSemaphoreWait>* waitSemaphores;
        std::vector<vk::Semaphore>* signalSemaphores;
        std::vector<uint64_t>* signalSemaphoreValues;

        if (drainAccumulator)
        {
            waitSemaphores = &m_WaitSemaphores;
            signalSemaphores = &m_SignalSemaphores;
            signalSemaphoreValues = &m_SignalSemaphoreValues;
        }
        else
        {
            waitSemaphores = &localWaitSemaphores;
            signalSemaphores = &localSignalSemaphores;
            signalSemaphoreValues = &localSignalSemaphoreValues;
        }

        if (extras)
        {
            assert(extras->numWaits == 0 || extras->waitSemaphores != nullptr);
            assert(extras->numSignals == 0 || extras->signalSemaphores != nullptr);
            for (uint32_t i = 0; i < extras->numWaits; ++i)
            {
                const vk::PipelineStageFlags2 stageMask = normalizeWaitStageMask(
                    extras->waitStageMasks
                        ? vk::PipelineStageFlags2(extras->waitStageMasks[i])
                        : vk::PipelineStageFlagBits2::eAllCommands);
                waitSemaphores->push_back(PendingSemaphoreWait{
                    vk::Semaphore(extras->waitSemaphores[i]),
                    extras->waitValues ? extras->waitValues[i] : 0ull,
                    stageMask });
            }
            for (uint32_t i = 0; i < extras->numSignals; ++i)
            {
                signalSemaphores->push_back(vk::Semaphore(extras->signalSemaphores[i]));
                signalSemaphoreValues->push_back(extras->signalValues
                    ? extras->signalValues[i] : 0ull);
            }
        }

        assert(signalSemaphores->size() == signalSemaphoreValues->size());

        std::vector<vk::CommandBufferSubmitInfo> commandBufferInfos(numCmd);

        // Reserve the next timeline value while holding m_Mutex, but publish it
        // to observers only after vkQueueSubmit2 succeeds. A failed submit must
        // return 0 and must not make an unsignaled value look submitted.
        const uint64_t submissionID =
            m_LastSubmittedID.load(std::memory_order_relaxed) + 1;

        for (size_t i = 0; i < numCmd; i++)
        {
            CommandList* commandList = checked_cast<CommandList*>(ppCmd[i]);
            TrackedCommandBufferPtr commandBuffer = commandList->getCurrentCmdBuf();

            commandBufferInfos[i] = vk::CommandBufferSubmitInfo()
                .setCommandBuffer(commandBuffer->cmdBuf)
                .setDeviceMask(0);
        }

        signalSemaphores->push_back(trackingSemaphore);
        signalSemaphoreValues->push_back(submissionID);

        std::vector<vk::SemaphoreSubmitInfo> waitInfos(waitSemaphores->size());
        for (size_t i = 0; i < waitInfos.size(); ++i)
        {
            waitInfos[i] = vk::SemaphoreSubmitInfo()
                .setSemaphore((*waitSemaphores)[i].semaphore)
                .setValue((*waitSemaphores)[i].value)
                .setStageMask((*waitSemaphores)[i].stageMask)
                .setDeviceIndex(0);
        }

        std::vector<vk::SemaphoreSubmitInfo> signalInfos(signalSemaphores->size());
        for (size_t i = 0; i < signalInfos.size(); ++i)
        {
            signalInfos[i] = vk::SemaphoreSubmitInfo()
                .setSemaphore((*signalSemaphores)[i])
                .setValue((*signalSemaphoreValues)[i])
                .setStageMask(vk::PipelineStageFlagBits2::eAllCommands)
                .setDeviceIndex(0);
        }

        auto submitInfo = vk::SubmitInfo2()
            .setWaitSemaphoreInfoCount(uint32_t(waitInfos.size()))
            .setPWaitSemaphoreInfos(waitInfos.empty() ? nullptr : waitInfos.data())
            .setCommandBufferInfoCount(uint32_t(commandBufferInfos.size()))
            .setPCommandBufferInfos(commandBufferInfos.empty() ? nullptr : commandBufferInfos.data())
            .setSignalSemaphoreInfoCount(uint32_t(signalInfos.size()))
            .setPSignalSemaphoreInfos(signalInfos.empty() ? nullptr : signalInfos.data());

        bool submitSucceeded = true;
        try {
            m_Queue.submit2(submitInfo);
        }
        catch (vk::DeviceLostError&)
        {
            m_Context.messageCallback->message(MessageSeverity::Error, "Device Removed!");
            // Preserve the legacy draining-submit behavior for existing callers,
            // which treat device loss as fatal and do not implement rollback.
            // Isolated worker submits have an explicit zero-on-failure contract.
            submitSucceeded = drainAccumulator;
        }
        catch (const vk::SystemError& error)
        {
            if (drainAccumulator)
                throw;

            const std::string message =
                std::string("Vulkan queue submission failed: ") + error.what();
            m_Context.messageCallback->message(
                MessageSeverity::Error, message.c_str());
            submitSucceeded = false;
        }

        if (drainAccumulator)
        {
            m_WaitSemaphores.clear();
            m_SignalSemaphores.clear();
            m_SignalSemaphoreValues.clear();
        }

        if (!submitSucceeded)
            return 0;

        m_LastSubmittedID.store(submissionID, std::memory_order_relaxed);
        for (size_t i = 0; i < numCmd; i++)
        {
            CommandList* commandList = checked_cast<CommandList*>(ppCmd[i]);
            TrackedCommandBufferPtr commandBuffer = commandList->getCurrentCmdBuf();
            commandBuffer->submissionID = submissionID;

            for (const auto& buffer : commandBuffer->referencedStagingBuffers)
            {
                buffer->lastUseQueue = m_QueueID;
                buffer->lastUseCommandListID = submissionID;
            }
        }

        return submissionID;
    }

    void Queue::returnCommandBufferToPool(TrackedCommandBufferPtr cb)
    {
        if (!cb)
            return;
        std::lock_guard lockGuard(m_Mutex);
        m_CommandBuffersPool.push_back(std::move(cb));
    }

    void Queue::updateTextureTileMappings(
        ITexture* _texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings,
        VkSemaphore signalSemaphore, uint64_t signalValue)
    {
        Texture* texture = checked_cast<Texture*>(_texture);

        std::vector<vk::SparseImageMemoryBind> sparseImageMemoryBinds;
        std::vector<vk::SparseMemoryBind> sparseMemoryBinds;

        vk::ImageCreateInfo& imageInfo = texture->imageInfo;
		vk::ImageAspectFlags textureAspectFlags = guessImageAspectFlags(imageInfo.format);

		// Required for extent and offset since they must be multiples of the tile dimensions
		uint32_t tileWidth = 1;
		uint32_t tileHeight = 1;
		uint32_t tileDepth = 1;

        // Mip tail info, required for resource offset
        vk::DeviceSize imageMipTailOffset = 0;
        vk::DeviceSize imageMipTailStride = 0;

        std::vector<vk::SparseImageFormatProperties> formatProperties = m_Context.physicalDevice.getSparseImageFormatProperties(imageInfo.format, imageInfo.imageType, imageInfo.samples, imageInfo.usage, imageInfo.tiling);
		std::vector<vk::SparseImageMemoryRequirements> memoryRequirements = m_Context.device.getImageSparseMemoryRequirements(texture->image);

		if (!formatProperties.empty())
		{
			tileWidth = formatProperties[0].imageGranularity.width;
			tileHeight = formatProperties[0].imageGranularity.height;
			tileDepth = formatProperties[0].imageGranularity.depth;
		}

        if (!memoryRequirements.empty())
        {
			imageMipTailOffset = memoryRequirements[0].imageMipTailOffset;
			imageMipTailStride = memoryRequirements[0].imageMipTailStride;
        }

        for (size_t i = 0; i < numTileMappings; i++)
        {
            uint32_t numRegions = tileMappings[i].numTextureRegions;
            Heap* heap = tileMappings[i].heap ? checked_cast<Heap*>(tileMappings[i].heap) : nullptr;
            vk::DeviceMemory deviceMemory = heap ? heap->memory : VK_NULL_HANDLE;

            for (uint32_t j = 0; j < numRegions; ++j)
            {
                const TiledTextureCoordinate& tiledTextureCoordinate = tileMappings[i].tiledTextureCoordinates[j];
                const TiledTextureRegion& tiledTextureRegion = tileMappings[i].tiledTextureRegions[j];

                if (tiledTextureRegion.tilesNum)
                {
                    sparseMemoryBinds.push_back(vk::SparseMemoryBind()
                        .setResourceOffset(imageMipTailOffset + tiledTextureCoordinate.arrayLevel * imageMipTailStride)
                        .setSize(tiledTextureRegion.tilesNum * texture->tileByteSize)
                        .setMemory(deviceMemory)
                        .setMemoryOffset(deviceMemory ? tileMappings[i].byteOffsets[j] : 0));
                }
                else
                {
                    vk::ImageSubresource subresource = {};
                    subresource.arrayLayer = tiledTextureCoordinate.arrayLevel;
                    subresource.mipLevel = tiledTextureCoordinate.mipLevel;
					subresource.aspectMask = textureAspectFlags; // Required for sparse binding

                    vk::Offset3D offset3D;
                    offset3D.x = tiledTextureCoordinate.x * tileWidth;
                    offset3D.y = tiledTextureCoordinate.y * tileHeight;
                    offset3D.z = tiledTextureCoordinate.z * tileHeight;

                    vk::Extent3D extent3D;
                    extent3D.width = tiledTextureRegion.width * tileWidth;
                    extent3D.height = tiledTextureRegion.height * tileHeight;
                    extent3D.depth = tiledTextureRegion.depth * tileDepth;

                    sparseImageMemoryBinds.push_back(vk::SparseImageMemoryBind()
                        .setSubresource(subresource)
                        .setOffset(offset3D)
                        .setExtent(extent3D)
                        .setMemory(deviceMemory)
                        .setMemoryOffset(deviceMemory ? tileMappings[i].byteOffsets[j] : 0));
                }
            }
        }

        vk::BindSparseInfo bindSparseInfo = {};

        vk::SparseImageMemoryBindInfo sparseImageMemoryBindInfo;
        if (!sparseImageMemoryBinds.empty())
        {
            sparseImageMemoryBindInfo.setImage(texture->image);
            sparseImageMemoryBindInfo.setBinds(sparseImageMemoryBinds);
            bindSparseInfo.setImageBinds(sparseImageMemoryBindInfo);
        }

        vk::SparseImageOpaqueMemoryBindInfo sparseImageOpaqueMemoryBindInfo;
        if (!sparseMemoryBinds.empty())
        {
            sparseImageOpaqueMemoryBindInfo.setImage(texture->image);
            sparseImageOpaqueMemoryBindInfo.setBinds(sparseMemoryBinds);
            bindSparseInfo.setImageOpaqueBinds(sparseImageOpaqueMemoryBindInfo);
        }

        // Optional timeline semaphore signal so the caller can wait on the
        // bind from a subsequent submit instead of stalling the host or
        // doing a device-wide waitForIdle.
        vk::Semaphore signalSems[1];
        uint64_t signalValues[1];
        vk::TimelineSemaphoreSubmitInfo timelineInfo;
        if (signalSemaphore != VK_NULL_HANDLE)
        {
            signalSems[0] = signalSemaphore;
            signalValues[0] = signalValue;
            bindSparseInfo.setSignalSemaphores(signalSems);
            timelineInfo.setSignalSemaphoreValues(signalValues);
            bindSparseInfo.setPNext(&timelineInfo);
        }

        // VkQueue external-sync: bindSparse on the same VkQueue as submit
        // (or presentKHR) needs serialization. Same mutex as Queue::submit.
        std::lock_guard lockGuard(m_Mutex);
        m_Queue.bindSparse(bindSparseInfo, vk::Fence());
    }

    uint64_t Queue::updateLastFinishedID()
    {
        const uint64_t finished = m_Context.device.getSemaphoreCounterValue(trackingSemaphore);
        m_LastFinishedID.store(finished, std::memory_order_release);
        return finished;
    }

    VkSemaphore Device::getQueueSemaphore(CommandQueue queueID)
    {
        Queue& queue = *m_Queues[uint32_t(queueID)];

        return queue.trackingSemaphore;
    }

    std::mutex& Device::getQueueMutex(CommandQueue queueID)
    {
        Queue& queue = *m_Queues[uint32_t(queueID)];
        return queue.getMutex();
    }

    void Device::queueWaitForSemaphore(CommandQueue waitQueueID, VkSemaphore semaphore, uint64_t value)
    {
        queueWaitForSemaphoreAtStage(
            waitQueueID, semaphore, value, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    }

    void Device::queueWaitForSemaphoreAtStage(
        CommandQueue waitQueueID, VkSemaphore semaphore, uint64_t value,
        VkPipelineStageFlags2 waitStageMask)
    {
        Queue& waitQueue = *m_Queues[uint32_t(waitQueueID)];

        waitQueue.addWaitSemaphore(semaphore, value, vk::PipelineStageFlags2(waitStageMask));
    }

    void Device::queueSignalSemaphore(CommandQueue executionQueueID, VkSemaphore semaphore, uint64_t value)
    {
        Queue& executionQueue = *m_Queues[uint32_t(executionQueueID)];

        executionQueue.addSignalSemaphore(semaphore, value);
    }

    void Device::queueWaitForCommandList(CommandQueue waitQueueID, CommandQueue executionQueueID, uint64_t instance)
    {
        queueWaitForCommandListAtStage(
            waitQueueID, executionQueueID, instance, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    }

    void Device::queueWaitForCommandListAtStage(
        CommandQueue waitQueueID, CommandQueue executionQueueID, uint64_t instance,
        VkPipelineStageFlags2 waitStageMask)
    {
        queueWaitForSemaphoreAtStage(
            waitQueueID, getQueueSemaphore(executionQueueID), instance, waitStageMask);
    }

    void Device::updateTextureTileMappings(ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings, CommandQueue executionQueue)
    {
        Queue& queue = *m_Queues[uint32_t(executionQueue)];

        queue.updateTextureTileMappings(texture, tileMappings, numTileMappings);
    }

    void Device::updateTextureTileMappingsSignal(
        ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings,
        CommandQueue executionQueue,
        VkSemaphore signalSemaphore, uint64_t signalValue)
    {
        Queue& queue = *m_Queues[uint32_t(executionQueue)];

        queue.updateTextureTileMappings(texture, tileMappings, numTileMappings, signalSemaphore, signalValue);
    }

    uint64_t Device::queueGetCompletedInstance(CommandQueue queue)
    {
        return m_Context.device.getSemaphoreCounterValue(getQueueSemaphore(queue));
    }

    bool Queue::pollCommandList(uint64_t commandListID)
    {
        if (commandListID > m_LastSubmittedID || commandListID == 0)
            return false;
        
        bool completed = getLastFinishedID() >= commandListID;
        if (completed)
            return true;

        completed = updateLastFinishedID() >= commandListID;
        return completed;
    }

    bool Queue::waitCommandList(uint64_t commandListID, uint64_t timeout)
    {
        if (commandListID > m_LastSubmittedID || commandListID == 0)
            return false;

        if (pollCommandList(commandListID))
            return true;

        std::array<const vk::Semaphore, 1> semaphores = { trackingSemaphore };
        std::array<uint64_t, 1> waitValues = { commandListID };

        auto waitInfo = vk::SemaphoreWaitInfo()
            .setSemaphores(semaphores)
            .setValues(waitValues);

        vk::Result result = m_Context.device.waitSemaphores(waitInfo, timeout);

        return (result == vk::Result::eSuccess);
    }
} // namespace nvrhi::vulkan
