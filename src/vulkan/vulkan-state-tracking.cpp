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
#include <nvrhi/common/misc.h>

namespace nvrhi::vulkan
{
    namespace detail
    {
        vk::ImageMemoryBarrier2 buildQueueOwnershipImageBarrier(
            vk::Image image,
            const vk::ImageSubresourceRange& subresources,
            const ResourceStateMapping& before,
            const ResourceStateMapping& after,
            uint32_t sourceQueueFamily,
            uint32_t destinationQueueFamily,
            bool release,
            bool sameFamily)
        {
            const vk::PipelineStageFlags2 sourceStages =
                sameFamily ? before.stageFlags
                    : (release ? before.stageFlags : after.stageFlags);
            const vk::AccessFlags2 sourceAccess =
                (sameFamily || release) ? before.accessMask : vk::AccessFlags2{};
            const vk::PipelineStageFlags2 destinationStages =
                sameFamily ? after.stageFlags
                    : (release ? before.stageFlags : after.stageFlags);
            const vk::AccessFlags2 destinationAccess =
                sameFamily ? after.accessMask
                    : (release ? vk::AccessFlags2{} : after.accessMask);

            return vk::ImageMemoryBarrier2()
                .setSrcStageMask(sourceStages)
                .setSrcAccessMask(sourceAccess)
                .setDstStageMask(destinationStages)
                .setDstAccessMask(destinationAccess)
                .setOldLayout(before.imageLayout)
                .setNewLayout(after.imageLayout)
                .setSrcQueueFamilyIndex(sameFamily
                    ? VK_QUEUE_FAMILY_IGNORED : sourceQueueFamily)
                .setDstQueueFamilyIndex(sameFamily
                    ? VK_QUEUE_FAMILY_IGNORED : destinationQueueFamily)
                .setImage(image)
                .setSubresourceRange(subresources);
        }

        vk::BufferMemoryBarrier2 buildQueueOwnershipBufferBarrier(
            vk::Buffer buffer,
            vk::DeviceSize size,
            const ResourceStateMapping& before,
            const ResourceStateMapping& after,
            uint32_t sourceQueueFamily,
            uint32_t destinationQueueFamily,
            bool release,
            bool sameFamily)
        {
            const vk::PipelineStageFlags2 sourceStages =
                sameFamily ? before.stageFlags
                    : (release ? before.stageFlags : after.stageFlags);
            const vk::AccessFlags2 sourceAccess =
                (sameFamily || release) ? before.accessMask : vk::AccessFlags2{};
            const vk::PipelineStageFlags2 destinationStages =
                sameFamily ? after.stageFlags
                    : (release ? before.stageFlags : after.stageFlags);
            const vk::AccessFlags2 destinationAccess =
                sameFamily ? after.accessMask
                    : (release ? vk::AccessFlags2{} : after.accessMask);

            return vk::BufferMemoryBarrier2()
                .setSrcStageMask(sourceStages)
                .setSrcAccessMask(sourceAccess)
                .setDstStageMask(destinationStages)
                .setDstAccessMask(destinationAccess)
                .setSrcQueueFamilyIndex(sameFamily
                    ? VK_QUEUE_FAMILY_IGNORED : sourceQueueFamily)
                .setDstQueueFamilyIndex(sameFamily
                    ? VK_QUEUE_FAMILY_IGNORED : destinationQueueFamily)
                .setBuffer(buffer)
                .setOffset(0)
                .setSize(size);
        }

        vk::DependencyFlags queueOwnershipDependencyFlags(
            bool sameFamily, bool maintenance8Enabled)
        {
            return !sameFamily && maintenance8Enabled
                ? vk::DependencyFlags(
                    vk::DependencyFlagBits::eQueueFamilyOwnershipTransferUseAllStagesKHR)
                : vk::DependencyFlags{};
        }
    }

    static vk::ImageSubresourceRange makeImageSubresourceRange(
        const Texture* texture,
        const TextureSubresourceSet& subresources)
    {
        const FormatInfo& formatInfo = getFormatInfo(texture->desc.format);
        vk::ImageAspectFlags aspectMask{};
        if (formatInfo.hasDepth) aspectMask |= vk::ImageAspectFlagBits::eDepth;
        if (formatInfo.hasStencil) aspectMask |= vk::ImageAspectFlagBits::eStencil;
        if (!aspectMask) aspectMask = vk::ImageAspectFlagBits::eColor;

        return vk::ImageSubresourceRange()
            .setBaseArrayLayer(subresources.baseArraySlice)
            .setLayerCount(subresources.numArraySlices)
            .setBaseMipLevel(subresources.baseMipLevel)
            .setLevelCount(subresources.numMipLevels)
            .setAspectMask(aspectMask);
    }

    static bool textureRangesOverlap(
        const TextureSubresourceSet& left,
        const TextureSubresourceSet& right)
    {
        const bool mipOverlap = left.baseMipLevel < right.baseMipLevel + right.numMipLevels
            && right.baseMipLevel < left.baseMipLevel + left.numMipLevels;
        const bool sliceOverlap = left.baseArraySlice < right.baseArraySlice + right.numArraySlices
            && right.baseArraySlice < left.baseArraySlice + left.numArraySlices;
        return mipOverlap && sliceOverlap;
    }

    static bool isValidResolvedTextureRange(
        const Texture* texture,
        const TextureSubresourceSet& subresources)
    {
        return subresources.numMipLevels > 0
            && subresources.baseMipLevel < texture->desc.mipLevels
            && subresources.numMipLevels
                <= texture->desc.mipLevels - subresources.baseMipLevel
            && subresources.numArraySlices > 0
            && subresources.baseArraySlice < texture->desc.arraySize
            && subresources.numArraySlices
                <= texture->desc.arraySize - subresources.baseArraySlice;
    }

    static bool validateQueueOwnershipTransfer(
        CommandList* commandList,
        Device* device,
        const VulkanContext& context,
        const QueueOwnershipTransferDesc& transfer,
        bool release,
        uint32_t& sourceQueueFamily,
        uint32_t& destinationQueueFamily)
    {
        if (!commandList || commandList->getDevice() != device)
        {
            context.error("Queue ownership transfer command list belongs to a different device");
            return false;
        }
        if (!commandList->getCurrentCmdBuf())
        {
            context.error("Queue ownership transfers must be recorded on an open command list");
            return false;
        }
        if (transfer.sourceQueue >= CommandQueue::Count
            || transfer.destinationQueue >= CommandQueue::Count)
        {
            context.error("Queue ownership transfer contains an invalid logical queue");
            return false;
        }
        if (transfer.sourceQueue == transfer.destinationQueue)
        {
            context.error("Queue ownership transfer source and destination queues must differ");
            return false;
        }
        if (transfer.stateBefore == ResourceStates::Unknown
            || transfer.stateAfter == ResourceStates::Unknown)
        {
            context.error("Queue ownership transfer states must be known");
            return false;
        }

        Queue* sourceQueue = device->getQueue(transfer.sourceQueue);
        Queue* destinationQueue = device->getQueue(transfer.destinationQueue);
        if (!sourceQueue || !destinationQueue)
        {
            context.error("Queue ownership transfer references an unavailable queue");
            return false;
        }

        const CommandQueue expectedQueue = release
            ? transfer.sourceQueue : transfer.destinationQueue;
        if (commandList->getDesc().queueType != expectedQueue)
        {
            context.error(release
                ? "Queue ownership release must be recorded on the source queue command list"
                : "Queue ownership acquire must be recorded on the destination queue command list");
            return false;
        }

        sourceQueueFamily = sourceQueue->getQueueFamilyIndex();
        destinationQueueFamily = destinationQueue->getQueueFamilyIndex();
        if (sourceQueueFamily != destinationQueueFamily
            && !context.extensions.KHR_maintenance8)
        {
            context.error(
                "Distinct-family queue ownership transfers require enabled "
                "VK_KHR_maintenance8; the legacy path synchronizes ownership "
                "operations at ALL_COMMANDS and is intentionally disabled");
            return false;
        }
        return true;
    }
    
    void CommandList::setResourceStatesForBindingSet(IBindingSet* _bindingSet)
    {
        setResourceStatesForBindingSetInternal(
            _bindingSet, false, ShaderType::All);
    }

    void CommandList::setResourceStatesForBindingSetInternal(
        IBindingSet* _bindingSet,
        bool automaticOnly,
        ShaderType pipelineStages)
    {
        if (_bindingSet == nullptr)
            return;
        if (_bindingSet->getDesc() == nullptr)
            return; // is bindless

        BindingSet* bindingSet = checked_cast<BindingSet*>(_bindingSet);
        ShaderType shaderStages = ShaderType::All;
        if (m_CommandListParameters.enableStageQualifiedBindingBarriers)
        {
            BindingLayout* layout = checked_cast<BindingLayout*>(
                bindingSet->layout.Get());
            shaderStages = resolveBindingBarrierShaderStages(
                layout->desc.visibility, pipelineStages);
        }

        for (auto bindingIndex : bindingSet->bindingsThatNeedTransitions)
        {
            const BindingSetItem& binding = bindingSet->desc.bindings[bindingIndex];

            if (automaticOnly && !binding.enableAutomaticTransitions)
                continue;

            switch(binding.type)  // NOLINT(clang-diagnostic-switch-enum)
            {
                case ResourceType::Texture_SRV:
                    requireTextureState(checked_cast<ITexture*>(binding.resourceHandle), binding.subresources, getTextureSrvState(binding), shaderStages);
                    break;

                case ResourceType::Texture_UAV:
                    requireTextureState(checked_cast<ITexture*>(binding.resourceHandle), binding.subresources, ResourceStates::UnorderedAccess, shaderStages);
                    break;

                case ResourceType::TypedBuffer_SRV:
                case ResourceType::StructuredBuffer_SRV:
                case ResourceType::RawBuffer_SRV:
                    requireBufferState(checked_cast<IBuffer*>(binding.resourceHandle), ResourceStates::ShaderResource, shaderStages);
                    break;

                case ResourceType::TypedBuffer_UAV:
                case ResourceType::StructuredBuffer_UAV:
                case ResourceType::RawBuffer_UAV:
                    requireBufferState(checked_cast<IBuffer*>(binding.resourceHandle), ResourceStates::UnorderedAccess, shaderStages);
                    break;

                case ResourceType::ConstantBuffer:
                    requireBufferState(checked_cast<IBuffer*>(binding.resourceHandle), ResourceStates::ConstantBuffer, shaderStages);
                    break;

                case ResourceType::RayTracingAccelStruct:
                    requireBufferState(checked_cast<AccelStruct*>(binding.resourceHandle)->dataBuffer, ResourceStates::AccelStructRead, shaderStages);

                default:
                    // do nothing
                    break;
            }
        }
    }

    void CommandList::insertResourceBarriersForBindingSets(
        const BindingSetVector& newBindings,
        const BindingSetVector& oldBindings,
        ShaderType pipelineStages,
        ShaderType previousPipelineStages)
    {
        uint32_t bindingUpdateMask = 0;

        if (m_BindingStatesDirty)
            bindingUpdateMask = ~0u;

        // The same binding set can be consumed by different active shader
        // stages when its layout visibility is broad. Revisit every set so
        // the state tracker unions those outstanding reads and widens any
        // still-pending destination barrier.
        if (m_CommandListParameters.enableStageQualifiedBindingBarriers
            && pipelineStages != previousPipelineStages)
        {
            bindingUpdateMask = ~0u;
        }

        if (bindingUpdateMask == 0)
            bindingUpdateMask = arrayDifferenceMask(newBindings, oldBindings);

        for (size_t i = 0; i < newBindings.size(); i++)
        {
            if (newBindings[i]->getDesc() == nullptr) // Ignore bindless sets
                continue;

            BindingSet const* bindingSet = checked_cast<BindingSet const*>(newBindings[i]);

            bool const updateThisSet = (bindingUpdateMask & (1u << i)) != 0;
            bool const refreshUavBarriers = bindingUpdateMask != 0 && bindingSet->hasUavBindings;
            if (updateThisSet || refreshUavBarriers || bindingSet->hasDepthReadOnlyAttachmentBindings)
                setResourceStatesForBindingSetInternal(
                    newBindings[i], true, pipelineStages);
        }
    }

    void CommandList::insertGraphicsResourceBarriers(const GraphicsState& state)
    {
        const ShaderType pipelineStages = state.pipeline
            ? checked_cast<GraphicsPipeline*>(state.pipeline)->shaderMask
            : ShaderType::All;
        const ShaderType previousPipelineStages = m_CurrentGraphicsState.pipeline
            ? checked_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline)->shaderMask
            : ShaderType::All;
        insertResourceBarriersForBindingSets(
            state.bindings,
            m_CurrentGraphicsState.bindings,
            pipelineStages,
            previousPipelineStages);

        if (state.indexBuffer.buffer && (m_BindingStatesDirty || state.indexBuffer.buffer != m_CurrentGraphicsState.indexBuffer.buffer))
        {
            requireBufferState(state.indexBuffer.buffer, ResourceStates::IndexBuffer);
        }

        if (m_BindingStatesDirty || arraysAreDifferent(state.vertexBuffers, m_CurrentGraphicsState.vertexBuffers))
        {
            for (const auto& vb : state.vertexBuffers)
            {
                requireBufferState(vb.buffer, ResourceStates::VertexBuffer);
            }
        }

        if (m_BindingStatesDirty || m_CurrentGraphicsState.framebuffer != state.framebuffer)
        {
            setResourceStatesForFramebuffer(state.framebuffer);
        }

        if (state.indirectParams && (m_BindingStatesDirty || state.indirectParams != m_CurrentGraphicsState.indirectParams))
        {
            requireBufferState(state.indirectParams, ResourceStates::IndirectArgument);
        }

        m_BindingStatesDirty = false;
    }

    void CommandList::insertComputeResourceBarriers(const ComputeState& state)
    {
        insertResourceBarriersForBindingSets(
            state.bindings,
            m_CurrentComputeState.bindings,
            ShaderType::Compute,
            ShaderType::Compute);

        if (state.indirectParams && (m_BindingStatesDirty || state.indirectParams != m_CurrentComputeState.indirectParams))
        {
            Buffer* indirectParams = checked_cast<Buffer*>(state.indirectParams);

            requireBufferState(indirectParams, ResourceStates::IndirectArgument);
        }

        m_BindingStatesDirty = false;
    }

    void CommandList::insertMeshletResourceBarriers(const MeshletState& state)
    {
        const ShaderType pipelineStages = state.pipeline
            ? checked_cast<MeshletPipeline*>(state.pipeline)->shaderMask
            : ShaderType::All;
        const ShaderType previousPipelineStages = m_CurrentMeshletState.pipeline
            ? checked_cast<MeshletPipeline*>(m_CurrentMeshletState.pipeline)->shaderMask
            : ShaderType::All;
        insertResourceBarriersForBindingSets(
            state.bindings,
            m_CurrentMeshletState.bindings,
            pipelineStages,
            previousPipelineStages);

        if (m_BindingStatesDirty || m_CurrentMeshletState.framebuffer != state.framebuffer)
        {
            setResourceStatesForFramebuffer(state.framebuffer);
        }

        if (state.indirectParams && (m_BindingStatesDirty || state.indirectParams != m_CurrentMeshletState.indirectParams))
        {
            requireBufferState(state.indirectParams, ResourceStates::IndirectArgument);
        }

        m_BindingStatesDirty = false;
    }

    void CommandList::insertRayTracingResourceBarriers(const rt::State& state)
    {
        insertResourceBarriersForBindingSets(
            state.bindings,
            m_CurrentRayTracingState.bindings,
            ShaderType::AllRayTracing,
            ShaderType::AllRayTracing);

        m_BindingStatesDirty = false;
    }

    void CommandList::requireTextureState(
        ITexture* _texture,
        TextureSubresourceSet subresources,
        ResourceStates state,
        ShaderType shaderStages)
    {
        Texture* texture = checked_cast<Texture*>(_texture);

        if (isTextureRangeReleased(texture, subresources))
        {
            reportReleasedResourceUse("texture", texture->desc.debugName);
            return;
        }

        m_StateTracker.requireTextureState(
            texture, subresources, state, shaderStages);
    }

    void CommandList::requireBufferState(
        IBuffer* _buffer,
        ResourceStates state,
        ShaderType shaderStages)
    {
        Buffer* buffer = checked_cast<Buffer*>(_buffer);

        if (m_ReleasedBuffers.find(buffer) != m_ReleasedBuffers.end())
        {
            reportReleasedResourceUse("buffer", buffer->desc.debugName);
            return;
        }

        m_StateTracker.requireBufferState(buffer, state, shaderStages);
    }

    bool CommandList::isTextureRangeReleased(
        Texture* texture,
        TextureSubresourceSet subresources) const
    {
        subresources = subresources.resolve(texture->desc, false);
        for (const ReleasedTextureRange& released : m_ReleasedTextureRanges)
        {
            if (released.texture == texture
                && textureRangesOverlap(released.subresources, subresources))
            {
                return true;
            }
        }
        return false;
    }

    void CommandList::reportReleasedResourceUse(
        const char* resourceKind,
        const std::string& debugName) const
    {
        std::stringstream message;
        message << "Attempted to use " << resourceKind << " "
            << utils::DebugNameToString(debugName)
            << " after releasing its queue-family ownership on this command list";
        m_Context.error(message.str());
    }

    bool CommandList::anyBarriers() const
    {
        return !m_StateTracker.getBufferBarriers().empty() || !m_StateTracker.getTextureBarriers().empty();
    }

    void CommandList::commitBarriersInternal()
    {
        std::vector<vk::ImageMemoryBarrier2> imageBarriers;
        std::vector<vk::BufferMemoryBarrier2> bufferBarriers;

        // Opt-in (CommandListParameters::collapseComputeOnlyBarrierStages): on a
        // command list whose accesses are all from compute shaders, collapse the
        // conservative ALL_COMMANDS stage scope (emitted by the ShaderResource /
        // UnorderedAccess / ConstantBuffer states) down to COMPUTE_SHADER. This
        // turns full-pipeline-drain barriers into compute-only execution
        // dependencies; access masks and image layouts are left untouched, so the
        // memory visibility and layout transitions are identical. Other stage
        // bits (e.g. eTransfer from CopyDest) are preserved as-is.
        const bool collapseToCompute = m_CommandListParameters.collapseComputeOnlyBarrierStages;
        auto narrowStages = [collapseToCompute](vk::PipelineStageFlags2 s) -> vk::PipelineStageFlags2 {
            if (collapseToCompute && (s & vk::PipelineStageFlagBits2::eAllCommands))
            {
                s &= ~vk::PipelineStageFlags2(vk::PipelineStageFlagBits2::eAllCommands);
                s |= vk::PipelineStageFlagBits2::eComputeShader;
            }
            return s;
        };

        for (const TextureBarrier& barrier : m_StateTracker.getTextureBarriers())
        {
            Texture* texture = static_cast<Texture*>(barrier.texture);

            ResourceStateMapping before = convertTextureState(
                barrier.stateBefore, texture->desc, barrier.shaderStagesBefore);
            ResourceStateMapping after = convertTextureState(
                barrier.stateAfter, texture->desc, barrier.shaderStagesAfter);

            assert(after.imageLayout != vk::ImageLayout::eUndefined);

            const FormatInfo& formatInfo = getFormatInfo(texture->desc.format);

            vk::ImageAspectFlags aspectMask = (vk::ImageAspectFlagBits)0;
            if (formatInfo.hasDepth) aspectMask |= vk::ImageAspectFlagBits::eDepth;
            if (formatInfo.hasStencil) aspectMask |= vk::ImageAspectFlagBits::eStencil;
            if (!aspectMask) aspectMask = vk::ImageAspectFlagBits::eColor;

            vk::ImageSubresourceRange subresourceRange = vk::ImageSubresourceRange()
                .setBaseArrayLayer(barrier.entireTexture ? 0 : barrier.arraySlice)
                .setLayerCount(barrier.entireTexture ? texture->desc.arraySize : 1)
                .setBaseMipLevel(barrier.entireTexture ? 0 : barrier.mipLevel)
                .setLevelCount(barrier.entireTexture ? texture->desc.mipLevels : 1)
                .setAspectMask(aspectMask);

            imageBarriers.push_back(vk::ImageMemoryBarrier2()
                .setSrcAccessMask(before.accessMask)
                .setDstAccessMask(after.accessMask)
                .setSrcStageMask(narrowStages(before.stageFlags))
                .setDstStageMask(narrowStages(after.stageFlags))
                .setOldLayout(before.imageLayout)
                .setNewLayout(after.imageLayout)
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setImage(texture->image)
                .setSubresourceRange(subresourceRange));
        }

        if (!imageBarriers.empty())
        {
            vk::DependencyInfo dep_info;
            dep_info.setImageMemoryBarriers(imageBarriers);

            m_CurrentCmdBuf->cmdBuf.pipelineBarrier2(dep_info);
        }

        imageBarriers.clear();

        for (const BufferBarrier& barrier : m_StateTracker.getBufferBarriers())
        {
            ResourceStateMapping before = convertResourceState(
                barrier.stateBefore, false, false, false,
                barrier.shaderStagesBefore);
            ResourceStateMapping after = convertResourceState(
                barrier.stateAfter, false, false, false,
                barrier.shaderStagesAfter);

            Buffer* buffer = static_cast<Buffer*>(barrier.buffer);

            bufferBarriers.push_back(vk::BufferMemoryBarrier2()
                .setSrcAccessMask(before.accessMask)
                .setDstAccessMask(after.accessMask)
                .setSrcStageMask(narrowStages(before.stageFlags))
                .setDstStageMask(narrowStages(after.stageFlags))
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setBuffer(buffer->buffer)
                .setOffset(0)
                .setSize(buffer->desc.byteSize));
        }

        if (!bufferBarriers.empty())
        {
            vk::DependencyInfo dep_info;
            dep_info.setBufferMemoryBarriers(bufferBarriers);

            m_CurrentCmdBuf->cmdBuf.pipelineBarrier2(dep_info);
        }
        bufferBarriers.clear();

        m_StateTracker.clearBarriers();
    }

    void CommandList::commitBarriers()
    {
        if (m_StateTracker.getBufferBarriers().empty() && m_StateTracker.getTextureBarriers().empty())
            return;

        endRenderPass();

        commitBarriersInternal();
    }

    bool CommandList::recordTextureQueueOwnershipTransfer(
        Texture* texture,
        TextureSubresourceSet subresources,
        const QueueOwnershipTransferDesc& transfer,
        bool release)
    {
        uint32_t sourceQueueFamily = VK_QUEUE_FAMILY_IGNORED;
        uint32_t destinationQueueFamily = VK_QUEUE_FAMILY_IGNORED;
        if (!validateQueueOwnershipTransfer(
            this, m_Device, m_Context, transfer, release,
            sourceQueueFamily, destinationQueueFamily))
        {
            return false;
        }
        if (!texture)
        {
            m_Context.error("Queue ownership transfer texture is null");
            return false;
        }
        if (texture->queueSharingMode == QueueSharingMode::Concurrent)
        {
            m_Context.error("Queue ownership transfer cannot target a concurrent-sharing texture");
            return false;
        }
        if (texture->queueSharingMode == QueueSharingMode::UnknownNative
            || !texture->managed)
        {
            m_Context.error("Queue ownership transfer cannot prove the sharing mode of a native texture");
            return false;
        }
        if (texture->desc.keepInitialState)
        {
            m_Context.error("Queue ownership transfer textures must disable keepInitialState");
            return false;
        }
        if (texture->permanentState != ResourceStates::Unknown)
        {
            m_Context.error("Queue ownership transfer cannot target a permanent-state texture");
            return false;
        }
        if (m_StateTracker.hasPendingPermanentTextureState(texture))
        {
            m_Context.error("Queue ownership transfer cannot follow a pending permanent-state texture transition");
            return false;
        }

        subresources = subresources.resolve(texture->desc, false);
        if (!isValidResolvedTextureRange(texture, subresources))
        {
            m_Context.error("Queue ownership transfer texture subresource range is empty or out of bounds");
            return false;
        }
        ShaderType effectiveBeforeStages = transfer.shaderStagesBefore;
        if (release)
        {
            if (isTextureRangeReleased(texture, subresources))
            {
                m_Context.error("Texture queue ownership was already released on this command list");
                return false;
            }
            if (!m_StateTracker.isTextureStateTracked(texture, subresources))
            {
                m_Context.error("Texture queue ownership release requires a tracked source state");
                return false;
            }
            for (ArraySlice arraySlice = subresources.baseArraySlice;
                 arraySlice < subresources.baseArraySlice + subresources.numArraySlices;
                 ++arraySlice)
            {
                for (MipLevel mipLevel = subresources.baseMipLevel;
                     mipLevel < subresources.baseMipLevel + subresources.numMipLevels;
                     ++mipLevel)
                {
                    if (m_StateTracker.getTextureSubresourceState(
                            texture, arraySlice, mipLevel) != transfer.stateBefore)
                    {
                        m_Context.error("Texture queue ownership release state does not match the tracked state");
                        return false;
                    }
                    effectiveBeforeStages = effectiveBeforeStages
                        | m_StateTracker.getTextureSubresourceShaderStages(
                            texture, arraySlice, mipLevel);
                }
            }
        }
        else if (m_StateTracker.isTextureStateTracked(texture, subresources))
        {
            m_Context.error("Texture queue ownership acquire must precede its first local use");
            return false;
        }

        const ResourceStateMapping before = convertTextureState(
            transfer.stateBefore, texture->desc, effectiveBeforeStages);
        const ResourceStateMapping after = convertTextureState(
            transfer.stateAfter, texture->desc, transfer.shaderStagesAfter);
        if (before.imageLayout == vk::ImageLayout::eUndefined
            || after.imageLayout == vk::ImageLayout::eUndefined)
        {
            m_Context.error("Queue ownership transfer texture states must map to concrete image layouts");
            return false;
        }

        endRenderPass();
        if (anyBarriers())
            commitBarriersInternal();

        const bool sameFamily = sourceQueueFamily == destinationQueueFamily;
        if (release || !sameFamily)
        {
            const vk::ImageMemoryBarrier2 barrier =
                detail::buildQueueOwnershipImageBarrier(
                    texture->image,
                    makeImageSubresourceRange(texture, subresources),
                    before, after,
                    sourceQueueFamily, destinationQueueFamily,
                    release, sameFamily);
            vk::DependencyInfo dependencyInfo;
            dependencyInfo.setDependencyFlags(
                detail::queueOwnershipDependencyFlags(
                    sameFamily, m_Context.extensions.KHR_maintenance8));
            dependencyInfo.setImageMemoryBarriers(barrier);
            m_CurrentCmdBuf->cmdBuf.pipelineBarrier2(dependencyInfo);
        }

        m_StateTracker.beginTrackingTextureState(
            texture, subresources, transfer.stateAfter,
            transfer.shaderStagesAfter);
        if (release)
            m_ReleasedTextureRanges.push_back({ texture, subresources });
        m_CurrentCmdBuf->referencedResources.push_back(texture);
        return true;
    }

    bool CommandList::recordBufferQueueOwnershipTransfer(
        Buffer* buffer,
        const QueueOwnershipTransferDesc& transfer,
        bool release)
    {
        uint32_t sourceQueueFamily = VK_QUEUE_FAMILY_IGNORED;
        uint32_t destinationQueueFamily = VK_QUEUE_FAMILY_IGNORED;
        if (!validateQueueOwnershipTransfer(
            this, m_Device, m_Context, transfer, release,
            sourceQueueFamily, destinationQueueFamily))
        {
            return false;
        }
        if (!buffer)
        {
            m_Context.error("Queue ownership transfer buffer is null");
            return false;
        }
        if (buffer->queueSharingMode == QueueSharingMode::Concurrent)
        {
            m_Context.error("Queue ownership transfer cannot target a concurrent-sharing buffer");
            return false;
        }
        if (buffer->queueSharingMode == QueueSharingMode::UnknownNative
            || !buffer->managed)
        {
            m_Context.error("Queue ownership transfer cannot prove the sharing mode of a native buffer");
            return false;
        }
        if (buffer->desc.keepInitialState)
        {
            m_Context.error("Queue ownership transfer buffers must disable keepInitialState");
            return false;
        }
        if (buffer->permanentState != ResourceStates::Unknown)
        {
            m_Context.error("Queue ownership transfer cannot target a permanent-state buffer");
            return false;
        }
        if (m_StateTracker.hasPendingPermanentBufferState(buffer))
        {
            m_Context.error("Queue ownership transfer cannot follow a pending permanent-state buffer transition");
            return false;
        }
        if (buffer->desc.isVolatile || buffer->desc.cpuAccess != CpuAccessMode::None)
        {
            m_Context.error("Queue ownership transfer does not support volatile or CPU-visible buffers");
            return false;
        }

        if (release)
        {
            if (m_ReleasedBuffers.find(buffer) != m_ReleasedBuffers.end())
            {
                m_Context.error("Buffer queue ownership was already released on this command list");
                return false;
            }
            if (!m_StateTracker.isBufferStateTracked(buffer))
            {
                m_Context.error("Buffer queue ownership release requires a tracked source state");
                return false;
            }
            if (m_StateTracker.getBufferState(buffer) != transfer.stateBefore)
            {
                m_Context.error("Buffer queue ownership release state does not match the tracked state");
                return false;
            }
        }
        else if (m_StateTracker.isBufferStateTracked(buffer))
        {
            m_Context.error("Buffer queue ownership acquire must precede its first local use");
            return false;
        }

        const ShaderType effectiveBeforeStages = release
            ? transfer.shaderStagesBefore | m_StateTracker.getBufferShaderStages(buffer)
            : transfer.shaderStagesBefore;
        const ResourceStateMapping before = convertResourceState(
            transfer.stateBefore, false, false, false,
            effectiveBeforeStages);
        const ResourceStateMapping after = convertResourceState(
            transfer.stateAfter, false, false, false,
            transfer.shaderStagesAfter);

        endRenderPass();
        if (anyBarriers())
            commitBarriersInternal();

        const bool sameFamily = sourceQueueFamily == destinationQueueFamily;
        if (release || !sameFamily)
        {
            const vk::BufferMemoryBarrier2 barrier =
                detail::buildQueueOwnershipBufferBarrier(
                    buffer->buffer, buffer->desc.byteSize,
                    before, after,
                    sourceQueueFamily, destinationQueueFamily,
                    release, sameFamily);
            vk::DependencyInfo dependencyInfo;
            dependencyInfo.setDependencyFlags(
                detail::queueOwnershipDependencyFlags(
                    sameFamily, m_Context.extensions.KHR_maintenance8));
            dependencyInfo.setBufferMemoryBarriers(barrier);
            m_CurrentCmdBuf->cmdBuf.pipelineBarrier2(dependencyInfo);
        }

        m_StateTracker.beginTrackingBufferState(
            buffer, transfer.stateAfter, transfer.shaderStagesAfter);
        if (release)
            m_ReleasedBuffers.insert(buffer);
        m_CurrentCmdBuf->referencedResources.push_back(buffer);
        return true;
    }

    bool Device::releaseTextureQueueOwnership(
        ICommandList* commandList, ITexture* texture,
        TextureSubresourceSet subresources,
        const QueueOwnershipTransferDesc& transfer)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Texture* vulkanTexture = dynamic_cast<Texture*>(texture);
        if (!vulkanCommandList || !vulkanTexture)
        {
            m_Context.error("Queue ownership release requires Vulkan command-list and texture objects");
            return false;
        }
        return vulkanCommandList->recordTextureQueueOwnershipTransfer(
            vulkanTexture, subresources, transfer, true);
    }

    bool Device::acquireTextureQueueOwnership(
        ICommandList* commandList, ITexture* texture,
        TextureSubresourceSet subresources,
        const QueueOwnershipTransferDesc& transfer)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Texture* vulkanTexture = dynamic_cast<Texture*>(texture);
        if (!vulkanCommandList || !vulkanTexture)
        {
            m_Context.error("Queue ownership acquire requires Vulkan command-list and texture objects");
            return false;
        }
        return vulkanCommandList->recordTextureQueueOwnershipTransfer(
            vulkanTexture, subresources, transfer, false);
    }

    bool Device::releaseBufferQueueOwnership(
        ICommandList* commandList, IBuffer* buffer,
        const QueueOwnershipTransferDesc& transfer)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Buffer* vulkanBuffer = dynamic_cast<Buffer*>(buffer);
        if (!vulkanCommandList || !vulkanBuffer)
        {
            m_Context.error("Queue ownership release requires Vulkan command-list and buffer objects");
            return false;
        }
        return vulkanCommandList->recordBufferQueueOwnershipTransfer(
            vulkanBuffer, transfer, true);
    }

    bool Device::acquireBufferQueueOwnership(
        ICommandList* commandList, IBuffer* buffer,
        const QueueOwnershipTransferDesc& transfer)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Buffer* vulkanBuffer = dynamic_cast<Buffer*>(buffer);
        if (!vulkanCommandList || !vulkanBuffer)
        {
            m_Context.error("Queue ownership acquire requires Vulkan command-list and buffer objects");
            return false;
        }
        return vulkanCommandList->recordBufferQueueOwnershipTransfer(
            vulkanBuffer, transfer, false);
    }

    void CommandList::beginTrackingTextureState(ITexture* _texture, TextureSubresourceSet subresources, ResourceStates stateBits)
    {
        beginTrackingTextureState(
            _texture, subresources, stateBits, ShaderType::All);
    }

    void CommandList::beginTrackingTextureState(
        ITexture* _texture,
        TextureSubresourceSet subresources,
        ResourceStates stateBits,
        ShaderType shaderStages)
    {
        Texture* texture = checked_cast<Texture*>(_texture);

        if (isTextureRangeReleased(texture, subresources))
        {
            reportReleasedResourceUse("texture", texture->desc.debugName);
            return;
        }

        m_StateTracker.beginTrackingTextureState(
            texture, subresources, stateBits, shaderStages);
    }

    void CommandList::beginTrackingBufferState(IBuffer* _buffer, ResourceStates stateBits)
    {
        beginTrackingBufferState(_buffer, stateBits, ShaderType::All);
    }

    void CommandList::beginTrackingBufferState(
        IBuffer* _buffer,
        ResourceStates stateBits,
        ShaderType shaderStages)
    {
        Buffer* buffer = checked_cast<Buffer*>(_buffer);

        if (m_ReleasedBuffers.find(buffer) != m_ReleasedBuffers.end())
        {
            reportReleasedResourceUse("buffer", buffer->desc.debugName);
            return;
        }

        m_StateTracker.beginTrackingBufferState(buffer, stateBits, shaderStages);
    }

    void CommandList::setTextureState(ITexture* _texture, TextureSubresourceSet subresources, ResourceStates stateBits)
    {
        setTextureState(_texture, subresources, stateBits, ShaderType::All);
    }

    void CommandList::setTextureState(
        ITexture* _texture,
        TextureSubresourceSet subresources,
        ResourceStates stateBits,
        ShaderType shaderStages)
    {
        Texture* texture = checked_cast<Texture*>(_texture);

        requireTextureState(texture, subresources, stateBits, shaderStages);

        if (m_CurrentCmdBuf)
            m_CurrentCmdBuf->referencedResources.push_back(texture);
    }

    void CommandList::setBufferState(IBuffer* _buffer, ResourceStates stateBits)
    {
        setBufferState(_buffer, stateBits, ShaderType::All);
    }

    void CommandList::setBufferState(
        IBuffer* _buffer,
        ResourceStates stateBits,
        ShaderType shaderStages)
    {
        Buffer* buffer = checked_cast<Buffer*>(_buffer);

        requireBufferState(buffer, stateBits, shaderStages);
        
        if (m_CurrentCmdBuf)
            m_CurrentCmdBuf->referencedResources.push_back(buffer);
    }
    
    void CommandList::setAccelStructState(rt::IAccelStruct* _as, ResourceStates stateBits)
    {
        setAccelStructState(_as, stateBits, ShaderType::All);
    }

    void CommandList::setAccelStructState(
        rt::IAccelStruct* _as,
        ResourceStates stateBits,
        ShaderType shaderStages)
    {
        AccelStruct* as = checked_cast<AccelStruct*>(_as);

        if (as->dataBuffer)
        {
            Buffer* buffer = checked_cast<Buffer*>(as->dataBuffer.Get());
            requireBufferState(buffer, stateBits, shaderStages);

            if (m_CurrentCmdBuf)
                m_CurrentCmdBuf->referencedResources.push_back(as);
        }
    }

    void CommandList::setPermanentTextureState(ITexture* _texture, ResourceStates stateBits)
    {
        Texture* texture = checked_cast<Texture*>(_texture);

        if (isTextureRangeReleased(texture, AllSubresources))
        {
            reportReleasedResourceUse("texture", texture->desc.debugName);
            return;
        }

        m_StateTracker.setPermanentTextureState(texture, AllSubresources, stateBits);

        if (m_CurrentCmdBuf)
            m_CurrentCmdBuf->referencedResources.push_back(texture);
    }

    void CommandList::setPermanentBufferState(IBuffer* _buffer, ResourceStates stateBits)
    {
        Buffer* buffer = checked_cast<Buffer*>(_buffer);

        if (m_ReleasedBuffers.find(buffer) != m_ReleasedBuffers.end())
        {
            reportReleasedResourceUse("buffer", buffer->desc.debugName);
            return;
        }

        m_StateTracker.setPermanentBufferState(buffer, stateBits);
        
        if (m_CurrentCmdBuf)
            m_CurrentCmdBuf->referencedResources.push_back(buffer);
    }

    ResourceStates CommandList::getTextureSubresourceState(ITexture* _texture, ArraySlice arraySlice, MipLevel mipLevel)
    {
        Texture* texture = checked_cast<Texture*>(_texture);

        return m_StateTracker.getTextureSubresourceState(texture, arraySlice, mipLevel);
    }

    ResourceStates CommandList::getBufferState(IBuffer* _buffer)
    {
        Buffer* buffer = checked_cast<Buffer*>(_buffer);

        return m_StateTracker.getBufferState(buffer);
    }

    void CommandList::setEnableAutomaticBarriers(bool enable)
    {
        m_EnableAutomaticBarriers = enable;
    }

    void CommandList::setEnableUavBarriersForTexture(ITexture* _texture, bool enableBarriers)
    {
        Texture* texture = checked_cast<Texture*>(_texture);

        m_StateTracker.setEnableUavBarriersForTexture(texture, enableBarriers);
    }

    void CommandList::setEnableUavBarriersForBuffer(IBuffer* _buffer, bool enableBarriers)
    {
        Buffer* buffer = checked_cast<Buffer*>(_buffer);

        m_StateTracker.setEnableUavBarriersForBuffer(buffer, enableBarriers);
    }

} // namespace nvrhi::vulkan
