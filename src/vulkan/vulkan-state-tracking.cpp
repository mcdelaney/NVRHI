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

    static bool isValidGraphTextureRange(
        const Texture* texture,
        const TextureSubresourceSet& subresources)
    {
        const TextureDesc& desc = texture->desc;
        if (desc.mipLevels == 0 || desc.arraySize == 0
            || subresources.baseMipLevel >= desc.mipLevels
            || subresources.baseArraySlice >= desc.arraySize)
        {
            return false;
        }
        const MipLevel mipCount =
            subresources.numMipLevels == TextureSubresourceSet::AllMipLevels
                ? desc.mipLevels - subresources.baseMipLevel
                : subresources.numMipLevels;
        const ArraySlice sliceCount =
            subresources.numArraySlices == TextureSubresourceSet::AllArraySlices
                ? desc.arraySize - subresources.baseArraySlice
                : subresources.numArraySlices;
        return mipCount > 0
            && mipCount <= desc.mipLevels - subresources.baseMipLevel
            && sliceCount > 0
            && sliceCount <= desc.arraySize - subresources.baseArraySlice;
    }

    constexpr uint32_t graphStateBits(ResourceStates state)
    {
        return uint32_t(state);
    }

    constexpr uint32_t c_GraphKnownStateBits =
        graphStateBits(ResourceStates::Common)
        | graphStateBits(ResourceStates::ConstantBuffer)
        | graphStateBits(ResourceStates::VertexBuffer)
        | graphStateBits(ResourceStates::IndexBuffer)
        | graphStateBits(ResourceStates::IndirectArgument)
        | graphStateBits(ResourceStates::ShaderResource)
        | graphStateBits(ResourceStates::UnorderedAccess)
        | graphStateBits(ResourceStates::RenderTarget)
        | graphStateBits(ResourceStates::DepthWrite)
        | graphStateBits(ResourceStates::DepthRead)
        | graphStateBits(ResourceStates::StreamOut)
        | graphStateBits(ResourceStates::CopyDest)
        | graphStateBits(ResourceStates::CopySource)
        | graphStateBits(ResourceStates::ResolveDest)
        | graphStateBits(ResourceStates::ResolveSource)
        | graphStateBits(ResourceStates::Present)
        | graphStateBits(ResourceStates::AccelStructRead)
        | graphStateBits(ResourceStates::AccelStructWrite)
        | graphStateBits(ResourceStates::AccelStructBuildInput)
        | graphStateBits(ResourceStates::AccelStructBuildBlas)
        | graphStateBits(ResourceStates::ShadingRateSurface)
        | graphStateBits(ResourceStates::OpacityMicromapWrite)
        | graphStateBits(ResourceStates::OpacityMicromapBuildInput)
        | graphStateBits(ResourceStates::ConvertCoopVecMatrixInput)
        | graphStateBits(ResourceStates::ConvertCoopVecMatrixOutput);

    constexpr uint32_t c_GraphTextureStateBits =
        graphStateBits(ResourceStates::Common)
        | graphStateBits(ResourceStates::ShaderResource)
        | graphStateBits(ResourceStates::UnorderedAccess)
        | graphStateBits(ResourceStates::RenderTarget)
        | graphStateBits(ResourceStates::DepthWrite)
        | graphStateBits(ResourceStates::DepthRead)
        | graphStateBits(ResourceStates::CopyDest)
        | graphStateBits(ResourceStates::CopySource)
        | graphStateBits(ResourceStates::ResolveDest)
        | graphStateBits(ResourceStates::ResolveSource)
        | graphStateBits(ResourceStates::Present)
        | graphStateBits(ResourceStates::ShadingRateSurface);

    constexpr uint32_t c_GraphBufferStateBits =
        graphStateBits(ResourceStates::Common)
        | graphStateBits(ResourceStates::ConstantBuffer)
        | graphStateBits(ResourceStates::VertexBuffer)
        | graphStateBits(ResourceStates::IndexBuffer)
        | graphStateBits(ResourceStates::IndirectArgument)
        | graphStateBits(ResourceStates::ShaderResource)
        | graphStateBits(ResourceStates::UnorderedAccess)
        | graphStateBits(ResourceStates::StreamOut)
        | graphStateBits(ResourceStates::CopyDest)
        | graphStateBits(ResourceStates::CopySource)
        | graphStateBits(ResourceStates::AccelStructBuildInput)
        | graphStateBits(ResourceStates::OpacityMicromapWrite)
        | graphStateBits(ResourceStates::OpacityMicromapBuildInput)
        | graphStateBits(ResourceStates::ConvertCoopVecMatrixInput)
        | graphStateBits(ResourceStates::ConvertCoopVecMatrixOutput);

    const ResourceStates c_GraphShaderVisibleStates =
        ResourceStates::ConstantBuffer
        | ResourceStates::ShaderResource
        | ResourceStates::UnorderedAccess
        | ResourceStates::AccelStructRead;

    bool validateGraphStateKind(
        const VulkanContext& context,
        ResourceStates state,
        bool texture,
        const TextureDesc* textureDesc,
        const char* label)
    {
        const uint32_t bits = graphStateBits(state);
        if (state == ResourceStates::Unknown
            || (bits & ~c_GraphKnownStateBits) != 0)
        {
            context.error(std::string(label) + " state is unknown or contains unsupported bits");
            return false;
        }
        if (((state & ResourceStates::Common) != 0
                && state != ResourceStates::Common)
            || ((state & ResourceStates::Present) != 0
                && state != ResourceStates::Present))
        {
            context.error(std::string(label) + " combines a standalone resource state");
            return false;
        }

        const uint32_t allowed = texture
            ? c_GraphTextureStateBits : c_GraphBufferStateBits;
        if ((bits & ~allowed) != 0)
        {
            context.error(std::string(label) + " state is incompatible with the resource kind");
            return false;
        }
        if (!texture)
            return true;

        const FormatInfo& formatInfo = getFormatInfo(textureDesc->format);
        const bool isDepthStencil = formatInfo.hasDepth || formatInfo.hasStencil;
        if (((state & (ResourceStates::DepthRead | ResourceStates::DepthWrite)) != 0
                && !isDepthStencil)
            || ((state & ResourceStates::RenderTarget) != 0
                && isDepthStencil))
        {
            context.error(std::string(label) + " state is incompatible with the texture format");
            return false;
        }
        if (textureDesc->useGeneralLayout)
            return true;

        enum class LayoutClass : uint8_t
        {
            None,
            ShaderRead,
            General,
            ColorAttachment,
            DepthAttachment,
            DepthReadOnly,
            TransferDestination,
            TransferSource,
            Present,
            ShadingRate,
        };
        LayoutClass layout = LayoutClass::None;
        const bool sampledDepthRead = isDepthStencil
            && (state & ResourceStates::ShaderResource) != 0
            && (state & ResourceStates::DepthRead) != 0;
        const auto mergeLayout = [&](LayoutClass candidate) {
            if (layout == LayoutClass::None || layout == candidate)
            {
                layout = candidate;
                return true;
            }
            return false;
        };
        bool valid = true;
        if ((state & ResourceStates::ShaderResource) != 0)
            valid &= mergeLayout(sampledDepthRead
                ? LayoutClass::DepthReadOnly : LayoutClass::ShaderRead);
        if ((state & ResourceStates::UnorderedAccess) != 0)
            valid &= mergeLayout(LayoutClass::General);
        if ((state & ResourceStates::RenderTarget) != 0)
            valid &= mergeLayout(LayoutClass::ColorAttachment);
        if ((state & ResourceStates::DepthWrite) != 0)
            valid &= mergeLayout(LayoutClass::DepthAttachment);
        if ((state & ResourceStates::DepthRead) != 0)
            valid &= mergeLayout(LayoutClass::DepthReadOnly);
        if ((state & (ResourceStates::CopyDest | ResourceStates::ResolveDest)) != 0)
            valid &= mergeLayout(LayoutClass::TransferDestination);
        if ((state & (ResourceStates::CopySource | ResourceStates::ResolveSource)) != 0)
            valid &= mergeLayout(LayoutClass::TransferSource);
        if ((state & ResourceStates::Present) != 0)
            valid &= mergeLayout(LayoutClass::Present);
        if ((state & ResourceStates::ShadingRateSurface) != 0)
            valid &= mergeLayout(LayoutClass::ShadingRate);
        if (!valid)
        {
            context.error(std::string(label) + " state maps to incompatible image layouts");
            return false;
        }
        return true;
    }

    bool validateGraphState(
        const VulkanContext& context,
        const GraphResourceState& state,
        bool texture,
        const TextureDesc* textureDesc,
        const char* label)
    {
        if (!validateGraphStateKind(
                context, state.state, texture, textureDesc, label))
        {
            return false;
        }
        if ((uint16_t(state.shaderStages) & ~uint16_t(ShaderType::All)) != 0)
        {
            context.error(std::string(label) + " contains unknown shader-stage bits");
            return false;
        }
        const bool shaderVisible =
            (state.state & c_GraphShaderVisibleStates) != 0;
        if (shaderVisible && state.shaderStages == ShaderType::None)
        {
            context.error(std::string(label) + " shader-visible state requires non-None shader stages");
            return false;
        }
        if (!shaderVisible && state.shaderStages != ShaderType::None)
        {
            context.error(std::string(label) + " fixed-function state must use ShaderType::None");
            return false;
        }
        return true;
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
        if (!commandList->isRecording())
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

        if (state.indirectCountBuffer
            && state.indirectCountBuffer != state.indirectParams
            && (m_BindingStatesDirty
                || state.indirectCountBuffer != m_CurrentGraphicsState.indirectCountBuffer))
        {
            requireBufferState(state.indirectCountBuffer, ResourceStates::IndirectArgument);
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

        if (state.indirectCountBuffer
            && state.indirectCountBuffer != state.indirectParams
            && (m_BindingStatesDirty
                || state.indirectCountBuffer != m_CurrentMeshletState.indirectCountBuffer))
        {
            requireBufferState(state.indirectCountBuffer, ResourceStates::IndirectArgument);
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
        ShaderType shaderStages,
        bool preserveReadOnlyDepthState)
    {
        Texture* texture = checked_cast<Texture*>(_texture);

        // Same-state declarations can widen/coalesce a pending explicit
        // dependency before it is emitted. A real state transition must begin
        // after the explicit dependency batch instead of sharing that batch.
        if (m_PendingBarriersAreMemoryDependencies)
        {
            const TextureSubresourceSet resolved =
                subresources.resolve(texture->desc, false);
            bool sameState = isValidResolvedTextureRange(texture, resolved)
                && m_StateTracker.isTextureStateTracked(texture, resolved);
            for (ArraySlice arraySlice = resolved.baseArraySlice;
                 sameState
                    && arraySlice < resolved.baseArraySlice + resolved.numArraySlices;
                 ++arraySlice)
            {
                for (MipLevel mipLevel = resolved.baseMipLevel;
                     mipLevel < resolved.baseMipLevel + resolved.numMipLevels;
                     ++mipLevel)
                {
                    if (m_StateTracker.getTextureSubresourceState(
                            texture, arraySlice, mipLevel) != state)
                    {
                        sameState = false;
                        break;
                    }
                }
            }
            if (!sameState)
            {
                endRenderPass();
                commitBarriersInternal();
            }
        }

        if (isTextureRangeReleased(texture, subresources))
        {
            reportReleasedResourceUse("texture", texture->desc.debugName);
            return;
        }

        m_StateTracker.requireTextureState(
            texture, subresources, state, shaderStages,
            preserveReadOnlyDepthState);
    }

    void CommandList::requireBufferState(
        IBuffer* _buffer,
        ResourceStates state,
        ShaderType shaderStages)
    {
        Buffer* buffer = checked_cast<Buffer*>(_buffer);

        if (m_PendingBarriersAreMemoryDependencies)
        {
            if (!m_StateTracker.isBufferStateTracked(buffer)
                || m_StateTracker.getBufferState(buffer) != state)
            {
                endRenderPass();
                commitBarriersInternal();
            }
        }

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

        if (!imageBarriers.empty() || !bufferBarriers.empty())
        {
            vk::DependencyInfo dep_info;
            dep_info.setImageMemoryBarriers(imageBarriers);
            dep_info.setBufferMemoryBarriers(bufferBarriers);

            m_CurrentCmdBuf->cmdBuf.pipelineBarrier2(dep_info);
        }

        m_StateTracker.clearBarriers();
        m_PendingBarriersAreMemoryDependencies = false;
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
        if (!texture->belongsTo(m_Context))
        {
            m_Context.error(
                "Queue ownership transfer texture belongs to a different device");
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
        if (!buffer->belongsTo(m_Context))
        {
            m_Context.error(
                "Queue ownership transfer buffer belongs to a different device");
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
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error(
                "Queue ownership release command list belongs to a different device");
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
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error(
                "Queue ownership acquire command list belongs to a different device");
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
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error(
                "Queue ownership release command list belongs to a different device");
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
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error(
                "Queue ownership acquire command list belongs to a different device");
            return false;
        }
        return vulkanCommandList->recordBufferQueueOwnershipTransfer(
            vulkanBuffer, transfer, false);
    }

    bool CommandList::recordTextureMemoryDependency(
        Texture* texture,
        TextureSubresourceSet subresources,
        const MemoryDependencyDesc& dependency)
    {
        if (!m_IsRecording)
        {
            m_Context.error(
                "Texture memory dependencies must be added to an open command list");
            return false;
        }
        if (!texture)
        {
            m_Context.error("Texture memory dependency texture is null");
            return false;
        }
        if (!texture->belongsTo(m_Context))
        {
            m_Context.error(
                "Texture memory dependency texture belongs to a different device");
            return false;
        }
        if (dependency.state == ResourceStates::Unknown)
        {
            m_Context.error("Texture memory dependency state must be known");
            return false;
        }
        if (texture->desc.keepInitialState)
        {
            m_Context.error(
                "Texture memory dependency textures must disable keepInitialState");
            return false;
        }
        if (texture->permanentState != ResourceStates::Unknown)
        {
            m_Context.error(
                "Texture memory dependency cannot target a permanent-state texture");
            return false;
        }
        if (m_StateTracker.hasPendingPermanentTextureState(texture))
        {
            m_Context.error(
                "Texture memory dependency cannot follow a pending permanent-state texture transition");
            return false;
        }

        subresources = subresources.resolve(texture->desc, false);
        if (!isValidResolvedTextureRange(texture, subresources))
        {
            m_Context.error(
                "Texture memory dependency subresource range is empty or out of bounds");
            return false;
        }
        if (isTextureRangeReleased(texture, subresources))
        {
            reportReleasedResourceUse("texture", texture->desc.debugName);
            return false;
        }
        if (!m_StateTracker.isTextureStateTracked(texture, subresources))
        {
            m_Context.error(
                "Texture memory dependency requires an explicitly tracked state");
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
                        texture, arraySlice, mipLevel) != dependency.state)
                {
                    m_Context.error(
                        "Texture memory dependency state does not match every tracked subresource");
                    return false;
                }
            }
        }

        const ResourceStateMapping after = convertTextureState(
            dependency.state, texture->desc, dependency.shaderStagesAfter);
        if (after.imageLayout == vk::ImageLayout::eUndefined)
        {
            m_Context.error(
                "Texture memory dependency state must map to a concrete image layout");
            return false;
        }
        Queue* recordingQueue =
            m_Device->getQueue(m_CommandListParameters.queueType);
        if (!recordingQueue
            || !detail::isWaitStageMaskSupported(
                after.stageFlags, recordingQueue->getQueueFlags()))
        {
            m_Context.error(
                "Texture memory dependency destination stage scope is unsupported by the recording queue family");
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
                const ShaderType effectiveBefore =
                    dependency.shaderStagesBefore
                    | m_StateTracker.getTextureSubresourceShaderStages(
                        texture, arraySlice, mipLevel);
                const ResourceStateMapping before = convertTextureState(
                    dependency.state, texture->desc, effectiveBefore);
                if (before.imageLayout == vk::ImageLayout::eUndefined
                    || !detail::isWaitStageMaskSupported(
                        before.stageFlags, recordingQueue->getQueueFlags()))
                {
                    m_Context.error(
                        "Texture memory dependency source stage scope is unsupported by the recording queue family");
                    return false;
                }
            }
        }

        endRenderPass();
        if (anyBarriers() && !m_PendingBarriersAreMemoryDependencies)
            commitBarriersInternal();
        if (!m_StateTracker.addTextureMemoryDependency(
                texture, subresources, dependency.state,
                dependency.shaderStagesBefore,
                dependency.shaderStagesAfter))
        {
            return false;
        }

        m_PendingBarriersAreMemoryDependencies = true;
        m_CurrentCmdBuf->referencedResources.push_back(texture);
        return true;
    }

    bool CommandList::recordBufferMemoryDependency(
        Buffer* buffer,
        const MemoryDependencyDesc& dependency)
    {
        if (!m_IsRecording)
        {
            m_Context.error(
                "Buffer memory dependencies must be added to an open command list");
            return false;
        }
        if (!buffer)
        {
            m_Context.error("Buffer memory dependency buffer is null");
            return false;
        }
        if (!buffer->belongsTo(m_Context))
        {
            m_Context.error(
                "Buffer memory dependency buffer belongs to a different device");
            return false;
        }
        if (dependency.state == ResourceStates::Unknown)
        {
            m_Context.error("Buffer memory dependency state must be known");
            return false;
        }
        if (buffer->desc.keepInitialState)
        {
            m_Context.error(
                "Buffer memory dependency buffers must disable keepInitialState");
            return false;
        }
        if (buffer->permanentState != ResourceStates::Unknown)
        {
            m_Context.error(
                "Buffer memory dependency cannot target a permanent-state buffer");
            return false;
        }
        if (m_StateTracker.hasPendingPermanentBufferState(buffer))
        {
            m_Context.error(
                "Buffer memory dependency cannot follow a pending permanent-state buffer transition");
            return false;
        }
        if (buffer->desc.isVolatile
            || buffer->desc.cpuAccess != CpuAccessMode::None)
        {
            m_Context.error(
                "Buffer memory dependency does not support volatile or CPU-visible buffers");
            return false;
        }
        if (m_ReleasedBuffers.find(buffer) != m_ReleasedBuffers.end())
        {
            reportReleasedResourceUse("buffer", buffer->desc.debugName);
            return false;
        }
        if (!m_StateTracker.isBufferStateTracked(buffer))
        {
            m_Context.error(
                "Buffer memory dependency requires an explicitly tracked state");
            return false;
        }
        if (m_StateTracker.getBufferState(buffer) != dependency.state)
        {
            m_Context.error(
                "Buffer memory dependency state does not match the tracked state");
            return false;
        }

        Queue* recordingQueue =
            m_Device->getQueue(m_CommandListParameters.queueType);
        const ShaderType effectiveBefore = dependency.shaderStagesBefore
            | m_StateTracker.getBufferShaderStages(buffer);
        const ResourceStateMapping before = convertResourceState(
            dependency.state, false, false, false, effectiveBefore);
        const ResourceStateMapping after = convertResourceState(
            dependency.state, false, false, false,
            dependency.shaderStagesAfter);
        if (!recordingQueue
            || !detail::isWaitStageMaskSupported(
                before.stageFlags, recordingQueue->getQueueFlags())
            || !detail::isWaitStageMaskSupported(
                after.stageFlags, recordingQueue->getQueueFlags()))
        {
            m_Context.error(
                "Buffer memory dependency stage scope is unsupported by the recording queue family");
            return false;
        }

        endRenderPass();
        if (anyBarriers() && !m_PendingBarriersAreMemoryDependencies)
            commitBarriersInternal();
        if (!m_StateTracker.addBufferMemoryDependency(
                buffer, dependency.state,
                dependency.shaderStagesBefore,
                dependency.shaderStagesAfter))
        {
            return false;
        }

        m_PendingBarriersAreMemoryDependencies = true;
        m_CurrentCmdBuf->referencedResources.push_back(buffer);
        return true;
    }

    bool CommandList::ensureTextureStateTracked(
        Texture* texture,
        TextureSubresourceSet subresources,
        const GraphResourceState& exactState)
    {
        if (!m_IsRecording)
        {
            m_Context.error("Graph texture state must be ensured on an open command list");
            return false;
        }
        if (!texture || !texture->belongsTo(m_Context))
        {
            m_Context.error("Graph texture state texture belongs to a different device");
            return false;
        }
        if (!texture->managed)
        {
            m_Context.error("Graph texture state requires a Vulkan-managed texture");
            return false;
        }
        if (texture->desc.keepInitialState)
        {
            m_Context.error("Graph texture state textures must disable keepInitialState");
            return false;
        }
        if (texture->permanentState != ResourceStates::Unknown
            || m_StateTracker.hasPendingPermanentTextureState(texture))
        {
            m_Context.error("Graph texture state cannot target a permanent-state texture");
            return false;
        }
        if (!validateGraphState(
                m_Context, exactState, true, &texture->desc,
                "Graph texture seed"))
        {
            return false;
        }

        if (!isValidGraphTextureRange(texture, subresources))
        {
            m_Context.error("Graph texture seed range is empty or out of bounds");
            return false;
        }
        subresources = subresources.resolve(texture->desc, false);
        if (!isValidResolvedTextureRange(texture, subresources))
        {
            m_Context.error("Graph texture seed range is empty or out of bounds");
            return false;
        }
        if (isTextureRangeReleased(texture, subresources))
        {
            reportReleasedResourceUse("texture", texture->desc.debugName);
            return false;
        }

        Queue* recordingQueue = m_Device->getQueue(m_CommandListParameters.queueType);
        const ResourceStateMapping mapping = convertTextureState(
            exactState.state, texture->desc, exactState.shaderStages);
        if (!recordingQueue
            || !detail::isWaitStageMaskSupported(
                mapping.stageFlags, recordingQueue->getQueueFlags()))
        {
            m_Context.error("Graph texture seed stage scope is unsupported by the recording queue family");
            return false;
        }

        bool sawUnknown = false;
        bool sawKnown = false;
        for (ArraySlice arraySlice = subresources.baseArraySlice;
             arraySlice < subresources.baseArraySlice + subresources.numArraySlices;
             ++arraySlice)
        {
            for (MipLevel mipLevel = subresources.baseMipLevel;
                 mipLevel < subresources.baseMipLevel + subresources.numMipLevels;
                 ++mipLevel)
            {
                const ResourceStates tracked = m_StateTracker.getTextureSubresourceState(
                    texture, arraySlice, mipLevel);
                if (tracked == ResourceStates::Unknown)
                {
                    sawUnknown = true;
                }
                else
                {
                    sawKnown = true;
                    if (tracked != exactState.state
                        || m_StateTracker.getTextureSubresourceShaderStages(
                            texture, arraySlice, mipLevel)
                            != exactState.shaderStages)
                    {
                        m_Context.error("Graph texture seed does not match the exact tracked state and stages");
                        return false;
                    }
                }
            }
        }
        if (sawUnknown && sawKnown)
        {
            m_Context.error("Graph texture seed range mixes known and unknown subresources");
            return false;
        }
        if (sawUnknown)
        {
            m_StateTracker.beginTrackingTextureState(
                texture, subresources, exactState.state,
                exactState.shaderStages);
        }
        m_CurrentCmdBuf->referencedResources.push_back(texture);
        return true;
    }

    bool CommandList::ensureBufferStateTracked(
        Buffer* buffer,
        const GraphResourceState& exactState)
    {
        if (!m_IsRecording)
        {
            m_Context.error("Graph buffer state must be ensured on an open command list");
            return false;
        }
        if (!buffer || !buffer->belongsTo(m_Context))
        {
            m_Context.error("Graph buffer state buffer belongs to a different device");
            return false;
        }
        if (!buffer->managed)
        {
            m_Context.error("Graph buffer state requires a Vulkan-managed buffer");
            return false;
        }
        if (buffer->desc.keepInitialState)
        {
            m_Context.error("Graph buffer state buffers must disable keepInitialState");
            return false;
        }
        if (buffer->permanentState != ResourceStates::Unknown
            || m_StateTracker.hasPendingPermanentBufferState(buffer))
        {
            m_Context.error("Graph buffer state cannot target a permanent-state buffer");
            return false;
        }
        if (buffer->desc.isVolatile
            || buffer->desc.cpuAccess != CpuAccessMode::None)
        {
            m_Context.error("Graph buffer state does not support volatile or CPU-visible buffers");
            return false;
        }
        if (m_ReleasedBuffers.find(buffer) != m_ReleasedBuffers.end())
        {
            reportReleasedResourceUse("buffer", buffer->desc.debugName);
            return false;
        }
        if (!validateGraphState(
                m_Context, exactState, false, nullptr,
                "Graph buffer seed"))
        {
            return false;
        }

        Queue* recordingQueue = m_Device->getQueue(m_CommandListParameters.queueType);
        const ResourceStateMapping mapping = convertResourceState(
            exactState.state, false, false, false,
            exactState.shaderStages);
        if (!recordingQueue
            || !detail::isWaitStageMaskSupported(
                mapping.stageFlags, recordingQueue->getQueueFlags()))
        {
            m_Context.error("Graph buffer seed stage scope is unsupported by the recording queue family");
            return false;
        }

        const ResourceStates tracked = m_StateTracker.getBufferState(buffer);
        if (tracked == ResourceStates::Unknown)
        {
            m_StateTracker.beginTrackingBufferState(
                buffer, exactState.state, exactState.shaderStages);
            m_CurrentCmdBuf->referencedResources.push_back(buffer);
            return true;
        }
        if (tracked != exactState.state
            || m_StateTracker.getBufferShaderStages(buffer)
                != exactState.shaderStages)
        {
            m_Context.error("Graph buffer seed does not match the exact tracked state and stages");
            return false;
        }
        m_CurrentCmdBuf->referencedResources.push_back(buffer);
        return true;
    }

    bool CommandList::verifyTextureStateTracked(
        Texture* texture,
        TextureSubresourceSet subresources,
        const GraphResourceState& exactState)
    {
        if (!m_IsRecording)
        {
            m_Context.error("Graph texture state must be verified on an open command list");
            return false;
        }
        if (!texture || !texture->belongsTo(m_Context))
        {
            m_Context.error("Graph texture verification texture belongs to a different device");
            return false;
        }
        if (!texture->managed)
        {
            m_Context.error("Graph texture verification requires a Vulkan-managed texture");
            return false;
        }
        if (texture->desc.keepInitialState)
        {
            m_Context.error("Graph texture verification textures must disable keepInitialState");
            return false;
        }
        if (texture->permanentState != ResourceStates::Unknown
            || m_StateTracker.hasPendingPermanentTextureState(texture))
        {
            m_Context.error("Graph texture verification cannot target a permanent-state texture");
            return false;
        }
        if (!validateGraphState(
                m_Context, exactState, true, &texture->desc,
                "Graph texture verification"))
        {
            return false;
        }

        if (!isValidGraphTextureRange(texture, subresources))
        {
            m_Context.error("Graph texture verification range is empty or out of bounds");
            return false;
        }
        subresources = subresources.resolve(texture->desc, false);
        if (!isValidResolvedTextureRange(texture, subresources))
        {
            m_Context.error("Graph texture verification range is empty or out of bounds");
            return false;
        }
        if (isTextureRangeReleased(texture, subresources))
        {
            reportReleasedResourceUse("texture", texture->desc.debugName);
            return false;
        }

        Queue* recordingQueue = m_Device->getQueue(m_CommandListParameters.queueType);
        const ResourceStateMapping mapping = convertTextureState(
            exactState.state, texture->desc, exactState.shaderStages);
        if (!recordingQueue
            || !detail::isWaitStageMaskSupported(
                mapping.stageFlags, recordingQueue->getQueueFlags()))
        {
            m_Context.error("Graph texture verification stage scope is unsupported by the recording queue family");
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
                const ResourceStates tracked =
                    m_StateTracker.getTextureSubresourceState(
                        texture, arraySlice, mipLevel);
                if (tracked == ResourceStates::Unknown)
                {
                    m_Context.error("Graph texture verification requires every addressed subresource state to be known");
                    return false;
                }
                if (tracked != exactState.state
                    || m_StateTracker.getTextureSubresourceShaderStages(
                        texture, arraySlice, mipLevel)
                        != exactState.shaderStages)
                {
                    m_Context.error("Graph texture verification does not match every exact tracked subresource state and stage scope");
                    return false;
                }
            }
        }

        m_CurrentCmdBuf->referencedResources.push_back(texture);
        return true;
    }

    bool CommandList::verifyBufferStateTracked(
        Buffer* buffer,
        const GraphResourceState& exactState)
    {
        if (!m_IsRecording)
        {
            m_Context.error("Graph buffer state must be verified on an open command list");
            return false;
        }
        if (!buffer || !buffer->belongsTo(m_Context))
        {
            m_Context.error("Graph buffer verification buffer belongs to a different device");
            return false;
        }
        if (!buffer->managed)
        {
            m_Context.error("Graph buffer verification requires a Vulkan-managed buffer");
            return false;
        }
        if (buffer->desc.keepInitialState)
        {
            m_Context.error("Graph buffer verification buffers must disable keepInitialState");
            return false;
        }
        if (buffer->permanentState != ResourceStates::Unknown
            || m_StateTracker.hasPendingPermanentBufferState(buffer))
        {
            m_Context.error("Graph buffer verification cannot target a permanent-state buffer");
            return false;
        }
        if (buffer->desc.isVolatile
            || buffer->desc.cpuAccess != CpuAccessMode::None)
        {
            m_Context.error("Graph buffer verification does not support volatile or CPU-visible buffers");
            return false;
        }
        if (m_ReleasedBuffers.find(buffer) != m_ReleasedBuffers.end())
        {
            reportReleasedResourceUse("buffer", buffer->desc.debugName);
            return false;
        }
        if (!validateGraphState(
                m_Context, exactState, false, nullptr,
                "Graph buffer verification"))
        {
            return false;
        }

        Queue* recordingQueue = m_Device->getQueue(m_CommandListParameters.queueType);
        const ResourceStateMapping mapping = convertResourceState(
            exactState.state, false, false, false,
            exactState.shaderStages);
        if (!recordingQueue
            || !detail::isWaitStageMaskSupported(
                mapping.stageFlags, recordingQueue->getQueueFlags()))
        {
            m_Context.error("Graph buffer verification stage scope is unsupported by the recording queue family");
            return false;
        }

        const ResourceStates tracked = m_StateTracker.getBufferState(buffer);
        if (tracked == ResourceStates::Unknown)
        {
            m_Context.error("Graph buffer verification requires the tracked state to be known");
            return false;
        }
        const ShaderType trackedShaderStages =
            m_StateTracker.getBufferShaderStages(buffer);
        if (tracked != exactState.state
            || trackedShaderStages != exactState.shaderStages)
        {
            m_Context.error(
                "Graph buffer verification mismatch for '"
                + buffer->desc.debugName
                + "': tracked state=" + std::to_string(uint32_t(tracked))
                + " stages=" + std::to_string(uint32_t(trackedShaderStages))
                + ", expected state=" + std::to_string(uint32_t(exactState.state))
                + " stages=" + std::to_string(uint32_t(exactState.shaderStages)));
            return false;
        }

        m_CurrentCmdBuf->referencedResources.push_back(buffer);
        return true;
    }

    bool CommandList::recordTextureStateTransition(
        Texture* texture,
        TextureSubresourceSet subresources,
        const GraphResourceStateTransition& transition)
    {
        if (!m_IsRecording)
        {
            m_Context.error("Graph texture transitions require an open command list");
            return false;
        }
        if (!texture || !texture->belongsTo(m_Context))
        {
            m_Context.error("Graph texture transition texture belongs to a different device");
            return false;
        }
        if (!texture->managed || texture->desc.keepInitialState)
        {
            m_Context.error("Graph texture transitions require a managed texture with keepInitialState disabled");
            return false;
        }
        if (texture->permanentState != ResourceStates::Unknown
            || m_StateTracker.hasPendingPermanentTextureState(texture))
        {
            m_Context.error("Graph texture transition cannot target a permanent-state texture");
            return false;
        }
        const GraphResourceState source {
            transition.stateBefore, transition.shaderStagesBefore };
        const GraphResourceState destination {
            transition.stateAfter, transition.shaderStagesAfter };
        if (!validateGraphState(
                m_Context, source, true, &texture->desc,
                "Graph texture transition source")
            || !validateGraphState(
                m_Context, destination, true, &texture->desc,
                "Graph texture transition destination"))
        {
            return false;
        }
        if (transition.stateBefore == transition.stateAfter)
        {
            m_Context.error("Graph texture transition must change logical state");
            return false;
        }

        if (!isValidGraphTextureRange(texture, subresources))
        {
            m_Context.error("Graph texture transition range is empty or out of bounds");
            return false;
        }
        subresources = subresources.resolve(texture->desc, false);
        if (!isValidResolvedTextureRange(texture, subresources))
        {
            m_Context.error("Graph texture transition range is empty or out of bounds");
            return false;
        }
        if (isTextureRangeReleased(texture, subresources))
        {
            reportReleasedResourceUse("texture", texture->desc.debugName);
            return false;
        }

        Queue* recordingQueue = m_Device->getQueue(m_CommandListParameters.queueType);
        const ResourceStateMapping after = convertTextureState(
            transition.stateAfter, texture->desc,
            transition.shaderStagesAfter);
        if (after.imageLayout == vk::ImageLayout::eUndefined
            || !recordingQueue
            || !detail::isWaitStageMaskSupported(
                after.stageFlags, recordingQueue->getQueueFlags()))
        {
            m_Context.error("Graph texture transition destination is unsupported by the recording queue family");
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
                        texture, arraySlice, mipLevel)
                        != transition.stateBefore
                    || m_StateTracker.getTextureSubresourceShaderStages(
                        texture, arraySlice, mipLevel)
                        != transition.shaderStagesBefore)
                {
                    m_Context.error("Graph texture transition source does not match every exact tracked subresource state and stage scope");
                    return false;
                }
                const ResourceStateMapping before = convertTextureState(
                    transition.stateBefore, texture->desc,
                    transition.shaderStagesBefore);
                if (!detail::isWaitStageMaskSupported(
                        before.stageFlags, recordingQueue->getQueueFlags()))
                {
                    m_Context.error("Graph texture transition source is unsupported by the recording queue family");
                    return false;
                }
            }
        }

        bool overlapsPendingTransition = false;
        for (const TextureBarrier& barrier : m_StateTracker.getTextureBarriers())
        {
            if (barrier.texture != texture)
                continue;
            if (barrier.entireTexture
                || (barrier.mipLevel >= subresources.baseMipLevel
                    && barrier.mipLevel
                        < subresources.baseMipLevel + subresources.numMipLevels
                    && barrier.arraySlice >= subresources.baseArraySlice
                    && barrier.arraySlice
                        < subresources.baseArraySlice + subresources.numArraySlices))
            {
                overlapsPendingTransition = true;
                break;
            }
        }
        if (overlapsPendingTransition)
        {
            endRenderPass();
            commitBarriersInternal();
        }

        requireTextureState(
            texture, subresources, transition.stateAfter,
            transition.shaderStagesAfter, false);
        for (ArraySlice arraySlice = subresources.baseArraySlice;
             arraySlice < subresources.baseArraySlice + subresources.numArraySlices;
             ++arraySlice)
        {
            for (MipLevel mipLevel = subresources.baseMipLevel;
                 mipLevel < subresources.baseMipLevel + subresources.numMipLevels;
                 ++mipLevel)
            {
                if (m_StateTracker.getTextureSubresourceState(
                        texture, arraySlice, mipLevel)
                        != transition.stateAfter
                    || m_StateTracker.getTextureSubresourceShaderStages(
                        texture, arraySlice, mipLevel)
                        != transition.shaderStagesAfter)
                {
                    m_Context.error("Graph texture transition did not produce its exact destination state and stages");
                    return false;
                }
            }
        }
        m_CurrentCmdBuf->referencedResources.push_back(texture);
        return true;
    }

    bool CommandList::recordBufferStateTransition(
        Buffer* buffer,
        const GraphResourceStateTransition& transition)
    {
        if (!m_IsRecording)
        {
            m_Context.error("Graph buffer transitions require an open command list");
            return false;
        }
        if (!buffer || !buffer->belongsTo(m_Context))
        {
            m_Context.error("Graph buffer transition buffer belongs to a different device");
            return false;
        }
        if (!buffer->managed || buffer->desc.keepInitialState)
        {
            m_Context.error("Graph buffer transitions require a managed buffer with keepInitialState disabled");
            return false;
        }
        if (buffer->permanentState != ResourceStates::Unknown
            || m_StateTracker.hasPendingPermanentBufferState(buffer))
        {
            m_Context.error("Graph buffer transition cannot target a permanent-state buffer");
            return false;
        }
        if (buffer->desc.isVolatile
            || buffer->desc.cpuAccess != CpuAccessMode::None)
        {
            m_Context.error("Graph buffer transition does not support volatile or CPU-visible buffers");
            return false;
        }
        if (m_ReleasedBuffers.find(buffer) != m_ReleasedBuffers.end())
        {
            reportReleasedResourceUse("buffer", buffer->desc.debugName);
            return false;
        }
        const GraphResourceState source {
            transition.stateBefore, transition.shaderStagesBefore };
        const GraphResourceState destination {
            transition.stateAfter, transition.shaderStagesAfter };
        if (!validateGraphState(
                m_Context, source, false, nullptr,
                "Graph buffer transition source")
            || !validateGraphState(
                m_Context, destination, false, nullptr,
                "Graph buffer transition destination"))
        {
            return false;
        }
        if (transition.stateBefore == transition.stateAfter)
        {
            m_Context.error("Graph buffer transition must change logical state");
            return false;
        }
        if (m_StateTracker.getBufferState(buffer) != transition.stateBefore
            || m_StateTracker.getBufferShaderStages(buffer)
                != transition.shaderStagesBefore)
        {
            m_Context.error("Graph buffer transition source does not match the exact tracked state and stages");
            return false;
        }

        Queue* recordingQueue = m_Device->getQueue(m_CommandListParameters.queueType);
        const ResourceStateMapping before = convertResourceState(
            transition.stateBefore, false, false, false,
            transition.shaderStagesBefore);
        const ResourceStateMapping after = convertResourceState(
            transition.stateAfter, false, false, false,
            transition.shaderStagesAfter);
        if (!recordingQueue
            || !detail::isWaitStageMaskSupported(
                before.stageFlags, recordingQueue->getQueueFlags())
            || !detail::isWaitStageMaskSupported(
                after.stageFlags, recordingQueue->getQueueFlags()))
        {
            m_Context.error("Graph buffer transition stage scope is unsupported by the recording queue family");
            return false;
        }

        const bool overlapsPendingTransition = std::any_of(
            m_StateTracker.getBufferBarriers().begin(),
            m_StateTracker.getBufferBarriers().end(),
            [buffer](const BufferBarrier& barrier) {
                return barrier.buffer == buffer;
            });
        if (overlapsPendingTransition)
        {
            endRenderPass();
            commitBarriersInternal();
        }

        requireBufferState(
            buffer, transition.stateAfter,
            transition.shaderStagesAfter);
        if (m_StateTracker.getBufferState(buffer) != transition.stateAfter
            || m_StateTracker.getBufferShaderStages(buffer)
                != transition.shaderStagesAfter)
        {
            m_Context.error("Graph buffer transition did not produce its exact destination state and stages");
            return false;
        }
        m_CurrentCmdBuf->referencedResources.push_back(buffer);
        return true;
    }

    bool Device::addTextureMemoryDependency(
        ICommandList* commandList,
        ITexture* texture,
        TextureSubresourceSet subresources,
        const MemoryDependencyDesc& dependency)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Texture* vulkanTexture = dynamic_cast<Texture*>(texture);
        if (!vulkanCommandList || !vulkanTexture)
        {
            m_Context.error(
                "Texture memory dependency requires Vulkan command-list and texture objects");
            return false;
        }
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error(
                "Texture memory dependency command list belongs to a different device");
            return false;
        }
        return vulkanCommandList->recordTextureMemoryDependency(
            vulkanTexture, subresources, dependency);
    }

    bool Device::addBufferMemoryDependency(
        ICommandList* commandList,
        IBuffer* buffer,
        const MemoryDependencyDesc& dependency)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Buffer* vulkanBuffer = dynamic_cast<Buffer*>(buffer);
        if (!vulkanCommandList || !vulkanBuffer)
        {
            m_Context.error(
                "Buffer memory dependency requires Vulkan command-list and buffer objects");
            return false;
        }
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error(
                "Buffer memory dependency command list belongs to a different device");
            return false;
        }
        return vulkanCommandList->recordBufferMemoryDependency(
            vulkanBuffer, dependency);
    }

    bool Device::ensureTextureStateTracked(
        ICommandList* commandList,
        ITexture* texture,
        TextureSubresourceSet subresources,
        const GraphResourceState& exactState)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Texture* vulkanTexture = dynamic_cast<Texture*>(texture);
        if (!vulkanCommandList || !vulkanTexture)
        {
            m_Context.error("Graph texture state requires Vulkan command-list and texture objects");
            return false;
        }
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error("Graph texture state command list belongs to a different device");
            return false;
        }
        return vulkanCommandList->ensureTextureStateTracked(
            vulkanTexture, subresources, exactState);
    }

    bool Device::ensureBufferStateTracked(
        ICommandList* commandList,
        IBuffer* buffer,
        const GraphResourceState& exactState)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Buffer* vulkanBuffer = dynamic_cast<Buffer*>(buffer);
        if (!vulkanCommandList || !vulkanBuffer)
        {
            m_Context.error("Graph buffer state requires Vulkan command-list and buffer objects");
            return false;
        }
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error("Graph buffer state command list belongs to a different device");
            return false;
        }
        return vulkanCommandList->ensureBufferStateTracked(
            vulkanBuffer, exactState);
    }

    bool Device::verifyTextureStateTracked(
        ICommandList* commandList,
        ITexture* texture,
        TextureSubresourceSet subresources,
        const GraphResourceState& exactState)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Texture* vulkanTexture = dynamic_cast<Texture*>(texture);
        if (!vulkanCommandList || !vulkanTexture)
        {
            m_Context.error("Graph texture verification requires Vulkan command-list and texture objects");
            return false;
        }
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error("Graph texture verification command list belongs to a different device");
            return false;
        }
        return vulkanCommandList->verifyTextureStateTracked(
            vulkanTexture, subresources, exactState);
    }

    bool Device::verifyBufferStateTracked(
        ICommandList* commandList,
        IBuffer* buffer,
        const GraphResourceState& exactState)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Buffer* vulkanBuffer = dynamic_cast<Buffer*>(buffer);
        if (!vulkanCommandList || !vulkanBuffer)
        {
            m_Context.error("Graph buffer verification requires Vulkan command-list and buffer objects");
            return false;
        }
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error("Graph buffer verification command list belongs to a different device");
            return false;
        }
        return vulkanCommandList->verifyBufferStateTracked(
            vulkanBuffer, exactState);
    }

    bool Device::transitionTextureState(
        ICommandList* commandList,
        ITexture* texture,
        TextureSubresourceSet subresources,
        const GraphResourceStateTransition& transition)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Texture* vulkanTexture = dynamic_cast<Texture*>(texture);
        if (!vulkanCommandList || !vulkanTexture)
        {
            m_Context.error("Graph texture transition requires Vulkan command-list and texture objects");
            return false;
        }
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error("Graph texture transition command list belongs to a different device");
            return false;
        }
        return vulkanCommandList->recordTextureStateTransition(
            vulkanTexture, subresources, transition);
    }

    bool Device::transitionBufferState(
        ICommandList* commandList,
        IBuffer* buffer,
        const GraphResourceStateTransition& transition)
    {
        CommandList* vulkanCommandList = dynamic_cast<CommandList*>(commandList);
        Buffer* vulkanBuffer = dynamic_cast<Buffer*>(buffer);
        if (!vulkanCommandList || !vulkanBuffer)
        {
            m_Context.error("Graph buffer transition requires Vulkan command-list and buffer objects");
            return false;
        }
        if (vulkanCommandList->getDevice() != this)
        {
            m_Context.error("Graph buffer transition command list belongs to a different device");
            return false;
        }
        return vulkanCommandList->recordBufferStateTransition(
            vulkanBuffer, transition);
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

        if (m_PendingBarriersAreMemoryDependencies)
        {
            endRenderPass();
            commitBarriersInternal();
        }

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

        if (m_PendingBarriersAreMemoryDependencies)
        {
            endRenderPass();
            commitBarriersInternal();
        }

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
