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
#include <algorithm>
#include <sstream>

namespace nvrhi::vulkan
{

    BindingLayoutHandle Device::createBindingLayout(const BindingLayoutDesc& desc)
    {
        BindingLayout* ret = new BindingLayout(m_Context, desc);

        ret->bake();

        // f111-pig: a binding set keeps one volatileConstantBuffers entry, and a
        // bind one dynamic offset, per dynamic uniform buffer element, array
        // sizes counted, in arrays of c_MaxVolatileConstantBuffersPerLayout.
        if (ret->dynamicBufferElements.size() > c_MaxVolatileConstantBuffersPerLayout)
        {
            std::stringstream ss;
            ss << "Binding layout has " << ret->dynamicBufferElements.size()
               << " volatile constant buffer elements (array sizes counted); the limit is "
               << c_MaxVolatileConstantBuffersPerLayout;
            m_Context.error(ss.str());
            delete ret;
            return nullptr;
        }

        // f111-pig: the longest run of null writes a set of this layout can need.
        uint32_t longestBinding = 0;
        for (const vk::DescriptorSetLayoutBinding& layoutBinding : ret->vulkanLayoutBindings)
            longestBinding = std::max(longestBinding, layoutBinding.descriptorCount);
        m_DescriptorSetAllocator.reserveNullDescriptors(longestBinding);

        return BindingLayoutHandle::Create(ret);
    }

    BindingLayoutHandle Device::createBindlessLayout(const BindlessLayoutDesc& desc)
    {
        BindingLayout* ret = new BindingLayout(m_Context, desc);

        ret->bake();

        return BindingLayoutHandle::Create(ret);
    }

    static uint32_t getRegisterOffsetForResourceType(VulkanBindingOffsets const& bindingOffsets, ResourceType type)
    {
        switch (type)
        {
        case ResourceType::Texture_SRV:
        case ResourceType::TypedBuffer_SRV:
        case ResourceType::StructuredBuffer_SRV:
        case ResourceType::RawBuffer_SRV:
        case ResourceType::RayTracingAccelStruct:
            return bindingOffsets.shaderResource;

        case ResourceType::Texture_UAV:
        case ResourceType::TypedBuffer_UAV:
        case ResourceType::StructuredBuffer_UAV:
        case ResourceType::RawBuffer_UAV:
            return bindingOffsets.unorderedAccess;

        case ResourceType::ConstantBuffer:
        case ResourceType::VolatileConstantBuffer:
        case ResourceType::PushConstants:
            return bindingOffsets.constantBuffer;
            break;

        case ResourceType::Sampler:
            return bindingOffsets.sampler;

        default:
            utils::InvalidEnum();
            return 0;
        }        
    }

    BindingLayout::BindingLayout(const VulkanContext& context, const BindingLayoutDesc& _desc)
        : desc(_desc)
        , isBindless(false)
        , m_Context(context)
    {
        vk::ShaderStageFlagBits shaderStageFlags = convertShaderTypeToShaderStageFlagBits(desc.visibility);

        // iterate over all binding types and add to map
        for (const BindingLayoutItem& binding : desc.bindings)
        {
            if (binding.type == ResourceType::PushConstants)
            {
                // Don't need any descriptors for the push constants
                continue;
            }

            vk::DescriptorType const descriptorType = convertResourceType(binding.type);
            uint32_t const descriptorCount = binding.size;
            uint32_t const registerOffset = getRegisterOffsetForResourceType(_desc.bindingOffsets, binding.type);

            const uint32_t bindingLocation = registerOffset + binding.slot;

            vk::DescriptorSetLayoutBinding descriptorSetLayoutBinding = vk::DescriptorSetLayoutBinding()
                .setBinding(bindingLocation)
                .setDescriptorCount(descriptorCount)
                .setDescriptorType(descriptorType)
                .setStageFlags(shaderStageFlags);

            vulkanLayoutBindings.push_back(descriptorSetLayoutBinding);
        }
    }

    BindingLayout::BindingLayout(const VulkanContext& context, const BindlessLayoutDesc& _desc)
        : bindlessDesc(_desc)
        , isBindless(true)
        , m_Context(context)
    {
        desc.visibility = bindlessDesc.visibility;
        vk::ShaderStageFlagBits shaderStageFlags = convertShaderTypeToShaderStageFlagBits(bindlessDesc.visibility);
        uint32_t bindingPoint = 0;
        uint32_t arraySize = bindlessDesc.maxCapacity;
        
        if (bindlessDesc.layoutType != BindlessLayoutDesc::LayoutType::Immutable)
        {
            if (!m_Context.extensions.EXT_mutable_descriptor_type)
            {
                m_Context.error("Mutable descriptor types are not supported by this device. VK_EXT_mutable_descriptor_type extension is required for mutable bindless layouts.");
            }

            if (bindlessDesc.registerSpaces.size() > 0)
            {
                m_Context.error("Mutable descriptor sets cannot specify register spaces");
            }

            vk::DescriptorSetLayoutBinding descriptorSetLayoutBinding = vk::DescriptorSetLayoutBinding()
                .setBinding(bindingPoint)
                .setDescriptorCount(arraySize)
                .setDescriptorType(vk::DescriptorType::eMutableEXT)
                .setStageFlags(shaderStageFlags);

            vulkanLayoutBindings.push_back(descriptorSetLayoutBinding);
        }
        else
        {
            // iterate over all binding types and add to map
            for (const BindingLayoutItem& space : bindlessDesc.registerSpaces)
            {
                vk::DescriptorType const descriptorType = convertResourceType(space.type);
                
                if (space.type == ResourceType::VolatileConstantBuffer)
                    m_Context.error("Volatile constant buffers are not supported in bindless layouts");

                vk::DescriptorSetLayoutBinding descriptorSetLayoutBinding = vk::DescriptorSetLayoutBinding()
                    .setBinding(bindingPoint)
                    .setDescriptorCount(arraySize)
                    .setDescriptorType(descriptorType)
                    .setStageFlags(shaderStageFlags);

                vulkanLayoutBindings.push_back(descriptorSetLayoutBinding);

                ++bindingPoint;
            }
        }
    }

    Object BindingLayout::getNativeObject(ObjectType objectType)
    {
        switch (objectType)
        {
        case ObjectTypes::VK_DescriptorSetLayout:
            return Object(descriptorSetLayout);
        default:
            return nullptr;
        }
    }

    vk::Result BindingLayout::bake()
    {
        // create the descriptor set layout object
        
        auto descriptorSetLayoutInfo = vk::DescriptorSetLayoutCreateInfo()
            .setBindingCount(uint32_t(vulkanLayoutBindings.size()))
            .setPBindings(vulkanLayoutBindings.data());

        std::vector<vk::DescriptorBindingFlags> bindFlag(vulkanLayoutBindings.size(), vk::DescriptorBindingFlagBits::ePartiallyBound);

        if (isBindless && bindlessDesc.enableUpdateAfterBind)
        {
            for (vk::DescriptorBindingFlags& flags : bindFlag)
                flags |= vk::DescriptorBindingFlagBits::eUpdateAfterBind;

            // Vulkan requires layouts containing UPDATE_AFTER_BIND bindings to
            // be allocated from an UPDATE_AFTER_BIND descriptor pool.
            descriptorSetLayoutInfo.setFlags(
                vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
        }

        if (isBindless && bindlessDesc.enableUpdateUnusedWhilePending)
        {
            for (vk::DescriptorBindingFlags& flags : bindFlag)
                flags |= vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
        }

        auto extendedInfo = vk::DescriptorSetLayoutBindingFlagsCreateInfo()
            .setBindingCount(uint32_t(vulkanLayoutBindings.size()))
            .setPBindingFlags(bindFlag.data());

        vk::DescriptorType cbvSrvUavTypes[] =
        {
            vk::DescriptorType::eSampledImage,
            vk::DescriptorType::eStorageImage,
            vk::DescriptorType::eUniformTexelBuffer,
            vk::DescriptorType::eStorageTexelBuffer,
            vk::DescriptorType::eUniformBuffer,
            vk::DescriptorType::eStorageBuffer
        };

        vk::DescriptorType counterTypes[] =
        {
            vk::DescriptorType::eStorageBuffer
        };

        vk::DescriptorType samplerTypes[] =
        {
            vk::DescriptorType::eSampler
        };

        auto cbvSrvUavTypesList = vk::MutableDescriptorTypeListEXT()
            .setDescriptorTypeCount(uint32_t(sizeof(cbvSrvUavTypes) / sizeof(cbvSrvUavTypes[0])))
            .setPDescriptorTypes(cbvSrvUavTypes);

        auto counterTypesList = vk::MutableDescriptorTypeListEXT()
            .setDescriptorTypeCount(uint32_t(sizeof(counterTypes) / sizeof(counterTypes[0])))
            .setPDescriptorTypes(counterTypes);

        auto samplerTypesList = vk::MutableDescriptorTypeListEXT()
            .setDescriptorTypeCount(uint32_t(sizeof(samplerTypes) / sizeof(samplerTypes[0])))
            .setPDescriptorTypes(samplerTypes);

        auto pMutableDescriptorTypeLists =
            bindlessDesc.layoutType == BindlessLayoutDesc::LayoutType::MutableCounters ? &counterTypesList :
            bindlessDesc.layoutType == BindlessLayoutDesc::LayoutType::MutableSampler ? &samplerTypesList :
            &cbvSrvUavTypesList;

        auto mutableDescriptorTypeCreateInfo = vk::MutableDescriptorTypeCreateInfoEXT()
            .setMutableDescriptorTypeListCount(1)
            .setPMutableDescriptorTypeLists(pMutableDescriptorTypeLists)
            .setPNext(&extendedInfo);

        if (isBindless)
        {
            if (bindlessDesc.layoutType != BindlessLayoutDesc::LayoutType::Immutable)
            {
                descriptorSetLayoutInfo.setPNext(&mutableDescriptorTypeCreateInfo);
            }
            else
            {
                descriptorSetLayoutInfo.setPNext(&extendedInfo);
            }
        }

        const vk::Result res = m_Context.device.createDescriptorSetLayout(&descriptorSetLayoutInfo,
                                                                        m_Context.allocationCallbacks,
                                                                        &descriptorSetLayout);
        CHECK_VK_RETURN(res)

        // count the number of descriptors required per type
        std::unordered_map<vk::DescriptorType, uint32_t> poolSizeMap;
        for (auto layoutBinding : vulkanLayoutBindings)
        {
            if (poolSizeMap.find(layoutBinding.descriptorType) == poolSizeMap.end())
            {
                poolSizeMap[layoutBinding.descriptorType] = 0;
            }

            poolSizeMap[layoutBinding.descriptorType] += layoutBinding.descriptorCount;
        }

        // compute descriptor pool size info
        for (auto poolSizeIter : poolSizeMap)
        {
            if (poolSizeIter.second > 0)
            {
                descriptorPoolSizeInfo.push_back(vk::DescriptorPoolSize()
                    .setType(poolSizeIter.first)
                    .setDescriptorCount(poolSizeIter.second));
            }
        }

        // f111-pig: number the descriptors for createBindingSet's null fill,
        // count them by type for the pools, and list the dynamic uniform buffer
        // elements in dynamic offset order.
        if (!isBindless)
        {
            pooledDescriptorCounts = DescriptorSetAllocator::countPerType(descriptorPoolSizeInfo);
            bindingFirstDescriptor.reserve(vulkanLayoutBindings.size());
            for (const vk::DescriptorSetLayoutBinding& layoutBinding : vulkanLayoutBindings)
            {
                bindingFirstDescriptor.push_back(numDescriptors);
                numDescriptors += layoutBinding.descriptorCount;

                if (layoutBinding.descriptorType == vk::DescriptorType::eUniformBufferDynamic)
                {
                    for (uint32_t element = 0; element < layoutBinding.descriptorCount; ++element)
                        dynamicBufferElements.push_back({ layoutBinding.binding, element });
                }
            }
            std::sort(dynamicBufferElements.begin(), dynamicBufferElements.end(),
                [](const DynamicBufferElement& a, const DynamicBufferElement& b)
                {
                    return a.binding != b.binding ? a.binding < b.binding : a.arrayElement < b.arrayElement;
                });
        }

        return vk::Result::eSuccess;
    }

    BindingLayout::~BindingLayout()
    {
        // No binding set refers to this layout any more, so nothing else can
        // touch recycledDescriptorSets.
        if (!recycledDescriptorSets.empty())
            m_Context.descriptorSetAllocator->releaseRecycled(*this);

        if (descriptorSetLayout)
        {
            m_Context.device.destroyDescriptorSetLayout(descriptorSetLayout, m_Context.allocationCallbacks);
            descriptorSetLayout = vk::DescriptorSetLayout();
        }
    }

    static Texture::TextureSubresourceViewType getTextureViewType(Format bindingFormat, Format textureFormat)
    {
        Format format = (bindingFormat == Format::UNKNOWN) ? textureFormat : bindingFormat;

        const FormatInfo& formatInfo = getFormatInfo(format);

        if (formatInfo.hasDepth)
            return Texture::TextureSubresourceViewType::DepthOnly;
        else if (formatInfo.hasStencil)
            return Texture::TextureSubresourceViewType::StencilOnly;
        else
            return Texture::TextureSubresourceViewType::AllAspects;
    }

    bool Device::precreateTextureSrvView(const BindingSetItem& binding)
    {
        if (binding.type != ResourceType::Texture_SRV)
        {
            m_Context.error(
                "precreateTextureSrvView requires a Texture_SRV binding");
            return false;
        }

        Texture* texture = dynamic_cast<Texture*>(binding.resourceHandle);
        if (!texture || !texture->belongsTo(m_Context) || !texture->image)
        {
            m_Context.error(
                "precreateTextureSrvView requires a texture from this Vulkan device");
            return false;
        }

        const TextureSubresourceSet subresource =
            binding.subresources.resolve(texture->desc, false);
        if (subresource.numMipLevels == 0 || subresource.numArraySlices == 0)
        {
            m_Context.error(
                "precreateTextureSrvView subresources do not intersect the texture");
            return false;
        }

        const auto textureViewType =
            getTextureViewType(binding.format, texture->desc.format);
        const auto* view = texture->getSubresourceView(
            subresource, binding.dimension, binding.format,
            vk::ImageUsageFlagBits::eSampled, textureViewType);
        return view && view->view;
    }

    // f111-pig: DescriptorSetAllocator, the shared pools binding sets take
    // their descriptor sets from.
    //
    // Driver cost, measured with VK_EXT_device_memory_report on an RTX 5090,
    // driver 616.92 (2026-10-19):
    //  - Creating a VkDescriptorPool takes no device memory, whatever its
    //    maxSets or descriptor counts. The driver backs it as sets are
    //    allocated from it, in 64 KiB chunks of whole sets, each set taking its
    //    descriptor bytes rounded up to 128 (1-8 storage buffers took 128
    //    bytes, 9-16 took 256, 17-24 took 384; the descriptor-buffer sizes the
    //    driver reports, 4 B per image or sampler, 8 B per uniform buffer or
    //    accel struct, 16 B per storage or texel buffer, match every set
    //    measured, and dynamic uniform buffers take none).
    //  - A pool keeps its chunks until it is destroyed. Freeing or resetting
    //    sets returns none, and a FREE_DESCRIPTOR_SET pool with unused
    //    capacity places a new set there rather than in a freed hole (free
    //    every other set of 2,048 and allocate 1,024 more: +256 KiB).
    //  - FREE_DESCRIPTOR_SET_BIT changes none of these numbers.
    //  - Host memory is committed at creation: ~0.9 KiB per maxSets slot,
    //    independent of the descriptor counts.
    // So a pool per binding set costs a whole 64 KiB chunk for a set of a few
    // hundred bytes at most.
    //
    // General pools, not a pool family per layout: a pool's floor is one
    // chunk, so per-layout pools would cost at least 64 KiB for every layout
    // with a live set, where general pools cost the sets' bytes plus one
    // partly filled chunk per pool.
    //
    // Recycled per layout, not freed: the next binding set of a layout reuses
    // a destroyed one's descriptor set, so churn makes no pool allocation or
    // free and, since a freed hole is not refilled, grows no pool. Only a dying
    // layout frees its kept sets into their pools, which is what the pools'
    // FREE_DESCRIPTOR_SET_BIT is for. So a layout holds, for its whole life,
    // as many sets as it ever had alive at once, at 128-384 bytes a set (a
    // pool per set would cost 64 KiB a set). An application that retires its
    // command lists between bursts of binding set creation bounds that to the
    // sets in flight.
    //
    // Null fill: a reused set still holds its previous owner's descriptors and
    // a fresh one holds undefined ones, so createBindingSet writes null into
    // every descriptor the binding set does not, which a shader then reads as
    // bound to nothing (nullDescriptor) instead of a destroyed view. The
    // layout numbers its descriptors once (BindingLayout::bake), a bitset over
    // that numbering marks what the binding set wrote, and each run of
    // unwritten elements in a binding takes one write. A dynamic uniform
    // buffer written null is bound at dynamic offset 0, so the offsets
    // vkCmdBindDescriptorSets takes always match the layout's count.
    //
    // c_SetsPerPool, 512 set slots per pool. A pool's device memory is the
    // bytes of every set allocated into it since its creation or last reset,
    // with one partly filled chunk at the end, capped by its per-type
    // capacity: the driver places new sets in unused capacity rather than in
    // freed holes (allocate 400, free 200, allocate 200: about 600 sets'
    // worth), and a pool that outlives a dying layout is not emptied, so it is
    // not reset; a reset keeps the chunks for the sets that follow. Its host
    // memory is ~0.9 KiB per slot from creation. While pools fill without
    // frees, S live sets in pools of N slots carry at most 64 KiB x ceil(S / N)
    // of partly filled chunks, plus the newest pool's unused slots, under
    // 0.9 KiB x N of host memory; the sum is least near N = sqrt(64 S / 0.9),
    // 8.4 sqrt(S): 285-326 for the 1,150-1,500 live sets measured in the
    // engine (F-16 flight, dense_mixed). 512 is the power of two above,
    // favouring device memory, the budgeted one.
    //
    // Per-type capacity of a pool, in each type: the larger of
    // c_SetsPerPool x c_CapacityHeadroom x max(1, ceil(the average count of
    // that type over the sets allocated so far, the requesting set included))
    // and the requesting set's own count. So a pool takes 512 sets of up to
    // twice the running average in every type; a set far above the average
    // can use up a type sooner, and the counter check then skips that pool
    // without asking the driver. A headroom of 2 absorbs the drift of later
    // sets from the average; the floor of one per slot keeps a type no set
    // has used yet from forcing a new pool when its first set arrives.
    // Unused capacity costs nothing on the driver measured, in device or host
    // memory. A driver that backs a pool when it is created pays, per pool,
    // the sum over types of that capacity times the type's descriptor size;
    // the floor alone is 2 x 512 x 76 B = 76 KiB at the descriptor sizes this
    // driver reports.
    static constexpr uint64_t c_CapacityHeadroom = 2;

    // A null acceleration structure descriptor. Its count must equal the
    // write's descriptorCount, so accel-struct runs are written one element
    // at a time.
    static const vk::AccelerationStructureKHR c_NullAccelStruct {};
    static const vk::WriteDescriptorSetAccelerationStructureKHR c_NullAccelStructWrite =
        vk::WriteDescriptorSetAccelerationStructureKHR()
        .setAccelerationStructureCount(1)
        .setPAccelerationStructures(&c_NullAccelStruct);

    // Names an object through the pointer form, which returns its result
    // instead of throwing; the name is a debugging hint, so the result is
    // ignored. A null name clears it.
    static void setDescriptorObjectName(const VulkanContext& context, vk::ObjectType type,
        uint64_t handle, const char* name)
    {
        if (!context.extensions.EXT_debug_utils)
            return;
        const auto nameInfo = vk::DebugUtilsObjectNameInfoEXT()
            .setObjectType(type)
            .setObjectHandle(handle)
            .setPObjectName(name);
        (void)context.device.setDebugUtilsObjectNameEXT(&nameInfo);
    }

    vk::Result DescriptorSetAllocator::init()
    {
        // nullDescriptor covers images, buffers, texel buffers and accel
        // structs; a sampler descriptor must name a sampler.
        const auto samplerInfo = vk::SamplerCreateInfo();
        const vk::Result res = m_Context.device.createSampler(&samplerInfo, m_Context.allocationCallbacks, &m_NullSampler);
        if (res != vk::Result::eSuccess)
            m_NullSampler = vk::Sampler();
        return res;
    }

    void DescriptorSetAllocator::reserveNullDescriptors(uint32_t count)
    {
        std::lock_guard lockGuard(m_Mutex);
        const NullDescriptorArrays* current = m_NullDescriptors.load(std::memory_order_relaxed);
        const size_t currentCount = current ? current->images.size() : 0;
        if (count <= currentCount)
            return;

        // At least doubling keeps every generation kept alive, together,
        // under twice the newest.
        const size_t newCount = std::max<size_t>(count, currentCount * 2);
        auto arrays = std::make_unique<NullDescriptorArrays>();
        arrays->images.assign(newCount, vk::DescriptorImageInfo().setImageLayout(vk::ImageLayout::eGeneral));
        arrays->samplers.assign(newCount, vk::DescriptorImageInfo().setSampler(m_NullSampler));
        arrays->buffers.assign(newCount, vk::DescriptorBufferInfo().setOffset(0).setRange(VK_WHOLE_SIZE));
        arrays->bufferViews.assign(newCount, vk::BufferView());
        m_NullDescriptors.store(arrays.get(), std::memory_order_release);
        m_NullDescriptorGenerations.push_back(std::move(arrays));
    }

    DescriptorSetAllocator::~DescriptorSetAllocator()
    {
        for (const Pool& pool : m_Pools)
            m_Context.device.destroyDescriptorPool(pool.pool, m_Context.allocationCallbacks);
        if (m_NullSampler)
            m_Context.device.destroySampler(m_NullSampler, m_Context.allocationCallbacks);
    }

    DescriptorSetAllocator::PerType DescriptorSetAllocator::countPerType(const std::vector<vk::DescriptorPoolSize>& poolSizes)
    {
        PerType counts {};
        for (const vk::DescriptorPoolSize& size : poolSizes)
        {
            for (size_t typeIndex = 0; typeIndex < c_PooledDescriptorTypes.size(); ++typeIndex)
            {
                if (c_PooledDescriptorTypes[typeIndex] == size.type)
                    counts[typeIndex] = size.descriptorCount;
            }
        }
        return counts;
    }

    vk::Result DescriptorSetAllocator::openPool(const PerType& setCounts)
    {
        Pool pool;
        static_vector<vk::DescriptorPoolSize, c_PooledDescriptorTypes.size()> poolSizes;
        const uint64_t sets = m_AllocatedSets + 1;
        for (size_t typeIndex = 0; typeIndex < c_PooledDescriptorTypes.size(); ++typeIndex)
        {
            const vk::DescriptorType type = c_PooledDescriptorTypes[typeIndex];
            if (type == vk::DescriptorType::eAccelerationStructureKHR && !m_Context.extensions.KHR_acceleration_structure)
                continue;

            // The running average, the requesting set included, rounded up.
            const uint64_t setCount = setCounts[typeIndex];
            const uint64_t averagePerSet = (m_AllocatedDescriptors[typeIndex] + setCount + sets - 1) / sets;
            const uint64_t capacity = std::max<uint64_t>(
                c_SetsPerPool * c_CapacityHeadroom * std::max<uint64_t>(1, averagePerSet), setCount);
            pool.capacity[typeIndex] = uint32_t(std::min<uint64_t>(capacity, std::numeric_limits<uint32_t>::max()));
            poolSizes.push_back(vk::DescriptorPoolSize()
                .setType(type)
                .setDescriptorCount(pool.capacity[typeIndex]));
        }

        auto poolInfo = vk::DescriptorPoolCreateInfo()
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
            .setMaxSets(c_SetsPerPool)
            .setPoolSizeCount(uint32_t(poolSizes.size()))
            .setPPoolSizes(poolSizes.data());

        const vk::Result res = m_Context.device.createDescriptorPool(&poolInfo, m_Context.allocationCallbacks, &pool.pool);
        if (res != vk::Result::eSuccess)
            return res;

        m_Pools.push_back(pool);
        m_PoolsWithRoom.push_back(uint32_t(m_Pools.size() - 1));
        setDescriptorObjectName(m_Context, vk::ObjectType::eDescriptorPool,
            reinterpret_cast<uint64_t>(static_cast<VkDescriptorPool>(pool.pool)), "NVRHI binding set descriptor pool");
        return res;
    }

    bool DescriptorSetAllocator::acquire(BindingLayout& layout, DescriptorSetAllocation& outAllocation)
    {
        const char* failure = nullptr;
        vk::Result failureResult = vk::Result::eSuccess;
        bool acquired = false;
        {
            std::lock_guard lockGuard(m_Mutex);
            acquired = acquireLocked(layout, outAllocation, failure, failureResult);
        }
        // After the lock: the message callback is the application's and may
        // take its own locks. Only a failed acquire reports: an empty pool
        // that refused the set does not matter once an older pool took it.
        //
        // The driver calls under m_Mutex can run a validation layer's
        // VkDebugUtilsMessenger synchronously: vkAllocateDescriptorSets here,
        // vkCreateDescriptorPool and its naming call in openPool, and
        // vkFreeDescriptorSets and vkResetDescriptorPool in releaseRecycled,
        // on binding layout destruction. So the application's messenger must
        // not take any lock that is held while binding sets or binding
        // layouts are created or destroyed.
        if (!acquired && failure)
            m_Context.error(std::string(failure) + ", VkResult = " + resultToString(VkResult(failureResult)));
        return acquired;
    }

    bool DescriptorSetAllocator::acquireLocked(BindingLayout& layout, DescriptorSetAllocation& outAllocation,
        const char*& failure, vk::Result& failureResult)
    {
        if (!layout.recycledDescriptorSets.empty())
        {
            outAllocation = layout.recycledDescriptorSets.back();
            layout.recycledDescriptorSets.pop_back();
            return true;
        }

        auto allocInfo = vk::DescriptorSetAllocateInfo()
            .setDescriptorSetCount(1)
            .setPSetLayouts(&layout.descriptorSetLayout);
        const PerType& setCounts = layout.pooledDescriptorCounts;

        // Counter room in every type, and smaller than any set the driver
        // refused from this pool despite counter room.
        auto hasRoom = [&](const Pool& pool)
        {
            if (layout.numDescriptors >= pool.refusedSetDescriptors)
                return false;
            for (size_t typeIndex = 0; typeIndex < c_PooledDescriptorTypes.size(); ++typeIndex)
            {
                if (setCounts[typeIndex] > pool.capacity[typeIndex] - pool.used[typeIndex])
                    return false;
            }
            return true;
        };

        // vk::Result::eSuccess fills outAllocation; a pool that reaches maxSets
        // leaves m_PoolsWithRoom (at listIndex) until a set is freed into it.
        auto allocateFrom = [&](size_t listIndex) -> vk::Result
        {
            const uint32_t poolIndex = m_PoolsWithRoom[listIndex];
            Pool& pool = m_Pools[poolIndex];
            allocInfo.setDescriptorPool(pool.pool);
            vk::DescriptorSet set;
            const vk::Result res = m_Context.device.allocateDescriptorSets(&allocInfo, &set);
            if (res != vk::Result::eSuccess)
                return res;

            for (size_t typeIndex = 0; typeIndex < c_PooledDescriptorTypes.size(); ++typeIndex)
            {
                pool.used[typeIndex] += setCounts[typeIndex];
                m_AllocatedDescriptors[typeIndex] += setCounts[typeIndex];
            }
            ++m_AllocatedSets;
            ++pool.allocatedSets;
            if (!pool.listed())
                m_PoolsWithRoom.erase(m_PoolsWithRoom.begin() + ptrdiff_t(listIndex));
            outAllocation.set = set;
            outAllocation.poolIndex = poolIndex;
            return res;
        };

        // Newest first, skipping pools that show no room for the set.
        bool emptyPoolRefused = false;
        for (size_t listIndex = m_PoolsWithRoom.size(); listIndex-- > 0;)
        {
            Pool& pool = m_Pools[m_PoolsWithRoom[listIndex]];
            if (!hasRoom(pool))
                continue;

            const vk::Result res = allocateFrom(listIndex);
            if (res == vk::Result::eSuccess)
                return true;
            if (res != vk::Result::eErrorOutOfPoolMemory && res != vk::Result::eErrorFragmentedPool)
            {
                failure = "vkAllocateDescriptorSets failed for a binding set";
                failureResult = res;
                return false;
            }

            if (pool.allocatedSets == 0)
            {
                // Vulkan guarantees a pool with no frees since its creation or
                // last reset does not fail an allocation for fragmentation, and
                // releaseRecycled resets a pool its frees empty, so every empty
                // pool has that guarantee. One refusing a set its sizes fit
                // means a new pool would refuse it too.
                failure = "An empty descriptor pool refused a binding set that fits its sizes";
                failureResult = res;
                emptyPoolRefused = true;
                continue;
            }

            // Fragmentation: sets at least this large skip the pool until a
            // set is freed into it; smaller ones may still fit.
            pool.refusedSetDescriptors = std::min(pool.refusedSetDescriptors, layout.numDescriptors);
        }

        // Opening a pool per retry of a set no pool takes would grow without bound.
        if (emptyPoolRefused)
            return false;

        const vk::Result openResult = openPool(setCounts);
        if (openResult != vk::Result::eSuccess)
        {
            failure = "vkCreateDescriptorPool failed for binding sets";
            failureResult = openResult;
            return false;
        }

        // The new pool holds the set by construction. On failure it stays
        // listed: it is empty and usable by other sets.
        const vk::Result res = allocateFrom(m_PoolsWithRoom.size() - 1);
        if (res == vk::Result::eSuccess)
            return true;
        failure = "A new descriptor pool refused the binding set it was sized for";
        failureResult = res;
        return false;
    }

    void DescriptorSetAllocator::appendNullWrites(const BindingLayout& layout, vk::DescriptorSet set,
        std::vector<vk::WriteDescriptorSet>& writes) const
    {
        const size_t numBindings = layout.vulkanLayoutBindings.size();
        if (layout.numDescriptors == 0)
            return;

        // One bit per layout descriptor, set where `writes` writes it. Each
        // thread keeps its scratch's capacity, so steady state allocates nothing.
        static thread_local std::vector<uint64_t> written;
        written.assign((layout.numDescriptors + 63) / 64, 0);
        uint32_t writtenCount = 0;

        // Writes mostly follow the layout's binding order, so each search
        // starts at the binding the previous write matched.
        size_t bindingIndex = 0;
        for (const vk::WriteDescriptorSet& write : writes)
        {
            size_t probe = 0;
            while (probe < numBindings && layout.vulkanLayoutBindings[bindingIndex].binding != write.dstBinding)
            {
                bindingIndex = (bindingIndex + 1) % numBindings;
                ++probe;
            }
            if (probe == numBindings)
                continue;

            const uint32_t elementCount = layout.vulkanLayoutBindings[bindingIndex].descriptorCount;
            const uint32_t elementEnd = std::min(write.dstArrayElement + write.descriptorCount, elementCount);
            for (uint32_t element = write.dstArrayElement; element < elementEnd; ++element)
            {
                const uint32_t descriptor = layout.bindingFirstDescriptor[bindingIndex] + element;
                const uint64_t bit = uint64_t(1) << (descriptor % 64);
                if ((written[descriptor / 64] & bit) == 0)
                {
                    written[descriptor / 64] |= bit;
                    ++writtenCount;
                }
            }
        }

        if (writtenCount == layout.numDescriptors)
            return;

        // createBindingLayout grew these to the layout's longest binding.
        const NullDescriptorArrays* nulls = m_NullDescriptors.load(std::memory_order_acquire);
        assert(nulls);

        auto isWritten = [&](uint32_t descriptor)
        {
            return (written[descriptor / 64] & (uint64_t(1) << (descriptor % 64))) != 0;
        };

        writes.reserve(writes.size() + (layout.numDescriptors - writtenCount));
        for (size_t i = 0; i < numBindings; ++i)
        {
            const vk::DescriptorSetLayoutBinding& layoutBinding = layout.vulkanLayoutBindings[i];
            const uint32_t firstDescriptor = layout.bindingFirstDescriptor[i];
            uint32_t element = 0;
            while (element < layoutBinding.descriptorCount)
            {
                if (isWritten(firstDescriptor + element))
                {
                    ++element;
                    continue;
                }

                const uint32_t runStart = element;
                while (element < layoutBinding.descriptorCount && !isWritten(firstDescriptor + element))
                    ++element;
                const uint32_t runLength = element - runStart;
                assert(runLength <= nulls->images.size());

                auto write = vk::WriteDescriptorSet()
                    .setDstSet(set)
                    .setDstBinding(layoutBinding.binding)
                    .setDstArrayElement(runStart)
                    .setDescriptorCount(runLength)
                    .setDescriptorType(layoutBinding.descriptorType);

                switch (layoutBinding.descriptorType)
                {
                case vk::DescriptorType::eSampler:
                    write.setPImageInfo(nulls->samplers.data());
                    break;
                case vk::DescriptorType::eSampledImage:
                case vk::DescriptorType::eStorageImage:
                    write.setPImageInfo(nulls->images.data());
                    break;
                case vk::DescriptorType::eUniformTexelBuffer:
                case vk::DescriptorType::eStorageTexelBuffer:
                    write.setPTexelBufferView(nulls->bufferViews.data());
                    break;
                case vk::DescriptorType::eUniformBuffer:
                case vk::DescriptorType::eUniformBufferDynamic:
                case vk::DescriptorType::eStorageBuffer:
                    write.setPBufferInfo(nulls->buffers.data());
                    break;
                case vk::DescriptorType::eAccelerationStructureKHR:
                    write.setDescriptorCount(1).setPNext(&c_NullAccelStructWrite);
                    for (uint32_t runElement = runStart; runElement < element; ++runElement)
                        writes.push_back(vk::WriteDescriptorSet(write).setDstArrayElement(runElement));
                    continue;
                default:
                    // convertResourceType produces no other type for a regular layout.
                    utils::InvalidEnum();
                    continue;
                }

                writes.push_back(write);
            }
        }
    }

    void DescriptorSetAllocator::recycle(BindingLayout& layout, const DescriptorSetAllocation& allocation)
    {
        std::lock_guard lockGuard(m_Mutex);
        layout.recycledDescriptorSets.push_back(allocation);
    }

    void DescriptorSetAllocator::releaseRecycled(BindingLayout& layout)
    {
        std::lock_guard lockGuard(m_Mutex);
        const PerType& setCounts = layout.pooledDescriptorCounts;
        for (const DescriptorSetAllocation& allocation : layout.recycledDescriptorSets)
        {
            Pool& pool = m_Pools[allocation.poolIndex];
            (void)m_Context.device.freeDescriptorSets(pool.pool, 1, &allocation.set);
            const bool wasListed = pool.listed();
            for (size_t typeIndex = 0; typeIndex < c_PooledDescriptorTypes.size(); ++typeIndex)
                pool.used[typeIndex] -= setCounts[typeIndex];
            --pool.allocatedSets;
            // The free makes room and may join the fragments.
            pool.refusedSetDescriptors = UINT32_MAX;
            if (pool.allocatedSets == 0)
            {
                // A pool with frees since its creation or last reset may fail
                // an allocation for fragmentation even when empty; a reset
                // restores Vulkan's guarantee that it does not, which is what
                // acquire's empty-pool rule relies on. The C entry point:
                // vkResetDescriptorPool cannot fail, and the vulkan-hpp
                // wrapper would throw on a failure, here under a lock and on a
                // layout's destruction path.
                VULKAN_HPP_DEFAULT_DISPATCHER.vkResetDescriptorPool(
                    VkDevice(m_Context.device), VkDescriptorPool(pool.pool), 0);
                // Every free took its set's descriptors off the counters.
                assert(std::all_of(pool.used.begin(), pool.used.end(), [](uint32_t used) { return used == 0; }));
            }
            if (!wasListed)
                m_PoolsWithRoom.push_back(allocation.poolIndex);
        }
        layout.recycledDescriptorSets.clear();
    }

    BindingSetHandle Device::createBindingSet(const BindingSetDesc& desc, IBindingLayout* _layout)
    {
        BindingLayout* layout = checked_cast<BindingLayout*>(_layout);

        DescriptorSetAllocation allocation;
        if (!m_DescriptorSetAllocator.acquire(*layout, allocation))
            return nullptr;

        BindingSet *ret = new BindingSet(m_Context);
        ret->desc = desc;
        ret->layout = layout;
        ret->descriptorSetAllocation = allocation;

        // collect all of the descriptor write data
        std::vector<vk::DescriptorImageInfo> descriptorImageInfo;
        std::vector<vk::DescriptorBufferInfo> descriptorBufferInfo;
        std::vector<vk::WriteDescriptorSet> descriptorWriteInfo;
        std::vector<vk::WriteDescriptorSetAccelerationStructureKHR> accelStructWriteInfo;
        descriptorImageInfo.reserve(desc.bindings.size());
        descriptorBufferInfo.reserve(desc.bindings.size());
        descriptorWriteInfo.reserve(desc.bindings.size());
        accelStructWriteInfo.reserve(desc.bindings.size());

        // f111-pig: volatile constant buffers as written; volatileConstantBuffers
        // takes them in the layout's dynamic offset order once all are known.
        struct VolatileBufferWrite
        {
            uint32_t binding;
            uint32_t arrayElement;
            Buffer* buffer;
        };
        static_vector<VolatileBufferWrite, c_MaxVolatileConstantBuffersPerLayout> volatileBufferWrites;

        auto generateWriteDescriptorData =
            // generates a vk::WriteDescriptorSet struct in descriptorWriteInfo
            [&](uint32_t bindingLocation,
                uint32_t arrayElement,
                vk::DescriptorType descriptorType,
                vk::DescriptorImageInfo *imageInfo,
                vk::DescriptorBufferInfo *bufferInfo,
                vk::BufferView *bufferView,
                const void* pNext = nullptr)
        {
            descriptorWriteInfo.push_back(
                vk::WriteDescriptorSet()
                .setDstSet(allocation.set)
                .setDstBinding(bindingLocation)
                .setDstArrayElement(arrayElement)
                .setDescriptorCount(1)
                .setDescriptorType(descriptorType)
                .setPImageInfo(imageInfo)
                .setPBufferInfo(bufferInfo)
                .setPTexelBufferView(bufferView)
                .setPNext(pNext)
            );
        };

        for (size_t bindingIndex = 0; bindingIndex < desc.bindings.size(); bindingIndex++)
        {
            const BindingSetItem& binding = desc.bindings[bindingIndex];

            if (binding.resourceHandle == nullptr)
            {
                continue;
            }

            ret->resources.push_back(binding.resourceHandle); // keep a strong reference to the resource

            vk::DescriptorType const descriptorType = convertResourceType(binding.type);
            uint32_t const registerOffset = getRegisterOffsetForResourceType(layout->desc.bindingOffsets, binding.type);
            
            switch (binding.type)
            {
            case ResourceType::Texture_SRV:
            {
                const auto texture = checked_cast<Texture *>(binding.resourceHandle);
                const ResourceStates requiredState = getTextureSrvState(binding);
                if (binding.depthReadOnlyAttachment)
                {
                    const FormatInfo& formatInfo = getFormatInfo(texture->desc.format);
                    assert(formatInfo.hasDepth || formatInfo.hasStencil);
                    (void)formatInfo;
                }

                const auto subresource = binding.subresources.resolve(texture->desc, false);
                const auto textureViewType = getTextureViewType(binding.format, texture->desc.format);
                const auto* view = texture->getSubresourceView(
                    subresource, binding.dimension, binding.format,
                    vk::ImageUsageFlagBits::eSampled, textureViewType);
                if (!view)
                {
                    delete ret;
                    return nullptr;
                }

                auto& imageInfo = descriptorImageInfo.emplace_back();
                imageInfo = vk::DescriptorImageInfo()
                    .setImageView(view->view)
                    .setImageLayout(convertTextureLayout(requiredState, texture->desc));

                generateWriteDescriptorData(
                    registerOffset + binding.slot,
                    binding.arrayElement,
                    descriptorType,
                    &imageInfo, nullptr, nullptr);

                if (!texture->permanentState)
                    ret->bindingsThatNeedTransitions.push_back(static_cast<uint16_t>(bindingIndex));
                else
                    verifyPermanentResourceState(texture->permanentState,
                        requiredState,
                        true, texture->desc.debugName, m_Context.messageCallback);
            }

            break;

            case ResourceType::Texture_UAV:
            {
                const auto texture = checked_cast<Texture *>(binding.resourceHandle);

                const auto subresource = binding.subresources.resolve(texture->desc, true);
                const auto textureViewType = getTextureViewType(binding.format, texture->desc.format);
                const auto* view = texture->getSubresourceView(
                    subresource, binding.dimension, binding.format,
                    vk::ImageUsageFlagBits::eStorage, textureViewType);
                if (!view)
                {
                    delete ret;
                    return nullptr;
                }

                auto& imageInfo = descriptorImageInfo.emplace_back();
                imageInfo = vk::DescriptorImageInfo()
                    .setImageView(view->view)
                    .setImageLayout(convertTextureLayout(ResourceStates::UnorderedAccess, texture->desc));

                generateWriteDescriptorData(
                    registerOffset + binding.slot,
                    binding.arrayElement,
                    descriptorType,
                    &imageInfo, nullptr, nullptr);

                if (!texture->permanentState)
                    ret->bindingsThatNeedTransitions.push_back(static_cast<uint16_t>(bindingIndex));
                else
                    verifyPermanentResourceState(texture->permanentState,
                        ResourceStates::UnorderedAccess,
                        true, texture->desc.debugName, m_Context.messageCallback);
            }

            break;

            case ResourceType::TypedBuffer_SRV:
            case ResourceType::TypedBuffer_UAV:
            {
                const auto buffer = checked_cast<Buffer *>(binding.resourceHandle);

                assert(buffer->desc.canHaveTypedViews);
                const bool isUAV = (binding.type == ResourceType::TypedBuffer_UAV);
                if (isUAV)
                    assert(buffer->desc.canHaveUAVs);

                Format format = binding.format;

                if (format == Format::UNKNOWN)
                {
                    format = buffer->desc.format;
                }

                auto vkformat = nvrhi::vulkan::convertFormat(format);
                const auto range = binding.range.resolve(buffer->desc);

                size_t viewInfoHash = 0;
                nvrhi::hash_combine(viewInfoHash, range.byteOffset);
                nvrhi::hash_combine(viewInfoHash, range.byteSize);
                nvrhi::hash_combine(viewInfoHash, (uint64_t)vkformat);

                const auto& bufferViewFound = buffer->viewCache.find(viewInfoHash);
                auto& bufferViewRef = (bufferViewFound != buffer->viewCache.end()) ? bufferViewFound->second : buffer->viewCache[viewInfoHash];
                if (bufferViewFound == buffer->viewCache.end())
                {
                    assert(format != Format::UNKNOWN);

                    auto bufferViewInfo = vk::BufferViewCreateInfo()
                        .setBuffer(buffer->buffer)
                        .setOffset(range.byteOffset)
                        .setRange(range.byteSize)
                        .setFormat(vk::Format(vkformat));

                    const vk::Result res = m_Context.device.createBufferView(&bufferViewInfo, m_Context.allocationCallbacks, &bufferViewRef);
                    ASSERT_VK_OK(res);
                }

                generateWriteDescriptorData(
                    registerOffset + binding.slot,
                    binding.arrayElement,
                    descriptorType,
                    nullptr, nullptr, &bufferViewRef);

                if (!buffer->permanentState)
                    ret->bindingsThatNeedTransitions.push_back(static_cast<uint16_t>(bindingIndex));
                else
                    verifyPermanentResourceState(buffer->permanentState, 
                        isUAV ? ResourceStates::UnorderedAccess : ResourceStates::ShaderResource,
                        false, buffer->desc.debugName, m_Context.messageCallback);
            }
            break;

            case ResourceType::StructuredBuffer_SRV:
            case ResourceType::StructuredBuffer_UAV:
            case ResourceType::RawBuffer_SRV:
            case ResourceType::RawBuffer_UAV:
            case ResourceType::ConstantBuffer:
            case ResourceType::VolatileConstantBuffer:
            {
                const auto buffer = checked_cast<Buffer *>(binding.resourceHandle);

                if (binding.type == ResourceType::StructuredBuffer_UAV || binding.type == ResourceType::RawBuffer_UAV)
                    assert(buffer->desc.canHaveUAVs);
                if (binding.type == ResourceType::StructuredBuffer_UAV || binding.type == ResourceType::StructuredBuffer_SRV)
                    assert(buffer->desc.structStride != 0);
                if (binding.type == ResourceType::RawBuffer_SRV|| binding.type == ResourceType::RawBuffer_UAV)
                    assert(buffer->desc.canHaveRawViews);

                const auto range = binding.range.resolve(buffer->desc);

                auto& bufferInfo = descriptorBufferInfo.emplace_back();
                bufferInfo = vk::DescriptorBufferInfo()
                    .setBuffer(buffer->buffer)
                    .setOffset(range.byteOffset)
                    .setRange(range.byteSize);

                assert(buffer->buffer);
                generateWriteDescriptorData(
                    registerOffset + binding.slot,
                    binding.arrayElement,
                    descriptorType,
                    nullptr, &bufferInfo, nullptr);

                if (binding.type == ResourceType::VolatileConstantBuffer)
                {
                    assert(buffer->desc.isVolatile);
                    // The layout holds at most this many dynamic elements
                    // (createBindingLayout), so a desc writing more is invalid.
                    if (volatileBufferWrites.size() < volatileBufferWrites.max_size())
                        volatileBufferWrites.push_back({ registerOffset + binding.slot, binding.arrayElement, buffer });
                    else
                        m_Context.error("Binding set writes more volatile constant buffers than its layout holds");
                }
                else
                {
                    if (!buffer->permanentState)
                        ret->bindingsThatNeedTransitions.push_back(static_cast<uint16_t>(bindingIndex));
                    else
                    {
                        ResourceStates requiredState;
                        if (binding.type == ResourceType::StructuredBuffer_UAV || binding.type == ResourceType::RawBuffer_UAV)
                            requiredState = ResourceStates::UnorderedAccess;
                        else if (binding.type == ResourceType::ConstantBuffer)
                            requiredState = ResourceStates::ConstantBuffer;
                        else
                            requiredState = ResourceStates::ShaderResource;

                        verifyPermanentResourceState(buffer->permanentState, requiredState,
                            false, buffer->desc.debugName, m_Context.messageCallback);
                    }
                }
            }

            break;

            case ResourceType::Sampler:
            {
                const auto sampler = checked_cast<Sampler *>(binding.resourceHandle);

                auto& imageInfo = descriptorImageInfo.emplace_back();
                imageInfo = vk::DescriptorImageInfo()
                    .setSampler(sampler->sampler);

                generateWriteDescriptorData(
                    registerOffset + binding.slot,
                    binding.arrayElement,
                    descriptorType,
                    &imageInfo, nullptr, nullptr);
            }

            break;

            case ResourceType::RayTracingAccelStruct:
            {
                const auto as = checked_cast<AccelStruct*>(binding.resourceHandle);

                auto& accelStructWrite = accelStructWriteInfo.emplace_back();
                accelStructWrite.accelerationStructureCount = 1;
                accelStructWrite.pAccelerationStructures = &as->accelStruct;

                generateWriteDescriptorData(
                    registerOffset + binding.slot,
                    binding.arrayElement,
                    descriptorType,
                    nullptr, nullptr, nullptr, &accelStructWrite);

                ret->bindingsThatNeedTransitions.push_back(static_cast<uint16_t>(bindingIndex));
            }

            break;

            case ResourceType::PushConstants:
                break;

            case ResourceType::None:
            case ResourceType::Count:
            default:
                utils::InvalidEnum();
                break;
            }

            // Update the hasUavBindings flag, it's cleaner to do it as a separate switch.
            switch (binding.type)
            {
            case ResourceType::Texture_UAV:
            case ResourceType::TypedBuffer_UAV:
            case ResourceType::StructuredBuffer_UAV:
            case ResourceType::RawBuffer_UAV:
            case ResourceType::SamplerFeedbackTexture_UAV:
                ret->hasUavBindings |= binding.enableAutomaticTransitions;
                break;
            default:
                break;
            }

            if (binding.type == ResourceType::Texture_SRV)
            {
                ret->hasDepthReadOnlyAttachmentBindings |=
                    binding.depthReadOnlyAttachment
                    && binding.enableAutomaticTransitions;
            }

        }

        // One entry per layout dynamic uniform buffer element, in dynamic offset
        // order: an element the binding set left unset is written null below
        // and bound at offset 0 (bindBindingSets).
        for (const BindingLayout::DynamicBufferElement& element : layout->dynamicBufferElements)
        {
            Buffer* buffer = nullptr;
            for (const VolatileBufferWrite& write : volatileBufferWrites)
            {
                if (write.binding == element.binding && write.arrayElement == element.arrayElement)
                    buffer = write.buffer;
            }
            ret->volatileConstantBuffers.push_back(buffer);
        }

        m_DescriptorSetAllocator.appendNullWrites(*layout, allocation.set, descriptorWriteInfo);
        m_Context.device.updateDescriptorSets(uint32_t(descriptorWriteInfo.size()), descriptorWriteInfo.data(), 0, nullptr);

        return BindingSetHandle::Create(ret);
    }

    BindingSet::~BindingSet()
    {
        // A binding set recorded with trackLiveness dies only once the command
        // lists holding it retire (their submissions' semaphore values have
        // completed), so no GPU work still reads the set when its next owner
        // rewrites it. One created with trackLiveness = false relies on the
        // application keeping it alive until that work finishes, as
        // destroying a descriptor pool would.
        if (descriptorSetAllocation.set)
        {
            // The next owner may never name the set; without this, validation
            // messages and captures would show this binding set's name. One
            // call measured 0.017 us without layers (RTX 5090, driver 616.92),
            // so the ~8 binding sets the engine churns a frame cost ~0.14 us.
            // It does not throw, which a destructor must not.
            setDescriptorObjectName(m_Context, vk::ObjectType::eDescriptorSet,
                reinterpret_cast<uint64_t>(static_cast<VkDescriptorSet>(descriptorSetAllocation.set)), nullptr);

            m_Context.descriptorSetAllocator->recycle(*checked_cast<BindingLayout*>(layout.Get()), descriptorSetAllocation);
            descriptorSetAllocation = DescriptorSetAllocation();
        }
    }

    Object BindingSet::getNativeObject(ObjectType objectType)
    {
        switch (objectType)
        {
        // VK_DescriptorPool: a binding set owns no pool; its set comes from a
        // pool shared by many binding sets.
        case ObjectTypes::VK_DescriptorSet:
            return Object(descriptorSetAllocation.set);
        default:
            return nullptr;
        }
    }

    DescriptorTableHandle Device::createDescriptorTable(IBindingLayout* _layout)
    { 
        BindingLayout* layout = checked_cast<BindingLayout*>(_layout);

        DescriptorTable* ret = new DescriptorTable(m_Context);
        ret->layout = layout;
        ret->capacity = layout->vulkanLayoutBindings[0].descriptorCount;

        const auto& descriptorSetLayout = layout->descriptorSetLayout;
        const auto& poolSizes = layout->descriptorPoolSizeInfo;

        // create descriptor pool to allocate a descriptor from
        auto poolInfo = vk::DescriptorPoolCreateInfo()
            .setPoolSizeCount(uint32_t(poolSizes.size()))
            .setPPoolSizes(poolSizes.data())
            .setMaxSets(1);

        if (layout->bindlessDesc.enableUpdateAfterBind)
            poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);

        vk::Result res = m_Context.device.createDescriptorPool(&poolInfo,
                                                             m_Context.allocationCallbacks,
                                                             &ret->descriptorPool);
        CHECK_VK_FAIL(res)

        // create the descriptor set
        auto descriptorSetAllocInfo = vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(ret->descriptorPool)
            .setDescriptorSetCount(1)
            .setPSetLayouts(&descriptorSetLayout);

        res = m_Context.device.allocateDescriptorSets(&descriptorSetAllocInfo,
            &ret->descriptorSet);
        CHECK_VK_FAIL(res)

        return DescriptorTableHandle::Create(ret);
    }

    DescriptorTable::~DescriptorTable()
    {
        if (descriptorPool)
        {
            m_Context.device.destroyDescriptorPool(descriptorPool, m_Context.allocationCallbacks);
            descriptorPool = vk::DescriptorPool();
            descriptorSet = vk::DescriptorSet();
        }
    }

    Object DescriptorTable::getNativeObject(ObjectType objectType)
    {
        switch (objectType)
        {
        case ObjectTypes::VK_DescriptorPool:
            return Object(descriptorPool);
        case ObjectTypes::VK_DescriptorSet:
            return Object(descriptorSet);
        default:
            return nullptr;
        }
    }

    void Device::resizeDescriptorTable(IDescriptorTable* _descriptorTable, uint32_t newSize, bool keepContents)
    {
        assert(newSize <= checked_cast<DescriptorTable*>(_descriptorTable)->layout->getBindlessDesc()->maxCapacity);
        (void)_descriptorTable;
        (void)newSize;
        (void)keepContents;
    }

    bool Device::writeDescriptorTable(IDescriptorTable* _descriptorTable, const BindingSetItem& binding)
    {
        DescriptorTable* descriptorTable = checked_cast<DescriptorTable*>(_descriptorTable);
        BindingLayout* layout = checked_cast<BindingLayout*>(descriptorTable->layout.Get());

        if (binding.slot >= descriptorTable->capacity)
            return false;

        if (binding.type == ResourceType::None)
        {
            // Vulkan doesn't support null descriptors, we use vk::DescriptorBindingFlagBits::ePartiallyBound
            return true;
        }

        // collect all of the descriptor write data
        static_vector<vk::DescriptorImageInfo, c_MaxBindlessRegisterSpaces> descriptorImageInfo;
        static_vector<vk::DescriptorBufferInfo, c_MaxBindlessRegisterSpaces> descriptorBufferInfo;
        static_vector<vk::WriteDescriptorSet, c_MaxBindlessRegisterSpaces> descriptorWriteInfo;

        auto generateWriteDescriptorData =
            // generates a vk::WriteDescriptorSet struct in descriptorWriteInfo
            [&](uint32_t bindingLocation,
                vk::DescriptorType descriptorType,
                vk::DescriptorImageInfo* imageInfo,
                vk::DescriptorBufferInfo* bufferInfo,
                vk::BufferView* bufferView)
        {
            descriptorWriteInfo.push_back(
                vk::WriteDescriptorSet()
                .setDstSet(descriptorTable->descriptorSet)
                .setDstBinding(bindingLocation)
                .setDstArrayElement(binding.slot)
                .setDescriptorCount(1)
                .setDescriptorType(descriptorType)
                .setPImageInfo(imageInfo)
                .setPBufferInfo(bufferInfo)
                .setPTexelBufferView(bufferView)
            );
        };

        bool imageViewCreationFailed = false;
        auto writeDescriptorForBinding = [&](const vk::DescriptorSetLayoutBinding& layoutBinding) -> void
        {
            if (imageViewCreationFailed)
                return;

            switch (binding.type)
            {
            case ResourceType::Texture_SRV:
            {
                const auto& texture = checked_cast<Texture*>(binding.resourceHandle);
                const ResourceStates requiredState = getTextureSrvState(binding);
                if (binding.depthReadOnlyAttachment)
                {
                    const FormatInfo& formatInfo = getFormatInfo(texture->desc.format);
                    assert(formatInfo.hasDepth || formatInfo.hasStencil);
                    (void)formatInfo;
                }

                const auto subresource = binding.subresources.resolve(texture->desc, false);
                const auto textureViewType = getTextureViewType(binding.format, texture->desc.format);
                const auto* view = texture->getSubresourceView(
                    subresource, binding.dimension, binding.format,
                    vk::ImageUsageFlagBits::eSampled, textureViewType);
                if (!view)
                {
                    imageViewCreationFailed = true;
                    return;
                }

                auto& imageInfo = descriptorImageInfo.emplace_back();
                imageInfo = vk::DescriptorImageInfo()
                    .setImageView(view->view)
                    .setImageLayout(convertTextureLayout(requiredState, texture->desc));

                generateWriteDescriptorData(layoutBinding.binding,
                    convertResourceType(binding.type),
                    &imageInfo, nullptr, nullptr);
            }
            break;

            case ResourceType::Texture_UAV:
            {
                const auto texture = checked_cast<Texture*>(binding.resourceHandle);

                const auto subresource = binding.subresources.resolve(texture->desc, true);
                const auto textureViewType = getTextureViewType(binding.format, texture->desc.format);
                const auto* view = texture->getSubresourceView(
                    subresource, binding.dimension, binding.format,
                    vk::ImageUsageFlagBits::eStorage, textureViewType);
                if (!view)
                {
                    imageViewCreationFailed = true;
                    return;
                }

                auto& imageInfo = descriptorImageInfo.emplace_back();
                imageInfo = vk::DescriptorImageInfo()
                    .setImageView(view->view)
                    .setImageLayout(convertTextureLayout(ResourceStates::UnorderedAccess, texture->desc));

                generateWriteDescriptorData(layoutBinding.binding,
                    convertResourceType(binding.type),
                    &imageInfo, nullptr, nullptr);
            }
            break;

            case ResourceType::TypedBuffer_SRV:
            case ResourceType::TypedBuffer_UAV:
            {
                const auto& buffer = checked_cast<Buffer*>(binding.resourceHandle);

                auto vkformat = nvrhi::vulkan::convertFormat(binding.format);

                const auto range = binding.range.resolve(buffer->desc);
                size_t viewInfoHash = 0;
                nvrhi::hash_combine(viewInfoHash, range.byteOffset);
                nvrhi::hash_combine(viewInfoHash, range.byteSize);
                nvrhi::hash_combine(viewInfoHash, (uint64_t)vkformat);

                const auto& bufferViewFound = buffer->viewCache.find(viewInfoHash);
                auto& bufferViewRef = (bufferViewFound != buffer->viewCache.end()) ? bufferViewFound->second : buffer->viewCache[viewInfoHash];
                if (bufferViewFound == buffer->viewCache.end())
                {
                    assert(binding.format != Format::UNKNOWN);

                    auto bufferViewInfo = vk::BufferViewCreateInfo()
                        .setBuffer(buffer->buffer)
                        .setOffset(range.byteOffset)
                        .setRange(range.byteSize)
                        .setFormat(vk::Format(vkformat));

                    vk::Result res = m_Context.device.createBufferView(&bufferViewInfo, m_Context.allocationCallbacks, &bufferViewRef);
                    ASSERT_VK_OK(res);
                }

                generateWriteDescriptorData(layoutBinding.binding,
                    convertResourceType(binding.type),
                    nullptr, nullptr, &bufferViewRef);
            }
            break;

            case ResourceType::StructuredBuffer_SRV:
            case ResourceType::StructuredBuffer_UAV:
            case ResourceType::RawBuffer_SRV:
            case ResourceType::RawBuffer_UAV:
            case ResourceType::ConstantBuffer:
            case ResourceType::VolatileConstantBuffer:
            {
                const auto buffer = checked_cast<Buffer*>(binding.resourceHandle);

                const auto range = binding.range.resolve(buffer->desc);

                auto& bufferInfo = descriptorBufferInfo.emplace_back();
                bufferInfo = vk::DescriptorBufferInfo()
                    .setBuffer(buffer->buffer)
                    .setOffset(range.byteOffset)
                    .setRange(range.byteSize);

                assert(buffer->buffer);
                generateWriteDescriptorData(layoutBinding.binding,
                    convertResourceType(binding.type),
                    nullptr, &bufferInfo, nullptr);
            }
            break;

            case ResourceType::Sampler:
            {
                const auto& sampler = checked_cast<Sampler*>(binding.resourceHandle);

                auto& imageInfo = descriptorImageInfo.emplace_back();
                imageInfo = vk::DescriptorImageInfo()
                    .setSampler(sampler->sampler);

                generateWriteDescriptorData(layoutBinding.binding,
                    convertResourceType(binding.type),
                    &imageInfo, nullptr, nullptr);
            }
            break;

            case ResourceType::RayTracingAccelStruct:
                utils::NotImplemented();
                break;

            case ResourceType::PushConstants:
                utils::NotSupported();
                break;

            case ResourceType::None:
            case ResourceType::Count:
            default:
                utils::InvalidEnum();
            }
        };

        if (layout->bindlessDesc.layoutType != BindlessLayoutDesc::LayoutType::Immutable)
        {
            // For mutable descriptor sets, there are no register spaces, so always use the first layout binding
            assert(layout->vulkanLayoutBindings.size() > 0);
            writeDescriptorForBinding(layout->vulkanLayoutBindings[0]);
        }
        else
        {
            // For regular bindless layouts, iterate through register spaces to find matching binding type
            for (uint32_t bindingLocation = 0; bindingLocation < uint32_t(layout->bindlessDesc.registerSpaces.size()); bindingLocation++)
            {
                if (layout->bindlessDesc.registerSpaces[bindingLocation].type == binding.type)
                {
                    const vk::DescriptorSetLayoutBinding& layoutBinding = layout->vulkanLayoutBindings[bindingLocation];
                    writeDescriptorForBinding(layoutBinding);
                }
            }
        }

        if (imageViewCreationFailed)
            return false;

        m_Context.device.updateDescriptorSets(uint32_t(descriptorWriteInfo.size()), descriptorWriteInfo.data(), 0, nullptr);

        return true;
    }

    void CommandList::bindBindingSets(vk::PipelineBindPoint bindPoint, vk::PipelineLayout pipelineLayout, const BindingSetVector& bindings, BindingVector<uint32_t> const& descriptorSetIdxToBindingIdx)
    {
        const uint32_t numBindings = (uint32_t)bindings.size();
        const uint32_t numDescriptorSets = descriptorSetIdxToBindingIdx.empty() ? numBindings : (uint32_t)descriptorSetIdxToBindingIdx.size();

        BindingVector<vk::DescriptorSet> descriptorSets;
        uint32_t nextDescriptorSetToBind = 0;
        // f111-pig: up to c_MaxBindingLayouts sets, each with at most
        // c_MaxVolatileConstantBuffersPerLayout dynamic elements
        // (createBindingLayout), so the offsets of any bind fit.
        static_vector<uint32_t, c_MaxBindingLayouts * c_MaxVolatileConstantBuffersPerLayout> dynamicOffsets;
        for (uint32_t i = 0; i < numDescriptorSets; ++i)
        {
            IBindingSet* bindingSetHandle = nullptr;
            if (descriptorSetIdxToBindingIdx.empty())
            {
                bindingSetHandle = bindings[i];
            }
            else if(descriptorSetIdxToBindingIdx[i] != 0xffffffff)
            {
                bindingSetHandle = bindings[descriptorSetIdxToBindingIdx[i]];
            }

            if (bindingSetHandle == nullptr)
            {
                // This is a hole in the descriptor sets, so bind the contiguous descriptor sets we've got so far
                if (!descriptorSets.empty())
                {
                    m_CurrentCmdBuf->cmdBuf.bindDescriptorSets(bindPoint, pipelineLayout,
                        /* firstSet = */ nextDescriptorSetToBind, uint32_t(descriptorSets.size()), descriptorSets.data(),
                        uint32_t(dynamicOffsets.size()), dynamicOffsets.data());

                    descriptorSets.resize(0);
                    dynamicOffsets.resize(0);
                }
                nextDescriptorSetToBind = i + 1;
            }
            else
            {
                const BindingSetDesc* desc = bindingSetHandle->getDesc();
                if (desc)
                {
                    BindingSet* bindingSet = checked_cast<BindingSet*>(bindingSetHandle);
                    descriptorSets.push_back(bindingSet->descriptorSetAllocation.set);

                    for (Buffer* constantBuffer : bindingSet->volatileConstantBuffers)
                    {
                        // f111-pig: an element the binding set left unset is a
                        // null descriptor; its offset only keeps the count.
                        if (!constantBuffer)
                        {
                            dynamicOffsets.push_back(0);
                            continue;
                        }

                        auto found = m_VolatileBufferStates.find(constantBuffer);
                        if (found == m_VolatileBufferStates.end())
                        {
                            std::stringstream ss;
                            ss << "Binding volatile constant buffer " << utils::DebugNameToString(constantBuffer->desc.debugName)
                                << " before writing into it is invalid.";
                            m_Context.error(ss.str());

                            dynamicOffsets.push_back(0); // use zero offset just to use something
                        }
                        else
                        {
                            uint32_t version = found->second.latestVersion;
                            uint64_t offset = version * constantBuffer->desc.byteSize;
                            assert(offset < std::numeric_limits<uint32_t>::max());
                            dynamicOffsets.push_back(uint32_t(offset));
                        }
                    }

                    if (desc->trackLiveness)
                        m_CurrentCmdBuf->referencedResources.push_back(bindingSetHandle);
                }
                else
                {
                    DescriptorTable* table = checked_cast<DescriptorTable*>(bindingSetHandle);
                    descriptorSets.push_back(table->descriptorSet);
                }
            }
        }
        if (!descriptorSets.empty())
        {
            // Bind the remaining sets
            m_CurrentCmdBuf->cmdBuf.bindDescriptorSets(bindPoint, pipelineLayout,
                /* firstSet = */ nextDescriptorSetToBind, uint32_t(descriptorSets.size()), descriptorSets.data(),
                uint32_t(dynamicOffsets.size()), dynamicOffsets.data());
        }
    }

    vk::Result createPipelineLayout(
        vk::PipelineLayout& outPipelineLayout,
        BindingVector<RefCountPtr<BindingLayout>>& outBindingLayouts,
        vk::ShaderStageFlags& outPushConstantVisibility,
        BindingVector<uint32_t>& outDescriptorSetIdxToBindingIdx,
        VulkanContext const& context,
        BindingLayoutVector const& inBindingLayouts)
    {
        // Establish if we're going to use outDescriptorSetIdxToBindingIdx
        // We do this if the layout descs specify registerSpaceIsDescriptorSet
        // (Validation ensures all the binding layouts have it set to the same value)
        bool createDescriptorSetIdxToBindingIdx = false;
        for (BindingLayoutHandle const& _layout : inBindingLayouts)
        {
            BindingLayout const* layout = checked_cast<BindingLayout const*>(_layout.Get());
            if (!layout->isBindless)
            {
                createDescriptorSetIdxToBindingIdx = layout->getDesc()->registerSpaceIsDescriptorSet;
                break;
            }
        }

        if (createDescriptorSetIdxToBindingIdx)
        {
            // Figure out how many descriptor sets we'll need in outBindingLayouts.
            // There's not necessarily a one-to-one relationship because there could potentially be
            // holes in binding layout.  E.g. if a binding layout uses register spaces 0 and 2
            // then we'll need to use 3 descriptor sets, with a hole at index 1 because Vulkan
            // descriptor set indices map to register spaces.
            // Bindless layouts are assumed to not need binding to specific descriptor set
            // indices, so we put those last
            uint32_t numRegularDescriptorSets = 0;
            for (BindingLayoutHandle const& _layout : inBindingLayouts)
            {
                BindingLayout const* layout = checked_cast<BindingLayout const*>(_layout.Get());
                if (!layout->isBindless)
                {
                    numRegularDescriptorSets = std::max(numRegularDescriptorSets, layout->getDesc()->registerSpace + 1);
                }
            }

            // Now create the layout
            outBindingLayouts.resize(numRegularDescriptorSets);
            outDescriptorSetIdxToBindingIdx.resize(numRegularDescriptorSets);
            for (uint32_t i = 0; i < numRegularDescriptorSets; ++i)
            {
                outDescriptorSetIdxToBindingIdx[i] = 0xffffffff;
            }
            for (uint32_t i = 0; i < (uint32_t)inBindingLayouts.size(); ++i)
            {
                BindingLayout* layout = checked_cast<BindingLayout*>(inBindingLayouts[i].Get());
                if (layout->isBindless)
                {
                    outBindingLayouts.push_back(layout);
                    // Let's always put the bindless ones at the end.
                    outDescriptorSetIdxToBindingIdx.push_back(i);
                }
                else
                {
                    uint32_t const descriptorSetIdx = layout->getDesc()->registerSpace;
                    // Can't have multiple binding sets with the same registerSpace
                    // Should not have passed validation in validatePipelineBindingLayouts
                    assert(outBindingLayouts[descriptorSetIdx] == nullptr);
                    outBindingLayouts[descriptorSetIdx] = layout;
                    outDescriptorSetIdxToBindingIdx[descriptorSetIdx] = i;
                }
            }
        }
        else
        {
            // Legacy behaviour mode, where we don't fill in outDescriptorSetIdxToBindingIdx
            // In this mode, there can be no holes in the binding layout
            for (const BindingLayoutHandle& _layout : inBindingLayouts)
            {
                BindingLayout* layout = checked_cast<BindingLayout*>(_layout.Get());
                outBindingLayouts.push_back(layout);
            }
        }

        BindingVector<vk::DescriptorSetLayout> descriptorSetLayouts;
        uint32_t pushConstantSize = 0;
        outPushConstantVisibility = vk::ShaderStageFlagBits();
        for (BindingLayout const* layout : outBindingLayouts)
        {
            if (layout)
            {
                descriptorSetLayouts.push_back(layout->descriptorSetLayout);

                if (!layout->isBindless)
                {
                    for (const BindingLayoutItem& item : layout->desc.bindings)
                    {
                        if (item.type == ResourceType::PushConstants)
                        {
                            pushConstantSize = item.size;
                            outPushConstantVisibility = convertShaderTypeToShaderStageFlagBits(layout->desc.visibility);
                            // assume there's only one push constant item in all layouts -- the validation layer makes sure of that
                            break;
                        }
                    }
                }
            }
            else
            {
                // Empty descriptor set
                descriptorSetLayouts.push_back(context.emptyDescriptorSetLayout);
            }
        }

        auto pushConstantRange = vk::PushConstantRange()
            .setOffset(0)
            .setSize(pushConstantSize)
            .setStageFlags(outPushConstantVisibility);

        auto pipelineLayoutInfo = vk::PipelineLayoutCreateInfo()
            .setSetLayoutCount(uint32_t(descriptorSetLayouts.size()))
            .setPSetLayouts(descriptorSetLayouts.data())
            .setPushConstantRangeCount(pushConstantSize ? 1 : 0)
            .setPPushConstantRanges(&pushConstantRange);

        vk::Result res = context.device.createPipelineLayout(&pipelineLayoutInfo,
            context.allocationCallbacks,
            &outPipelineLayout);

        return res;
    }

} // namespace nvrhi::vulkan
