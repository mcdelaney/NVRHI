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
    ComputePipelineHandle Device::createComputePipeline(const ComputePipelineDesc& desc)
    {
        vk::Result res;

        assert(desc.CS);
        
        ComputePipeline *pso = new ComputePipeline(m_Context);
        pso->desc = desc;

        res = createPipelineLayout(
            pso->pipelineLayout,
            pso->pipelineBindingLayouts,
            pso->pushConstantVisibility,
            pso->descriptorSetIdxToBindingIdx,
            m_Context,
            desc.bindingLayouts);
        CHECK_VK_FAIL(res)

        Shader* CS = checked_cast<Shader*>(desc.CS.Get());

        // See createGraphicsPipeline() for a more expanded implementation
        // of shader specializations with multiple shaders in the pipeline

        size_t numShaders = 0;
        size_t numShadersWithSpecializations = 0;
        size_t numSpecializationConstants = 0;

        countSpecializationConstants(CS, numShaders, numShadersWithSpecializations, numSpecializationConstants);

        assert(numShaders == 1);

        // The create info lives on the heap: an offloaded compile reads it
        // after this function returns, and the stage info points into the
        // specialization vectors, whose storage a move of the struct keeps.
        auto deferred = std::make_unique<ComputePipeline::DeferredCompile>();

        deferred->specInfos.reserve(numShadersWithSpecializations);
        deferred->specMapEntries.reserve(numSpecializationConstants);
        deferred->specData.reserve(numSpecializationConstants);

        auto shaderStageInfo = makeShaderStageCreateInfo(CS,
            deferred->specInfos, deferred->specMapEntries, deferred->specData);

        deferred->pipelineInfo = vk::ComputePipelineCreateInfo()
                                .setStage(shaderStageInfo)
                                .setLayout(pso->pipelineLayout);

        PipelineCompileOffloadDesc offload;
        {
            std::lock_guard<std::mutex> lock(m_PipelineCompileOffload.mutex);
            offload = m_PipelineCompileOffload.desc;
            if (offload.submit)
                ++m_PipelineCompileOffload.outstanding;
        }
        if (offload.submit)
        {
            deferred->compileScope = offload.compileScope;
            deferred->scopeUser = offload.user;
            pso->deferred = std::move(deferred);
            pso->compileState.store(ComputePipeline::Queued, std::memory_order_relaxed);
            // The job's reference: the pipeline outlives a job that has not run
            // even when every handle to it is dropped first.
            pso->AddRef();
            ComputePipelineHandle handle = ComputePipelineHandle::Create(pso);
            offload.submit(offload.user, &ComputePipeline::runSubmittedCompile, pso);
            return handle;
        }

        res = m_Context.device.createComputePipelines(m_Context.pipelineCache,
                                                    1, &deferred->pipelineInfo,
                                                    m_Context.allocationCallbacks,
                                                    &pso->pipeline);

        CHECK_VK_FAIL(res)

        return ComputePipelineHandle::Create(pso);
    }

    void Device::beginPipelineCompileOffload(const PipelineCompileOffloadDesc& desc)
    {
        std::lock_guard<std::mutex> lock(m_PipelineCompileOffload.mutex);
        assert(!m_PipelineCompileOffload.desc.submit && desc.submit);
        m_PipelineCompileOffload.desc = desc;
        m_PipelineCompileOffload.failed = 0;
    }

    bool Device::endPipelineCompileOffload()
    {
        std::unique_lock<std::mutex> lock(m_PipelineCompileOffload.mutex);
        m_PipelineCompileOffload.desc = PipelineCompileOffloadDesc();
        m_PipelineCompileOffload.compiled.wait(lock,
            [this] { return m_PipelineCompileOffload.outstanding == 0; });
        const bool allCompiled = m_PipelineCompileOffload.failed == 0;
        m_PipelineCompileOffload.failed = 0;
        return allCompiled;
    }

    void ComputePipeline::runSubmittedCompile(void* jobContext)
    {
        ComputePipeline* pso = static_cast<ComputePipeline*>(jobContext);
        if (pso->tryClaim())
            pso->compile();
        pso->Release();
    }

    bool ComputePipeline::tryClaim()
    {
        uint8_t expected = Queued;
        return compileState.compare_exchange_strong(expected, Compiling, std::memory_order_acquire);
    }

    void ComputePipeline::claimOrWait()
    {
        if (tryClaim())
        {
            compile();
            return;
        }
        PipelineCompileOffload& offload = *m_Context.pipelineCompileOffload;
        std::unique_lock<std::mutex> lock(offload.mutex);
        offload.compiled.wait(lock,
            [this] { return compileState.load(std::memory_order_acquire) == Compiled; });
    }

    void ComputePipeline::compile()
    {
        const char* shaderName = desc.CS->getDesc().debugName.c_str();
        if (deferred->compileScope)
            deferred->compileScope(deferred->scopeUser, shaderName);
        const vk::Result res = m_Context.device.createComputePipelines(m_Context.pipelineCache,
                                                    1, &deferred->pipelineInfo,
                                                    m_Context.allocationCallbacks,
                                                    &pipeline);
        if (deferred->compileScope)
            deferred->compileScope(deferred->scopeUser, nullptr);
        deferred.reset();

        const bool compiled = res == vk::Result::eSuccess;
        if (!compiled)
        {
            pipeline = nullptr;
            m_Context.error(std::string("Offloaded compute pipeline compile failed for shader '")
                + shaderName + "': " + resultToString(VkResult(res)));
        }

        PipelineCompileOffload& offload = *m_Context.pipelineCompileOffload;
        {
            std::lock_guard<std::mutex> lock(offload.mutex);
            compileState.store(Compiled, std::memory_order_release);
            --offload.outstanding;
            if (!compiled)
                ++offload.failed;
        }
        offload.compiled.notify_all();
    }

    ComputePipeline::~ComputePipeline()
    {
        // A queued compile holds a reference, so the last release happens
        // after whichever thread compiled it has finished.
        assert(compileState.load(std::memory_order_acquire) == Compiled);

        if (pipeline)
        {
            m_Context.device.destroyPipeline(pipeline, m_Context.allocationCallbacks);
            pipeline = nullptr;
        }

        if (pipelineLayout)
        {
            m_Context.device.destroyPipelineLayout(pipelineLayout, m_Context.allocationCallbacks);
            pipelineLayout = nullptr;
        }
    }

    Object ComputePipeline::getNativeObject(ObjectType objectType)
    {
        switch (objectType)
        {
        case ObjectTypes::VK_PipelineLayout:
            return Object(pipelineLayout);
        case ObjectTypes::VK_Pipeline:
            ensureCompiled();
            return Object(pipeline);
        default:
            return nullptr;
        }
    }

    void CommandList::setComputeState(const ComputeState& state)
    {
        endRenderPass();

        assert(m_CurrentCmdBuf);

        ComputePipeline* pso = checked_cast<ComputePipeline*>(state.pipeline);

        if (m_EnableAutomaticBarriers)
        {
            insertComputeResourceBarriers(state);
        }

        if (m_CurrentComputeState.pipeline != state.pipeline)
        {
            pso->ensureCompiled();
            m_CurrentCmdBuf->cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, pso->pipeline);

            m_CurrentCmdBuf->referencedResources.push_back(state.pipeline);
        }

        if (state.indirectParams && state.indirectParams != m_CurrentComputeState.indirectParams)
        {
            m_CurrentCmdBuf->referencedResources.push_back(state.indirectParams);
        }

        if (arraysAreDifferent(m_CurrentComputeState.bindings, state.bindings) || m_AnyVolatileBufferWrites)
        {
            bindBindingSets(vk::PipelineBindPoint::eCompute, pso->pipelineLayout, state.bindings, pso->descriptorSetIdxToBindingIdx);
        }

        m_CurrentPipelineLayout = pso->pipelineLayout;
        m_CurrentPushConstantsVisibility = pso->pushConstantVisibility;

        commitBarriers();

        m_CurrentGraphicsState = GraphicsState();
        m_CurrentComputeState = state;
        m_CurrentMeshletState = MeshletState();
        m_CurrentRayTracingState = rt::State();
        m_AnyVolatileBufferWrites = false;
    }

    void CommandList::updateComputeVolatileBuffers()
    {
        if (m_AnyVolatileBufferWrites && m_CurrentComputeState.pipeline)
        {
            ComputePipeline* pso = checked_cast<ComputePipeline*>(m_CurrentComputeState.pipeline);

            bindBindingSets(vk::PipelineBindPoint::eCompute, pso->pipelineLayout, m_CurrentComputeState.bindings, pso->descriptorSetIdxToBindingIdx);

            m_AnyVolatileBufferWrites = false;
        }
    }

    void CommandList::dispatch(uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
    {
        RecordedCommandScope recordedCommand(m_StateTracker, __func__);
        assert(m_CurrentCmdBuf);

        updateComputeVolatileBuffers();

        m_CurrentCmdBuf->cmdBuf.dispatch(groupsX, groupsY, groupsZ);
    }

    void CommandList::dispatchIndirect(uint32_t offsetBytes)
    {
        RecordedCommandScope recordedCommand(m_StateTracker, __func__);
        assert(m_CurrentCmdBuf);

        updateComputeVolatileBuffers();

        Buffer* indirectParams = checked_cast<Buffer*>(m_CurrentComputeState.indirectParams);
        assert(indirectParams);

        m_CurrentCmdBuf->cmdBuf.dispatchIndirect(indirectParams->buffer, offsetBytes);
    }

} // namespace nvrhi::vulkan
