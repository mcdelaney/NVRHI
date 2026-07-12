// Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.

#pragma once

#include <vulkan/vulkan.hpp>

namespace nvrhi::vulkan::detail
{
    inline bool isWaitStageMaskSupported(
        vk::PipelineStageFlags2 stageMask,
        vk::QueueFlags queueFlags)
    {
        const VkPipelineStageFlags2 rawMask = static_cast<VkPipelineStageFlags2>(stageMask);
        const VkQueueFlags rawQueueFlags = static_cast<VkQueueFlags>(queueFlags);

        VkPipelineStageFlags2 allowed =
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT |
            VK_PIPELINE_STAGE_2_HOST_BIT |
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        if ((rawQueueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
        {
            allowed |=
                VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
                VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |
                VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
                VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
                VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
                VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT |
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR |
                VK_PIPELINE_STAGE_2_FRAGMENT_DENSITY_PROCESS_BIT_EXT |
                VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                VK_PIPELINE_STAGE_2_SUBPASS_SHADER_BIT_HUAWEI |
                VK_PIPELINE_STAGE_2_INVOCATION_MASK_BIT_HUAWEI |
                VK_PIPELINE_STAGE_2_CLUSTER_CULLING_SHADER_BIT_HUAWEI;
        }

        if ((rawQueueFlags & VK_QUEUE_COMPUTE_BIT) != 0)
        {
            allowed |=
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
        }

        if ((rawQueueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != 0)
        {
            allowed |=
                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT |
                VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT |
                VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT;
        }

        if ((rawQueueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) != 0)
        {
            allowed |=
                VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT |
                VK_PIPELINE_STAGE_2_COPY_BIT |
                VK_PIPELINE_STAGE_2_RESOLVE_BIT |
                VK_PIPELINE_STAGE_2_BLIT_BIT |
                VK_PIPELINE_STAGE_2_CLEAR_BIT |
                VK_PIPELINE_STAGE_2_COPY_INDIRECT_BIT_KHR |
                VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR |
                VK_PIPELINE_STAGE_2_CONVERT_COOPERATIVE_VECTOR_MATRIX_BIT_NV;
        }

        return rawMask != 0 && (rawMask & ~allowed) == 0;
    }
}
