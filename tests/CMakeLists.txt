add_executable(nvrhi_vk_submit_sync_test
    vulkan-submit-sync-test.cpp)

target_link_libraries(nvrhi_vk_submit_sync_test PRIVATE
    nvrhi
    ${nvrhi_vulkan_target}
    Vulkan::Vulkan)
target_include_directories(nvrhi_vk_submit_sync_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/vulkan)

if (MSVC)
    target_compile_definitions(nvrhi_vk_submit_sync_test PRIVATE NOMINMAX)
endif()

add_test(NAME nvrhi_vk_submit_sync COMMAND nvrhi_vk_submit_sync_test)
set_tests_properties(nvrhi_vk_submit_sync PROPERTIES TIMEOUT 30)
if (WIN32 AND DEFINED ENV{VULKAN_SDK})
    set_tests_properties(nvrhi_vk_submit_sync PROPERTIES
        ENVIRONMENT "VK_ADD_LAYER_PATH=$ENV{VULKAN_SDK}/Bin")
endif()
