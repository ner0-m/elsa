add_subdirectory(common)
add_subdirectory(projectors)

if(WITH_CUDA)
    add_subdirectory(projectors_cuda)
endif()

# setup the interface library which includes all the sub-libraries

message(STATUS "registered elsa targets:")
message(STATUS "${ELSA_REGISTERED_TARGETS}")

list(APPEND DEPS elsa_xray_common elsa_projectors)
if(WITH_CUDA)
    list(APPEND DEPS elsa_projectors_cuda)
endif()

add_library(elsa_xray INTERFACE)
add_library(elsa::xray ALIAS elsa_xray)

# link against all modules
target_link_libraries(elsa_xray INTERFACE ${DEPS})
target_include_directories(
    elsa_xray INTERFACE $<INSTALL_INTERFACE:include/elsa/xray-ct> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

write_module_config(xray DEPENDENCIES ${DEPS})

REGISTER_MODULE(xray)

install_elsa_module(xray elsa_xray elsa_xrayTargets)
