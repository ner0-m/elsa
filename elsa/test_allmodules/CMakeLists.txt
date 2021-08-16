# Check for CUDA first
if(${ELSA_BUILD_CUDA_PROJECTORS})
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    endif()
endif()

# set sources
set(SOURCES
    integration_main.cpp
    SetupHelpers.cpp)

set(HEADERS
    SetupHelpers.h
    LoggingHelpers.h
    TestDriver.h)

# Create executable and link to elsa::all
add_executable(integration-tests ${SOURCES})
target_link_libraries(integration-tests elsa::all)
target_compile_features(integration-tests PUBLIC cxx_std_17)
