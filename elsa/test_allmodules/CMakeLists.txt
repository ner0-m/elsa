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

# link Quickvec for DataHandlerGPU
# Also add the necessary compile definition for quickvec
if(${ELSA_BUILD_WITH_QUICKVEC})
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(integration-tests PUBLIC ELSA_ENABLE_CUDA_VECTOR)
    # target_link_libraries(integration-tests PUBLIC "Quickvec::quickvec")
    target_compile_features(integration-tests PUBLIC cuda_std_17)
endif()