# build the shearlet example program
add_example(shearlet_example shearlet.cpp)

if(WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        # build the GPU projector speed test program
        add_example(speed_test speed_test.cpp)
    endif()
endif()
