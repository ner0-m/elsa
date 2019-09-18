cmake_minimum_required(VERSION 3.9)

# build the 2d example program
add_executable(example2d example2d.cpp)
target_link_libraries(example2d elsa_core elsa_projectors elsa_generators elsa_io elsa_solvers)
target_compile_features(example2d PUBLIC cxx_std_17)

if (BUILD_ELSA_PROJECTORS_CUDA)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        # build the 3d example program
        add_executable(example3d example3d.cpp)
        target_link_libraries(example3d elsa_core elsa_projectors_cuda elsa_generators elsa_io elsa_solvers)
        target_compile_features(example3d PUBLIC cxx_std_17)

        # build the GPU projector speed test program
        add_executable(speed_test speed_test.cpp)
        target_link_libraries(speed_test elsa_generators elsa_projectors_cuda elsa_logging)
        target_compile_features(speed_test PUBLIC cxx_std_17)
    endif()
endif()