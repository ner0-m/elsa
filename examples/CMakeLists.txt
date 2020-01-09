cmake_minimum_required(VERSION 3.9)

# define a target for all examples
add_custom_target(examples)

# build the 2d example program
add_executable(example2d example2d.cpp)
target_link_libraries(example2d elsa::all)
target_compile_features(example2d PUBLIC cxx_std_17)
add_dependencies(examples example2d)


if(ELSA_BUILD_CUDA_PROJECTORS)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        # build the 3d example program
        add_executable(example3d example3d.cpp)
        target_link_libraries(example3d elsa::all)
        target_compile_features(example3d PUBLIC cxx_std_17)
        add_dependencies(examples example3d)

        # build the GPU projector speed test program
        add_executable(speed_test speed_test.cpp)
        target_link_libraries(speed_test elsa::all)
        target_compile_features(speed_test PUBLIC cxx_std_17)
        add_dependencies(examples speed_test)
    endif()
endif()
