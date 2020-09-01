cmake_minimum_required(VERSION 3.9)

# enable ctest and Catch test discovery
include(CTest)
include(Catch)

add_custom_target(build-benchmarks) 
 
macro(ELSA_BENCHMARK NAME)
    # Add source to all sources, so we can create a common target to build and run the benchmarks
    SET(BENCHMARK_SOURCES ${BENCHMARK_SOURCES} bench_${NAME}.cpp)

    # create the test executable
    add_executable(bench_${NAME} EXCLUDE_FROM_ALL bench_${NAME}.cpp bench_main.cpp)
    # add catch and the corresponding elsa library
    target_link_libraries(bench_${NAME} PRIVATE Catch2::Catch2 elsa::all)
     
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA) 
        find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
        target_include_directories(bench_${NAME}
            PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
     
    # enable C++17
    target_compile_features(bench_${NAME} PUBLIC cxx_std_17)
    # include helpers
    target_include_directories(bench_${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/elsa/test_routines/)
     
    # Add benchmark to build-benchmarks 
    add_dependencies(build-benchmarks bench_${NAME}) 
endmacro(ELSA_BENCHMARK)

ELSA_BENCHMARK(RayGenerationBench)
ELSA_BENCHMARK(Projectors)
ELSA_BENCHMARK(Intersection)
ELSA_BENCHMARK(DataContainerConstruction)
ELSA_BENCHMARK(ExpressionTemplates)
 
# Add a single executable for all benchmarks, as CTest removes a lot of the output
add_executable(bench_all EXCLUDE_FROM_ALL bench_main.cpp ${BENCHMARK_SOURCES})
# add catch and the corresponding elsa library
target_link_libraries(bench_all PRIVATE Catch2::Catch2 elsa::all)
# enable C++17
target_compile_features(bench_all PUBLIC cxx_std_17)
# include helpers
target_include_directories(bench_all PRIVATE ${CMAKE_SOURCE_DIR}/elsa/test_routines/)

# Add the custom target to run all the benchmarks
add_custom_target(run-benchmarks
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bench_all
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
        COMMENT "Run benchmarks")
add_dependencies(run-benchmarks bench_all)
