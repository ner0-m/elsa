# Example for an allocation pattern where a cache resource is appropriate
add_example(cache_resource cache_resource.cpp)

# Example for an allocation pattern where a region resource is appropriate
add_example(region_resource region_resource.cpp)

# Example for using the dedicated memory resources and scoped mr
add_example(memory_resource memory_resource.cpp)
if(ELSA_CUDA_ENABLED)
    set_source_files_properties(memory_resource.cpp PROPERTIES LANGUAGE CUDA)
endif()
