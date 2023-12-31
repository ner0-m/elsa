# ######################################################################################################################
# Check if elsa_arch_type is set to auto, if yes, then the current CUDA capabilities are checked added to
function(set_cuda_arch_type elsa_arch_type)
    string(TOLOWER "${elsa_arch_type}" _arch_type)

    if(${_arch_type} STREQUAL "auto")
        cuda_detect_installed_gpus(_detected_capabilities)
        string(STRIP "${_detected_capabilities}" _detected_capabilities)
        if(CUDA_GPU_DETECT_OUTPUT)
            message(STATUS "Automatically detected GPU architectures: ${_detected_capabilities}")
        else()
            message(STATUS "Common architectures: ${_detected_capabilities}")
        endif()
        string(REPLACE " " ";" _target_gpus "${_detected_capabilities}")
        list(REMOVE_DUPLICATES _target_gpus)
    else()
        set(_target_gpus "${elsa_arch_type}")
    endif()

    string(REPLACE "." "" _target_gpus "${_target_gpus}")
    # TODO: Remove me once we upgrade to version 3.18
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
        string(REPLACE " " ";" _target_gpus "${_target_gpus}")
        string(REGEX REPLACE "([1-9][0-9]+)[+]PTX" "\\1-virtual" _target_gpus "${_target_gpus}")
        string(REGEX REPLACE "([1-9][0-9]+)([^-])" "\\1-real\\2" _target_gpus "${_target_gpus}")
        set(CMAKE_CUDA_ARCHITECTURES "${_target_gpus}" PARENT_SCOPE)
    else()
        string(REPLACE " " ";" _target_gpus "${_target_gpus}")
        foreach(target_gpu ${_target_gpus})
            if(target_gpu MATCHES "[1-9][0-9]+[+]PTX")
                string(REGEX REPLACE "([1-9][0-9]+)[+]PTX" "\\1" target_gpu "${target_gpu}")
                set(CMAKE_CUDA_FLAGS
                    "${CMAKE_CUDA_FLAGS} --generate-code arch=compute_${target_gpu},code=[sm_${target_gpu},compute_${target_gpu}]"
                )
            else()
                set(CMAKE_CUDA_FLAGS
                    "${CMAKE_CUDA_FLAGS} --generate-code arch=compute_${target_gpu},code=[sm_${target_gpu}]"
                )
            endif()
        endforeach()
    endif()
endfunction()
