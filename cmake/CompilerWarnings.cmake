# Add warnings to a target.
#
# Call like: add_warnings_to_target(<target>)
#
# Arguments:
#
# * WARNINGS_AS_ERRORS: set the necessary flags to error on warnings
# * LEVEL: base|all base is a rather small set of important warnings, all is very aggressive with many warnings
#
# Adapted from: https://github.com/lefticus/cpp_starter_project/blob/master/cmake/CompilerWarnings.cmake
function(set_target_warnings target)
    set(options WARNINGS_AS_ERRORS)
    set(oneValueArgs LEVEL)
    set(multiValueArgs)
    cmake_parse_arguments(add_warnings "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(warning_level "base")

    # If it's defined use given warning level, default to base
    if(DEFINED add_warnings_LEVEL)
        string(TOLOWER ${add_warnings_LEVEL} warning_level)
    endif()

    set(base_msvc_warnings /W4 # Baseline reasonable warnings
                           /permissive- # standards conformance mode for MSVC compiler.
    )
    set(all_msvc_warnings
        ${base_msvc_warnings}
        /w14242 # 'identifier': conversion from 'type1' to 'type1', possible loss of data
        /w14254 # 'operator': conversion from 'type1:field_bits' to 'type2:field_bits', possible loss of data
        /w14263 # 'function': member function does not override any base class virtual member function
        /w14265 # 'classname': class has virtual functions, but destructor is not virtual instances of this class may
                # not be destructed correctly
        /w14287 # 'operator': unsigned/negative constant mismatch
        /we4289 # nonstandard extension used: 'variable': loop control variable declared in the for-loop is used outside
                # the for-loop scope
        /w14296 # 'operator': expression is always 'boolean_value'
        /w14311 # 'variable': pointer truncation from 'type1' to 'type2'
        /w14545 # expression before comma evaluates to a function which is missing an argument list
        /w14546 # function call before comma missing argument list
        /w14547 # 'operator': operator before comma has no effect; expected operator with side-effect
        /w14549 # 'operator': operator before comma has no effect; did you intend 'operator'?
        /w14555 # expression has no effect; expected expression with side- effect
        /w14619 # pragma warning: there is no warning number 'number'
        /w14640 # Enable warning on thread un-safe static member initialization
        /w14826 # Conversion from 'type1' to 'type_2' is sign-extended. This may cause unexpected runtime behavior.
        /w14905 # wide string literal cast to 'LPSTR'
        /w14906 # string literal cast to 'LPWSTR'
        /w14928 # illegal copy-initialization; more than one user-defined conversion has been implicitly applied
    )

    set(base_gcc_warnings
        -Wall # included -Wmisleading-indentation, -Wmost, -Wparentheses, -Wswitch, -Wswitch-bool.
        -Wextra # reasonable and standard
        -Wfatal-errors
    )

    if(WARNINGS_AS_ERRORS)
        set(base_gcc_warnings ${base_gcc_warnings} -Werror)
        set(msvc_base_warnings ${base_msvc_warnings} /WX)
    endif()

    # if(${ELSA_CUDA_ENABLED}) # NVCC emits lots of compiler-specific code, so we omit pedantic when cuda is enabled
    # else()

    set(all_clang_warnings
        ${base_gcc_warnings}
        -Wshadow # warn the user if a variable declaration shadows one from a parent context
        -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual destructor. This helps
                           # catch hard to track down memory errors
        -Wold-style-cast # warn for c-style casts
        -Wcast-align # warn for potential performance problem casts
        -Wunused # warn on anything being unused
        -Woverloaded-virtual # warn if you overload (not override) a virtual function
        -Wsign-conversion # warn on sign conversions
        -Wnull-dereference # warn if a null dereference is detected
        -Wformat=2 # warn on security issues around functions that format output (ie printf)
    )

    set(all_gcc_warnings
        ${base_gcc_warnings}
        -Wduplicated-cond # warn if if / else chain has duplicated conditions
        -Wduplicated-branches # warn if if / else branches have duplicated code
        -Wlogical-op # warn about logical operations being used where bitwise were probably wanted
        -Wuseless-cast # warn if you perform a cast to the same type
    )

    if(NOT ${warning_level} STREQUAL "base" AND NOT ${warning_level} STREQUAL "all")
        message(WARNING "Unknown warning level ${add_warnings_LEVEL}")
    endif()

    if(MSVC)
        set(project_warnings ${${warning_level}_msvc_warnings})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(project_warnings ${${warning_level}_clang_warnings})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(project_warnings ${${warning_level}_gcc_warnings})
    else()
        message(AUTHOR_WARNING "No compiler warnings set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
    endif()

    target_compile_options(${target} INTERFACE ${project_warnings})
endfunction()
