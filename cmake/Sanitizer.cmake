# Code heavily influenced by https://github.com/arsenm/sanitizers-cmake/ original license MIT License

set(ELSA_SANITIZE_ADDRESS OFF)
set(ELSA_SANITIZE_MEMORY OFF)
set(ELSA_SANITIZE_UNDEFINED OFF)
set(ELSA_SANITIZE_THREAD OFF)

if(ELSA_SANITIZER MATCHES "([Aa]ddress);([Uu]ndefined)" OR ELSA_SANITIZER MATCHES "([Uu]ndefined);([Aa]ddress)")
    set(ELSA_SANITIZE_ADDRESS ON)
    set(ELSA_SANITIZE_UNDEFINED ON)
elseif(ELSA_SANITIZER MATCHES "[Aa]ddress")
    set(ELSA_SANITIZE_ADDRESS ON)
elseif(ELSA_SANITIZER MATCHES "[Mm]emory")
    set(ELSA_SANITIZE_MEMORY ON)
elseif(ELSA_SANITIZER MATCHES "([Uu]ndefined)")
    set(ELSA_SANITIZE_UNDEFINED ON)
elseif(ELSA_SANITIZER MATCHES "([Tt]hread)")
    set(ELSA_SANITIZE_THREAD ON)
endif()

# Helper function to get the language of a source file.
function(sanitizer_lang_of_source FILE RETURN_VAR)
    get_filename_component(LONGEST_EXT "${FILE}" EXT)
    # If extension is empty return. This can happen for extensionless headers
    if("${LONGEST_EXT}" STREQUAL "")
        set(${RETURN_VAR} "" PARENT_SCOPE)
        return()
    endif()
    # Get shortest extension as some files can have dot in their names
    string(REGEX REPLACE "^.*(\\.[^.]+)$" "\\1" FILE_EXT ${LONGEST_EXT})
    string(TOLOWER "${FILE_EXT}" FILE_EXT)
    string(SUBSTRING "${FILE_EXT}" 1 -1 FILE_EXT)

    get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    foreach(LANG ${ENABLED_LANGUAGES})
        list(FIND CMAKE_${LANG}_SOURCE_FILE_EXTENSIONS "${FILE_EXT}" TEMP)
        if(NOT ${TEMP} EQUAL -1)
            set(${RETURN_VAR} "${LANG}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${RETURN_VAR} "" PARENT_SCOPE)
endfunction()

# Helper function to get compilers used by a target.
function(sanitizer_target_compilers TARGET RETURN_VAR)
    # Check if all sources for target use the same compiler. If a target uses e.g. C and Fortran mixed and uses
    # different compilers (e.g. clang and gfortran) this can trigger huge problems, because different compilers may use
    # different implementations for sanitizers.
    set(BUFFER "")
    get_target_property(TSOURCES ${TARGET} SOURCES)
    foreach(FILE ${TSOURCES})
        # If expression was found, FILE is a generator-expression for an object library. Object libraries will be
        # ignored.
        string(REGEX MATCH "TARGET_OBJECTS:([^ >]+)" _file ${FILE})
        if("${_file}" STREQUAL "")
            sanitizer_lang_of_source(${FILE} LANG)
            if(LANG)
                list(APPEND BUFFER ${CMAKE_${LANG}_COMPILER_ID})
            endif()
        endif()
    endforeach()

    list(REMOVE_DUPLICATES BUFFER)
    set(${RETURN_VAR} "${BUFFER}" PARENT_SCOPE)
endfunction()

# Helper function to check compiler flags for language compiler.
function(sanitizer_check_compiler_flag FLAG LANG VARIABLE)
    if(${LANG} STREQUAL "CXX")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("${FLAG}" ${VARIABLE})
    endif()
endfunction()

# Helper function to test compiler flags.
function(sanitizer_check_compiler_flags FLAG_CANDIDATES NAME PREFIX)
    set(CMAKE_REQUIRED_QUIET ${${PREFIX}_FIND_QUIETLY})

    get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    foreach(LANG ${ENABLED_LANGUAGES})
        if(${LANG} STREQUAL "NONE")
            continue()
        endif()

        # Sanitizer flags are not dependent on language, but the used compiler. So instead of searching flags foreach
        # language, search flags foreach compiler used.
        set(COMPILER ${CMAKE_${LANG}_COMPILER_ID})
        if(NOT DEFINED ${PREFIX}_${COMPILER}_FLAGS)
            foreach(FLAG ${FLAG_CANDIDATES})
                if(NOT CMAKE_REQUIRED_QUIET)
                    message(STATUS "Try ${COMPILER} ${NAME} flag = [${FLAG}]")
                endif()

                set(CMAKE_REQUIRED_FLAGS "${FLAG}")
                unset(${PREFIX}_FLAG_DETECTED CACHE)
                sanitizer_check_compiler_flag("${FLAG}" ${LANG} ${PREFIX}_FLAG_DETECTED)

                if(${PREFIX}_FLAG_DETECTED)
                    # If compiler is a GNU compiler, search for static flag, if SANITIZE_LINK_STATIC is enabled.
                    if(SANITIZE_LINK_STATIC AND (${COMPILER} STREQUAL "GNU"))
                        string(TOLOWER ${PREFIX} PREFIX_lower)
                        sanitizer_check_compiler_flag(
                            "-static-lib${PREFIX_lower}" ${LANG} ${PREFIX}_STATIC_FLAG_DETECTED
                        )

                        if(${PREFIX}_STATIC_FLAG_DETECTED)
                            set(FLAG "-static-lib${PREFIX_lower} ${FLAG}")
                        endif()
                    endif()

                    set(${PREFIX}_${COMPILER}_FLAGS "${FLAG}" CACHE STRING "${NAME} flags for ${COMPILER} compiler.")
                    mark_as_advanced(${PREFIX}_${COMPILER}_FLAGS)

                    message(STATUS "Activating ${PREFIX} flag = [${FLAG}]")
                    break()
                endif()
            endforeach()

            if(NOT ${PREFIX}_FLAG_DETECTED)
                set(${PREFIX}_${COMPILER}_FLAGS "" CACHE STRING "${NAME} flags for ${COMPILER} compiler.")
                mark_as_advanced(${PREFIX}_${COMPILER}_FLAGS)

                message(WARNING "${NAME} is not available for ${COMPILER} "
                                "compiler. Targets using this compiler will be " "compiled without ${NAME}."
                )
            endif()
        endif()
    endforeach()
endfunction()

# Helper to assign sanitizer flags for TARGET.
function(sanitizer_add_flags TARGET NAME PREFIX)
    # Get list of compilers used by target and check, if sanitizer is available for this target. Other compiler checks
    # like check for conflicting compilers will be done in add_sanitizers function.
    sanitizer_target_compilers(${TARGET} TARGET_COMPILER)
    list(LENGTH TARGET_COMPILER NUM_COMPILERS)
    if("${${PREFIX}_${TARGET_COMPILER}_FLAGS}" STREQUAL "")
        return()
    endif()

    # Set compile- and link-flags for target.
    set_property(TARGET ${TARGET} APPEND_STRING PROPERTY COMPILE_FLAGS " ${${PREFIX}_${TARGET_COMPILER}_FLAGS}")
    set_property(TARGET ${TARGET} APPEND_STRING PROPERTY COMPILE_FLAGS " ${SanBlist_${TARGET_COMPILER}_FLAGS}")
    set_property(TARGET ${TARGET} APPEND_STRING PROPERTY LINK_FLAGS " ${${PREFIX}_${TARGET_COMPILER}_FLAGS}")
endfunction()

function(add_sanitize_address TARGET)
    if(ELSA_SANITIZE_ADDRESS AND (ELSA_SANITIZE_THREAD OR ELSA_SANITIZE_MEMORY))
        message(FATAL_ERROR "AddressSanitizer is not compatible with " "ThreadSanitizer or MemorySanitizer.")
    endif()

    if(NOT ELSA_SANITIZE_ADDRESS)
        return()
    endif()

    set(FLAG_CANDIDATES
        # Clang 3.2+ use this version. The no-omit-frame-pointer option is optional.
        "-g -fsanitize=address -fno-omit-frame-pointer" "-g -fsanitize=address"
        # Older deprecated flag for ASan
        "-g -faddress-sanitizer"
    )

    sanitizer_check_compiler_flags("${FLAG_CANDIDATES}" "AddressSanitizer" "ASan")
    sanitizer_add_flags(${TARGET} "AddressSanitizer" "ASan")
endfunction()

function(add_sanitize_undefined TARGET)
    if(NOT ELSA_SANITIZE_UNDEFINED)
        return()
    endif()

    set(FLAG_CANDIDATES "-g -fsanitize=undefined")
    sanitizer_check_compiler_flags("${FLAG_CANDIDATES}" "UndefinedBehaviorSanitizer" "UBSan")

    sanitizer_add_flags(${TARGET} "UndefinedBehaviorSanitizer" "UBSan")
endfunction()

function(add_sanitize_address_and_ub TARGET)
    if(ELSA_SANITIZE_ADDRESS AND (ELSA_SANITIZE_THREAD OR ELSA_SANITIZE_MEMORY))
        message(FATAL_ERROR "AddressSanitizer is not compatible with " "ThreadSanitizer or MemorySanitizer.")
    endif()

    if(NOT ELSA_SANITIZE_ADDRESS)
        return()
    endif()

    set(FLAG_CANDIDATES # Clang 3.2+ use this version. The no-omit-frame-pointer option is optional.
        "-g -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer"
        "-g -fsanitize=address -fsanitize=undefined"
    )

    sanitizer_check_compiler_flags("${FLAG_CANDIDATES}" "AddressSanitizer" "ASan")
    sanitizer_add_flags(${TARGET} "AddressSanitizer" "ASan")
endfunction()

function(add_sanitize_thread TARGET)
    if(ELSA_SANITIZE_THREAD AND ELSA_SANITIZE_MEMORY)
        message(FATAL_ERROR "ThreadSanitizer is not compatible with " "MemorySanitizer.")
    endif()

    if(NOT ELSA_SANITIZE_THREAD)
        return()
    endif()

    set(FLAG_CANDIDATES "-g -fsanitize=thread")

    if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
        message(WARNING "ThreadSanitizer disabled for target ${TARGET} because "
                        "ThreadSanitizer is supported for Linux systems and macOS only."
        )
        set(ELSA_SANITIZE_THREAD Off CACHE BOOL "Enable ThreadSanitizer for sanitized targets." FORCE)
    elseif(NOT ${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        message(WARNING "ThreadSanitizer disabled for target ${TARGET} because "
                        "ThreadSanitizer is supported for 64bit systems only."
        )
        set(ELSA_SANITIZE_THREAD Off CACHE BOOL "Enable ThreadSanitizer for sanitized targets." FORCE)
    else()
        sanitizer_check_compiler_flags("${FLAG_CANDIDATES}" "ThreadSanitizer" "TSan")
    endif()

    sanitizer_add_flags(${TARGET} "ThreadSanitizer" "TSan")
endfunction()

function(add_sanitize_memory TARGET)
    if(NOT ELSA_SANITIZE_MEMORY)
        return()
    endif()

    set(FLAG_CANDIDATES "-g -fsanitize=memory")

    if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        message(WARNING "MemorySanitizer disabled for target ${TARGET} because "
                        "MemorySanitizer is supported for Linux systems only."
        )
        set(ELSA_SANITIZE_MEMORY Off CACHE BOOL "Enable MemorySanitizer for sanitized targets." FORCE)
    elseif(NOT ${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        message(WARNING "MemorySanitizer disabled for target ${TARGET} because "
                        "MemorySanitizer is supported for 64bit systems only."
        )
        set(ELSA_SANITIZE_MEMORY Off CACHE BOOL "Enable MemorySanitizer for sanitized targets." FORCE)
    else()
        sanitizer_check_compiler_flags("${FLAG_CANDIDATES}" "MemorySanitizer" "MSan")
    endif()

    sanitizer_add_flags(${TARGET} "MemorySanitizer" "MSan")
endfunction()

function(add_sanitizers)
    # If no sanitizer is enabled, return immediately.
    if(NOT
       (ELSA_SANITIZE_ADDRESS
        OR ELSA_SANITIZE_MEMORY
        OR ELSA_SANITIZE_UNDEFINED
        OR ELSA_SANITIZE_THREAD)
    )
        return()
    endif()

    foreach(TARGET ${ARGV})
        # Check if this target will be compiled by exactly one compiler. Other- wise sanitizers can't be used and a
        # warning should be printed once.
        get_target_property(TARGET_TYPE ${TARGET} TYPE)
        if(TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
            message(WARNING "Can't use any sanitizers for target ${TARGET}, "
                            "because it is an interface library and cannot be " "compiled directly."
            )
            return()
        endif()
        sanitizer_target_compilers(${TARGET} TARGET_COMPILER)
        list(LENGTH TARGET_COMPILER NUM_COMPILERS)
        if(NUM_COMPILERS GREATER 1)
            message(WARNING "Can't use any sanitizers for target ${TARGET}, "
                            "because it will be compiled by incompatible compilers. "
                            "Target will be compiled without sanitizers."
            )
            return()

        elseif(NUM_COMPILERS EQUAL 0)
            # If the target is compiled by no or no known compiler, give a warning.
            message(WARNING "Sanitizers for target ${TARGET} may not be"
                            " usable, because it uses no or an unknown compiler. "
                            "This is a false warning for targets using only " "object lib(s) as input."
            )
            return()
        endif()

        # Add sanitizers for target.
        add_sanitize_address(${TARGET})
        add_sanitize_undefined(${TARGET})
        add_sanitize_thread(${TARGET})
        add_sanitize_memory(${TARGET})
    endforeach()
endfunction(add_sanitizers)
