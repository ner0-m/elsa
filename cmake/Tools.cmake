# Code heavily influenced by https://github.com/StableCoder/cmake-scripts/
#   original copyright by George Cave - gcave@stablecoder.ca
#   original license Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# Helper append function
function(append value)
  foreach(variable ${ARGN})
    set(${variable}
        "${${variable}} ${value}"
        PARENT_SCOPE)
  endforeach(variable)
endfunction()

#
# clang-tidy
#

# Adds clang-tidy checks to the compilation, with the given arguments being used
# as the options set.
macro(clang_tidy)
  if(ELSA_CLANG_TIDY AND CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE} ${ARGN})
  endif()
endmacro()

# Find clang-tidy
find_program(CLANG_TIDY_EXE NAMES "clang-tidy-8")
mark_as_advanced(FORCE CLANG_TIDY_EXE)
if(CLANG_TIDY_EXE)
  message(STATUS "clang-tidy-8 found: ${CLANG_TIDY_EXE}")
  if(NOT ELSA_CLANG_TIDY)
    message(STATUS "clang-tidy-8 NOT ENABLED via 'ELSA_CLANG_TIDY' variable!")
    set(CMAKE_CXX_CLANG_TIDY
            ""
            CACHE STRING "" FORCE) # delete it
  endif()
elseif(ELSA_CLANG_TIDY)
  message(SEND_ERROR "Cannot enable clang-tidy-8, as executable not found!")
  set(CMAKE_CXX_CLANG_TIDY
          ""
          CACHE STRING "" FORCE) # delete it
else()
  message(STATUS "clang-tidy-8 not found!")
  set(CMAKE_CXX_CLANG_TIDY
          ""
          CACHE STRING "" FORCE) # delete it
endif()

#
# clang-format
#
find_program(CLANG_FORMAT_EXE "clang-format-8")
mark_as_advanced(FORCE CLANG_FORMAT_EXE)
if(CLANG_FORMAT_EXE)
  message(STATUS "clang-format-8 found: ${CLANG_FORMAT_EXE}")
else()
  message(STATUS "clang-format-8 not found!")
endif()

# Generates a 'format' target using a custom name, files, and include
# directories all being parameters.
#
# Do note that in order for sources to be inherited properly, the source paths
# must be reachable from where the macro is called, or otherwise require a full
# path for proper inheritance.
#
# ~~~
# Required:
# TARGET_NAME - The name of the target to create.
#
# Optional: ARGN - The list of targets OR files to format. Relative and absolute
# paths are accepted.
# ~~~
function(clang_format TARGET_NAME)
  if(CLANG_FORMAT_EXE)
    set(FORMAT_FILES)
    # Check through the ARGN's, determine existent files
    foreach(item IN LISTS ARGN)
      if(TARGET ${item})
        # If the item is a target, then we'll attempt to grab the associated
        # source files from it.
        get_target_property(_TARGET_TYPE ${item} TYPE)
        if(NOT _TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
          get_property(
            _TEMP
            TARGET ${item}
            PROPERTY SOURCES)
          foreach(iter IN LISTS _TEMP)
            if(EXISTS ${iter})
              set(FORMAT_FILES ${FORMAT_FILES} ${iter})
            endif()
          endforeach()
        endif()
      elseif(EXISTS ${item})
        # Check if it's a full file path
        set(FORMAT_FILES ${FORMAT_FILES} ${item})
      elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${item})
        # Check if it's based on the current source dir
        set(FORMAT_FILES ${FORMAT_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/${item})
      endif()
    endforeach()

    # Make the target
    if(FORMAT_FILES)
      if(NOT TARGET ${TARGET_NAME})
        add_custom_target(${TARGET_NAME} COMMAND ${CLANG_FORMAT_EXE} -i
                                                 -style=file ${FORMAT_FILES})
      endif()
    endif()

  endif()
endfunction()

if(ELSA_SANITIZER)
  if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    message(WARNING "Sanitizers should preferably run in Debug mode.")
  endif()

  append("-fno-omit-frame-pointer -fno-sanitize-recover=all" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

  if(UNIX)
    set(USING_CLANG FALSE)
    if(CMAKE_C_COMPILER_ID MATCHES "(Apple)?[Cc]lang" OR CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
      set(USING_CLANG TRUE)
    endif()
    set(USING_GNU FALSE)
    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
      set(USING_GNU TRUE)
    endif()

    if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
      append("-O1" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    endif()

    # Address and UB sanitizer
    if(ELSA_SANITIZER MATCHES "([Aa]ddress);([Uu]ndefined)"
       OR ELSA_SANITIZER MATCHES "([Uu]ndefined);([Aa]ddress)")
      message(STATUS "Building with Address, Undefined sanitizers")
      append("-fsanitize=address,undefined" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

    # Address sanitizer
    elseif(ELSA_SANITIZER MATCHES "([Aa]ddress)")
      message(STATUS "Building with Address sanitizer")
      append("-fsanitize=address" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

    # Memory sanitizer (only Clang)
    elseif(ELSA_SANITIZER MATCHES "([Mm]emory([Ww]ith[Oo]rigins)?)")
      if(USING_CLANG AND NOT USING_GNU)
        append("-fsanitize=memory" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
        if(ELSA_SANITIZER MATCHES "([Mm]emory[Ww]ith[Oo]rigins)")
          message(STATUS "Building with MemoryWithOrigins sanitizer")
          append("-fsanitize-memory-track-origins" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
        else()
          message(STATUS "Building with Memory sanitizer")
        endif()
      else()
        message(STATUS "Cannot use Memory sanitizer with GNU compiler. Use Address instead.")
      endif()

    # UB sanitizer
    elseif(ELSA_SANITIZER MATCHES "([Uu]ndefined)")
      message(STATUS "Building with Undefined sanitizer")
      append("-fsanitize=undefined" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
      if(EXISTS "${BLACKLIST_FILE}")
        append("-fsanitize-blacklist=${BLACKLIST_FILE}" CMAKE_C_FLAGS
               CMAKE_CXX_FLAGS)
      endif()

    # Thread sanitizer
    elseif(ELSA_SANITIZER MATCHES "([Tt]hread)")
      message(STATUS "Building with Thread sanitizer")
      append("-fsanitize=thread" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
      
    else()
      message(
        FATAL_ERROR "Unsupported value of ELSA_SANITIZER: ${ELSA_SANITIZER}")
    endif()

  elseif(MSVC) # if(UNIX)
    if(ELSA_SANITIZER MATCHES "([Aa]ddress)")
      message(STATUS "Building with Address sanitizer")
      append("-fsanitize=address" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    else()
      message(FATAL_ERROR
          "This sanitizer is not yet supported in the MSVC environment: ${ELSA_SANITIZER}"
      )
    endif()

  else() # elseif(MSVC)
    message(FATAL_ERROR "ELSA_SANITIZER is not supported on this platform.")
  endif() # if (UNIX)

endif() # if(ELSA_SANITIZER)
