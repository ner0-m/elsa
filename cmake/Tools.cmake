# Code heavily influenced by https://github.com/StableCoder/cmake-scripts/
#   original copyright by George Cave - gcave@stablecoder.ca
#   original license Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

function(append value)
  foreach(variable ${ARGN})
    set(${variable}
        "${${variable}} ${value}"
        PARENT_SCOPE)
  endforeach(variable)
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
