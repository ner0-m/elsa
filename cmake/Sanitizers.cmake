
if(ELSA_SANITIZE_THREAD AND ELSA_SANITIZE_ADDRESS)
    message(FATAL_ERROR "AddressSanitizer is not compatible with ThreadSanitizer.")
endif()

if(ELSA_SANITIZE_ADDRESS)
    message(STATUS "Address and undefiened-behavior sanitizer enabled")
    set(SANITIZERS "leak,address,undefined,shift,integer-divide-by-zero,unreachable,vla-bound,null,return,signed-integer-overflow")
    set(SANITIZER_FLAGS "-ggdb -O1 -fno-sanitize-recover=all -fno-omit-frame-pointer -fsanitize=${SANITIZERS}")
endif()

if(ELSA_SANITIZE_THREAD)
    message(STATUS "Thread sanitizer enabled")
    set(SANITIZER_FLAGS "-ggdb -O1 -fno-sanitize-recover=all -fno-omit-frame-pointer -fsanitize=thread")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${SANITIZER_FLAGS}")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} ${SANITIZER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_LINKER_FLAGS_DEBUG}")