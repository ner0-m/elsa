# list all the headers of the module
set(MODULE_HEADERS EDFHandler.h MHDHandler.h PGMHandler.h IO.h ioUtils.h)

# list all the code files of the module
set(MODULE_SOURCES EDFHandler.cpp MHDHandler.cpp PGMHandler.cpp IO.cpp ioUtils.cpp)

list(APPEND MODULE_PUBLIC_DEPS elsa_core elsa_logging)
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    io "${MODULE_HEADERS}" "${MODULE_SOURCES}" PUBLIC_DEPS ${MODULE_PUBLIC_DEPS} PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

write_module_config(${ELSA_MODULE_NAME} DEPENDENCIES elsa_core elsa_logging)
