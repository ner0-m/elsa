# list all the headers of the module
set(MODULE_HEADERS Logger.h LogGuard.h Timer.h)

# list all the code files of the module
set(MODULE_SOURCES Logger.cpp LogGuard.cpp Timer.cpp)

list(APPEND MODULE_PUBLIC_DEPS "spdlog::spdlog_header_only")
list(APPEND MODULE_PRIVATE_DEPS)

ADD_ELSA_MODULE(
    logging "${MODULE_HEADERS}" "${MODULE_SOURCES}" INSTALL_DIR PUBLIC_DEPS ${MODULE_PUBLIC_DEPS}
    PRIVATE_DEPS ${MODULE_PRIVATE_DEPS}
)

write_module_config(${ELSA_MODULE_NAME})
