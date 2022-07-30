pybind11_add_module(_core MODULE pyelsa/main.cpp)
install(TARGETS _core DESTINATION .)
