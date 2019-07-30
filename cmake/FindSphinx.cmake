# find the Sphinx executable
# inspired from https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/

find_program(SPHINX_EXECUTABLE
        NAMES sphinx-build
        DOC "Path to the Sphinx documentation generator")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Sphinx
        "Sphinx executable (sphinx-build) not found"
        SPHINX_EXECUTABLE)
