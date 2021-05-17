function(write_module_config module_name)
    if(ELSA_INSTALL)
        # Parse arguments
        set(options)
        set(oneValueArgs)
        set(multiValueArgs DEPENDENCIES)
        cmake_parse_arguments(INSTALL_MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
         
        # set up the target/library for make install
        include(GNUInstallDirs)
        include(CMakePackageConfigHelpers)

        # This uses INSTALL_MODULE_DEPEDENCIES! to replace in the .in file
        configure_package_config_file(
            ${PROJECT_SOURCE_DIR}/cmake/elsaComponentConfig.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/elsa/elsa_${module_name}Config.cmake
            INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
        )

        # install the config files
        install(
                FILES
                ${CMAKE_CURRENT_BINARY_DIR}/elsa/elsa_${module_name}Config.cmake
                DESTINATION ${INSTALL_CONFIG_DIR}
        )
    endif() 
endfunction()

# install an elsa module
function(install_elsa_module ELSA_MODULE_NAME ELSA_MODULE_TARGET_NAME ELSA_MODULE_EXPORT_TARGET)
    if(ELSA_INSTALL)
        # This is required so that the exported target has the name core and not elsa_core
        set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES EXPORT_NAME ${ELSA_MODULE_NAME})

        include(GNUInstallDirs)
        # install the module
        install(TARGETS ${ELSA_MODULE_TARGET_NAME}
                EXPORT ${ELSA_MODULE_EXPORT_TARGET}
                # INCLUDES DESTINATION include
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
        # install the header files
        install(FILES ${MODULE_HEADERS}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/elsa/${ELSA_MODULE_NAME}
        )
        # create the config file for the module
        install(EXPORT ${ELSA_MODULE_EXPORT_TARGET}
                FILE ${ELSA_MODULE_EXPORT_TARGET}.cmake
                NAMESPACE elsa::
                DESTINATION ${INSTALL_CONFIG_DIR}
        )
    endif(ELSA_INSTALL)
endfunction()

# Install a module using a directory
# Install the target "ELSA_MODULE_TARGET_NAME" with the exported name "ELSA_MODULE_EXPORT_TARGET"
# by installing all the files given in "MODULE_DIRECTORY" to installprefix/elsa/module_name
# 
# This method preserves all the hierarchical structures of the directory (sub folders are preserved)
function(install_elsa_module_dir ELSA_MODULE_NAME ELSA_MODULE_TARGET_NAME ELSA_MODULE_EXPORT_TARGET MODULE_DIRECTORY)
    if(ELSA_INSTALL)
        # This is required so that the exported target has the name core and not elsa_core
        set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES EXPORT_NAME ${ELSA_MODULE_NAME})

        include(GNUInstallDirs)
        # install the module
        install(TARGETS ${ELSA_MODULE_TARGET_NAME}
                EXPORT ${ELSA_MODULE_EXPORT_TARGET}
                INCLUDES DESTINATION include
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
        # install the header files
	install(DIRECTORY ${MODULE_DIRECTORY}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/elsa/${ELSA_MODULE_NAME}
		FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
        )
        # create the config file for the module
        install(EXPORT ${ELSA_MODULE_EXPORT_TARGET}
                FILE ${ELSA_MODULE_EXPORT_TARGET}.cmake
                NAMESPACE elsa::
                DESTINATION ${INSTALL_CONFIG_DIR}
        )
    endif(ELSA_INSTALL)
endfunction()
