# install an elsa module

function(InstallElsaModule ELSA_MODULE_NAME ELSA_MODULE_TARGET_NAME ELSA_MODULE_EXPORT_TARGET ELSA_COMPONENT_DEPENDENCIES)
    if(ELSA_INSTALL)
        # This is required so that the exported target has the name core and not elsa_core
        set_target_properties(${ELSA_MODULE_TARGET_NAME} PROPERTIES EXPORT_NAME ${ELSA_MODULE_NAME})

        message(STATUS "Dependencies for ${ELSA_MODULE_TARGET_NAME}: ${ELSA_COMPONENT_DEPENDENCIES}")

        include(GNUInstallDirs)
        include(CMakePackageConfigHelpers)

        #Create config file for each component
        configure_package_config_file(
            ${PROJECT_SOURCE_DIR}/cmake/elsaComponentConfig.cmake.in 
            ${PROJECT_BINARY_DIR}/elsa/elsa_${ELSA_MODULE_NAME}Config.cmake 
            INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
        )

        #and install it to lib/cmake/
        install(FILES 
            ${PROJECT_BINARY_DIR}/elsa/elsa_${ELSA_MODULE_NAME}Config.cmake 
            DESTINATION ${INSTALL_CONFIG_DIR}
        )


        # install the module
        install(TARGETS ${ELSA_MODULE_TARGET_NAME}
                EXPORT ${ELSA_MODULE_EXPORT_TARGET}
                INCLUDES DESTINATION include
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
