# install an elsa module

function(InstallElsaModule ELSA_MODULE_NAME ELSA_MODULE_TARGET_NAME ELSA_MODULE_EXPORT_TARGET)
    if(ELSA_INSTALL)
        #This is required so that the exported target has the name core and not elsa_core
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
        install(FILES ${MODULE_HEADERS}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${ELSA_MODULE_NAME}
        )
        # create the config file for the module
        install(EXPORT ${ELSA_MODULE_EXPORT_TARGET}
                FILE ${ELSA_MODULE_EXPORT_TARGET}.cmake
                NAMESPACE elsa::
                DESTINATION ${INSTALL_CONFIG_DIR}
        )
    endif(ELSA_INSTALL)
endfunction()
