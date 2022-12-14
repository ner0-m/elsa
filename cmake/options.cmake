# commandline option enhancements
# cmake only supports binary ON OFF option(),
# but we want 3 states: on, off, if_available - to allow autodetection and explicit (de)activation.
#
# definition must be a variable named WANT_THING which one has to default if not set.


# records whether the option NAME is enabled
# sets WITH_${VARNAME} to HAVE
# errors if WANT_${VARNAME} conflicts with HAVE
function(have_config_option NAME VARNAME HAVE)
    set(WANT "${WANT_${VARNAME}}")
    set(WITH_${VARNAME} "${HAVE}" PARENT_SCOPE)

    if(HAVE)
        set_property(GLOBAL APPEND PROPERTY ELSA_CONFIG_OPTIONS_ENABLED "${NAME}")

        if(NOT WANT)
            message(FATAL_ERROR "${NAME}: WANT_${VARNAME}=${WANT}, but WITH_${VARNAME}=${HAVE}")
        endif()
    else()
        set_property(GLOBAL APPEND PROPERTY ELSA_CONFIG_OPTIONS_DISABLED "${NAME}")

        if(WANT STREQUAL "if_available")
            message(STATUS "optional dependency is unavailable: ${NAME}")
        elseif(WANT)
            message(FATAL_ERROR "${NAME}: WANT_${VARNAME}=${WANT}, but WITH_${VARNAME}=${HAVE}")
        endif()
    endif()
endfunction()

function(print_config_options)
    get_property(enabled_opts GLOBAL PROPERTY ELSA_CONFIG_OPTIONS_ENABLED)
    get_property(disabled_opts GLOBAL PROPERTY ELSA_CONFIG_OPTIONS_DISABLED)

    message("    config options: |")
    message("                    | enabled:")
    if(enabled_opts)
        foreach(opt ${enabled_opts})
            message("                    | ${opt}")
        endforeach()
    else()
        message("                    | <none>")
    endif()

    message("                    |")
    message("                    | disabled:")
    if(disabled_opts)
        foreach(opt ${disabled_opts})
            message("                    | ${opt}")
        endforeach()
    else()
        message("                    | <none>")
    endif()
endfunction()
