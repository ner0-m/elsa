# Install script for directory: /home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/libelsa_core.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/elsa/core" TYPE FILE FILES
    "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/elsa.h"
    "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/Cloneable.h"
    "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/DataDescriptor.h"
    "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/BlockDescriptor.h"
    "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/DataContainer.h"
    "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/DataContainerIterator.h"
    "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/DataHandler.h"
    "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/DataHandlerCPU.h"
    "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/LinearOperator.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/cmake/elsa_coreTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}/cmake/elsa_coreTargets.cmake"
         "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/CMakeFiles/Export/_cmake/elsa_coreTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}/cmake/elsa_coreTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}/cmake/elsa_coreTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/cmake/elsa_coreTargets.cmake")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/cmake" TYPE FILE FILES "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/CMakeFiles/Export/_cmake/elsa_coreTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/cmake/elsa_coreTargets-release.cmake")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "/cmake" TYPE FILE FILES "/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/CMakeFiles/Export/_cmake/elsa_coreTargets-release.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/beccy/Dokumente/Master BMC/2.Semester/BAReloaded/Local Libraries/elsa/elsa/core/tests/cmake_install.cmake")

endif()

