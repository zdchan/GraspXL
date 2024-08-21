# Install script for directory: /home/huizhang/intern/raisim_grasp_arctic

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/huizhang/intern/raisim/raisim_build")
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

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/huizhang/intern/raisim/raisim_build/include")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/huizhang/intern/raisim/raisim_build" TYPE DIRECTORY FILES "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/include")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/huizhang/intern/raisim/raisim_build/lib/libraisim.so;/home/huizhang/intern/raisim/raisim_build/lib/libraisim.so.1.1.6;/home/huizhang/intern/raisim/raisim_build/lib/libraisimMine.so;/home/huizhang/intern/raisim/raisim_build/lib/libraisimODE.so;/home/huizhang/intern/raisim/raisim_build/lib/libraisimODE.so.1.1.6;/home/huizhang/intern/raisim/raisim_build/lib/libraisimPng.so;/home/huizhang/intern/raisim/raisim_build/lib/libraisimZ.so;/home/huizhang/intern/raisim/raisim_build/lib/raisim.mexa64")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/huizhang/intern/raisim/raisim_build/lib" TYPE FILE FILES
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/libraisim.so"
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/libraisim.so.1.1.6"
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/libraisimMine.so"
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/libraisimODE.so"
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/libraisimODE.so.1.1.6"
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/libraisimPng.so"
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/libraisimZ.so"
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/raisim.mexa64"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/huizhang/intern/raisim/raisim_build/lib/cmake")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/huizhang/intern/raisim/raisim_build/lib" TYPE DIRECTORY FILES "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/huizhang/intern/raisim/raisim_build/lib/python3.8/site-packages/raisimpy.cpython-35m-x86_64-linux-gnu.so;/home/huizhang/intern/raisim/raisim_build/lib/python3.8/site-packages/raisimpy.cpython-36m-x86_64-linux-gnu.so;/home/huizhang/intern/raisim/raisim_build/lib/python3.8/site-packages/raisimpy.cpython-37m-x86_64-linux-gnu.so;/home/huizhang/intern/raisim/raisim_build/lib/python3.8/site-packages/raisimpy.cpython-38-x86_64-linux-gnu.so;/home/huizhang/intern/raisim/raisim_build/lib/python3.8/site-packages/raisimpy.cpython-39-x86_64-linux-gnu.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/huizhang/intern/raisim/raisim_build/lib/python3.8/site-packages" TYPE FILE FILES
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/raisimpy.cpython-35m-x86_64-linux-gnu.so"
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/raisimpy.cpython-36m-x86_64-linux-gnu.so"
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/raisimpy.cpython-37m-x86_64-linux-gnu.so"
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/raisimpy.cpython-38-x86_64-linux-gnu.so"
    "/home/huizhang/intern/raisim_grasp_arctic/raisim/linux/lib/raisimpy.cpython-39-x86_64-linux-gnu.so"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/raisim" TYPE FILE FILES "/home/huizhang/intern/raisim_grasp_arctic/package.xml")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/huizhang/intern/raisim_grasp_arctic/raisimGymTorch/examples/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/huizhang/intern/raisim_grasp_arctic/raisimGymTorch/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
