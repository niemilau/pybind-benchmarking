cmake_minimum_required(VERSION 3.15)
project(nanobind-tester)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.9 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Get numpy CFLAGS, include dirs in particular
execute_process(
  COMMAND numpy-config --cflags
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE numpy_CFLAGS)

# add raw numpy flags to all targets
add_compile_options(${numpy_CFLAGS})

set(EXT_INSTALL_DIR ${CMAKE_SOURCE_DIR})
include_directories(src/)

add_subdirectory(src/nanobind)
add_subdirectory(src/cpython)
add_subdirectory(src/pybind)

install(TARGETS nanobind_ext DESTINATION ${EXT_INSTALL_DIR})
