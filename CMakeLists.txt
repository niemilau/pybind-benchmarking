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

# Find nanobind. This assumes pip or conda installed nanobind so we use the Python executable to locate it
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Get numpy CFLAGS, include dirs in particular
execute_process(
  COMMAND numpy-config --cflags
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE numpy_CFLAGS)

# add raw numpy flags to all targets
add_compile_options(${numpy_CFLAGS})

# NOTE: by default nanobind_add_module() compiles the module with -Os, ie. optimized for binary size.
# This is because usually the ext module only wraps around other code that is presumably compiled with -O3.
# See https://nanobind.readthedocs.io/en/latest/building.html#building-an-extension.
# We can compile also the binding code with -O3 as follows:
nanobind_add_module(nanobind_ext NOMINSIZE src/nanobind_ext.cpp)

#nanobind_add_module(my_ext src/ext.cpp)

set(EXT_INSTALL_DIR ${CMAKE_SOURCE_DIR})
include_directories(src/)

add_subdirectory(src/cpython)
# Separate module with pybind11 for performance comparison
add_subdirectory(src/pybind)

install(TARGETS nanobind_ext DESTINATION ${EXT_INSTALL_DIR})
