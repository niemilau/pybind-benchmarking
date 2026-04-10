# Find nanobind. This assumes pip or conda installed nanobind so we use the Python executable to locate it
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)


FILE(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*)

# NOTE: by default nanobind_add_module() compiles the module with -Os, ie. optimized for binary size.
# This is because usually the ext module only wraps around other code that is presumably compiled with -O3.
# See https://nanobind.readthedocs.io/en/latest/building.html#building-an-extension.
# We can compile also the binding code with -O3 as follows:
nanobind_add_module(nanobind_ext NOMINSIZE ${SOURCES})

#nanobind_add_module(my_ext src/ext.cpp)

install(TARGETS nanobind_ext DESTINATION ${EXT_INSTALL_DIR})
