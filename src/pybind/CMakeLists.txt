execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE pybind11_ROOT)

find_package(pybind11 CONFIG REQUIRED)

FILE(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*)
pybind11_add_module(pybind_ext ${SOURCES})

install(TARGETS pybind_ext DESTINATION ${EXT_INSTALL_DIR})
