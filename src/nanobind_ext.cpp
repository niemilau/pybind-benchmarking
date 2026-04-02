#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#define _TESTER_SHOULD_IMPORT_NUMPY
#include "numpy_includes.hpp"

NB_MODULE(nanobind_ext, m)
{
    import_array1();
}

