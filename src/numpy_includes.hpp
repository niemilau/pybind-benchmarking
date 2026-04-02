#pragma once

#ifndef _TESTER_SHOULD_IMPORT_NUMPY
    #define NO_IMPORT_ARRAY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL TESTER_ARRAY_API
#include <numpy/arrayobject.h>
