#pragma once

#include "numpy_includes.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tester
{

/* Copy some cheap metadata from input arrays here and return to python.
This should ensure the compiler doesn't completely optimize stuff out.
*/
struct ArrayMetadata
{
    ArrayMetadata() noexcept {}

    // Construct manually from PyArrayObject*
    ArrayMetadata(PyArrayObject* a) noexcept;

    uintptr_t data_ptr;
    int32_t ndim;
    bool c_contig;

    //
};

} // namespace tester
