#include "tester_common.hpp"

namespace tester
{

ArrayMetadata::ArrayMetadata(PyArrayObject* a) noexcept
{
    data_ptr = reinterpret_cast<uintptr_t>(PyArray_DATA(a));
    ndim = PyArray_NDIM(a);
}

}
