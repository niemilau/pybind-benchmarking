#pragma once

#include <Python.h>
#include "numpy_includes.hpp"
#include "array_meta.hpp"

inline DType dtype_from_numpy(int typenum)
{
    switch (typenum)
    {
        case NPY_FLOAT:    return DType::Float32;
        case NPY_DOUBLE:   return DType::Float64;
        case NPY_CFLOAT:   return DType::Complex64;
        case NPY_CDOUBLE:  return DType::Complex128;
        case NPY_INT32:    return DType::Int32;
        case NPY_INT64:    return DType::Int64;
        case NPY_UINT32:   return DType::UInt32;
        case NPY_UINT64:   return DType::UInt64;
        default:           return DType::Unknown;
    }
}

// Returns false and sets a Python TypeError if obj is not a numpy array.
inline bool to_meta(PyObject* obj, ArrayMeta& out)
{
    if (!PyArray_Check(obj))
    {
        PyErr_SetString(PyExc_TypeError, "expected a numpy ndarray");
        return false;
    }

    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(obj);

    out.data     = PyArray_DATA(arr);
    out.ndim     = static_cast<int>(PyArray_NDIM(arr));
    out.itemsize = static_cast<ssize_t>(PyArray_ITEMSIZE(arr));
    out.dtype    = dtype_from_numpy(PyArray_TYPE(arr));
    out.device   = Device::CPU;

    npy_intp* shape   = PyArray_SHAPE(arr);
    npy_intp* strides = PyArray_STRIDES(arr);
    for (int i = 0; i < out.ndim && i < 8; ++i)
    {
        out.shape[i]   = static_cast<ssize_t>(shape[i]);
        out.strides[i] = static_cast<ssize_t>(strides[i]);
    }
    return true;
}
