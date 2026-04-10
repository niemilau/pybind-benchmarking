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
inline bool to_meta(PyObject* obj, ArrayMeta& m)
{
    if (!PyArray_Check(obj)) { PyErr_SetString(PyExc_TypeError, "expected ndarray"); return false; }
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(obj);
    m.data     = PyArray_DATA(arr);
    m.ndim     = static_cast<int>(PyArray_NDIM(arr));
    m.itemsize = static_cast<ssize_t>(PyArray_ITEMSIZE(arr));
    m.dtype    = dtype_from_numpy(PyArray_TYPE(arr));
    m.device   = Device::CPU;
    m.shape    = reinterpret_cast<const ssize_t*>(PyArray_SHAPE(arr));
    m.strides  = reinterpret_cast<const ssize_t*>(PyArray_STRIDES(arr));
    return true;
}
