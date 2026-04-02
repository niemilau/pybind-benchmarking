#pragma once

#include "ndarray.hpp"
#include "array_meta.hpp"

using namespace tester;

// Map your dtype tags → DType enum.
// Extend as you add more element types to NDArray.
template<typename T> constexpr DType dtype_of();
template<> constexpr DType dtype_of<float>()                  { return DType::Float32;    }
template<> constexpr DType dtype_of<double>()                 { return DType::Float64;    }
template<> constexpr DType dtype_of<std::complex<float>>()    { return DType::Complex64;  }
template<> constexpr DType dtype_of<std::complex<double>>()   { return DType::Complex128; }
template<> constexpr DType dtype_of<int32_t>()                { return DType::Int32;      }
template<> constexpr DType dtype_of<int64_t>()                { return DType::Int64;      }

inline DType dtype_from_pybind(const pybind11::dtype& dt)
{
    switch (dt.kind())
    {
        case 'f':
            switch (dt.itemsize())
            {
                case 4:  return DType::Float32;
                case 8:  return DType::Float64;
                default: return DType::Unknown;
            }
        case 'c':
            switch (dt.itemsize())
            {
                case 8:  return DType::Complex64;
                case 16: return DType::Complex128;
                default: return DType::Unknown;
            }
        case 'i':
            switch (dt.itemsize())
            {
                case 4:  return DType::Int32;
                case 8:  return DType::Int64;
                default: return DType::Unknown;
            }
        case 'u':
            switch (dt.itemsize())
            {
                case 4:  return DType::UInt32;
                case 8:  return DType::UInt64;
                default: return DType::Unknown;
            }
        default:
            return DType::Unknown;
    }
}

template<typename... Args>
ArrayMeta to_meta(const NDArray<Args...>& a)
{
    ArrayMeta m;
    m.data     = const_cast<void*>(static_cast<const void*>(a.data()));
    m.ndim     = static_cast<int>(a.ndim());
    m.itemsize = static_cast<ssize_t>(a.itemsize());

    // Dtype: if NDArray carries the scalar type as first template arg, use dtype_of<>.
    // Otherwise fall back to runtime query.
    using Scalar = typename NDArray<Args...>::Scalar;
    if constexpr (!std::is_void_v<Scalar>)
    {
        m.dtype = dtype_of<Scalar>();
    }
    else
    {
        m.dtype = dtype_from_pybind(a.dtype());
    }

    // Not implemented in NDArray:
    m.device = Device::Unknown;
    //m.device = (a.device() == 0 /*cpu tag*/) ? Device::CPU : Device::GPU;

    for (int i = 0; i < m.ndim && i < 8; ++i)
    {
        m.shape[i]   = static_cast<ssize_t>(a.shape(i));
        m.strides[i] = static_cast<ssize_t>(a.stride(i));
    }
    return m;
}
