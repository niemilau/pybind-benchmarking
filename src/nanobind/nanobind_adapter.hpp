#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "array_meta.hpp"

namespace nb = nanobind;

// nb::ndarray dtype is a DLPack dtype struct: code + bits + lanes.
inline DType dtype_from_nb(const nb::dlpack::dtype& dt)
{
    using C = nb::dlpack::dtype_code;
    switch (dt.code)
    {
        case (uint8_t)C::Float:
            switch (dt.bits)
            {
                case 32:  return DType::Float32;
                case 64:  return DType::Float64;
                default:  return DType::Unknown;
            }
        case (uint8_t)C::Complex:
            switch (dt.bits)
            {
                case 64:  return DType::Complex64;
                case 128: return DType::Complex128;
                default:  return DType::Unknown;
            }
        case (uint8_t)C::Int:
            switch (dt.bits)
            {
                case 32:  return DType::Int32;
                case 64:  return DType::Int64;
                default:  return DType::Unknown;
            }
        case (uint8_t)C::UInt:
            switch (dt.bits)
            {
                case 32:  return DType::UInt32;
                case 64:  return DType::UInt64;
                default:  return DType::Unknown;
            }
        default:
            return DType::Unknown;
    }
}

template<typename... Args>
ArrayMeta to_meta(const nb::ndarray<Args...>& a)
{
    ArrayMeta m;
    m.data     = static_cast<void*>(a.data());
    m.ndim     = static_cast<int>(a.ndim());
    m.itemsize = static_cast<ssize_t>(a.itemsize());
    m.dtype    = dtype_from_nb(a.dtype());
    m.device   = (a.device_type() == nb::device::cpu::value)
                     ? Device::CPU : Device::GPU;
    // DLPack stores shape/strides as int64_t* — safe to alias as ssize_t*
    // on any platform where ssize_t is 64-bit (all targets you care about).
    m.shape   = reinterpret_cast<const ssize_t*>(a.shape_ptr());
    // FIXME: nanobind strides are in number of elements, not in bytes
    m.strides = reinterpret_cast<const ssize_t*>(a.stride_ptr());
    return m;
}
