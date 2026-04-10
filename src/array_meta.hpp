#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

// ──────────────────────────────────────────────────────────────
// Neutral POD representation of array metadata.
// Filled in by each binding adapter; benchmark kernels only see this.
// ──────────────────────────────────────────────────────────────

enum class DType : uint8_t
{
    Float32, Float64, Complex64, Complex128,
    Int32, Int64, UInt32, UInt64, Unknown
};

enum class Device : uint8_t { CPU, GPU, Unknown };

struct ArrayMeta
{
    void*           data     = nullptr;
    int             ndim     = 0;
    // Keep shape/strides as non-owning views so that we avoid copying them when benchmarking
    const ssize_t*  shape    = nullptr;  // non-owning
    const ssize_t*  strides  = nullptr;  // non-owning
    ssize_t         itemsize = 0;
    DType           dtype    = DType::Unknown;
    Device          device   = Device::Unknown;
};

// ──────────────────────────────────────────────────────────────
// Benchmark kernels — intentionally trivial so binding overhead dominates.
// Return values prevent dead-code elimination.
// ──────────────────────────────────────────────────────────────

namespace bench
{

// 1. Pure no-op: measures raw call + argument-unpacking overhead only.
inline void noop(const ArrayMeta&) {}

// 2. Read a single scalar field (cheapest metadata access).
inline int read_ndim(const ArrayMeta& m) { return m.ndim; }

// 3. Read full shape into a local array (touches all shape words).
inline ssize_t read_shape_sum(const ArrayMeta& m)
{
    ssize_t s = 0;
    for (int i = 0; i < m.ndim; ++i)
    {
        s += m.shape[i];
    }
    return s;
}

// 4. Read strides (often the same cost as shape but a separate access pattern).
inline ssize_t read_stride_sum(const ArrayMeta& m)
{
    ssize_t s = 0;
    for (int i = 0; i < m.ndim; ++i)
    {
        s += m.strides[i];
    }
    return s;
}

// 5. Runtime dtype check — raises on mismatch.
inline void check_dtype(const ArrayMeta& m, DType expected)
{
    if (m.dtype != expected)
    {
        throw std::runtime_error("dtype mismatch");
    }
}

// 6. Runtime ndim check.
inline void check_ndim(const ArrayMeta& m, int expected)
{
    if (m.ndim != expected)
    {
        throw std::runtime_error("ndim mismatch: got "
            + std::to_string(m.ndim)
            + ", expected " + std::to_string(expected));
    }
}

// 7. Runtime C-contiguity check (walks strides).
inline bool check_c_contig(const ArrayMeta& m)
{
    ssize_t expected = m.itemsize;
    for (int i = m.ndim - 1; i >= 0; --i)
    {
        if (m.strides[i] != expected)
        {
            return false;
        }
        expected *= m.shape[i];
    }
    return true;
}

// 8. Runtime F-contiguity check.
inline bool check_f_contig(const ArrayMeta& m)
{
    ssize_t expected = m.itemsize;
    for (int i = 0; i < m.ndim; ++i)
    {
        if (m.strides[i] != expected)
        {
            return false;
        }
        expected *= m.shape[i];
    }
    return true;
}

// 9. Combined full validation — what real library code typically does.
inline void check_full(const ArrayMeta& m,
                       DType expected_dtype,
                       int   expected_ndim,
                       Device expected_device = Device::CPU)
{
    check_dtype(m, expected_dtype);
    check_ndim(m, expected_ndim);
    if (m.device != expected_device)
    {
        throw std::runtime_error("wrong device");
    }
    if (!check_c_contig(m))
    {
        throw std::runtime_error("array is not C-contiguous");
    }
}

// 10. Touch the first element's address (simulate a pointer validity check
//     without actually dereferencing — pointer arithmetic only).
inline bool check_data_ptr(const ArrayMeta& m)
{
    return m.data != nullptr;
}

} // namespace bench
