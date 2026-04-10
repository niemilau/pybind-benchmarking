#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#define _TESTER_SHOULD_IMPORT_NUMPY
#include "numpy_includes.hpp"

#include "array_meta.hpp"
#include "nanobind_adapter.hpp"

#include <complex>

namespace nb = nanobind;
using namespace bench;

// ─────────────────────────────────────────────────────────────────────────────
// Wrap helpers — mirror the pybind ones exactly, just s/NDArray/nb::ndarray/
// ─────────────────────────────────────────────────────────────────────────────

template<typename... Args>
auto wrap_noop()
{
    return [](const nb::ndarray<Args...>& a) { noop(to_meta(a)); };
}

template<typename... Args>
auto wrap_read_ndim()
{
    return [](const nb::ndarray<Args...>& a) { return read_ndim(to_meta(a)); };
}

template<typename... Args>
auto wrap_read_shape_sum()
{
    return [](const nb::ndarray<Args...>& a) { return read_shape_sum(to_meta(a)); };
}

template<typename... Args>
auto wrap_check_full_rt(DType dt, int nd)
{
    return [dt, nd](const nb::ndarray<Args...>& a) { check_full(to_meta(a), dt, nd); };
}

template<typename... Args>
auto wrap_check_full_typed()
{
    return [](const nb::ndarray<Args...>& a) { noop(to_meta(a)); };
}

NB_MODULE(nanobind_ext, m)
{
    // Import numpy for consistency with the other test cases, even though this module doesn't really need it
    import_array1();

    // ── Group 1: unconstrained input, varying body cost ───────────────────
    m.def("noop_any",            wrap_noop<>());
    m.def("read_ndim_any",       wrap_read_ndim<>());
    m.def("read_shape_sum_any",  wrap_read_shape_sum<>());
    m.def("read_stride_sum_any", [](const nb::ndarray<>& a)
    {
        return read_stride_sum(to_meta(a));
    });
    m.def("check_data_ptr_any",  [](const nb::ndarray<>& a)
    {
        return check_data_ptr(to_meta(a));
    });

    // ── Group 2: runtime checks — body does the validation ────────────────
    m.def("check_dtype_rt",    [](const nb::ndarray<>& a)
    {
        check_dtype(to_meta(a), DType::Float64);
    });
    m.def("check_ndim_rt",     [](const nb::ndarray<>& a)
    {
        check_ndim(to_meta(a), 3);
    });
    m.def("check_c_contig_rt", [](const nb::ndarray<>& a)
    {
        return check_c_contig(to_meta(a));
    });
    m.def("check_full_rt",     wrap_check_full_rt<>(DType::Float64, 3));

    // ── Group 3: type-constrained — binding layer enforces, body is a no-op ─
    m.def("noop_f64_3d_cc",
        wrap_noop<double, nb::ndim<3>, nb::c_contig>());
    m.def("noop_cf128_2x3_fc_cpu",
        wrap_noop<std::complex<double>, nb::f_contig, nb::device::cpu, nb::shape<2, 3>>());
    m.def("check_full_typed_f64_3d",
        wrap_check_full_typed<double, nb::ndim<3>, nb::c_contig>());

    // ── Group 4: multi-array — does overhead scale with argument count? ────
    m.def("noop_two_arrays",  [](const nb::ndarray<>& a, const nb::ndarray<>& b)
    {
        noop(to_meta(a)); noop(to_meta(b));
    });
    m.def("noop_four_arrays",
        [](const nb::ndarray<>& a, const nb::ndarray<>& b,
           const nb::ndarray<>& c, const nb::ndarray<>& d)
    {
        noop(to_meta(a)); noop(to_meta(b));
        noop(to_meta(c)); noop(to_meta(d));
    });

    // ── Group 5: scalar-returning ──────────────────────────────────────────
    m.def("return_ndim",      wrap_read_ndim<>());
    m.def("return_shape_sum", wrap_read_shape_sum<>());
    m.def("return_itemsize",  [](const nb::ndarray<>& a)
    {
        return to_meta(a).itemsize;
    });
}
