#include <pybind11/pybind11.h>

#define _TESTER_SHOULD_IMPORT_NUMPY
#include "numpy_includes.hpp"

#include "ndarray.hpp"
#include "ndarray_adapter.hpp"
#include "array_meta.hpp"

#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace bench;
using namespace tester;

// ──────────────────────────────────────────────────────────────
// Helpers: thin lambda wrappers so each binding is one line.
// The "unconstrained" variants accept any array; the "typed" variants
// let the binding layer enforce constraints before the body runs.
// ──────────────────────────────────────────────────────────────

template<typename... Args>
auto wrap_noop()
{
    return [](const NDArray<Args...>& a) { noop(to_meta(a)); };
}
template<typename... Args>
auto wrap_read_ndim()
{
    return [](const NDArray<Args...>& a) { return read_ndim(to_meta(a)); };
}
template<typename... Args>
auto wrap_read_shape_sum()
{
    return [](const NDArray<Args...>& a) { return read_shape_sum(to_meta(a)); };
}
template<typename... Args>
auto wrap_check_full_rt(DType dt, int nd)
{
    // "rt" = runtime check: constraints passed as values, not baked into type.
    return [dt, nd](const NDArray<Args...>& a) { check_full(to_meta(a), dt, nd); };
}
template<typename... Args>
auto wrap_check_full_typed()
{
    // "typed": binding layer already enforced dtype/ndim/contiguity via NDArray<Args...>.
    // Body intentionally does nothing extra — we're measuring binding overhead only.
    return [](const NDArray<Args...>& a) { noop(to_meta(a)); };
}


PYBIND11_MODULE(pybind_ext, m)
{
    import_array1();

    // ── Group 1: unconstrained array, varying body cost ───────────────────
    // Baseline: pure call overhead with the least work possible in the body.
    m.def("noop_any",            wrap_noop<>());
    m.def("read_ndim_any",       wrap_read_ndim<>());
    m.def("read_shape_sum_any",  wrap_read_shape_sum<>());
    m.def("read_stride_sum_any", [](const NDArray<>& a)
    {
        return read_stride_sum(to_meta(a));
    });
    m.def("check_data_ptr_any",  [](const NDArray<>& a)
    {
        return check_data_ptr(to_meta(a));
    });

    // ── Group 2: runtime checks — body does the validation
    m.def("check_dtype_rt",    [](const NDArray<>& a)
    {
        check_dtype(to_meta(a), DType::Float64);
    });
    m.def("check_ndim_rt",     [](const NDArray<>& a)
    {
        check_ndim(to_meta(a), 3);
    });
    m.def("check_c_contig_rt", [](const NDArray<>& a)
    {
        return check_c_contig(to_meta(a));
    });
    m.def("check_full_rt", wrap_check_full_rt<>(DType::Float64, 3));

    // ── Group 3: type-constrained — binding layer enforces, body is a no-op ─
    // Compare these against their Group-1 runtime counterparts to isolate
    // where pybind11 vs nanobind spend time on constraint checking.
    m.def("noop_f64_3d_cc",
        wrap_noop<double, c_contig, ndim<3>>());
    m.def("noop_cf128_2x3_fc_cpu",
        wrap_noop<std::complex<double>, f_contig, device::cpu, shape<2,3>>());
    m.def("check_full_typed_f64_3d",
        wrap_check_full_typed<double, c_contig, ndim<3>>());

    // ── Group 4: multi-array — measures per-argument overhead scaling ──────
    // Lets you check whether cost is O(1) or O(n_args).
    m.def("noop_two_arrays", [](const NDArray<>& a, const NDArray<>& b)
    {
        noop(to_meta(a)); noop(to_meta(b));
    });
    m.def("noop_four_arrays",
        [](const NDArray<>& a, const NDArray<>& b,
           const NDArray<>& c, const NDArray<>& d) {
            noop(to_meta(a)); noop(to_meta(b));
            noop(to_meta(c)); noop(to_meta(d));
    });

    // ── Group 5: scalar-returning — ensures Python doesn't short-circuit ──
    // Returns a value so Python must actually process the result.
    m.def("return_ndim",      wrap_read_ndim<>());
    m.def("return_shape_sum", wrap_read_shape_sum<>());
    m.def("return_itemsize",  [](const NDArray<>& a)
    {
        return to_meta(a).itemsize;
    });
}
