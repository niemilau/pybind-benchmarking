#define _TESTER_SHOULD_IMPORT_NUMPY
#include "numpy_includes.hpp"

#include "array_meta.hpp"
#include "numpy_adapter.hpp"

using namespace bench;

// ─────────────────────────────────────────────────────────────────────────────
// Macro: boilerplate for a function that accepts N positional PyObjects,
// converts each to ArrayMeta, runs Body, and returns Py_None.
// On error in to_meta the TypeError is already set; we just return NULL.
// ─────────────────────────────────────────────────────────────────────────────

#define META1(varname, arg)         \
    ArrayMeta varname;              \
    if (!to_meta(arg, varname))     \
        return nullptr;

// ─────────────────────────────────────────────────────────────────────────────
// Group 1 — unconstrained input, varying body cost
// ─────────────────────────────────────────────────────────────────────────────

static PyObject* py_noop_any(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    noop(m);
    Py_RETURN_NONE;
}

static PyObject* py_read_ndim_any(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    return PyLong_FromLong(read_ndim(m));
}

static PyObject* py_read_shape_sum_any(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    return PyLong_FromSsize_t(read_shape_sum(m));
}

static PyObject* py_read_stride_sum_any(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    return PyLong_FromSsize_t(read_stride_sum(m));
}

static PyObject* py_check_data_ptr_any(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    return PyBool_FromLong(check_data_ptr(m) ? 1 : 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2 — runtime checks in body
// ─────────────────────────────────────────────────────────────────────────────

static PyObject* py_check_dtype_rt(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    try { check_dtype(m, DType::Float64); }
    catch (const std::exception& e)
    {
        PyErr_SetString(PyExc_TypeError, e.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject* py_check_ndim_rt(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    try { check_ndim(m, 3); }
    catch (const std::exception& e)
    {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject* py_check_c_contig_rt(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    return PyBool_FromLong(check_c_contig(m) ? 1 : 0);
}

static PyObject* py_check_full_rt(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    try { check_full(m, DType::Float64, 3); }
    catch (const std::exception& e)
    {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3 — type-constrained variants
//
// Raw CPython has no template machinery to enforce dtype/contiguity at binding
// time, so we replicate the checks manually.  This is intentional: it shows
// the cost a hand-rolled C extension would actually pay.
// ─────────────────────────────────────────────────────────────────────────────

// Require: float64, 3-D, C-contiguous.
static PyObject* py_noop_f64_3d_cc(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    if (!PyArray_Check(a))
    {
        PyErr_SetString(PyExc_TypeError, "expected ndarray");
        return nullptr;
    }
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(a);
    if (PyArray_TYPE(arr) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "expected float64 array");
        return nullptr;
    }
    if (PyArray_NDIM(arr) != 3)
    {
        PyErr_SetString(PyExc_ValueError, "expected 3-D array");
        return nullptr;
    }
    if (!PyArray_IS_C_CONTIGUOUS(arr))
    {
        PyErr_SetString(PyExc_ValueError, "expected C-contiguous array");
        return nullptr;
    }
    META1(m, a)
    noop(m);
    Py_RETURN_NONE;
}

// Require: complex128, shape (2,3), Fortran-contiguous, CPU (always true for NumPy).
static PyObject* py_noop_cf128_2x3_fc_cpu(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    if (!PyArray_Check(a))
    {
        PyErr_SetString(PyExc_TypeError, "expected ndarray");
        return nullptr;
    }
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(a);
    if (PyArray_TYPE(arr) != NPY_CDOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "expected complex128 array");
        return nullptr;
    }
    if (PyArray_NDIM(arr) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "expected 2-D array");
        return nullptr;
    }
    npy_intp* shape = PyArray_SHAPE(arr);
    if (shape[0] != 2 || shape[1] != 3)
    {
        PyErr_SetString(PyExc_ValueError, "expected shape (2, 3)");
        return nullptr;
    }
    if (!PyArray_IS_F_CONTIGUOUS(arr))
    {
        PyErr_SetString(PyExc_ValueError, "expected Fortran-contiguous array");
        return nullptr;
    }
    META1(m, a)
    noop(m);
    Py_RETURN_NONE;
}

// Typed full-check: same constraints as noop_f64_3d_cc, body calls check_full.
static PyObject* py_check_full_typed_f64_3d(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    if (!PyArray_Check(a))
    {
        PyErr_SetString(PyExc_TypeError, "expected ndarray");
        return nullptr;
    }
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(a);
    if (PyArray_TYPE(arr) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "expected float64 array");
        return nullptr;
    }
    if (PyArray_NDIM(arr) != 3)
    {
        PyErr_SetString(PyExc_ValueError, "expected 3-D array");
        return nullptr;
    }
    if (!PyArray_IS_C_CONTIGUOUS(arr))
    {
        PyErr_SetString(PyExc_ValueError, "expected C-contiguous array");
        return nullptr;
    }
    META1(m, a)
    try { check_full(m, DType::Float64, 3); }
    catch (const std::exception& e)
    {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4 — multiple array arguments
// ─────────────────────────────────────────────────────────────────────────────

static PyObject* py_noop_two_arrays(PyObject*, PyObject* args)
{
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b)) return nullptr;
    META1(ma, a)
    META1(mb, b)
    noop(ma); noop(mb);
    Py_RETURN_NONE;
}

static PyObject* py_noop_four_arrays(PyObject*, PyObject* args)
{
    PyObject *a, *b, *c, *d;
    if (!PyArg_ParseTuple(args, "OOOO", &a, &b, &c, &d)) return nullptr;
    META1(ma, a) META1(mb, b) META1(mc, c) META1(md, d)
    noop(ma); noop(mb); noop(mc); noop(md);
    Py_RETURN_NONE;
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 5 — scalar returns
// ─────────────────────────────────────────────────────────────────────────────

static PyObject* py_return_ndim(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    return PyLong_FromLong(read_ndim(m));
}

static PyObject* py_return_shape_sum(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    return PyLong_FromSsize_t(read_shape_sum(m));
}

static PyObject* py_return_itemsize(PyObject*, PyObject* args)
{
    PyObject* a;
    if (!PyArg_ParseTuple(args, "O", &a)) return nullptr;
    META1(m, a)
    return PyLong_FromSsize_t(m.itemsize);
}

// ─────────────────────────────────────────────────────────────────────────────
// Method table
// ─────────────────────────────────────────────────────────────────────────────

static PyMethodDef cpython_ext_methods[] = {
    // Group 1
    { "noop_any",             py_noop_any,            METH_VARARGS, nullptr },
    { "read_ndim_any",        py_read_ndim_any,        METH_VARARGS, nullptr },
    { "read_shape_sum_any",   py_read_shape_sum_any,   METH_VARARGS, nullptr },
    { "read_stride_sum_any",  py_read_stride_sum_any,  METH_VARARGS, nullptr },
    { "check_data_ptr_any",   py_check_data_ptr_any,   METH_VARARGS, nullptr },
    // Group 2
    { "check_dtype_rt",       py_check_dtype_rt,       METH_VARARGS, nullptr },
    { "check_ndim_rt",        py_check_ndim_rt,        METH_VARARGS, nullptr },
    { "check_c_contig_rt",    py_check_c_contig_rt,    METH_VARARGS, nullptr },
    { "check_full_rt",        py_check_full_rt,        METH_VARARGS, nullptr },
    // Group 3
    { "noop_f64_3d_cc",            py_noop_f64_3d_cc,            METH_VARARGS, nullptr },
    { "noop_cf128_2x3_fc_cpu",     py_noop_cf128_2x3_fc_cpu,     METH_VARARGS, nullptr },
    { "check_full_typed_f64_3d",   py_check_full_typed_f64_3d,   METH_VARARGS, nullptr },
    // Group 4
    { "noop_two_arrays",      py_noop_two_arrays,      METH_VARARGS, nullptr },
    { "noop_four_arrays",     py_noop_four_arrays,     METH_VARARGS, nullptr },
    // Group 5
    { "return_ndim",          py_return_ndim,          METH_VARARGS, nullptr },
    { "return_shape_sum",     py_return_shape_sum,     METH_VARARGS, nullptr },
    { "return_itemsize",      py_return_itemsize,      METH_VARARGS, nullptr },
    // Sentinel
    { nullptr, nullptr, 0, nullptr }
};


static PyModuleDef cpython_ext_module = {
    PyModuleDef_HEAD_INIT,
    "cpython_ext",   // module name
    nullptr,         // docstring
    -1,              // per-interpreter state size (-1 = global state)
    cpython_ext_methods,
    nullptr, nullptr, nullptr, nullptr
};

PyMODINIT_FUNC PyInit_cpython_ext()
{
    import_array();
    return PyModule_Create(&cpython_ext_module);
}
