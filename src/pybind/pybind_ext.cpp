#include <pybind11/pybind11.h>

#define _TESTER_SHOULD_IMPORT_NUMPY
#include "numpy_includes.hpp"

#include <pybind11/numpy.h>
#include "ndarray.hpp"
#include "tester_common.hpp"

namespace py = pybind11;

namespace tester
{

template<typename... Args>
static ArrayMetadata extract_metadata(const NDArray<Args...>& a)
{

    ArrayMetadata metadata;
    metadata.data_ptr = reinterpret_cast<uintptr_t>(a.data());
    metadata.ndim = a.ndim();

    return metadata;
}

static ArrayMetadata extract_metadata_manual(py::handle obj)
{
    if (PyArray_Check(obj.ptr()))
    {
        //return ArrayMetadata(reinterpret_cast<PyArrayObject*>(obj.ptr()));
        return ArrayMetadata();
    }
    else
    {
        throw std::runtime_error("Not a numpy array");
    }
}

PYBIND11_MODULE(pybind_ext, m)
{
    // Numpy array imports
    import_array1();

    namespace py = pybind11;

    py::class_<ArrayMetadata>(m, "ArrayMetadata")
        .def_readwrite("data_ptr", &ArrayMetadata::data_ptr)
        .def_readwrite("ndim", &ArrayMetadata::ndim);

    m.def("extract_metadata_manual", &extract_metadata_manual);

    m.def("extract_metadata_any", &extract_metadata<>);
    m.def("extract_metadata_cpu", &extract_metadata<device::cpu>);
    m.def("extract_metadata_gpu", &extract_metadata<device::gpu>);
    m.def("extract_metadata_float", &extract_metadata<float>);
    m.def("extract_metadata_complexdouble", &extract_metadata<std::complex<double>>);
    m.def("extract_metadata_c_contig", &extract_metadata<c_contig>);
    m.def("extract_metadata_2D", &extract_metadata<ndim<2>>);
    m.def("extract_metadata_shape_1_2_3", &extract_metadata<shape<1, 2, 3>>);
    m.def("extract_metadata_shape_any", &extract_metadata<shape<-1>>);
    m.def("extract_metadata_shape_2_any", &extract_metadata<shape<2, -1>>);
    m.def("extract_metadata_double_c_contig_3D", &extract_metadata<double, c_contig, ndim<3>>);
    m.def("extract_metadata_complexdouble_f_contig_cpu_shape_2_3", &extract_metadata<std::complex<double>, f_contig, device::cpu, shape<2, 3>>);
    m.def("extract_metadata_complexdouble_c_contig_cpu_shape_2_3", &extract_metadata<std::complex<double>, c_contig, device::cpu, shape<2, 3>>);
    m.def("extract_metadata_complexfloat_c_contig_gpu_shape_any_any", &extract_metadata<std::complex<float>, c_contig, device::gpu, shape<-1, -1>>);
}

} // namespace tester
