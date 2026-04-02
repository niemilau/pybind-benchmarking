import nanobind_ext
import pybind_ext
import numpy as np
from time import time

a = np.empty((2, 3), dtype=np.float32)
metadata = pybind_ext.extract_metadata_float(a)

a = np.empty((2, 3), dtype=np.complex128)

nrepeat = 200

t1 = time()
for i in range(nrepeat):
    #metadata = pybind_ext.extract_metadata_complexdouble_c_contig_cpu_shape_2_3(a)
    metadata = pybind_ext.extract_metadata_any(a)
t2 = time()
print(f"Pybind + templated ndarray: {(t2-t1) * 1e9 / nrepeat} ns")

t1 = time()
for i in range(nrepeat):
    metadata = pybind_ext.extract_metadata_manual(a)
t2 = time()
print(f"Pybind + manual PyArrayObject: {(t2-t1) * 1e9 / nrepeat} ns")
