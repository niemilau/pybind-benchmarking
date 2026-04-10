# nanobind-tester

Comparing performance of nanobind vs pybind11 (custom NDArray wrapper) vs raw CPython when passing Numpy arrays to C++.

**CAUTION:** The benchmarking code and test cases are vibe coded and have not yet been fully scrutinized. The manual `ndarray.hpp` wrapper used with pybind11 is handwritten.


## Building

`cmake -DCMAKE_BUILD_TYPE=Release -B build && cmake --build build && cmake --install build`

Produces one Python extension module for each benchmark case. The install step puts them in the project root folder.


## Running

Run with `python bench.py`. Use `python bench.py --help` to see available command line options.


## Sample output

`python bench.py --warmup 500 --runs 500 --iters 1000`

Which gives:

```
Benchmark                                     pybind min       pybind mean±σ  nanobind min     nanobind mean±σ   cpython min      cpython mean±σ
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

  Group 1 · Unconstrained input, varying body cost
    noop_any                                       153.4          162.5±12.4           187.6           192.3±8.2            70.5           74.2±10.2 !
    read_ndim_any                                  155.3          165.4±17.4 !         195.2          202.2±13.8            71.6            74.9±7.9 !
    read_shape_sum_any                             160.9          169.5±20.9 !         196.1          203.0±27.0 !          72.0           76.2±23.5 !
    read_stride_sum_any                            190.3          200.7±18.2           191.6           197.5±9.9            81.1            84.7±8.5
    check_data_ptr_any                             165.9          179.4±29.5 !         190.7           195.4±4.0            70.5            73.5±7.0

  Group 2 · Runtime validation in body
    check_dtype_rt                                 156.6          166.6±24.8 !         186.9          193.1±31.7 !          70.2            73.6±7.6 !
    check_ndim_rt                                  155.3          165.4±17.3 !         190.7          198.7±23.7 !          71.5            74.5±6.3
    check_c_contig_rt                              161.5          169.7±14.8           192.2          197.6±12.4            73.1           77.0±17.7 !
    check_full_rt                                  156.5          166.1±17.3 !             —                   —            71.8            74.6±4.2
    check_dtype_rt/fail                           2952.4        3083.2±286.1          2590.6         2654.9±75.4           965.8         1006.4±62.1

  Group 3 · Type-constrained (binding layer validates)
    noop_f64_3d_cc                                 165.1          175.7±14.6           191.3          198.3±25.9 !          70.2            73.1±6.8
    noop_cf128_2x3_fc_cpu                          154.7          162.9±18.6 !         171.4          193.7±10.1            72.1            74.9±3.6
    noop_cf128_2x3_cc_cpu                          153.5          161.8±14.0           187.8           193.2±7.2            70.8            74.4±9.3 !
    check_full_typed_f64_3d                        149.7          157.5±21.1 !         191.0           196.3±9.7            73.1            76.2±3.9
    noop_f64_3d_cc/reject_dtype                  52341.1      54950.0±4167.0           509.0          527.4±24.8           177.0          186.9±27.5 !

  Group 4 · Multiple array arguments
    noop_two_arrays                                192.5          203.9±21.2 !         294.1          302.4±20.4            73.6            76.9±6.3
    noop_four_arrays                               298.5          313.8±27.6           502.1          514.9±24.6            83.6           87.6±15.6 !

  Group 5 · Scalar return values
    return_ndim                                    156.1          164.2±13.9           194.6           201.1±4.8            71.4            74.9±6.4
    return_shape_sum                               160.1          169.4±19.5 !         195.9          201.8±20.4 !          71.8            75.4±8.3 !
    return_itemsize                                157.6          166.3±24.5 !         195.1          201.1±12.4            71.6           75.8±16.1 !

  All times in nanoseconds (ns).  '!' = CV > 10 %, measurement may be noisy.
  Runs: 500 × iters_per_run (min is the robust estimator).
```