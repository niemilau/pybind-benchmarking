# bench.py
"""
Microbenchmark driver for Python binding overhead comparison.

Usage:
    python bench.py                     # all available backends
    python bench.py --backend pybind    # single backend
    python bench.py --warmup 500 --runs 5000
    python bench.py --format csv        # machine-readable output

Backends are loaded opportunistically — a missing .so is a warning, not a crash.
"""
from __future__ import annotations

import argparse
import importlib
import statistics
import sys
import timeit
import types
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Array fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_f64_3d_cc(shape=(4, 5, 6)) -> np.ndarray:
    """float64, 3-D, C-contiguous (row-major)."""
    return np.zeros(shape, dtype=np.float64)

def make_f64_3d_fc(shape=(4, 5, 6)) -> np.ndarray:
    """float64, 3-D, Fortran-contiguous (column-major)."""
    return np.asfortranarray(np.zeros(shape, dtype=np.float64))

def make_cf128_2x3_fc() -> np.ndarray:
    """complex128, shape (2, 3), Fortran-contiguous."""
    return np.asfortranarray(np.zeros((2, 3), dtype=np.complex128))

def make_cf128_2x3_cc() -> np.ndarray:
    """complex128, shape (2, 3), C-contiguous."""
    return np.asfortranarray(np.zeros((2, 3), dtype=np.complex128))

def make_f64_1d() -> np.ndarray:
    return np.zeros(16, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark case description
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchCase:
    name: str
    fn: Callable
    args: tuple = field(default_factory=tuple)
    group: str = ""
    # If True, the call is expected to raise — we time it but mark it specially.
    expect_raise: bool = False


def _sink(x: Any) -> None:
    """Absorb return values so the compiler/interpreter can't elide the call."""
    # Assign to a global that is never read after the loop.
    # CPython always executes the assignment so this is reliable.
    global _last_result
    _last_result = x

_last_result: Any = None


# ─────────────────────────────────────────────────────────────────────────────
# Collect cases for a given extension module
# ─────────────────────────────────────────────────────────────────────────────

def collect_cases(mod: types.ModuleType) -> list[BenchCase]:
    """
    Build the benchmark case list from a loaded extension module.

    Every function in the module must exist; a missing attribute raises
    AttributeError immediately so mismatches surface at collection time,
    not mid-run.
    """
    f = mod  # shorthand

    # Shared fixtures — created once, reused across all cases.
    arr_any          = make_f64_3d_cc()
    arr_f64_3d_cc    = make_f64_3d_cc()
    arr_f64_3d_fc    = make_f64_3d_fc()
    arr_cf128_2x3_fc = make_cf128_2x3_fc()
    arr_cf128_2x3_cc = make_cf128_2x3_cc()

    return [
        # ── Group 1: unconstrained input, varying body cost ───────────────
        # Isolates how much overhead the binding layer adds for the cheapest
        # possible body.  All accept any array so no conversion happens.
        BenchCase("noop_any",             f.noop_any,             (arr_any,),  group="1_unconstrained"),
        BenchCase("read_ndim_any",        f.read_ndim_any,        (arr_any,),  group="1_unconstrained"),
        BenchCase("read_shape_sum_any",   f.read_shape_sum_any,   (arr_any,),  group="1_unconstrained"),
        BenchCase("read_stride_sum_any",  f.read_stride_sum_any,  (arr_any,),  group="1_unconstrained"),
        BenchCase("check_data_ptr_any",   f.check_data_ptr_any,   (arr_any,),  group="1_unconstrained"),

        # ── Group 2: runtime checks — body does the validation ────────────
        # Compare against Group 3 (typed variants) to see whether baking
        # constraints into the C++ type is cheaper than checking at runtime.
        BenchCase("check_dtype_rt",     f.check_dtype_rt,     (arr_f64_3d_cc,),   group="2_runtime_check"),
        BenchCase("check_ndim_rt",      f.check_ndim_rt,      (arr_f64_3d_cc,),   group="2_runtime_check"),
        BenchCase("check_c_contig_rt",  f.check_c_contig_rt,  (arr_f64_3d_cc,),   group="2_runtime_check"),
        BenchCase("check_full_rt",      f.check_full_rt,      (arr_f64_3d_cc,),   group="2_runtime_check"),
        # Same function, wrong dtype — measures the error-path cost.
        BenchCase("check_dtype_rt/fail", f.check_dtype_rt,    (arr_cf128_2x3_cc,),
                  group="2_runtime_check", expect_raise=True),

        # ── Group 3: type-constrained — binding layer enforces, body is noop
        # The binding itself must reject wrong input, so we also probe the
        # rejection path for each typed variant.
        BenchCase("noop_f64_3d_cc",
                  f.noop_f64_3d_cc, (arr_f64_3d_cc,), group="3_typed"),
        BenchCase("noop_cf128_2x3_fc_cpu",
                  f.noop_cf128_2x3_fc_cpu, (arr_cf128_2x3_fc,), group="3_typed"),
        BenchCase("noop_cf128_2x3_cc_cpu",
                  f.noop_cf128_2x3_fc_cpu, (arr_cf128_2x3_cc,), group="3_typed"),
        BenchCase("check_full_typed_f64_3d",
                  f.check_full_typed_f64_3d, (arr_f64_3d_cc,), group="3_typed"),
        # Rejection: pass wrong dtype to a typed function.
        BenchCase("noop_f64_3d_cc/reject_dtype",
                  f.noop_f64_3d_cc, (arr_cf128_2x3_fc,),
                  group="3_typed", expect_raise=True),
        # Rejection: pass F-contiguous where C-contiguous required.
        BenchCase("noop_f64_3d_cc/reject_order",
                  f.noop_f64_3d_cc, (arr_f64_3d_fc,),
                  group="3_typed", expect_raise=True),

        # ── Group 4: multi-array — does overhead scale with argument count? ─
        BenchCase("noop_two_arrays",  f.noop_two_arrays,
                  (arr_any, arr_any),                             group="4_multi_arg"),
        BenchCase("noop_four_arrays", f.noop_four_arrays,
                  (arr_any, arr_any, arr_any, arr_any),           group="4_multi_arg"),

        # ── Group 5: scalar returns — Python must process the result ────────
        BenchCase("return_ndim",      f.return_ndim,      (arr_any,), group="5_return_value"),
        BenchCase("return_shape_sum", f.return_shape_sum, (arr_any,), group="5_return_value"),
        BenchCase("return_itemsize",  f.return_itemsize,  (arr_any,), group="5_return_value"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Timing engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    backend: str
    case: BenchCase
    n_runs: int
    times_ns: list[float]   # one entry per run, each is the mean of n_iter calls

    @property
    def min_ns(self) -> float:
        return min(self.times_ns)

    @property
    def mean_ns(self) -> float:
        return statistics.mean(self.times_ns)

    @property
    def stdev_ns(self) -> float:
        return statistics.stdev(self.times_ns) if len(self.times_ns) > 1 else 0.0

    @property
    def cv_pct(self) -> float:
        """Coefficient of variation — high values flag noisy measurements."""
        return 100.0 * self.stdev_ns / self.mean_ns if self.mean_ns else 0.0


def run_case(
    case: BenchCase,
    warmup: int,
    runs: int,
    iters_per_run: int,
) -> tuple[list[float], str | None]:
    """
    Returns (times_ns_per_run, error_message_or_None).

    For expect_raise cases we verify the exception fires on every call and
    time the full raise+catch cycle.
    """
    fn = case.fn
    args = case.args

    if case.expect_raise:
        def stmt():
            try:
                fn(*args)
            except Exception:
                pass
    else:
        def stmt():
            _sink(fn(*args))

    # Probe once to catch mismatches before wasting time on warmup.
    try:
        if case.expect_raise:
            fn(*args)
            return [], f"expected exception but call succeeded"
        else:
            _sink(fn(*args))
    except Exception as exc:
        if not case.expect_raise:
            return [], f"unexpected exception: {exc}"

    # Warmup — fills caches, lets the interpreter JIT (if ever) settle.
    timer = timeit.Timer(stmt)
    timer.timeit(warmup)

    # Measurement: take `runs` independent samples, each averaged over
    # `iters_per_run` calls.  Storing all samples lets us compute stdev and
    # spot outliers rather than committing to a single aggregate upfront.
    ns_per_run = [
        timer.timeit(iters_per_run) / iters_per_run * 1e9
        for _ in range(runs)
    ]
    return ns_per_run, None


def benchmark_backend(
    backend_name: str,
    mod: types.ModuleType,
    warmup: int,
    runs: int,
    iters_per_run: int,
    verbose: bool = False,
) -> list[BenchResult]:
    cases = collect_cases(mod)

    # One Timer per case, created upfront
    timers = {}
    for case in cases:
        if case.expect_raise:
            def make_stmt(fn, args):
                def stmt():
                    try: fn(*args)
                    except Exception: pass
                return stmt
            timers[case.name] = timeit.Timer(make_stmt(case.fn, case.args))
        else:
            def make_stmt(fn, args):
                def stmt(): _sink(fn(*args))
                return stmt
            timers[case.name] = timeit.Timer(make_stmt(case.fn, case.args))

    # Warmup all cases before measuring any of them
    for case in cases:
        timers[case.name].timeit(warmup)

    # Collect samples by interleaving: one run per case, repeat `runs` times.
    # This distributes cache/thermal effects evenly across all cases.
    samples: dict[str, list[float]] = {case.name: [] for case in cases}
    for _ in range(runs):
        for case in cases:
            t = timers[case.name].timeit(iters_per_run)
            samples[case.name].append(t / iters_per_run * 1e9)

    results = []
    for case in cases:
        if case.name not in samples:
            continue
        results.append(BenchResult(
            backend=backend_name,
            case=case,
            n_runs=runs,
            times_ns=samples[case.name],
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output formatters
# ─────────────────────────────────────────────────────────────────────────────

_GROUP_LABELS = {
    "1_unconstrained": "Group 1 · Unconstrained input, varying body cost",
    "2_runtime_check": "Group 2 · Runtime validation in body",
    "3_typed":         "Group 3 · Type-constrained (binding layer validates)",
    "4_multi_arg":     "Group 4 · Multiple array arguments",
    "5_return_value":  "Group 5 · Scalar return values",
}

def _ns(val: float) -> str:
    return f"{val:8.1f}"


def print_table(all_results: list[BenchResult], backends: list[str]) -> None:
    # Pivot: case_name → backend → result
    by_case: dict[str, dict[str, BenchResult]] = {}
    by_group: dict[str, list[str]] = {}

    for r in all_results:
        by_case.setdefault(r.case.name, {})[r.backend] = r
        by_group.setdefault(r.case.group, [])
        if r.case.name not in by_group[r.case.group]:
            by_group[r.case.group].append(r.case.name)

    col_w = 22
    name_w = 42

    # Header
    header = f"{'Benchmark':<{name_w}}" + "".join(
        f"{'min ns':>{col_w//2}}{'mean±σ':>{col_w - col_w//2}}"
        .replace("min ns", f"{b} min").replace("mean±σ", f"{b} mean±σ")
        for b in backends
    )
    # Simpler header:
    backend_cols = "  ".join(f"{b+' min':>{12}}  {b+' mean±σ':>{18}}" for b in backends)
    print()
    print(f"{'Benchmark':<{name_w}}  {backend_cols}")
    print("─" * (name_w + 2 + (12 + 2 + 18 + 2) * len(backends)))

    for group_key in sorted(by_group):
        label = _GROUP_LABELS.get(group_key, group_key)
        print(f"\n  {label}")

        for case_name in by_group[group_key]:
            row = f"    {case_name:<{name_w - 4}}"
            for b in backends:
                if b in by_case.get(case_name, {}):
                    r = by_case[case_name][b]
                    mean_sd = f"{r.mean_ns:.1f}±{r.stdev_ns:.1f}"
                    noise_flag = " !" if r.cv_pct > 10 else "  "
                    row += f"  {r.min_ns:>12.1f}  {mean_sd:>18}{noise_flag}"
                else:
                    row += f"  {'—':>12}  {'—':>18}  "
            print(row)

    print()
    print("  All times in nanoseconds (ns).  '!' = CV > 10 %, measurement may be noisy.")
    print(f"  Runs: {all_results[0].n_runs} × iters_per_run (min is the robust estimator).")


def print_csv(all_results: list[BenchResult]) -> None:
    print("backend,group,case,expect_raise,min_ns,mean_ns,stdev_ns,cv_pct,n_runs")
    for r in all_results:
        print(",".join([
            r.backend,
            r.case.group,
            r.case.name,
            str(r.case.expect_raise),
            f"{r.min_ns:.3f}",
            f"{r.mean_ns:.3f}",
            f"{r.stdev_ns:.3f}",
            f"{r.cv_pct:.2f}",
            str(r.n_runs),
        ]))


# ─────────────────────────────────────────────────────────────────────────────
# Backend registry — extend this dict when nanobind / cpython are ready
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_BACKENDS: dict[str, str] = {
    "pybind":   "pybind_ext",
    "nanobind": "nanobind_ext",   # not yet built — skipped automatically
    "cpython":  "cpython_ext",
}


def load_backends(requested: list[str]) -> dict[str, types.ModuleType]:
    loaded: dict[str, types.ModuleType] = {}
    for name in requested:
        module_name = KNOWN_BACKENDS.get(name)
        if module_name is None:
            print(f"[warn] Unknown backend '{name}' — skipping.", file=sys.stderr)
            continue
        try:
            loaded[name] = importlib.import_module(module_name)
            print(f"[ok]   Loaded backend '{name}' ({module_name})")
        except ImportError as exc:
            print(f"[skip] Backend '{name}' not available: {exc}", file=sys.stderr)
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", nargs="+", default=list(KNOWN_BACKENDS),
                   metavar="NAME",
                   help="Backends to run (default: all known). "
                        f"Choices: {list(KNOWN_BACKENDS)}")
    p.add_argument("--warmup", type=int, default=200,
                   help="Number of warmup calls before timing (default: 200)")
    p.add_argument("--runs", type=int, default=10,
                   help="Number of independent timing runs (default: 10)")
    p.add_argument("--iters", type=int, default=1000,
                   help="Calls per run — mean is taken (default: 1000)")
    p.add_argument("--format", choices=["table", "csv"], default="table",
                   help="Output format (default: table)")
    p.add_argument("--verbose", action="store_true",
                   help="Print each case as it runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\nBinding benchmark  —  numpy {np.__version__}  Python {sys.version.split()[0]}")
    print(f"warmup={args.warmup}  runs={args.runs}  iters/run={args.iters}\n")

    backends = load_backends(args.backend)
    if not backends:
        sys.exit("No backends could be loaded — nothing to benchmark.")

    all_results: list[BenchResult] = []
    for name, mod in backends.items():
        print(f"\nBenchmarking '{name}' ...")
        results = benchmark_backend(
            backend_name=name,
            mod=mod,
            warmup=args.warmup,
            runs=args.runs,
            iters_per_run=args.iters,
            verbose=args.verbose,
        )
        all_results.extend(results)
        print(f"  Done — {len(results)} cases measured.")

    print("\n" + "═" * 80)
    if args.format == "csv":
        print_csv(all_results)
    else:
        print_table(all_results, backends=list(backends))


if __name__ == "__main__":
    main()
