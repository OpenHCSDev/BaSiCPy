Python 3.14 numerical checkpoint
================================

Parent integration owner, 2026-09-30. PR1 source base is
29f0a327d2069324f13d0c6fa0809a5c7566a6d4. No production code or dependency
installation changed. Existing Python 3.14.7 and installed scientific packages
were used directly with this isolated worktree's ``src`` on PYTHONPATH.

Actual packages: JAX/JAXlib0.9.2, NumPy2.5.3, SciPy1.18.1,
scikit-image0.26.0, Pydantic2.13.5, pooch1.9.0. CPU platform, BLAS/OpenMP
threads1, disabled pytest plugin autoload and source bytecode/cache writing.
Fit process affinity is one CPU. No datasets, downloads, providers or GUI.

Real checks
-----------

* Existing public JAX inverse-DCT/SciPy comparisons and odd/even 1D/2D/3D
  round trips: 3PASS,9.83s pytest,10.72s process,545160KiB peakRSS.
* New bounded 24-observation32x32 synthetic checks: 3PASS,7.83s pytest,
  9.15s process,450640KiB peakRSS. Real 2D/3D fit-transform preserves
  floating measurement units and shapes, fitted field RMSE<0.15, zero
  disabled darkfield, saved model/settings/profiles reload and correction
  agree. Stationary biology produces materially worse recovered shading
  than moving objects, exposing identifiability rather than promising it.

Original logs/XML and process resource measurements are retained at
``/home/ts/wt/openhcs-issue-batch-20260929/basicpy-parent-numeric-20260930/``.
Each process exited0. Final published-test rerun: 6PASS,13.96s pytest,
15.97s process,690508KiB peakRSS, zero swaps, covering both suites in one
process. The tests use existing model declared attributes directly (BOUND-7).
No new model, field store,
registry, solver, fallback or compatibility layer is introduced (BOUND-2,
IMPL-13, TIME-1). Tests exercise the original BaSiC and JaxDCT owners.

Reproduce from this worktree using the existing interpreter::

    PYTHONPATH=$PWD/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
      python3.14 -B -m pytest -p no:cacheprovider \
      tests/test_public_jax_dct.py tests/test_python314_fit.py -q

Boundaries
----------

This proves real import and bounded CPU numerical behavior on the existing
Python3.14 environment, including persisted profiles. It does not prove a
fresh dependency resolver, minimum-version matrix, CUDA, autotuning, fitted
additive darkfield, real acquisitions, or OpenHCS compiled/MCP execution.
No installed user entrypoint or blind harness changed. PR217 and paired
ArrayBridge integration remain separate acceptance boundaries.
