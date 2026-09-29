# JAX prototype modernization for OpenHCS

This is a draft source checkpoint, not an installed or fit-validated release.
Integration owner: Linnaeus (OpenHCS backend/dependencies, issue
[OpenHCS #213](https://github.com/OpenHCSDev/openhcs/issues/213)).

## Provenance and API

Reuses the clean `trissim/BaSiCPy` JAX prototype at `ae2c647`, including Tristan's
`fc9da06` dependency change and `ae2c647` public `functools.wraps` change.
The LADMAP/approximate optimization and correction formula are not replaced.
`BaSiC.fit_transform(images, timelapse=False)`, `FittingMode`, `flatfield`,
`darkfield`, model settings and saved profile names retain their existing owners.
Images are independent observations `(N,Y,X)` or `(N,Z,Y,X)`, never a mixture
of channels. No pseudo-BaSiC algorithm or permissive alternate import is added.

Current [peng-lab 2.x](https://github.com/peng-lab/BaSiCPy) uses PyTorch and a
different model/API, including `is_timelapse` rather than `timelapse`.
Its [PR #174](https://github.com/peng-lab/BaSiCPy/pull/174) already addresses
modern SciPy for that implementation; [PR #170](https://github.com/peng-lab/BaSiCPy/pull/170)
targets Python 3.12/3.13. This fork preserves the existing JAX implementation
instead of duplicating those PyTorch changes or accidentally calling its API.

## Candidate dependency matrix

The running OpenHCS venv is Python 3.12.3 with JAX/JAXlib 0.9.2; BaSiCPy is not
installed. `/usr/bin/python3.14` is Python 3.14.7, linux-x86_64. Neither this
fork's dependencies nor its model have been imported on 3.14 yet.

| Owner | Source/metadata basis | Draft constraint |
| --- | --- | --- |
| JAX | [0.9.2 metadata](https://pypi.org/pypi/jax/0.9.2/json); [JAXlib cp314 wheels](https://pypi.org/pypi/jaxlib/0.9.2/json) | `jax>=0.9.2,<0.10`; JAX owns matching JAXlib |
| NumPy | JAX 0.9.2 requires NumPy 2 | `numpy>=2.0` |
| SciPy | [1.16.2 cp314 wheels](https://pypi.org/pypi/scipy/1.16.2/json) | `scipy>=1.16.2`; no old `scipy<1.13` cap |
| scikit-image | [0.26.0 cp314 wheels](https://pypi.org/pypi/scikit-image/0.26.0/json); 0.25.2 has none | `scikit-image>=0.26.0,<0.27` |
| Pydantic | [2.12 adds Python 3.14 support](https://pydantic.dev/articles/pydantic-v2-12-release) | `pydantic>=2.12,<3` |

Wheel availability and syntax are not import, numerical, GPU, or application
readiness evidence. These ranges are reviewable candidates pending a real
isolated Python 3.14 resolve/import/fit. Autotuning's old Hyperactive dependency
is owned by the existing `autotune` method and optional extra, not imported to
perform an ordinary fit. Autotuning is not part of the readiness claim.

## Focused ownership audit

Scope: changed JAX inverse-DCT owner, model/Pydantic boundary, dependency/test
configuration. This is source/AST/catalog review, not a full NRA proof/scan.

- IMPL-13 / TIME-1: existing `JaxDCT` now delegates to JAX's
  [public `idctn`](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.fft.idctn.html).
  The copied private-JAX helper and its tests are deleted in place; no fallback.
- TIME-6: removed nonexistent module-level DCT exports, not new aliases.
- BOUND-2: model settings and JAX pytree metadata use Pydantic's current
  `model_dump`/`model_copy`/`model_dump_json`, with existing model types retained.
- New-case check: another DCT backend belongs to the existing `DCT` family and
  derived `DCT_BACKENDS`. It needs no caller-side roster or copied inverse-DCT
  implementation. A new model field remains declared on `BaSiC`/`BaseFit`; their
  dumps project it without another schema list. The wrapper must deliberately
  expose a useful new user parameter, not mirror the entire model.

Inherited resize/fitting-mode dispatch and the unfinished historical CLI are
outside this focused patch. No claim of a globally clean codebase is made.

## Evidence and pending acceptance

Executed with no numerical imports or environment changes:

```
/usr/bin/python3.14 scripts/check_source_contracts.py
```

Four source tests pass (0.052 seconds): syntax, public inverse-DCT owner,
dependency contract and current Pydantic owner API. This does not execute DCT
or fit. `tests/test_public_jax_dct.py` contains the real SciPy comparison and
round-trip regression for odd/even 1D/2D/3D arrays; execution is pending.

Next shared slot, after parent #151/#212 acceptance and explicit handoff:

1. Verify isolated import paths, resolve actual cp314 dependencies and import.
2. Bound CPU threads, use synthetic independent same-channel shaded fields,
   run real DCT regressions and fit-transform; inspect both fitted fields and
   report elapsed time, peak/RSS/PSS/private/swap memory.
3. Compare recovered shading and moving synthetic objects; include a stationary
   object negative control to expose biological leakage/identifiability limits.
4. Verify paired OpenHCS compiled execution and real MCP discovery/signature/
   execute/artifact inspection. No blind biology inputs or installed changes.

All PRs remain draft; no installation, Python 3.14 fit, CUDA support or
OpenHCS entrypoint acceptance is claimed here.
