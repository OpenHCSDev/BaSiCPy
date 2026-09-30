OpenHCS distribution checkpoint
==============================

``openhcs-basicpy`` 1.3.0 packages the existing OpenHCSDev JAX fork as an
ordinary registry dependency. The Python package remains ``basicpy``;
no algorithm, dispatcher, compatibility alias or second implementation is
introduced. Do not install upstream ``basicpy`` beside this distribution: both
own that Python package. Scientific attribution and the MIT license are retained.

The VERSION file owns the release version. Existing release automation builds
on Python 3.12, then uses the organization's existing PyPI credential. OpenHCS
must not merge an unavailable dependency or put a direct Git URL into published
metadata. A successful wheel build is not registry publication.

Actual local evidence
---------------------

A built universal wheel was installed into an owned isolated target (no
dependency downloads and no shared environment changes). Both existing Python
3.12 and Python 3.14 imported ``basicpy`` from that target, with distribution
metadata identifying ``openhcs-basicpy`` 1.3.0. Six tests passed on each:
public JAX inverse-DCT parity, 2D/3D ensemble fits, preserved measurement units,
saved-model/profile reload, and the stationary-biology negative control.
Python 3.12: 13.24s; Python 3.14: 12.18s. The supported scikit-image range includes
the actual Python 3.12 environment's 0.25.2 and the modern 0.26 line.

This does not establish CUDA, autotuning, a minimum-version matrix, OpenHCS
compiled/MCP routing, real biological acceptance, or ordinary shared-install
activation. Durable wheel/XML receipts are retained under
``/home/ts/wt/openhcs-issue-batch-20260929/basicpy-project-dependency-20260930``;
the disposable installed target belongs to the parent dependency task at
``/home/ts/.cache/agent-scratch/basicpy-package-20260930/installed``.

Corrective publication checkpoint
--------------------------------

The advertised console script failed its actual installed entrypoint check:
``basicpy --help`` imported a removed ``Device`` type. Source inspection also
found that its processing body was only TODO comments and a no-op return.
The queued 1.3.0 release was cancelled before any job steps ran; its original
tag and evidence are retained, not overwritten. No 1.3.0 publication is claimed.

Version 1.3.1 removes that dead module and console-script declaration in place.
The functioning BaSiC Python API remains unchanged. A standalone CLI is not
implemented as an unrelated expansion of the OpenHCS dependency request.

Actual corrective acceptance: build from a fresh sdist, install the new wheel
into a separate owned target, then seven tests pass on Python3.14 in12.58s.
The new installed-distribution test verifies no advertised console command and
no dead CLI module. All functioning Python module bytes match the earlier
wheel that passed numerical tests on both Python3.12 and3.14; the only removed
Python source is the unimplemented CLI. Thus no numerical owner was rewritten.
