OpenHCS distribution checkpoint
==============================

``openhcs-basicpy`` 1.3.0 packages the existing OpenHCSDev JAX fork as an
ordinary registry dependency. The Python package and CLI remain ``basicpy``;
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
