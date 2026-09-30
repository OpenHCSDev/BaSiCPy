"""Public JAX DCT owner matches SciPy on odd/even multidimensional shapes."""

import numpy as np
import pytest
import scipy.fft

from basicpy.tools.dct_tools import JaxDCT


@pytest.mark.parametrize("shape", [(7,), (4, 7), (3, 4, 7)])
def test_public_jax_inverse_dct_matches_scipy_and_round_trips(shape):
    values = np.linspace(-3, 10, np.prod(shape), dtype=np.float32).reshape(shape)
    actual = JaxDCT.idctnd(values, len(shape))
    expected = scipy.fft.idctn(values, norm="ortho", type=2)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        JaxDCT.idctnd(JaxDCT.dctnd(values), len(shape)),
        values,
        rtol=2e-5,
        atol=2e-5,
    )
