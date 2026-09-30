"""Bounded, offline fits through the real BaSiC model and persisted profiles."""

import numpy as np
import pytest

from basicpy import BaSiC


def shaded_observations(*, stationary=False, volume=False):
    yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
    flatfield = 1.2 - 0.3 * (yy**2 + xx**2) + 0.1 * xx
    flatfield /= flatfield.mean()
    rng = np.random.default_rng(213)
    frames = []
    for index in range(24):
        background = 3000 + 50 * index
        if stationary:
            signal = background * np.exp(-((xx - 0.2)**2 + (yy + 0.1)**2) / 0.09)
        else:
            cx, cy = rng.uniform(-0.8, 0.8, size=2)
            signal = 5000 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / 0.008)
        frames.append(np.rint((background + signal) * flatfield).astype(np.uint16))
    images = np.stack(frames)
    if volume:
        images = np.stack((images, images), axis=1)
        flatfield = np.stack((flatfield, flatfield))
    return images, flatfield


def fit_model(images):
    model = BaSiC(max_iterations=100, working_size=None, max_workers=1)
    corrected = np.asarray(model.fit_transform(images, timelapse=False))
    return model, corrected


@pytest.mark.parametrize("volume", [False, True])
def test_real_fit_and_saved_profiles_preserve_measurement_units(tmp_path, volume):
    images, truth = shaded_observations(volume=volume)
    model, corrected = fit_model(images)
    assert corrected.shape == images.shape
    assert model.flatfield.shape == model.darkfield.shape == images.shape[1:]
    assert np.issubdtype(corrected.dtype, np.floating)
    assert np.isfinite(corrected).all()
    np.testing.assert_allclose(model.darkfield, 0, atol=1e-6)
    np.testing.assert_allclose(corrected, images / model.flatfield, rtol=1e-5, atol=1e-3)
    assert np.sqrt(np.mean((model.flatfield - truth)**2)) < 0.15
    assert np.any(corrected != np.floor(corrected))

    destination = tmp_path / "model"
    model.save_model(destination)
    restored = BaSiC.load_model(destination)
    assert restored.model_dump() == model.model_dump()
    np.testing.assert_array_equal(restored.flatfield, model.flatfield)
    np.testing.assert_array_equal(restored.darkfield, model.darkfield)
    np.testing.assert_array_equal(restored.baseline, model.baseline)
    np.testing.assert_allclose(restored.transform(images), corrected, rtol=1e-6, atol=1e-3)


def test_stationary_signal_is_not_mistaken_for_evidence_of_identifiability():
    moving, truth = shaded_observations()
    stationary, _ = shaded_observations(stationary=True)
    moving_model, _ = fit_model(moving)
    stationary_model, _ = fit_model(stationary)
    moving_error = np.sqrt(np.mean((moving_model.flatfield - truth)**2))
    stationary_error = np.sqrt(np.mean((stationary_model.flatfield - truth)**2))
    assert stationary_error > moving_error + 0.02
