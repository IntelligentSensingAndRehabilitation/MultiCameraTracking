"""Tests for EasymocapTracking retry logic on zero tracks and LinAlgError."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Stub modules that require DB connections before importing the function under test
for _mod_name in [
    "pose_pipeline",
    "multi_camera.datajoint.multi_camera_dj",
]:
    if _mod_name not in sys.modules:
        _stub = ModuleType(_mod_name)
        for _attr in [
            "Video",
            "VideoInfo",
            "BottomUpPeople",
            "schema",
            "MultiCameraRecording",
            "SingleCameraVideo",
            "Calibration",
            "CalibratedRecording",
        ]:
            _stub.__dict__[_attr] = MagicMock()
        sys.modules[_mod_name] = _stub

from multi_camera.datajoint.easymocap import _try_tracking_configs  # noqa: E402

MODULE = "multi_camera.datajoint.easymocap"


def _make_results_with_ids(ids: list[int]) -> list[list[dict]]:
    return [[{"id": pid} for pid in ids]]


def _make_empty_results() -> list[list[dict]]:
    return [[]]


@pytest.fixture
def dataset():
    return MagicMock()


@patch(f"{MODULE}.TRACKING_CONFIGS", ["config_a.yml", "config_b.yml"])
class TestTryTrackingConfigs:
    def test_zero_tracks_triggers_retry(self, dataset):
        tracking_fn = MagicMock(
            side_effect=[
                _make_empty_results(),
                _make_results_with_ids([0, 1]),
            ]
        )
        result = _try_tracking_configs(dataset, tracking_fn)

        assert tracking_fn.call_count == 2
        assert result.results is not None
        assert result.num_tracks == 2
        assert result.config_path == "config_b.yml"
        assert result.failures == ["0_tracks"]

    def test_all_configs_zero_tracks(self, dataset):
        tracking_fn = MagicMock(return_value=_make_empty_results())
        result = _try_tracking_configs(dataset, tracking_fn)

        assert tracking_fn.call_count == 2
        assert result.results is None
        assert result.failures == ["0_tracks", "0_tracks"]

    def test_all_svd_errors(self, dataset):
        tracking_fn = MagicMock(
            side_effect=np.linalg.LinAlgError("SVD did not converge")
        )
        result = _try_tracking_configs(dataset, tracking_fn)

        assert tracking_fn.call_count == 2
        assert result.results is None
        assert result.failures == ["SVD_convergence", "SVD_convergence"]

    def test_mixed_failures_uses_first_success(self, dataset):
        with patch(f"{MODULE}.TRACKING_CONFIGS", ["a.yml", "b.yml", "c.yml"]):
            tracking_fn = MagicMock(
                side_effect=[
                    np.linalg.LinAlgError("SVD did not converge"),
                    _make_empty_results(),
                    _make_results_with_ids([0]),
                ]
            )
            result = _try_tracking_configs(dataset, tracking_fn)

        assert tracking_fn.call_count == 3
        assert result.results is not None
        assert result.num_tracks == 1
        assert result.config_path == "c.yml"
        assert result.failures == ["SVD_convergence", "0_tracks"]

    def test_all_mixed_failures(self, dataset):
        tracking_fn = MagicMock(
            side_effect=[
                np.linalg.LinAlgError("SVD did not converge"),
                _make_empty_results(),
            ]
        )
        result = _try_tracking_configs(dataset, tracking_fn)

        assert tracking_fn.call_count == 2
        assert result.results is None
        assert result.failures == ["SVD_convergence", "0_tracks"]

    def test_first_config_succeeds(self, dataset):
        tracking_fn = MagicMock(return_value=_make_results_with_ids([0, 1, 2]))
        result = _try_tracking_configs(dataset, tracking_fn)

        assert tracking_fn.call_count == 1
        assert result.results is not None
        assert result.num_tracks == 3
        assert result.config_path == "config_a.yml"
        assert result.failures == []
