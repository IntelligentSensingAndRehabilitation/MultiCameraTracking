"""Shared test scaffolding for acquisition tests.

PySpin and simple_pyspin can't be installed in the dev container (they ship as
binary wheels tied to the host Spinnaker SDK), so we stub the symbols imported
at module load time. Tests that need richer behaviour can override the stubs
locally before importing the backend.
"""

from __future__ import annotations

import sys
import types


def _install_pyspin_stubs() -> None:
    if "PySpin" in sys.modules and "simple_pyspin" in sys.modules:
        return

    if "simple_pyspin" not in sys.modules:
        stub = types.ModuleType("simple_pyspin")

        class _Camera:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("PySpin stub")

        stub.Camera = _Camera  # type: ignore[attr-defined]
        stub._SYSTEM = None  # type: ignore[attr-defined]
        stub.list_cameras = lambda: []  # type: ignore[attr-defined]
        sys.modules["simple_pyspin"] = stub

    if "PySpin" not in sys.modules:
        stub = types.ModuleType("PySpin")

        class _InterfaceEventHandler:
            def __init__(self, *args, **kwargs):
                pass

        stub.InterfaceEventHandler = _InterfaceEventHandler  # type: ignore[attr-defined]
        stub.IsReadable = lambda node: False  # type: ignore[attr-defined]
        stub.CStringPtr = lambda node: None  # type: ignore[attr-defined]
        sys.modules["PySpin"] = stub


_install_pyspin_stubs()
