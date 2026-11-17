"""Japanese vehicle and license plate OCR using the released INT8 cascade."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

__version__ = "0.6.5"
MODEL_NAME = "vehicle-plate-ocr-jp-v1"

_PUBLIC = {
    "VehiclePlateVideoPipeline": "vpocr_det.pipeline",
    "MultiRateEdgeCascadePipeline": "vpocr_det.pipeline",
    "MFPlateOCRSessionV2": "vpocr_ocr.session_v2",
    "SessionConfigV2": "vpocr_ocr.session_v2",
    "TerminalCalibrationV2": "vpocr_ocr.session_v2",
    "OCRDecisionV2": "vpocr_ocr.session_v2",
}
__all__ = ["__version__", "MODEL_NAME", "create_predictor", *_PUBLIC]


def __getattr__(name: str):
    """Keep version and CLI metadata available without importing torch or ORT."""
    if name in _PUBLIC:
        value = getattr(import_module(_PUBLIC[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    if (repo / "pyproject.toml").is_file() and (repo / "web/models").is_dir():
        return repo
    bundled = Path(__file__).resolve().parent / "assets"
    if (bundled / "web/models").is_dir():
        return bundled
    raise FileNotFoundError("Model bundle missing; supply root containing web/models/")


def create_predictor(root: str | Path | None = None, config: dict | None = None):
    """Load the all-INT8 CPU predictor with terminal-only character readout.

    An unconfigured research checkout uses its original frozen adapter. A fresh
    clone or installed wheel uses the flat bundled inventory. Public overrides
    accept detector, cutout, stage2, ocr_dir and calibration paths beneath
    web/models/; absolute paths, traversal and external symlinks are rejected.
    No checkpoints, labels, FP32 graphs or network downloads are used.
    """
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be a dictionary or None")
    resolved = _resolve_root(root)
    if (resolved / "out/models/onnx_temporal_det_nano/temporal_det_nano.int8.onnx").is_file() and not config:
        # Resolve the frozen file directly so the installed CLI also works when
        # invoked from outside the checkout (tools/ is not a wheel package).
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location(
            "_vehicle_plate_ocr_jp_v1_frozen_adapter",
            resolved / "tools/frozen_gen6_adapter.py",
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the repository frozen adapter")
        adapter = module_from_spec(spec)
        spec.loader.exec_module(adapter)
        return adapter.create_predictor(resolved, config or {})
    from ._runtime import create_predictor as bundled_predictor
    return bundled_predictor(resolved, config or {})
