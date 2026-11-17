"""Public INT8 deployment, matching the frozen Gen-6 collector and EOF contract.

Adapted for the flat redistributable web/models inventory in v0.6.5.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch


DEFAULTS = {
    "detector": "web/models/temporal_det_nano.int8.onnx",
    "cutout": "web/models/vehicle_cutout_nano.int8.onnx",
    "stage2": "web/models/stage2.int8.onnx",
    "ocr_dir": "web/models",
    "calibration": "web/models/gen6.json",
}


def _artifact(root: Path, raw: str, subtree: str, *, directory=False) -> Path:
    """Reject traversal, absolute paths and symlinks before reading artifacts."""
    if not isinstance(raw, str):
        raise ValueError("Artifact paths must be repository-relative strings")
    rel = Path(raw)
    if rel.is_absolute() or ".." in rel.parts or not rel.is_relative_to(subtree):
        raise ValueError("Artifact outside frozen deployment inventory")
    target = root / rel
    if any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
        raise ValueError("Symlinked deployment artifacts are forbidden")
    if not target.resolve().is_relative_to((root / subtree).resolve()):
        raise ValueError("Artifact outside frozen deployment inventory")
    if not (target.is_dir() if directory else target.is_file()):
        raise FileNotFoundError(f"Missing frozen artifact: {rel}")
    return target


def _check_graph(root: Path, path: Path) -> None:
    """ORT must never follow an external tensor reference outside the inventory."""
    import onnx
    from onnx.external_data_helper import _get_all_tensors

    if not path.name.endswith(".int8.onnx"):
        raise ValueError("Frozen inference requires INT8 graphs")
    graph = onnx.load(str(path), load_external_data=False)
    for tensor in _get_all_tensors(graph):
        for entry in tensor.external_data:
            if entry.key == "location":
                external = Path(entry.value)
                if external.is_absolute() or ".." in external.parts:
                    raise ValueError("External tensor outside frozen inventory")
                _artifact(root, (path.parent / external).relative_to(root).as_posix(), "web/models")


class _ONNXOCRModel(torch.nn.Module):
    """Tensor interface for MFPlateOCRSessionV2 backed exclusively by ORT graphs.

    Pipeline owns the normal sealed MF sessions; this bridge shares read-only ORT
    graphs across tracks and does no character projection in encode/step/risk.
    """
    is_v2 = True

    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        shape = runtime.enc_session.get_inputs()[0].shape
        self.config = SimpleNamespace(input_hw=tuple(shape[-2:]), canonical_margin=0.05)

    @staticmethod
    def _run(session, values):
        feed = {meta.name: value.detach().cpu().numpy()
                for meta, value in zip(session.get_inputs(), values)}
        return tuple(torch.from_numpy(value) for value in session.run(None, feed))

    def init_memory(self, batch_size, device=None):
        from vpocr_ocr.model_v2 import MemoryStateV2
        if batch_size != 1 or torch.device(device or "cpu").type != "cpu":
            raise ValueError("Frozen OCR collectors require single-track CPU state")
        return MemoryStateV2(*(torch.from_numpy(a) for a in self.runtime._empty_state()))

    def encode_frames(self, crops, sides, geometry):
        return self._run(self.runtime.enc_session, (crops, sides, geometry))

    def step_memory(self, *values):
        from vpocr_ocr.model_v2 import MemoryStateV2
        return MemoryStateV2(*self._run(self.runtime.step_session, values))

    def stopping_risk_head(self, quality):
        return self._run(self.runtime.risk_session, (quality,))[0]

    def decode_memory(self, *values):
        return self._run(self.runtime.read_session, values)


def _terminal_predictions(decisions) -> list[dict]:
    predictions = []
    for dec in decisions.values():
        if dec.status != "ACCEPTED" or dec.plate_text is None:
            continue
        full = dec.confidence_tier == "FULL_4FIELD"
        if "serial" not in dec.verified_fields and not full:
            continue
        serial = _canonical_serial(dec.plate_text.serial)
        if serial is None:
            continue
        row = {"serial": serial}
        if full:
            text = _canonical_full_text(dec.plate_text.full_text)
            if text is not None:
                row["full_text"] = text
        predictions.append(row)
    return predictions


def _canonical_serial(value):
    """The frozen gate's four-ASCII-digit normalization contract."""
    if type(value) is int:
        return f"{value:04d}" if 0 <= value <= 9999 else None
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize("NFKC", value).strip()
    if not re.fullmatch(r"[ .·・]*[0-9]{1,4}(?:[ -]?[0-9]{1,4})?", value):
        return None
    digits = re.sub(r"[^0-9]", "", value)
    return digits.zfill(4) if 1 <= len(digits) <= 4 else None


def _canonical_full_text(value):
    if not isinstance(value, str) or not value.strip():
        return None
    text = "".join(unicodedata.normalize("NFKC", value).split())
    match = re.fullmatch(r"(.*?[^0-9.·・-])([.·・]*[0-9]+(?:-[0-9]+)?)", text)
    if match:
        serial = _canonical_serial(match[2])
        return match[1] + serial if serial is not None else None
    return text


class INT8Predictor:
    def __init__(self, pipeline_factory):
        self._pipeline_factory = pipeline_factory

    def predict_clip(self, path: Path) -> list[dict]:
        # Only the caller-supplied clip is opened; no directory/label discovery.
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("Cannot decode supplied video clip")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Video clip lacks a valid frame rate")
            pipeline = self._pipeline_factory()
            index = 0
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                pipeline.process_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), timestamp_s=index / fps)
                if any(s.num_readout_calls for s in pipeline._ocr_sessions.values()):
                    raise RuntimeError("Character readout during frozen streaming")
                index += 1
            if index == 0:
                raise ValueError("Video clip contains no decodable frames")
            return _terminal_predictions(pipeline.finalize_tracks())
        finally:
            capture.release()


class _ClipCollectors(dict):
    """Retain retired collectors and defer pipeline retirement reads to EOF.

    Relinking still operates on the original MF sessions and disjoint factors.
    The pipeline may cache a collecting placeholder on retirement; EOF uses the
    retained terminal capability, never that placeholder. Merged-away IDs are
    excluded by the pipeline's active/retired inventories.
    """
    def __init__(self):
        super().__init__()
        self.retained = {}
        self.terminal_reads = {}

    def __setitem__(self, key, session):
        if session not in self.terminal_reads:
            self.terminal_reads[session] = session.finish
            session.finish = session._placeholder
        self.retained[key] = session
        super().__setitem__(key, session)

    def finalize(self, keys):
        return {key: self.terminal_reads[self.retained[key]]() for key in keys}


def create_predictor(repo_root: Path, config: dict) -> INT8Predictor:
    """Build the deployed pipeline solely from frozen graphs and calibration."""
    import onnxruntime as ort
    from vpocr_det.detector import UnifiedTemporalDetORT
    from vpocr_det.pipeline import MultiRateEdgeCascadePipeline
    from vpocr_det.quad_cropper import STAGE2_OUTPUT_NAMES
    from vpocr_det.tracker import TrackerCalibration, PlateTrackStabilizer, MultiVehicleKalmanTracker
    from vpocr_ocr.onnx_export_v2 import OrtMFPlateOCRSessionV2
    from vpocr_ocr.session_v2 import SessionConfigV2, TerminalCalibrationV2
    from vpocr_ocr.vocab import PlateVocab

    if not isinstance(config, dict) or config.keys() - DEFAULTS.keys():
        raise ValueError("Unknown frozen adapter configuration")
    root = Path(repo_root).resolve()
    paths = {key: _artifact(root, config.get(key, default),
                           "web/models",
                           directory=key == "ocr_dir")
             for key, default in DEFAULTS.items()}
    graphs = [paths[k] for k in ("detector", "cutout", "stage2")]
    graphs += [_artifact(root, (paths["ocr_dir"] / f"{part}.int8.onnx").relative_to(root).as_posix(), "web/models")
               for part in ("frame_encoder", "memory_step", "memory_readout", "stopping_risk")]
    for graph in graphs:
        _check_graph(root, graph)
    policy = json.loads(paths["calibration"].read_text())
    if not isinstance(policy, dict) or policy.keys() - {"terminal", "tracker", "session", "pipeline"}:
        raise ValueError("Invalid frozen calibration schema")
    terminal = TerminalCalibrationV2(**policy["terminal"])
    tracker_cal = TrackerCalibration(**policy["tracker"])
    cfg = SessionConfigV2(**policy.get("session", {}), calibration=terminal)
    if cfg.risk_calibrated:
        raise ValueError("Frozen streaming must collect until clip finalization")
    vocab = PlateVocab()
    det = UnifiedTemporalDetORT(paths["detector"], paths["cutout"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    stage2 = ort.InferenceSession(str(paths["stage2"]), options, providers=["CPUExecutionProvider"])
    if ({meta.name for meta in stage2.get_outputs()} != set(STAGE2_OUTPUT_NAMES)
            or [meta.name for meta in stage2.get_inputs()] != ["fovea_rois", "surround_rois", "sampling_policy"]):
        raise ValueError("Stage-2 ONNX lacks the complete Gen-6 observation contract")
    runtime = OrtMFPlateOCRSessionV2(paths["ocr_dir"], vocab=vocab, config=cfg,
                                   use_int8=True, providers=["CPUExecutionProvider"], intra_op_threads=1)
    model = _ONNXOCRModel(runtime).eval()
    pipeline_options = policy.get("pipeline", {})
    allowed = {"plate_thresh", "min_is_plate_prob", "enable_far_zoom", "heartbeat_interval",
               "max_heartbeat_interval", "overload_track_threshold", "max_retired_history"}
    if not isinstance(pipeline_options, dict) or pipeline_options.keys() - allowed:
        raise ValueError("Invalid frozen pipeline options")

    class ClipPipeline(MultiRateEdgeCascadePipeline):
        def finalize_tracks(self):
            keys = dict.fromkeys([*self._retired_decisions, *self._ocr_sessions])
            return self._ocr_sessions.finalize(keys)

    def pipeline_factory():
        pipeline = ClipPipeline(det, stage2, device="cpu", ocr_model=model,
            ocr_vocab=vocab, ocr_session_config=cfg, **pipeline_options)
        pipeline._ocr_sessions = _ClipCollectors()
        pipeline.tracker = PlateTrackStabilizer(calibration=tracker_cal)
        pipeline.vehicle_tracker = MultiVehicleKalmanTracker(calibration=tracker_cal)
        return pipeline

    return INT8Predictor(pipeline_factory)
