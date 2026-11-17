"""Fast release inventory, grammar parity, fresh-clone loading and CLI checks."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import zlib

import numpy as np
import pytest

import vehicle_plate_ocr_jp_v1 as package
from vehicle_plate_ocr_jp_v1 import cli

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "web/models"
ARTIFACTS = tuple(f"{name}.int8.onnx" for name in cli.GRAPH_NAMES) + cli.CALIBRATION_NAMES


def test_release_inventory_and_size():
    for name in ("LICENSE", "NOTICE", "web/models/grammar_tables.json"):
        assert (ROOT / name).is_file(), name
    assert "Version 2.0, January 2004" in (ROOT / "LICENSE").read_text()
    assert "END OF TERMS AND CONDITIONS" in (ROOT / "LICENSE").read_text()
    assert "vehicle-plate-ocr-jp-v1 v0.6.5" in (ROOT / "NOTICE").read_text()
    assert all((MODELS / name).stat().st_size > 0 for name in ARTIFACTS)
    assert sum((MODELS / name).stat().st_size for name in ARTIFACTS) <= 15 * 1024 * 1024
    assert (MODELS / "grammar_tables.json").stat().st_size < 150_000


def test_exact_terminal_grammar_tables():
    from vpocr_ocr.session_v2 import _terminal_tables
    from vpocr_ocr.vocab import JPPlateGrammar, PlateVocab

    vocab = PlateVocab()
    regions, classes, kana, _, valid = _terminal_tables(JPPlateGrammar(vocab))
    table = json.loads((MODELS / "grammar_tables.json").read_text())
    assert table["tokens"] == [vocab.id_to_token[i] for i in range(254)]
    assert table["pad_id"] == vocab.pad_id
    assert table["dot_id"] == vocab.dot_id
    assert table["plate_types"] == [p.value for p in vocab.plate_types]
    assert len(table["plate_types"]) == 5
    np.testing.assert_array_equal(table["regions"], regions)
    np.testing.assert_array_equal(table["classes"], classes)
    np.testing.assert_array_equal(table["kana_ids"], kana)
    assert table["valid_encoding"] == "zlib-packbits-little-c-order"
    packed = zlib.decompress(base64.b64decode(table["valid_base64"], validate=True))
    restored = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little", count=valid.size)
    assert table["valid_shape"] == list(valid.shape)
    np.testing.assert_array_equal(restored.reshape(table["valid_shape"]).astype(bool), valid)
    decode = lambda ids: "".join(vocab.id_to_token[int(i)] for i in ids if i != vocab.pad_id)
    assert table["region_names"] == [decode(row) for row in regions]
    assert table["class_strings"] == [decode(row) for row in classes]
    assert table["kana_chars"] == [decode([i]) for i in kana]


def test_bundled_predictor_in_fresh_clone(tmp_path, monkeypatch):
    from vpocr_det.pipeline import MultiRateEdgeCascadePipeline
    from vehicle_plate_ocr_jp_v1._runtime import INT8Predictor

    shutil.copytree(MODELS, tmp_path / "web/models")
    monkeypatch.chdir(tmp_path)
    # Also exercise default repository discovery and uncached vocabulary loading.
    module_path = tmp_path / "src/vehicle_plate_ocr_jp_v1/__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "vehicle-plate-ocr-jp-v1"\n')
    monkeypatch.setattr(package, "__file__", str(module_path))
    from vpocr_synth.plate.data_loader import PlateDataLoader
    for name in ("_cached_regions", "_cached_region_map", "_cached_charset"):
        monkeypatch.setattr(PlateDataLoader, name, None)
    assert not (tmp_path / "out").exists()
    predictor = package.create_predictor()
    assert isinstance(predictor, INT8Predictor)
    pipeline = predictor._pipeline_factory()
    assert isinstance(pipeline, MultiRateEdgeCascadePipeline)
    assert pipeline.ocr_session_config.risk_calibrated is False
    assert callable(predictor.predict_clip)


def test_default_predictor_and_cli(capsys):
    predictor = package.create_predictor()
    assert callable(predictor.predict_clip)
    with pytest.raises(SystemExit) as caught:
        cli.main(["--version"])
    assert caught.value.code == 0
    assert capsys.readouterr().out.strip() == "vehicle-plate-ocr-jp-v1 0.6.5"
    assert cli.main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["total_bytes"] == sum((MODELS / name).stat().st_size for name in ARTIFACTS)
    assert info["contract"]["streaming_character_readouts"] == 0
    assert info["contract"]["states"] == ["COLLECTING", "SEALED", "FINALIZED"]


def test_invalid_public_configuration_fails_closed():
    with pytest.raises(ValueError, match="Unknown frozen adapter configuration"):
        package.create_predictor(config={"checkpoint": "unused.pt"})
    with pytest.raises(ValueError, match="outside frozen deployment inventory"):
        package.create_predictor(config={"detector": "../escape.int8.onnx"})


def test_copy_matches_research_inventory_when_present():
    sources = (
        *(ROOT / "out/models/onnx_temporal_det_nano" / name for name in ARTIFACTS[:2]),
        ROOT / "out/models/onnx_det_v4/stage2.int8.onnx",
        *(ROOT / "out/models/onnx_v2_nano" / name for name in ARTIFACTS[3:7]),
        *(ROOT / "out/calibration" / name for name in cli.CALIBRATION_NAMES),
    )
    for source, name in zip(sources, ARTIFACTS, strict=True):
        if source.is_file():
            assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256((MODELS / name).read_bytes()).digest()


def test_public_reexports():
    from vpocr_det.pipeline import VehiclePlateVideoPipeline, MultiRateEdgeCascadePipeline
    from vpocr_ocr.session_v2 import MFPlateOCRSessionV2, OCRDecisionV2, SessionConfigV2, TerminalCalibrationV2
    for cls in (VehiclePlateVideoPipeline, MultiRateEdgeCascadePipeline, MFPlateOCRSessionV2,
                OCRDecisionV2, SessionConfigV2, TerminalCalibrationV2):
        assert getattr(package, cls.__name__) is cls
    assert package.__version__ == "0.6.5"
    assert package.MODEL_NAME == "vehicle-plate-ocr-jp-v1"


def test_synthetic_browser_demo_has_accepted_terminal_prediction():
    """The shipped example must work through the public, pixel-only predictor."""
    predictions = package.create_predictor().predict_clip(
        ROOT / "web/samples/synthetic_gate_demo.mp4")
    assert predictions, "The browser sample must yield an ACCEPTED terminal prediction"
