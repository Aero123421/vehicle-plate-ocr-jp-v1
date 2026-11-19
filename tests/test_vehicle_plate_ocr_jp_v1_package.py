"""Fast release inventory, grammar parity, fresh-clone loading and CLI checks."""
import base64
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import zlib

import numpy as np
import pytest

import vehicle_plate_ocr_jp_v1 as package
from vehicle_plate_ocr_jp_v1 import cli

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "web/models"
ARTIFACTS = tuple(f"{name}.int8.onnx" for name in cli.GRAPH_NAMES) + cli.CALIBRATION_NAMES


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    """Keep this targeted smoke suite within its short CPU-only budget."""
    import torch
    import cv2
    previous = torch.get_num_threads()
    previous_cv = cv2.getNumThreads()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    yield
    torch.set_num_threads(previous)
    cv2.setNumThreads(previous_cv)


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


def test_browser_isolation_and_local_only_contract():
    worker = ROOT / "web/coi-sw.js"
    assert worker.is_file()
    source = worker.read_text()
    # Escapes keep private identifiers out of the public release itself.
    forbidden = re.compile(r"\x6b\x74\x72\x61\x63\x65|\x6b\x74\x6f\x70", re.I)
    assert not forbidden.search(source)
    assert "Apache-2.0" in source
    for header in ("Cross-Origin-Opener-Policy", "Cross-Origin-Embedder-Policy",
                   "Cross-Origin-Resource-Policy"):
        assert header in source
    html = (ROOT / "web/index.html").read_text()
    assert "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'none'" in html
    assert '<input id="thorough" type="checkbox">' in html
    assert 'preload="auto" controls' in html


@pytest.mark.parametrize("name", ("train_det_quad", "train_ocr", "calibrate_policy"))
def test_synthetic_tools_help_without_models_or_writes(name):
    # Block heavy imports, artifact reads and filesystem mutations in the child.
    # Checking the real entry point also covers parsing before training imports.
    guard = r'''
import os, runpy, sys
def audit(event, args):
    if event == "import" and args[0].split(".")[0] in {
        "torch", "numpy", "cv2", "onnx", "onnxruntime", "vpocr_det", "vpocr_ocr", "vpocr_synth"
    }:
        raise AssertionError("Help imported a training or model dependency")
    if event == "open":
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            raise AssertionError("Help opened a file for writing")
        if isinstance(path, str) and path.endswith((".onnx", ".pt", ".json")):
            raise AssertionError("Help read a model or calibration artifact")
    if event in {"os.mkdir", "os.remove", "os.rename", "os.rmdir", "os.link", "os.symlink"}:
        raise AssertionError("Help changed the filesystem")
sys.addaudithook(audit)
sys.argv = [sys.argv[1], "--help"]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", guard, str(ROOT / "tools" / f"{name}.py")],
        cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=2,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not result.stderr


def load_calibration_tool():
    spec = importlib.util.spec_from_file_location("public_calibration_tool", ROOT / "tools/calibrate_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_audit_only_in_fresh_clone(tmp_path, monkeypatch, capsys):
    tool = load_calibration_tool()
    shutil.copytree(MODELS, tmp_path / "web/models")
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).digest()
              for p in (tmp_path / "web/models").iterdir()}
    assert tool.main(["--audit-only"]) == 0
    output = capsys.readouterr().out
    for name in cli.GRAPH_NAMES:
        assert f"web/models/{name}.int8.onnx:" in output
    expected = sum((MODELS / name).stat().st_size for name in ARTIFACTS)
    assert expected <= 15 * 1024**2
    assert round(expected / 1024**2, 2) == 13.72
    assert f"TOTAL: {expected} bytes" in output
    assert "15.000 MiB" in output
    assert "Synthetic MP4 smoke passed:" in output
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".calibration-smoke-*"))
    assert before == {p.name: hashlib.sha256(p.read_bytes()).digest()
                      for p in (tmp_path / "web/models").iterdir()}


def test_calibration_nested_layout_and_policy_validation(tmp_path, monkeypatch):
    tool = load_calibration_tool()
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    layout = ("onnx_temporal_det_nano", "onnx_temporal_det_nano", "onnx_det_v4",
              *("onnx_v2_nano",) * 4)
    for name, folder in zip(cli.GRAPH_NAMES, layout, strict=True):
        path = tmp_path / "out/models" / folder / f"{name}.int8.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for name in cli.CALIBRATION_NAMES[:2]:
        path = tmp_path / "out/calibration" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    graphs, sidecars = tool.deployment_paths(Path("out/models"))
    assert len(graphs) == 7 and len(sidecars) == 2
    assert graphs[3].parent == tmp_path / "out/models/onnx_v2_nano"
    assert sidecars[1] == tmp_path / "out/calibration/stage2_int8_cal.json"
    policy = json.loads((MODELS / "gen6.json").read_text())
    tool.validate_policy(policy)
    policy["session"]["risk_calibrated"] = True
    with pytest.raises(ValueError, match="terminal-only"):
        tool.validate_policy(policy)


def test_vendored_wasm_preserves_initialized_memory():
    """The literal-string repack must preserve the complete runtime data image."""
    path = ROOT / "web/vendor/ort-wasm-simd-threaded.wasm"
    raw = path.read_bytes()

    def read(data, position, signed=False):
        value = shift = 0
        while True:
            byte = data[position]
            position += 1
            value |= (byte & 127) << shift
            shift += 7
            if byte < 128:
                if signed and byte & 64:
                    value -= 1 << shift
                return value, position

    assert raw[:8] == b"\x00asm\x01\x00\x00\x00"
    sections = {}
    position = 8
    while position < len(raw):
        tag = raw[position]
        size, start = read(raw, position + 1)
        sections[tag] = raw[start:start + size]
        position = start + size
    assert position == len(raw)
    count, position = read(sections[11], 0)
    assert count == read(sections[12], 0)[0] == 133
    segments = []
    for _ in range(count):
        flag, position = read(sections[11], position)
        assert flag == 1
        size, position = read(sections[11], position)
        segments.append(sections[11][position:position + size])
        position += size
    assert position == len(sections[11])
    function_count, position = read(sections[10], 0)
    for index in range(function_count):
        size, position = read(sections[10], position)
        if index == 11926:
            initializer = sections[10][position:position + size]
        position += size
    memory = bytearray(1_078_380)
    copies = re.compile(
        rb"\x41([\x80-\xff]{0,4}[\x00-\x7f])\x41\x00\x41"
        rb"([\x80-\xff]{0,4}[\x00-\x7f])\xfc\x08([\x80-\xff]{0,4}[\x00-\x7f])\x00"
    )
    seen = []
    for match in copies.finditer(initializer):
        address = read(match[1], 0, signed=True)[0]
        size = read(match[2], 0, signed=True)[0]
        index = read(match[3], 0)[0]
        assert len(segments[index]) == size
        memory[address:address + size] = segments[index]
        seen.append(index)
    assert sorted(seen) == list(range(count))
    assert hashlib.sha256(memory).hexdigest() == "553fda883e23105b005126465d3159664cd7687ba32ac70888b0a77c67acf440"
    node = shutil.which("node")
    if node:
        result = subprocess.run(
            [node, "-e", "if (!WebAssembly.validate(require('fs').readFileSync(process.argv[1]))) process.exit(1)", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
