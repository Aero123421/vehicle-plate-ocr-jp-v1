"""Deterministically export the Python terminal grammar; never writes sealed files."""
from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import numpy as np
from vpocr_ocr.session_v2 import _terminal_tables
from vpocr_ocr.vocab import JPPlateGrammar, PlateVocab


def export(destination: Path) -> None:
    vocab = PlateVocab()
    regions, classes, kana, _, valid = _terminal_tables(JPPlateGrammar(vocab))

    def text(ids):
        return "".join(vocab.id_to_token[int(i)] for i in ids if int(i) != vocab.pad_id)

    tables = {
        "schema_version": 1,
        "tokens": [vocab.id_to_token[i] for i in range(len(vocab.id_to_token))],
        "pad_id": int(vocab.pad_id), "dot_id": int(vocab.dot_id),
        "plate_types": [p.value for p in vocab.plate_types],
        "regions": regions.tolist(), "region_names": [text(row) for row in regions],
        "classes": classes.tolist(), "class_strings": [text(row) for row in classes],
        "kana_ids": kana.tolist(), "kana_chars": [text([i]) for i in kana],
        "valid_shape": list(valid.shape),
        "valid_encoding": "zlib-packbits-little-c-order",
        "valid_base64": base64.b64encode(zlib.compress(np.packbits(valid.reshape(-1), bitorder="little").tobytes(), level=9)).decode("ascii"),
    }
    payload = json.dumps(tables, ensure_ascii=False, separators=(",", ":")) + "\n"
    assert len(tables["tokens"]) == 254 and len(tables["plate_types"]) == 5
    assert len(payload.encode()) < 150_000
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    export(Path(__file__).resolve().parents[1] / "web/models/grammar_tables.json")
