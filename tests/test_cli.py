"""Unit tests for CLI commands."""

from pathlib import Path
import pytest
from vpocr_synth.cli import cmd_charset, cmd_plates, cmd_preview, cmd_validate
import argparse


def test_cli_plates(tmp_path: Path):
    out_dir = tmp_path / "test_plates"
    args = argparse.Namespace(
        n=5,
        out=str(out_dir),
        seed=123,
        scale=1.0,
        format="png",
    )
    rc = cmd_plates(args)
    assert rc == 0
    assert (out_dir / "labels.jsonl").exists()
    assert (out_dir / "images" / "000000.png").exists()

    with open(out_dir / "labels.jsonl", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 5


def test_cli_preview(tmp_path: Path):
    out_file = tmp_path / "test_preview.png"
    args = argparse.Namespace(
        n=4,
        out=str(out_file),
        seed=42,
        cols=2,
        scale=1.0,
        with_boxes=True,
    )
    rc = cmd_preview(args)
    assert rc == 0
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_cli_charset(tmp_path: Path):
    out_file = tmp_path / "test_charset.txt"
    args = argparse.Namespace(out=str(out_file))
    rc = cmd_charset(args)
    assert rc == 0
    assert out_file.exists()
    with open(out_file, encoding="utf-8") as f:
        chars = f.read().splitlines()
    assert len(chars) > 200


def test_cli_validate():
    args_valid = argparse.Namespace(text="品川 599 さ 12-34")
    assert cmd_validate(args_valid) == 0

    args_invalid = argparse.Namespace(text="東京 500 あ 12-34")
    assert cmd_validate(args_invalid) == 1


def test_cli_sequences(tmp_path: Path):
    from vpocr_synth.cli import cmd_sequences
    out_dir = tmp_path / "test_seq"
    args = argparse.Namespace(
        n=2,
        out=str(out_dir),
        seed=10,
        preview_count=1,
    )
    rc = cmd_sequences(args)
    assert rc == 0
    assert (out_dir / "sequences.jsonl").exists()
    assert (out_dir / "preview_filmstrip.png").exists()
    assert (out_dir / "seq_000010" / "meta.json").exists()


def test_cli_degrade(tmp_path: Path):
    from vpocr_synth.cli import cmd_degrade, cmd_sequences
    seq_dir = tmp_path / "seq_in"
    args_seq = argparse.Namespace(
        n=1,
        out=str(seq_dir),
        seed=20,
        preview_count=1,
    )
    cmd_sequences(args_seq)

    cam_dir = tmp_path / "seq_cam"
    args_deg = argparse.Namespace(
        in_dir=str(seq_dir),
        out=str(cam_dir),
        profiles=1,
        seed=0,
    )
    # simulate the lambda wrapper
    setattr(args_deg, "in", str(seq_dir))
    rc = cmd_degrade(args_deg)
    assert rc == 0
    assert (cam_dir / "degraded_sequences.jsonl").exists()
    assert (cam_dir / "comparison_preview.png").exists()
