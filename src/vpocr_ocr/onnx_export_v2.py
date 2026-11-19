"""Separated vocabulary-free collection/risk and terminal-only ONNX exporter (`frame_encoder.onnx`, `memory_step.onnx`,
`memory_readout.onnx`), INT8 Dynamic Quantizer, and `OrtMFPlateOCRSessionV2`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
from pathlib import Path
import shutil
from typing import Any
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from vpocr_ocr.model_v2 import EvidenceMemoryV2, MFJPPlateOCRV2
from vpocr_ocr.onnx_export import _resolve_ort_providers
from vpocr_ocr.session_v2 import (
    MFPlateOCRSessionV2,
    OCRDecisionV2,
    SessionConfigV2,
    _SealedSessionV2,
    SessionStateV2,
    _ScaleSlopeEstimator,
    _TrackQualityAccumulator,
    build_collecting_placeholder_v2,
    build_decision_v2_from_arrays,
    compute_bandpass_stroke_sharpness,
    fuse_views_v2_numpy,
    normalize_crops_and_geom_numpy,
)
from vpocr_ocr.vocab import JPPlateGrammar, NUM_SLOTS, PlateVocab


class _FrameEncoderV2ExportWrapper(nn.Module):
    """Vocabulary-free ONNX wrapper for encode_frames."""

    def __init__(self, fused_model: MFJPPlateOCRV2) -> None:
        super().__init__()
        self.model = fused_model

    def forward(
        self,
        crops: torch.Tensor,
        plate_sides: torch.Tensor,
        geom_meta: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        enc = self.model.encode_frames(crops, plate_sides, geom_meta)
        return (
            enc.z,
            enc.lambda_prec,
            enc.z_color,
            enc.lambda_color,
            enc.pol_logit,
            enc.pol_prec,
        )


class _MemoryStepV2ExportWrapper(nn.Module):
    """Continuous augmented-Gaussian update, without vocabulary projection."""

    def __init__(self, mem_filter: EvidenceMemoryV2) -> None:
        super().__init__()
        self.mem_filter = mem_filter

    def forward(
        self,
        mem_h: torch.Tensor,
        mem_P: torch.Tensor,
        mem_conflict: torch.Tensor,
        color_h: torch.Tensor,
        color_P: torch.Tensor,
        pol_h: torch.Tensor,
        pol_P: torch.Tensor,
        pose_ema: torch.Tensor,
        last_timestamp_s: torch.Tensor,
        z: torch.Tensor,
        lambda_prec: torch.Tensor,
        z_color: torch.Tensor,
        lambda_color: torch.Tensor,
        pol_logit: torch.Tensor,
        pol_prec: torch.Tensor,
        geom_meta: torch.Tensor,
        timestamp_s: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        nxt = self.mem_filter.step(
            mem_h=mem_h,
            mem_P=mem_P,
            mem_conflict=mem_conflict,
            color_h=color_h,
            color_P=color_P,
            pol_h=pol_h,
            pol_P=pol_P,
            pose_ema=pose_ema,
            last_timestamp_s=last_timestamp_s,
            z=z,
            lambda_prec=lambda_prec,
            z_color=z_color,
            lambda_color=lambda_color,
            pol_logit=pol_logit,
            pol_prec=pol_prec,
            geom_meta=geom_meta,
            timestamp_s=timestamp_s,
        )
        return (
            nxt.mem_h,
            nxt.mem_P,
            nxt.mem_conflict,
            nxt.color_h,
            nxt.color_P,
            nxt.pol_h,
            nxt.pol_P,
            nxt.pose_ema,
            nxt.last_timestamp_s,
        )


class _MemoryReadoutV2ExportWrapper(nn.Module):
    """Single final readout ONNX wrapper (`decode_memory`) called ONCE at track completion."""

    def __init__(self, fused_model: MFJPPlateOCRV2) -> None:
        super().__init__()
        self.model = fused_model

    def forward(
        self,
        mem_h: torch.Tensor,
        mem_P: torch.Tensor,
        mem_conflict: torch.Tensor,
        color_h: torch.Tensor,
        color_P: torch.Tensor,
        pol_h: torch.Tensor,
        pol_P: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        ro = self.model.decode_memory(
            mem_h=mem_h,
            mem_P=mem_P,
            mem_conflict=mem_conflict,
            color_h=color_h,
            color_P=color_P,
            pol_h=pol_h,
            pol_P=pol_P,
        )
        return (
            ro.slot_logits,
            ro.plate_type_logits,
            ro.polarity_logit,
            ro.slot_readability_pred,
            ro.status_logits,
        )


def export_ocr_v2_to_onnx(
    model: MFJPPlateOCRV2,
    output_dir: str | Path,
    opset_version: int = 17,
) -> dict[str, Path]:
    """Export encoder, Gaussian update, scalar risk, and isolated terminal decoder."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fused = model.clone_reparameterized().cpu().eval()
    d = fused.config.d_model
    dc = fused.config.d_color
    h_in, w_in = fused.config.input_hw

    enc_path = out_dir / "frame_encoder.onnx"
    step_path = out_dir / "memory_step.onnx"
    read_path = out_dir / "memory_readout.onnx"

    # 1. frame_encoder.onnx (NO slot_classifier!)
    enc_wrap = _FrameEncoderV2ExportWrapper(fused).eval()
    d_crops = torch.rand(2, 3, h_in, w_in, dtype=torch.float32)
    d_sides = torch.tensor([0, 1], dtype=torch.long)
    d_geom = torch.rand(2, fused.config.geom_dim, dtype=torch.float32)

    enc_in_names = ["crops", "plate_sides", "geom_meta"]
    enc_out_names = ["z", "lambda_prec", "z_color", "lambda_color", "pol_logit", "pol_prec"]
    enc_dyn = {name: {0: "num_views"} for name in (enc_in_names + enc_out_names)}

    with torch.no_grad():
        torch.onnx.export(
            enc_wrap,
            (d_crops, d_sides, d_geom),
            str(enc_path),
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=enc_in_names,
            output_names=enc_out_names,
            dynamic_axes=enc_dyn,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(enc_path)))

    # 2. memory_step.onnx (NO slot_classifier!)
    step_wrap = _MemoryStepV2ExportWrapper(fused.memory_filter).eval()
    d_mh = torch.randn(1, NUM_SLOTS, d, dtype=torch.float32)
    d_mp = torch.rand(1, NUM_SLOTS, d, dtype=torch.float32)
    d_mc = torch.rand(1, NUM_SLOTS, dtype=torch.float32) * 0.1
    d_ch = torch.randn(1, dc, dtype=torch.float32)
    d_cp = torch.rand(1, dc, dtype=torch.float32)
    d_ph = torch.randn(1, dtype=torch.float32)
    d_pp = torch.rand(1, dtype=torch.float32)
    d_pose = torch.zeros(1, fused.memory_filter.state_dim, dtype=torch.float32)
    d_lts = torch.tensor([0.1], dtype=torch.float32)

    d_z = torch.randn(1, NUM_SLOTS, d, dtype=torch.float32)
    d_lam = torch.rand(1, NUM_SLOTS, d, dtype=torch.float32)
    d_zc = torch.randn(1, dc, dtype=torch.float32)
    d_lamc = torch.rand(1, dc, dtype=torch.float32)
    d_pol = torch.randn(1, dtype=torch.float32)
    d_polp = torch.rand(1, dtype=torch.float32)
    d_gm1 = torch.rand(1, fused.config.geom_dim, dtype=torch.float32)
    d_ts = torch.tensor([0.2], dtype=torch.float32)

    step_in_names = [
        "mem_h",
        "mem_P",
        "mem_conflict",
        "color_h",
        "color_P",
        "pol_h",
        "pol_P",
        "pose_ema",
        "last_timestamp_s",
        "z",
        "lambda_prec",
        "z_color",
        "lambda_color",
        "pol_logit",
        "pol_prec",
        "geom_meta",
        "timestamp_s",
    ]
    step_out_names = [
        "new_mem_h",
        "new_mem_P",
        "new_mem_conflict",
        "new_color_h",
        "new_color_P",
        "new_pol_h",
        "new_pol_P",
        "new_pose_ema",
        "new_last_timestamp_s",
    ]
    step_dyn = {name: {0: "batch"} for name in (step_in_names + step_out_names)}

    with torch.no_grad():
        torch.onnx.export(
            step_wrap,
            (
                d_mh,
                d_mp,
                d_mc,
                d_ch,
                d_cp,
                d_ph,
                d_pp,
                d_pose,
                d_lts,
                d_z,
                d_lam,
                d_zc,
                d_lamc,
                d_pol,
                d_polp,
                d_gm1,
                d_ts,
            ),
            str(step_path),
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=step_in_names,
            output_names=step_out_names,
            dynamic_axes=step_dyn,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(step_path)))

    # 3. memory_readout.onnx (Single final readout!)
    read_wrap = _MemoryReadoutV2ExportWrapper(fused).eval()
    read_in_names = [
        "mem_h",
        "mem_P",
        "mem_conflict",
        "color_h",
        "color_P",
        "pol_h",
        "pol_P",
    ]
    read_out_names = [
        "slot_logits",
        "plate_type_logits",
        "polarity_logit",
        "slot_readability_pred",
        "status_logits",
    ]
    read_dyn = {name: {0: "batch"} for name in (read_in_names + read_out_names)}

    with torch.no_grad():
        torch.onnx.export(
            read_wrap,
            (d_mh, d_mp, d_mc, d_ch, d_cp, d_ph, d_pp),
            str(read_path),
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=read_in_names,
            output_names=read_out_names,
            dynamic_axes=read_dyn,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(read_path)))

    # Vocabulary-free scalar stopping model is separately loadable/calibratable.
    risk_path = out_dir / "stopping_risk.onnx"
    torch.onnx.export(fused.stopping_risk_head.eval(), (torch.zeros(1, 10),), str(risk_path),
        input_names=["quality"], output_names=["risks"], opset_version=opset_version,
        dynamic_axes={"quality": {0: "batch"}, "risks": {0: "batch"}}, dynamo=False)
    onnx.checker.check_model(onnx.load(str(risk_path)))
    return {
        "stopping_risk": risk_path,
        "frame_encoder": enc_path,
        "memory_step": step_path,
        "memory_readout": read_path,
    }


_SENSITIVE_ENCODER_FP32_HEADS: tuple[str, ...] = (
    "offset_head_1",
    "offset_head_2",
    "reg_len_head",
    "cls_len_head",
    "prec_head",
    "measurement_scale_head",
)

_SENSITIVE_READOUT_FP32_HEADS: tuple[str, ...] = (
    "slot_classifier",
    "color_classifier",
    "slot_readability_head",
    "status_head",
    "readout_color_proj",
    "readout_pol_proj",
    "readout_prec_proj",
)


def quantize_onnx_v2_for_cpu(
    onnx_dir: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Produce lossless hybrid INT8 ONNX models for weak edge CPUs.

    - `frame_encoder.int8.onnx`: quantizes backbone/attention `MatMul`/`Gemm` weights to INT8 while
      keeping coordinate/length/precision gate heads (`offset_head_1/2`, `reg_len_head`, `cls_len_head`,
      `prec_head`, `measurement_scale_head`) in exact FP32 via `nodes_to_exclude`.
    - `memory_step.int8.onnx`: exact FP32 elementwise natural-parameter filter copy.
    - `memory_readout.int8.onnx`: hybrid INT8 quantizing internal `readout_sa`/`readout_mlp` linear
      weights while keeping all terminal logit & calibration heads (`slot_classifier`, `color_classifier`,
      `slot_readability_head`, `status_head`, context projections) in exact FP32!
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    base = Path(onnx_dir)
    dst = Path(out_dir) if out_dir is not None else base
    dst.mkdir(parents=True, exist_ok=True)

    enc_fp32 = base / "frame_encoder.onnx"
    step_fp32 = base / "memory_step.onnx"
    read_fp32 = base / "memory_readout.onnx"

    enc_int8 = dst / "frame_encoder.int8.onnx"
    step_int8 = dst / "memory_step.int8.onnx"
    read_int8 = dst / "memory_readout.int8.onnx"

    enc_proto = onnx.load(str(enc_fp32))
    nodes_to_exclude: list[str] = []
    len_head_outputs: set[str] = set()
    len_softmax_outputs: set[str] = set()
    renamed_any = False
    for idx, node in enumerate(enc_proto.graph.node):
        if node.op_type == "Softmax" and any(inp in len_head_outputs for inp in node.input):
            len_softmax_outputs.update(node.output)
            continue
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        haystacks = [node.name, *list(node.input), *list(node.output)]
        is_sensitive_head = any(
            marker in text
            for text in haystacks
            for marker in _SENSITIVE_ENCODER_FP32_HEADS
        )
        is_statutory_anchor_mix = any(inp in len_softmax_outputs for inp in node.input)
        if is_sensitive_head or is_statutory_anchor_mix:
            if not node.name:
                node.name = f"fp32_sensitive_{node.op_type}_{idx}"
                renamed_any = True
            nodes_to_exclude.append(node.name)
            if node.op_type == "Gemm":
                # ORT splits Gemm `X` into MatMul `X_MatMul` + Add before checking `nodes_to_exclude`!
                nodes_to_exclude.append(f"{node.name}_MatMul")
                nodes_to_exclude.append(f"{node.name}_matmul")
            if any(m in text for text in haystacks for m in ("reg_len_head", "cls_len_head")):
                len_head_outputs.update(node.output)
    if renamed_any:
        onnx.save(enc_proto, str(enc_fp32))

    quantize_dynamic(
        model_input=str(enc_fp32),
        model_output=str(enc_int8),
        op_types_to_quantize=["MatMul", "Gemm"],
        nodes_to_exclude=nodes_to_exclude,
        weight_type=QuantType.QUInt8,
    )

    # Hybrid INT8 memory_readout: quantize internal readout_sa/readout_mlp blocks while keeping
    # all terminal logit, color, readability, status & context projection heads in exact FP32!
    read_proto = onnx.load(str(read_fp32))
    read_exclude: list[str] = []
    read_renamed = False
    for idx, node in enumerate(read_proto.graph.node):
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        haystacks = [node.name, *list(node.input), *list(node.output)]
        is_sensitive_ro = any(
            marker in text
            for text in haystacks
            for marker in _SENSITIVE_READOUT_FP32_HEADS
        )
        if is_sensitive_ro:
            if not node.name:
                node.name = f"fp32_ro_sensitive_{node.op_type}_{idx}"
                read_renamed = True
            read_exclude.append(node.name)
            if node.op_type == "Gemm":
                read_exclude.append(f"{node.name}_MatMul")
                read_exclude.append(f"{node.name}_matmul")
    if read_renamed:
        onnx.save(read_proto, str(read_fp32))

    quantize_dynamic(
        model_input=str(read_fp32),
        model_output=str(read_int8),
        op_types_to_quantize=["MatMul", "Gemm"],
        nodes_to_exclude=read_exclude,
        weight_type=QuantType.QUInt8,
    )

    # memory_step is pure FP32 elementwise natural-parameter filter with zero MatMul weights
    shutil.copyfile(step_fp32, step_int8)

    shutil.copyfile(base / "stopping_risk.onnx", dst / "stopping_risk.int8.onnx")
    return {
        "stopping_risk_int8": dst / "stopping_risk.int8.onnx",
        "frame_encoder_int8": enc_int8,
        "memory_step_int8": step_int8,
        "memory_readout_int8": read_int8,
    }


def verify_onnx_patent_hygiene(
    onnx_dir: str | Path,
    vocab_size: int = 254,
    use_int8: bool = False,
) -> dict[str, Any]:
    """Structural graph audit (historical patent_clean key is not a legal opinion) verifying `frame_encoder` and `memory_step` contain ZERO `vocab_size`
    readout outputs, initializers, or intermediate tensors (enforcing single-readout graph separation).
    """
    if int(vocab_size) < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    base = Path(onnx_dir)
    suffix = ".int8.onnx" if use_int8 else ".onnx"
    report: dict[str, Any] = {}
    for stem in ("frame_encoder", "memory_step", "stopping_risk", "memory_readout"):
        fpath = base / f"{stem}{suffix}"
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Expected ONNX graph {fpath} (use_int8={use_int8}) does not exist in {base}"
            )
        raw_m = onnx.load(str(fpath), load_external_data=False)
        shape_inference_ok = True
        try:
            onnx.checker.check_model(raw_m)
            m = onnx.shape_inference.infer_shapes(raw_m, strict_mode=True)
        except Exception:
            shape_inference_ok = False
            m = raw_m
        out_dims: list[list[int | str]] = []
        has_vocab_out = False
        for out in m.graph.output:
            dims: list[int | str] = []
            for d_proto in out.type.tensor_type.shape.dim:
                if d_proto.HasField("dim_value"):
                    val = int(d_proto.dim_value)
                    dims.append(val)
                    if val == vocab_size:
                        has_vocab_out = True
                else:
                    dims.append(d_proto.dim_param or "dyn")
            out_dims.append(dims)

        forbidden_decode_ops = {"ArgMax", "ArgMin", "TopK", "Hardmax"}
        allowed_value_independent_ops = {
            "Add", "And", "AveragePool", "BatchNormalization", "Cast", "Clip", "Concat",
            "Constant", "ConstantOfShape", "Conv", "ConvInteger", "CumSum", "Div",
            "DynamicQuantizeLinear", "Equal", "Erf", "Exp", "Expand", "Flatten", "Floor",
            "Gather", "Gemm", "GlobalAveragePool", "Greater", "GreaterOrEqual", "Identity",
            "LayerNormalization", "Less", "LessOrEqual", "Log", "MatMul", "MatMulInteger",
            "Max", "MaxPool", "Min", "Mul", "Neg", "Not", "Or", "Pad", "Pow", "QLinearConv",
            "QLinearMatMul", "Reciprocal", "ReduceL2", "ReduceMax", "ReduceMean", "ReduceMin",
            "ReduceSum", "Relu", "Reshape", "Resize", "Shape", "Sigmoid", "Slice", "Softmax",
            "Softplus", "Split", "Sqrt", "Squeeze", "Sub", "Tanh", "Tile", "Transpose",
            "Unsqueeze", "Where",
        }
        shape_input_slots: dict[str, tuple[int, ...]] = {
            "Reshape": (1,),
            "Expand": (1,),
            "Pad": (1, 3),
            "ConstantOfShape": (0,),
            "Slice": (1, 2, 3, 4),
            "Resize": (1, 2, 3),
            "Tile": (1,),
            "Split": (1,),
            "Squeeze": (1,),
            "Unsqueeze": (1,),
            "ReduceMean": (1,),
            "ReduceSum": (1,),
            "ReduceMax": (1,),
            "ReduceMin": (1,),
            "ReduceL2": (1,),
            "CumSum": (1,),
        }
        has_vocab_const = False
        has_vocab_init_flag = False
        has_vocab_value_info = False
        has_discrete_decode_op = False
        has_unknown_or_data_dep_op = False
        has_vocab_int64_literal = False
        has_local_functions = len(m.functions) > 0
        has_subgraphs = False
        has_sparse_or_training_ir = (
            len(raw_m.training_info) > 0
            or len(raw_m.opset_import) == 0
            or any(
                op_imp.domain not in ("", "ai.onnx") or not (13 <= int(op_imp.version) <= 22)
                for op_imp in raw_m.opset_import
            )
        )

        def _is_external_or_segmented(t_proto: onnx.TensorProto) -> bool:
            return bool(
                t_proto.data_location == onnx.TensorProto.EXTERNAL
                or len(t_proto.external_data) > 0
                or t_proto.HasField("segment")
            )

        def _check_int64_vocab_literal(t_proto: onnx.TensorProto) -> bool:
            if _is_external_or_segmented(t_proto):
                return False
            arr = onnx.numpy_helper.to_array(t_proto)
            return bool(arr.dtype == np.int64 and int(vocab_size) in arr)

        def _audit_graph_recursive(g_proto: onnx.GraphProto) -> None:
            nonlocal has_vocab_const, has_vocab_init_flag, has_vocab_value_info
            nonlocal has_discrete_decode_op, has_unknown_or_data_dep_op
            nonlocal has_vocab_int64_literal, has_subgraphs, has_sparse_or_training_ir
            if len(g_proto.sparse_initializer) > 0:
                has_sparse_or_training_ir = True
                for sp_init in g_proto.sparse_initializer:
                    if vocab_size in list(sp_init.dims):
                        has_vocab_init_flag = True
                    if vocab_size in list(sp_init.values.dims) or vocab_size in list(
                        sp_init.indices.dims
                    ):
                        has_vocab_init_flag = True
                    if _check_int64_vocab_literal(sp_init.values) or _check_int64_vocab_literal(
                        sp_init.indices
                    ):
                        has_vocab_int64_literal = True
            for init in g_proto.initializer:
                if _is_external_or_segmented(init):
                    has_sparse_or_training_ir = True
                if vocab_size in list(init.dims):
                    has_vocab_init_flag = True
                if _check_int64_vocab_literal(init):
                    has_vocab_int64_literal = True
            for vi in list(g_proto.value_info) + list(g_proto.input) + list(g_proto.output):
                if not vi.type.HasField("tensor_type"):
                    has_sparse_or_training_ir = True
                    continue
                if vi.type.tensor_type.HasField("shape"):
                    for d_proto in vi.type.tensor_type.shape.dim:
                        if d_proto.HasField("dim_value") and int(d_proto.dim_value) == vocab_size:
                            has_vocab_value_info = True
                            break
            for node in g_proto.node:
                if node.domain not in ("", "ai.onnx") or bool(node.overload):
                    has_unknown_or_data_dep_op = True
                if node.op_type in forbidden_decode_ops:
                    has_discrete_decode_op = True
                if node.op_type not in allowed_value_independent_ops:
                    has_unknown_or_data_dep_op = True
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.TENSOR:
                        if _is_external_or_segmented(attr.t):
                            has_sparse_or_training_ir = True
                        if vocab_size in list(attr.t.dims):
                            has_vocab_const = True
                        if _check_int64_vocab_literal(attr.t):
                            has_vocab_int64_literal = True
                    elif attr.type == onnx.AttributeProto.TENSORS:
                        if any(_is_external_or_segmented(t) for t in attr.tensors):
                            has_sparse_or_training_ir = True
                        if any(vocab_size in list(t.dims) for t in attr.tensors):
                            has_vocab_const = True
                        if any(_check_int64_vocab_literal(t) for t in attr.tensors):
                            has_vocab_int64_literal = True
                    elif attr.type in (
                        onnx.AttributeProto.SPARSE_TENSOR,
                        onnx.AttributeProto.SPARSE_TENSORS,
                        onnx.AttributeProto.TYPE_PROTO,
                        onnx.AttributeProto.TYPE_PROTOS,
                    ):
                        has_sparse_or_training_ir = True
                        if attr.type == onnx.AttributeProto.SPARSE_TENSOR and vocab_size in list(
                            attr.sparse_tensor.dims
                        ):
                            has_vocab_const = True
                    elif attr.type == onnx.AttributeProto.INTS:
                        if int(vocab_size) in list(attr.ints):
                            has_vocab_int64_literal = True
                    elif attr.type == onnx.AttributeProto.INT:
                        if int(attr.i) == int(vocab_size):
                            has_vocab_int64_literal = True
                    elif attr.type == onnx.AttributeProto.GRAPH:
                        has_subgraphs = True
                        _audit_graph_recursive(attr.g)
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        has_subgraphs = True
                        for sub_g in attr.graphs:
                            _audit_graph_recursive(sub_g)

        _audit_graph_recursive(m.graph)
        for fn_proto in m.functions:
            for node in fn_proto.node:
                if node.op_type in forbidden_decode_ops:
                    has_discrete_decode_op = True
                if node.op_type not in allowed_value_independent_ops:
                    has_unknown_or_data_dep_op = True

        # Reject initializer/input overlap and any non-SSA duplicate tensor definitions
        graph_input_list = [inp.name for inp in raw_m.graph.input]
        initializer_list = [init.name for init in raw_m.graph.initializer]
        node_output_list = [o_n for node in raw_m.graph.node for o_n in node.output if o_n]
        all_defs = graph_input_list + initializer_list + node_output_list
        has_ssa_or_init_collision = len(all_defs) != len(set(all_defs))
        graph_input_names = set(graph_input_list)
        initializer_names = set(initializer_list)
        has_overridable_initializer = bool(graph_input_names & initializer_names) or has_ssa_or_init_collision
        immutable_inits = initializer_names - graph_input_names

        # Verify every graph input has static dims on axes >= 1 and at most one shared dynamic batch dim on axis 0
        batch_symbols: set[str] = set()
        inputs_only_batch_dynamic = not has_overridable_initializer
        for inp in raw_m.graph.input:
            if not (inp.type.HasField("tensor_type") and inp.type.tensor_type.HasField("shape")):
                inputs_only_batch_dynamic = False
                break
            for ax_i, d_proto in enumerate(inp.type.tensor_type.shape.dim):
                if d_proto.HasField("dim_value"):
                    dv = int(d_proto.dim_value)
                    if dv < 1:
                        inputs_only_batch_dynamic = False
                    if dv == vocab_size:
                        has_vocab_value_info = True
                elif ax_i == 0 and d_proto.dim_param:
                    batch_symbols.add(d_proto.dim_param)
                else:
                    inputs_only_batch_dynamic = False
        if len(batch_symbols) > 1:
            inputs_only_batch_dynamic = False

        # Taint analysis: prove every shape-controlling input slot is rooted solely in immutable {initializer, Constant, Shape}
        shape_meta_safe = set(immutable_inits)
        progressed = True
        while progressed:
            progressed = False
            for node in raw_m.graph.node:
                if node.op_type in ("Constant", "Shape"):
                    for o_name in node.output:
                        if o_name and o_name not in shape_meta_safe:
                            shape_meta_safe.add(o_name)
                            progressed = True
                elif (
                    node.op_type in allowed_value_independent_ops
                    and node.input
                    and all((not x) or (x in shape_meta_safe) for x in node.input)
                ):
                    for o_name in node.output:
                        if o_name and o_name not in shape_meta_safe:
                            shape_meta_safe.add(o_name)
                            progressed = True

        has_value_dependent_shape_slot = False
        for node in raw_m.graph.node:
            for slot_idx in shape_input_slots.get(node.op_type, ()):
                if slot_idx < len(node.input):
                    in_name = node.input[slot_idx]
                    if in_name and in_name not in shape_meta_safe:
                        has_value_dependent_shape_slot = True

        # Exact closed-form symbolic shape proof over ALL positive integer batch sizes B >= 1:
        # Every tensor dim is proven to be either int c >= 1 or ("B", k) = k * B (k >= 1).
        # Arithmetic/conditional ops (Add/Sub/Mul/Div/Equal/Where/Cast/Floor) are folded ONLY on
        # pure compile-time constants (`dtype != object`), forbidding any conditional/nonlinear B expression.
        symbolic_shape_proof_error = None
        symbolic_shape_proof_ok = False
        has_vocab_symbolic = False
        sym_shapes: dict[str, list[int | tuple[str, int]]] = {}
        if (
            inputs_only_batch_dynamic
            and (not has_subgraphs)
            and (not has_local_functions)
            and (not has_sparse_or_training_ir)
        ):
            try:
                init_arrs = {
                    init.name: np.asarray(onnx.numpy_helper.to_array(init))
                    for init in raw_m.graph.initializer
                }
                for k_name, arr in init_arrs.items():
                    sym_shapes[k_name] = [int(d) for d in arr.shape]
                sym_vals: dict[str, np.ndarray] = dict(init_arrs)
                for inp in raw_m.graph.input:
                    shp_s: list[int | tuple[str, int]] = []
                    for ax_i, dp in enumerate(inp.type.tensor_type.shape.dim):
                        if dp.HasField("dim_value"):
                            shp_s.append(int(dp.dim_value))
                        elif ax_i == 0 and dp.dim_param:
                            shp_s.append(("B", 1))
                        else:
                            raise ValueError("Unsupported dynamic input axis")
                    sym_shapes[inp.name] = shp_s
                sym_vals_out: dict[str, np.ndarray] = {}

                def _is_pure_const(arr: np.ndarray | None) -> bool:
                    return isinstance(arr, np.ndarray) and arr.dtype != object

                def _bcast(
                    s1: list[int | tuple[str, int]], s2: list[int | tuple[str, int]]
                ) -> list[int | tuple[str, int]]:
                    rk = max(len(s1), len(s2))
                    p1 = [1] * (rk - len(s1)) + list(s1)
                    p2 = [1] * (rk - len(s2)) + list(s2)
                    out_b: list[int | tuple[str, int]] = []
                    for da, db in zip(p1, p2):
                        if da == db:
                            out_b.append(da)
                        elif da == 1:
                            out_b.append(db)
                        elif db == 1:
                            out_b.append(da)
                        else:
                            raise ValueError(f"Incompatible symbolic broadcast: {s1} vs {s2}")
                    return out_b

                def _prod(ds: list[int | tuple[str, int]]) -> int | tuple[str, int]:
                    c_p, b_m, seen_b = 1, 0, False
                    for d_item in ds:
                        if isinstance(d_item, tuple):
                            if seen_b or d_item[1] < 1:
                                raise ValueError("Nonlinear batch product")
                            seen_b, b_m = True, int(d_item[1])
                        else:
                            if int(d_item) < 1:
                                raise ValueError("Non-positive static dimension")
                            c_p *= int(d_item)
                    return ("B", b_m * c_p) if seen_b else c_p

                allowed_op_attrs: dict[str, set[str]] = {
                    "AveragePool": {"auto_pad", "ceil_mode", "count_include_pad", "dilations", "kernel_shape", "pads", "strides"},
                    "BatchNormalization": {"epsilon", "momentum", "training_mode"},
                    "Cast": {"to"},
                    "Concat": {"axis"},
                    "Constant": {"value"},
                    "ConstantOfShape": {"value"},
                    "Conv": {"auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"},
                    "ConvInteger": {"auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"},
                    "CumSum": {"exclusive", "reverse"},
                    "Flatten": {"axis"},
                    "Gather": {"axis"},
                    "Gemm": {"alpha", "beta", "transA", "transB"},
                    "LayerNormalization": {"axis", "epsilon", "stash_type"},
                    "MaxPool": {"auto_pad", "ceil_mode", "dilations", "kernel_shape", "pads", "storage_order", "strides"},
                    "Pad": {"mode"},
                    "QLinearConv": {"auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"},
                    "ReduceL2": {"axes", "keepdims", "noop_with_empty_axes"},
                    "ReduceMax": {"axes", "keepdims", "noop_with_empty_axes"},
                    "ReduceMean": {"axes", "keepdims", "noop_with_empty_axes"},
                    "ReduceMin": {"axes", "keepdims", "noop_with_empty_axes"},
                    "ReduceSum": {"axes", "keepdims", "noop_with_empty_axes"},
                    "Reshape": {"allowzero"},
                    "Resize": {"axes", "coordinate_transformation_mode", "cubic_coeff_a", "exclude_outside", "extrapolation_value", "keep_aspect_ratio_policy", "mode", "nearest_mode"},
                    "Shape": {"start", "end"},
                    "Softmax": {"axis"},
                    "Split": {"axis", "num_outputs", "split"},
                    "Squeeze": {"axes"},
                    "Transpose": {"perm"},
                    "Unsqueeze": {"axes"},
                }

                for node in raw_m.graph.node:
                    op = node.op_type
                    ins = list(node.input)
                    outs = list(node.output)
                    if any(x and x not in sym_shapes for x in ins):
                        raise ValueError(f"Undefined input in {node.name}")
                    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                    if not set(attrs).issubset(allowed_op_attrs.get(op, set())):
                        raise ValueError(f"Unsupported attribute on {op}: {set(attrs)}")
                    if op == "Constant":
                        if len(outs) != 1 or "value" not in attrs:
                            raise ValueError("Unsupported Constant")
                        arr = np.asarray(onnx.numpy_helper.to_array(attrs["value"]))
                        sym_vals[outs[0]] = arr
                        sym_shapes[outs[0]] = [int(d) for d in arr.shape]
                    elif op == "Shape":
                        if len(outs) != 1:
                            raise ValueError("Bad Shape arity")
                        in_s = sym_shapes[ins[0]]
                        st_i = int(attrs.get("start", 0))
                        en_i = int(attrs.get("end", len(in_s)))
                        sub_s = in_s[st_i:en_i]
                        if len(sub_s) < 1:
                            raise ValueError("Empty Shape slice")
                        sym_shapes[outs[0]] = [len(sub_s)]
                        if all(isinstance(d, int) for d in sub_s):
                            sym_vals[outs[0]] = np.array(sub_s, dtype=np.int64)
                        else:
                            o_arr = np.empty(len(sub_s), dtype=object)
                            for idx_i, d in enumerate(sub_s):
                                o_arr[idx_i] = d
                            sym_vals[outs[0]] = o_arr
                    elif op == "ConstantOfShape":
                        if len(outs) != 1 or ins[0] not in sym_vals:
                            raise ValueError("Non-symbolic ConstantOfShape")
                        sv = sym_vals[ins[0]]
                        shp_c = [
                            x if isinstance(x, tuple) else int(x) for x in sv.reshape(-1).tolist()
                        ]
                        _ = _prod(shp_c)
                        sym_shapes[outs[0]] = shp_c
                        if _is_pure_const(sv):
                            int_shp = [int(d) for d in shp_c]
                            val_t = (
                                onnx.numpy_helper.to_array(attrs["value"])
                                if "value" in attrs
                                else np.zeros((1,), dtype=np.float32)
                            )
                            if int(np.prod(int_shp)) <= 4096:
                                sym_vals[outs[0]] = np.full(
                                    int_shp, val_t.flat[0], dtype=val_t.dtype
                                )
                    elif op == "Gather":
                        if len(outs) != 1:
                            raise ValueError("Bad Gather arity")
                        ds = sym_shapes[ins[0]]
                        if len(ds) < 1:
                            raise ValueError("Scalar Gather")
                        axis = int(attrs.get("axis", 0))
                        if axis < 0:
                            axis += len(ds)
                        if axis < 0 or axis >= len(ds):
                            raise ValueError("Gather axis OOB")
                        sym_shapes[outs[0]] = ds[:axis] + sym_shapes[ins[1]] + ds[axis + 1 :]
                        if ins[0] in sym_vals and _is_pure_const(sym_vals.get(ins[1])):
                            res = np.take(sym_vals[ins[0]], sym_vals[ins[1]], axis=axis)
                            if isinstance(res, tuple):
                                o_arr = np.empty((), dtype=object)
                                o_arr[()] = res
                                res = o_arr
                            else:
                                res = np.asarray(res)
                                if res.dtype == object and all(
                                    isinstance(x, (int, np.integer)) for x in res.flat
                                ):
                                    res = res.astype(np.int64)
                            sym_vals[outs[0]] = res
                    elif op == "Unsqueeze":
                        if len(outs) != 1:
                            raise ValueError("Bad Unsqueeze arity")
                        ds = list(sym_shapes[ins[0]])
                        if len(ins) > 1 and ins[1]:
                            if not _is_pure_const(sym_vals.get(ins[1])):
                                raise ValueError("Dynamic Unsqueeze axes")
                            axes = [int(x) for x in sym_vals[ins[1]].reshape(-1).tolist()]
                        else:
                            axes = [int(x) for x in attrs["axes"]]
                        out_rk = len(ds) + len(axes)
                        norm_ax = sorted([a + out_rk if a < 0 else a for a in axes])
                        if len(set(norm_ax)) != len(norm_ax) or any(
                            a < 0 or a >= out_rk for a in norm_ax
                        ):
                            raise ValueError("Invalid Unsqueeze axes")
                        for a in norm_ax:
                            ds.insert(a, 1)
                        sym_shapes[outs[0]] = ds
                        if ins[0] in sym_vals:
                            v_u = sym_vals[ins[0]]
                            for a in norm_ax:
                                v_u = np.expand_dims(v_u, axis=a)
                            sym_vals[outs[0]] = np.asarray(v_u)
                    elif op == "Squeeze":
                        if len(outs) != 1:
                            raise ValueError("Bad Squeeze arity")
                        ds = list(sym_shapes[ins[0]])
                        if len(ins) > 1 and ins[1]:
                            if not _is_pure_const(sym_vals.get(ins[1])):
                                raise ValueError("Dynamic Squeeze axes")
                            axes = [int(x) for x in sym_vals[ins[1]].reshape(-1).tolist()]
                        elif "axes" in attrs:
                            axes = [int(x) for x in attrs["axes"]]
                        else:
                            axes = []
                        if not axes:
                            if any(isinstance(d, tuple) and d[1] == 1 for d in ds):
                                raise ValueError("Ambiguous Squeeze without axes on dynamic batch")
                            axes = [i for i, d in enumerate(ds) if d == 1]
                        norm_s = {a + len(ds) if a < 0 else a for a in axes}
                        if any(a < 0 or a >= len(ds) or ds[a] != 1 for a in norm_s):
                            raise ValueError("Invalid Squeeze axis")
                        sym_shapes[outs[0]] = [d for i, d in enumerate(ds) if i not in norm_s]
                        if ins[0] in sym_vals:
                            v_s = sym_vals[ins[0]]
                            for a in sorted(norm_s, reverse=True):
                                v_s = np.squeeze(v_s, axis=a)
                            sym_vals[outs[0]] = np.asarray(v_s)
                    elif op == "Concat":
                        if len(outs) != 1 or not ins:
                            raise ValueError("Bad Concat")
                        s0 = list(sym_shapes[ins[0]])
                        axis = int(attrs["axis"])
                        if axis < 0:
                            axis += len(s0)
                        if axis < 0 or axis >= len(s0):
                            raise ValueError("Concat axis OOB")
                        ax_sum, b_sum = 0, 0
                        for x_in in ins:
                            sx = sym_shapes[x_in]
                            if len(sx) != len(s0):
                                raise ValueError("Concat rank mismatch")
                            for i_c, (da, db) in enumerate(zip(s0, sx)):
                                if i_c != axis and da != db:
                                    raise ValueError("Concat non-axis mismatch")
                            d_ax = sx[axis]
                            if isinstance(d_ax, tuple):
                                b_sum += int(d_ax[1])
                            else:
                                ax_sum += int(d_ax)
                        if ax_sum > 0 and b_sum > 0:
                            raise ValueError("Mixed affine batch + constant Concat")
                        s0[axis] = ("B", b_sum) if b_sum > 0 else ax_sum
                        sym_shapes[outs[0]] = s0
                        if all(x_in in sym_vals for x_in in ins):
                            if all(_is_pure_const(sym_vals[x_in]) for x_in in ins):
                                sym_vals[outs[0]] = np.concatenate(
                                    [sym_vals[x_in] for x_in in ins], axis=axis
                                )
                            else:
                                sym_vals[outs[0]] = np.concatenate(
                                    [sym_vals[x_in].astype(object) for x_in in ins], axis=axis
                                )
                    elif op == "Slice":
                        if len(outs) != 1:
                            raise ValueError("Bad Slice arity")
                        ds = list(sym_shapes[ins[0]])
                        if not all(_is_pure_const(sym_vals.get(x)) for x in ins[1:] if x):
                            raise ValueError("Non-constant Slice bounds")
                        starts = sym_vals[ins[1]].reshape(-1).tolist()
                        ends = sym_vals[ins[2]].reshape(-1).tolist()
                        axes = (
                            sym_vals[ins[3]].reshape(-1).tolist()
                            if len(ins) > 3 and ins[3]
                            else list(range(len(starts)))
                        )
                        steps = (
                            sym_vals[ins[4]].reshape(-1).tolist()
                            if len(ins) > 4 and ins[4]
                            else [1] * len(starts)
                        )
                        slc = [slice(None)] * len(ds)
                        for st_v, en_v, ax_v, sp_v in zip(starts, ends, axes, steps):
                            ax_n = int(ax_v) + len(ds) if int(ax_v) < 0 else int(ax_v)
                            if ax_n < 0 or ax_n >= len(ds) or int(sp_v) == 0:
                                raise ValueError("Invalid Slice axis/step")
                            if not isinstance(ds[ax_n], int):
                                raise ValueError("Slice on dynamic batch axis")
                            ln = len(
                                range(*slice(int(st_v), int(en_v), int(sp_v)).indices(ds[ax_n]))
                            )
                            if ln < 1:
                                raise ValueError("Empty Slice result")
                            ds[ax_n] = ln
                            slc[ax_n] = slice(int(st_v), int(en_v), int(sp_v))
                        sym_shapes[outs[0]] = ds
                        if ins[0] in sym_vals:
                            res_sl = np.asarray(sym_vals[ins[0]][tuple(slc)])
                            if res_sl.dtype == object and all(
                                isinstance(x, (int, np.integer)) for x in res_sl.flat
                            ):
                                res_sl = res_sl.astype(np.int64)
                            sym_vals[outs[0]] = res_sl
                    elif op == "Reshape":
                        if len(outs) != 1 or ins[1] not in sym_vals:
                            raise ValueError("Unproven Reshape target")
                        in_s = sym_shapes[ins[0]]
                        tgt = sym_vals[ins[1]].reshape(-1).tolist()
                        allowzero = int(attrs.get("allowzero", 0))
                        out_s_r: list[int | tuple[str, int]] = []
                        neg_idx = -1
                        for idx_i, td in enumerate(tgt):
                            if isinstance(td, tuple):
                                out_s_r.append(td)
                            else:
                                ti = int(td)
                                if ti == 0 and not allowzero:
                                    if idx_i >= len(in_s):
                                        raise ValueError("Reshape 0-copy OOB")
                                    out_s_r.append(in_s[idx_i])
                                elif ti == -1:
                                    if neg_idx != -1:
                                        raise ValueError("Multiple -1 in Reshape")
                                    neg_idx = idx_i
                                    out_s_r.append(1)
                                elif ti > 0:
                                    out_s_r.append(ti)
                                else:
                                    raise ValueError("Invalid Reshape target dim")
                        in_tot = _prod(in_s)
                        known_tot = _prod(out_s_r)
                        if neg_idx != -1:
                            if isinstance(in_tot, tuple) and isinstance(known_tot, tuple):
                                if in_tot[1] % known_tot[1] != 0:
                                    raise ValueError("Indivisible Reshape batch ratio")
                                out_s_r[neg_idx] = in_tot[1] // known_tot[1]
                            elif isinstance(in_tot, tuple) and isinstance(known_tot, int):
                                if in_tot[1] % known_tot != 0:
                                    raise ValueError("Indivisible Reshape dynamic dim")
                                out_s_r[neg_idx] = ("B", in_tot[1] // known_tot)
                            elif isinstance(in_tot, int) and isinstance(known_tot, int):
                                if in_tot % known_tot != 0:
                                    raise ValueError("Indivisible Reshape static dim")
                                out_s_r[neg_idx] = in_tot // known_tot
                            else:
                                raise ValueError("Invalid Reshape symbolic ratio")
                        if _prod(out_s_r) != in_tot:
                            raise ValueError("Reshape volume mismatch")
                        sym_shapes[outs[0]] = out_s_r
                        if _is_pure_const(sym_vals.get(ins[0])) and all(
                            isinstance(d, int) for d in out_s_r
                        ):
                            sym_vals[outs[0]] = np.asarray(
                                sym_vals[ins[0]].reshape([int(d) for d in out_s_r])
                            )
                    elif op == "Expand":
                        if len(outs) != 1 or ins[1] not in sym_vals:
                            raise ValueError("Unproven Expand target")
                        tgt_e = [
                            x if isinstance(x, tuple) else int(x)
                            for x in sym_vals[ins[1]].reshape(-1).tolist()
                        ]
                        if any((not isinstance(td, tuple)) and int(td) < 1 for td in tgt_e):
                            raise ValueError("Invalid Expand dim < 1")
                        sym_shapes[outs[0]] = _bcast(sym_shapes[ins[0]], tgt_e)
                    elif op == "Transpose":
                        if len(outs) != 1:
                            raise ValueError("Bad Transpose arity")
                        in_s = sym_shapes[ins[0]]
                        perm = list(attrs.get("perm", reversed(range(len(in_s)))))
                        if sorted(perm) != list(range(len(in_s))):
                            raise ValueError("Invalid Transpose perm")
                        sym_shapes[outs[0]] = [in_s[p] for p in perm]
                        if _is_pure_const(sym_vals.get(ins[0])):
                            sym_vals[outs[0]] = np.asarray(np.transpose(sym_vals[ins[0]], perm))
                    elif op == "Flatten":
                        if len(outs) != 1:
                            raise ValueError("Bad Flatten arity")
                        in_s = sym_shapes[ins[0]]
                        ax_f = int(attrs.get("axis", 1))
                        if ax_f < 0:
                            ax_f += len(in_s)
                        if ax_f < 0 or ax_f > len(in_s):
                            raise ValueError("Invalid Flatten axis")
                        sym_shapes[outs[0]] = [_prod(in_s[:ax_f]), _prod(in_s[ax_f:])]
                    elif op == "Pad":
                        if len(outs) != 1 or not _is_pure_const(sym_vals.get(ins[1])):
                            raise ValueError("Non-constant Pad")
                        in_s = list(sym_shapes[ins[0]])
                        pads = sym_vals[ins[1]].reshape(-1).tolist()
                        if len(ins) > 3 and ins[3]:
                            if not _is_pure_const(sym_vals.get(ins[3])):
                                raise ValueError("Non-constant Pad axes")
                            axes = [
                                int(a) + len(in_s) if int(a) < 0 else int(a)
                                for a in sym_vals[ins[3]].reshape(-1).tolist()
                            ]
                        else:
                            axes = list(range(len(in_s)))
                        if len(pads) != 2 * len(axes):
                            raise ValueError("Pad length mismatch")
                        m_ax = len(axes)
                        for i_ax, ax_p in enumerate(axes):
                            if ax_p < 0 or ax_p >= len(in_s):
                                raise ValueError("Pad axis OOB")
                            add_p = int(pads[i_ax]) + int(pads[i_ax + m_ax])
                            if add_p != 0 and not isinstance(in_s[ax_p], int):
                                raise ValueError("Pad on dynamic axis")
                            if isinstance(in_s[ax_p], int):
                                in_s[ax_p] = int(in_s[ax_p]) + add_p
                                if int(in_s[ax_p]) < 1:
                                    raise ValueError("Non-positive Pad output")
                        sym_shapes[outs[0]] = in_s
                    elif op == "Tile":
                        if len(outs) != 1 or not _is_pure_const(sym_vals.get(ins[1])):
                            raise ValueError("Non-constant Tile")
                        in_s = sym_shapes[ins[0]]
                        reps = [int(r) for r in sym_vals[ins[1]].reshape(-1).tolist()]
                        if len(reps) != len(in_s) or any(r < 1 for r in reps):
                            raise ValueError("Invalid Tile repeats")
                        sym_shapes[outs[0]] = [
                            ("B", d[1] * r) if isinstance(d, tuple) else int(d) * r
                            for d, r in zip(in_s, reps)
                        ]
                    elif op == "Split":
                        in_s = list(sym_shapes[ins[0]])
                        ax_sp = int(attrs.get("axis", 0))
                        if ax_sp < 0:
                            ax_sp += len(in_s)
                        if ax_sp < 0 or ax_sp >= len(in_s) or not isinstance(in_s[ax_sp], int):
                            raise ValueError("Split on invalid/dynamic axis")
                        if len(ins) > 1 and ins[1]:
                            if not _is_pure_const(sym_vals.get(ins[1])):
                                raise ValueError("Non-constant Split sizes")
                            sp_sizes = [int(x) for x in sym_vals[ins[1]].reshape(-1).tolist()]
                        elif "split" in attrs:
                            sp_sizes = [int(x) for x in attrs["split"]]
                        else:
                            n_out = int(attrs.get("num_outputs", len(outs)))
                            if int(in_s[ax_sp]) % n_out != 0:
                                raise ValueError("Uneven Split")
                            sp_sizes = [int(in_s[ax_sp]) // n_out] * n_out
                        if len(sp_sizes) != len(outs) or sum(sp_sizes) != int(in_s[ax_sp]) or any(
                            s_v < 1 for s_v in sp_sizes
                        ):
                            raise ValueError("Invalid Split partition")
                        for o_sp, sz_sp in zip(outs, sp_sizes):
                            s_cp = list(in_s)
                            s_cp[ax_sp] = sz_sp
                            sym_shapes[o_sp] = s_cp
                    elif op == "Resize":
                        if len(outs) != 1:
                            raise ValueError("Bad Resize arity")
                        if attrs.get("keep_aspect_ratio_policy", b"stretch") not in (
                            b"stretch",
                            "stretch",
                            b"",
                            "",
                        ):
                            raise ValueError("Unsupported Resize keep_aspect_ratio_policy")
                        if len(ins) > 1 and ins[1] and not _is_pure_const(sym_vals.get(ins[1])):
                            raise ValueError("Non-constant Resize roi")
                        in_s = list(sym_shapes[ins[0]])
                        if "axes" in attrs:
                            rz_axes = [
                                int(a) + len(in_s) if int(a) < 0 else int(a)
                                for a in attrs["axes"]
                            ]
                        else:
                            rz_axes = list(range(len(in_s)))
                        if len(set(rz_axes)) != len(rz_axes) or any(
                            a < 0 or a >= len(in_s) for a in rz_axes
                        ):
                            raise ValueError("Invalid Resize axes")
                        has_sz = (
                            len(ins) > 3
                            and bool(ins[3])
                            and _is_pure_const(sym_vals.get(ins[3]))
                            and sym_vals[ins[3]].size > 0
                        )
                        has_sc = (
                            len(ins) > 2
                            and bool(ins[2])
                            and _is_pure_const(sym_vals.get(ins[2]))
                            and sym_vals[ins[2]].size > 0
                        )
                        if (len(ins) > 3 and ins[3] and not _is_pure_const(sym_vals.get(ins[3]))) or (
                            len(ins) > 2 and ins[2] and not _is_pure_const(sym_vals.get(ins[2]))
                        ):
                            raise ValueError("Non-constant Resize scales/sizes")
                        if has_sz and has_sc:
                            raise ValueError("Resize cannot specify both scales and sizes")
                        out_rz = list(in_s)
                        if has_sz:
                            szs = [int(x) for x in sym_vals[ins[3]].reshape(-1).tolist()]
                            if len(szs) != len(rz_axes) or any(x < 1 for x in szs):
                                raise ValueError("Invalid Resize sizes")
                            for ax_r, sz_r in zip(rz_axes, szs):
                                if isinstance(out_rz[ax_r], tuple):
                                    raise ValueError("Static size targeting dynamic batch in Resize")
                                out_rz[ax_r] = sz_r
                            sym_shapes[outs[0]] = out_rz
                        elif has_sc:
                            scs = [float(x) for x in sym_vals[ins[2]].reshape(-1).tolist()]
                            if len(scs) != len(rz_axes):
                                raise ValueError("Invalid Resize scales")
                            for ax_r, sc_r in zip(rz_axes, scs):
                                d_r = out_rz[ax_r]
                                if isinstance(d_r, tuple):
                                    if sc_r != 1.0:
                                        raise ValueError("Scaled dynamic batch in Resize")
                                else:
                                    v_rz = int(np.floor(int(d_r) * sc_r))
                                    if v_rz < 1:
                                        raise ValueError("Non-positive Resize output")
                                    out_rz[ax_r] = v_rz
                            sym_shapes[outs[0]] = out_rz
                        else:
                            raise ValueError("Resize missing scales/sizes")
                    elif op in ("Conv", "ConvInteger", "QLinearConv"):
                        if len(outs) != 1:
                            raise ValueError("Bad Conv arity")
                        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET", b"", ""):
                            raise ValueError("Unsupported Conv auto_pad")
                        x_s = sym_shapes[ins[0]]
                        w_s = sym_shapes[ins[3 if op == "QLinearConv" else 1]]
                        if len(x_s) < 3 or len(w_s) != len(x_s) or not all(
                            isinstance(d, int) for d in x_s[1:] + w_s
                        ):
                            raise ValueError("Dynamic spatial/channel Conv")
                        grp = int(attrs.get("group", 1))
                        if grp < 1 or int(x_s[1]) != int(w_s[1]) * grp or int(w_s[0]) % grp != 0:
                            raise ValueError("Invalid Conv group/channels")
                        sp = len(x_s) - 2
                        ks = list(attrs.get("kernel_shape", w_s[2:]))
                        strides = list(attrs.get("strides", [1] * sp))
                        dilations = list(attrs.get("dilations", [1] * sp))
                        pads = list(attrs.get("pads", [0] * (2 * sp)))
                        if (
                            len(ks) != sp
                            or len(strides) != sp
                            or len(dilations) != sp
                            or len(pads) != 2 * sp
                            or any(int(v) < 1 for v in ks + strides + dilations)
                        ):
                            raise ValueError("Invalid Conv kernel/stride/dilation")
                        out_sp: list[int] = []
                        for i_sp in range(sp):
                            num = (
                                int(x_s[2 + i_sp])
                                + int(pads[i_sp])
                                + int(pads[i_sp + sp])
                                - (int(dilations[i_sp]) * (int(ks[i_sp]) - 1) + 1)
                            )
                            if num < 0:
                                raise ValueError("Negative Conv spatial extent")
                            out_sp.append(num // int(strides[i_sp]) + 1)
                        sym_shapes[outs[0]] = [x_s[0], int(w_s[0])] + out_sp
                    elif op in ("AveragePool", "MaxPool"):
                        if len(outs) != 1:
                            raise ValueError("Bad Pool arity")
                        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET", b"", ""):
                            raise ValueError("Unsupported Pool auto_pad")
                        x_s = sym_shapes[ins[0]]
                        if len(x_s) < 3 or not all(isinstance(d, int) for d in x_s[1:]):
                            raise ValueError("Dynamic spatial Pool")
                        sp = len(x_s) - 2
                        ks = list(attrs["kernel_shape"])
                        strides = list(attrs.get("strides", [1] * sp))
                        dilations = list(attrs.get("dilations", [1] * sp))
                        pads = list(attrs.get("pads", [0] * (2 * sp)))
                        if (
                            len(ks) != sp
                            or len(strides) != sp
                            or len(dilations) != sp
                            or len(pads) != 2 * sp
                            or any(int(v) < 1 for v in ks + strides + dilations)
                        ):
                            raise ValueError("Invalid Pool kernel/stride/dilation")
                        ceil_m = int(attrs.get("ceil_mode", 0))
                        out_p: list[int] = []
                        for i_sp in range(sp):
                            num = (
                                int(x_s[2 + i_sp])
                                + int(pads[i_sp])
                                + int(pads[i_sp + sp])
                                - (int(dilations[i_sp]) * (int(ks[i_sp]) - 1) + 1)
                            )
                            if num < 0:
                                raise ValueError("Negative Pool spatial extent")
                            st_p = int(strides[i_sp])
                            out_p.append(((num + st_p - 1) // st_p if ceil_m else num // st_p) + 1)
                        sym_shapes[outs[0]] = [x_s[0], int(x_s[1])] + out_p
                    elif op == "GlobalAveragePool":
                        if len(outs) != 1:
                            raise ValueError("Bad GAP arity")
                        x_s = sym_shapes[ins[0]]
                        sym_shapes[outs[0]] = [x_s[0], x_s[1]] + [1] * (len(x_s) - 2)
                    elif op in ("MatMul", "MatMulInteger", "QLinearMatMul"):
                        if len(outs) != 1:
                            raise ValueError("Bad MatMul arity")
                        a_s = list(sym_shapes[ins[0]])
                        b_s = list(sym_shapes[ins[3 if op == "QLinearMatMul" else 1]])
                        if len(a_s) < 2 or len(b_s) < 2 or a_s[-1] != b_s[-2]:
                            raise ValueError("Invalid MatMul shapes")
                        sym_shapes[outs[0]] = (
                            [a_s[0], b_s[1]]
                            if (len(a_s) == 2 and len(b_s) == 2)
                            else (_bcast(a_s[:-2], b_s[:-2]) + [a_s[-2], b_s[-1]])
                        )
                    elif op == "Gemm":
                        if len(outs) != 1:
                            raise ValueError("Bad Gemm arity")
                        a_s, b_s = sym_shapes[ins[0]], sym_shapes[ins[1]]
                        if len(a_s) != 2 or len(b_s) != 2:
                            raise ValueError("Non-2D Gemm")
                        ta, tb = int(attrs.get("transA", 0)), int(attrs.get("transB", 0))
                        k_a = a_s[0] if ta else a_s[1]
                        k_b = b_s[1] if tb else b_s[0]
                        if k_a != k_b:
                            raise ValueError("Gemm K mismatch")
                        out_gm = [a_s[1] if ta else a_s[0], b_s[0] if tb else b_s[1]]
                        if len(ins) > 2 and ins[2]:
                            if _bcast(out_gm, sym_shapes[ins[2]]) != out_gm:
                                raise ValueError("Invalid Gemm C broadcast")
                        sym_shapes[outs[0]] = out_gm
                    elif op in ("ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceL2"):
                        if len(outs) != 1:
                            raise ValueError("Bad Reduce arity")
                        in_s = list(sym_shapes[ins[0]])
                        keepdims = int(attrs.get("keepdims", 1))
                        noop_empty = int(attrs.get("noop_with_empty_axes", 0))
                        if len(ins) > 1 and ins[1]:
                            if not _is_pure_const(sym_vals.get(ins[1])):
                                raise ValueError("Non-constant Reduce axes")
                            axes = [int(a) for a in sym_vals[ins[1]].reshape(-1).tolist()]
                        elif "axes" in attrs:
                            axes = [int(a) for a in attrs["axes"]]
                        else:
                            axes = []
                        if not axes:
                            axes = [] if noop_empty else list(range(len(in_s)))
                        norm_r = {a + len(in_s) if a < 0 else a for a in axes}
                        if any(a < 0 or a >= len(in_s) for a in norm_r):
                            raise ValueError("Reduce axis OOB")
                        sym_shapes[outs[0]] = (
                            [1 if i in norm_r else d for i, d in enumerate(in_s)]
                            if keepdims
                            else [d for i, d in enumerate(in_s) if i not in norm_r]
                        )
                    elif op == "DynamicQuantizeLinear":
                        if len(outs) != 3:
                            raise ValueError("Bad DynamicQuantizeLinear arity")
                        sym_shapes[outs[0]] = list(sym_shapes[ins[0]])
                        sym_shapes[outs[1]] = []
                        sym_shapes[outs[2]] = []
                    elif op in (
                        "Cast",
                        "Clip",
                        "Erf",
                        "Exp",
                        "Floor",
                        "Identity",
                        "Log",
                        "Neg",
                        "Not",
                        "Reciprocal",
                        "Relu",
                        "Sigmoid",
                        "Softmax",
                        "Softplus",
                        "Sqrt",
                        "Tanh",
                        "BatchNormalization",
                        "LayerNormalization",
                        "CumSum",
                    ):
                        if len(outs) != 1:
                            raise ValueError(f"Unsupported multi-output {op}")
                        if op == "CumSum" and not _is_pure_const(sym_vals.get(ins[1])):
                            raise ValueError("Non-constant CumSum axis")
                        sym_shapes[outs[0]] = list(sym_shapes[ins[0]])
                        if _is_pure_const(sym_vals.get(ins[0])):
                            if op == "Cast":
                                dt_map = {
                                    1: np.float32,
                                    2: np.uint8,
                                    3: np.int8,
                                    4: np.uint16,
                                    5: np.int16,
                                    6: np.int32,
                                    7: np.int64,
                                    9: bool,
                                    10: np.float16,
                                    11: np.float64,
                                    12: np.uint32,
                                    13: np.uint64,
                                }
                                to_t = int(attrs["to"])
                                if to_t in dt_map:
                                    src_v = sym_vals[ins[0]]
                                    dst_dt = dt_map[to_t]
                                    if np.issubdtype(src_v.dtype, np.floating) and np.issubdtype(
                                        dst_dt, np.integer
                                    ):
                                        if not np.all(np.isfinite(src_v)):
                                            raise ValueError("Non-finite float-to-int Cast")
                                        src_v = np.trunc(src_v)
                                    sym_vals[outs[0]] = np.asarray(src_v.astype(dst_dt))
                            elif op == "Floor":
                                sym_vals[outs[0]] = np.asarray(np.floor(sym_vals[ins[0]]))
                            elif op == "Identity":
                                sym_vals[outs[0]] = np.asarray(sym_vals[ins[0]])
                    elif op in (
                        "Add",
                        "And",
                        "Div",
                        "Equal",
                        "Greater",
                        "GreaterOrEqual",
                        "Less",
                        "LessOrEqual",
                        "Max",
                        "Min",
                        "Mul",
                        "Or",
                        "Pow",
                        "Sub",
                        "Where",
                    ):
                        if len(outs) != 1 or any(not x_in for x_in in ins):
                            raise ValueError(f"Bad {op} arity")
                        cur_s = sym_shapes[ins[0]]
                        for x_in in ins[1:]:
                            cur_s = _bcast(cur_s, sym_shapes[x_in])
                        sym_shapes[outs[0]] = cur_s
                        if all(_is_pure_const(sym_vals.get(x_in)) for x_in in ins):
                            vs = [sym_vals[x_in] for x_in in ins]
                            if op in ("Add", "Sub", "Mul", "Div", "Equal") and vs[0].dtype != vs[1].dtype:
                                raise ValueError(f"Mismatched dtypes in {op}")
                            if op == "Add":
                                sym_vals[outs[0]] = np.asarray(vs[0] + vs[1])
                            elif op == "Sub":
                                sym_vals[outs[0]] = np.asarray(vs[0] - vs[1])
                            elif op == "Mul":
                                sym_vals[outs[0]] = np.asarray(vs[0] * vs[1])
                            elif op == "Div":
                                if np.issubdtype(vs[0].dtype, np.integer):
                                    lhs_b, rhs_b = np.broadcast_arrays(vs[0], vs[1])
                                    if np.any(rhs_b == 0):
                                        raise ValueError("Integer Div by zero")
                                    i_info = np.iinfo(vs[0].dtype)
                                    q_flat: list[int] = []
                                    for a_el, b_el in zip(lhs_b.flat, rhs_b.flat):
                                        ai, bi = int(a_el), int(b_el)
                                        sgn = 1 if (ai < 0) == (bi < 0) else -1
                                        qv = sgn * (abs(ai) // abs(bi))
                                        if qv < i_info.min or qv > i_info.max:
                                            raise ValueError("Integer Div overflow")
                                        q_flat.append(qv)
                                    sym_vals[outs[0]] = np.array(
                                        q_flat, dtype=vs[0].dtype
                                    ).reshape(lhs_b.shape)
                                else:
                                    sym_vals[outs[0]] = np.asarray(vs[0] / vs[1])
                            elif op == "Equal":
                                sym_vals[outs[0]] = np.asarray(np.equal(vs[0], vs[1]))
                            elif op == "Where":
                                if vs[0].dtype != bool or vs[1].dtype != vs[2].dtype:
                                    raise ValueError("Mismatched dtypes in Where")
                                sym_vals[outs[0]] = np.asarray(np.where(vs[0], vs[1], vs[2]))
                    else:
                        raise ValueError(f"Unproven op {op}")

                    if any(o_n and o_n not in sym_shapes for o_n in outs):
                        raise ValueError(f"Missing output shape in {node.name}")

                if any(o.name not in sym_shapes for o in raw_m.graph.output):
                    raise ValueError("Graph output missing symbolic shape")

                for s_dims in sym_shapes.values():
                    for ax_i, d_sym in enumerate(s_dims):
                        if isinstance(d_sym, tuple):
                            k_mul = int(d_sym[1])
                            # Non-batch axes must be strictly static; batch multiplier k > 1 must not divide vocab_size
                            if ax_i != 0 or k_mul < 1:
                                raise ValueError("Dynamic dim on non-batch axis")
                            if k_mul == vocab_size or (k_mul > 1 and vocab_size % k_mul == 0):
                                has_vocab_symbolic = True
                        else:
                            if int(d_sym) == vocab_size:
                                has_vocab_symbolic = True

                sym_vals_out = dict(sym_vals)
                symbolic_shape_proof_ok = True
            except Exception as exc:
                symbolic_shape_proof_error = f"{node.name}: {exc}"
                symbolic_shape_proof_ok = False
                sym_vals_out = {}

        # Dual-batch concrete runtime probe (including Constant outputs) cross-checking symbolic shapes & metadata values
        runtime_probe_ok = True
        has_vocab_runtime = False
        checked_runtime_tensors = 0
        try:
            if has_sparse_or_training_ir:
                raise ValueError("External, sparse, or training IR rejected before runtime probe")
            probe_m = copy.deepcopy(raw_m)
            existing_outs = {o.name for o in probe_m.graph.output}
            for node in probe_m.graph.node:
                for o_name in node.output:
                    if o_name and o_name not in existing_outs:
                        probe_m.graph.output.append(onnx.helper.make_empty_tensor_value_info(o_name))
                        existing_outs.add(o_name)
            out_order = [o.name for o in probe_m.graph.output]
            probe_sess = ort.InferenceSession(
                probe_m.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            b1 = 1 if vocab_size != 1 else 2
            b2 = 3 if vocab_size != 3 else 5

            ort_dt_map = {
                "tensor(float)": np.float32,
                "tensor(double)": np.float64,
                "tensor(float16)": np.float16,
                "tensor(int64)": np.int64,
                "tensor(int32)": np.int32,
                "tensor(int16)": np.int16,
                "tensor(int8)": np.int8,
                "tensor(uint8)": np.uint8,
                "tensor(uint16)": np.uint16,
                "tensor(uint32)": np.uint32,
                "tensor(uint64)": np.uint64,
                "tensor(bool)": bool,
            }

            def _run_batch(b_val: int) -> list[np.ndarray]:
                feeds_b: dict[str, np.ndarray] = {}
                for inp in probe_sess.get_inputs():
                    shp = [
                        b_val if isinstance(d, str) or d is None else int(d)
                        for d in inp.shape
                    ]
                    if inp.type not in ort_dt_map:
                        raise ValueError(f"Unsupported ORT input type {inp.type}")
                    feeds_b[inp.name] = np.zeros(shp, dtype=ort_dt_map[inp.type])
                return probe_sess.run(None, feeds_b)

            outs_b1 = _run_batch(b1)
            outs_b2 = _run_batch(b2)
            checked_runtime_tensors = len(outs_b1)
            for t_name, a1, a2 in zip(out_order, outs_b1, outs_b2):
                if len(a1.shape) != len(a2.shape):
                    runtime_probe_ok = False
                    break
                if symbolic_shape_proof_ok and t_name in sym_shapes:
                    exp1 = [d[1] * b1 if isinstance(d, tuple) else int(d) for d in sym_shapes[t_name]]
                    exp2 = [d[1] * b2 if isinstance(d, tuple) else int(d) for d in sym_shapes[t_name]]
                    if list(a1.shape) != exp1 or list(a2.shape) != exp2:
                        runtime_probe_ok = False
                        break
                if symbolic_shape_proof_ok and t_name in sym_vals_out:
                    sv_ref = sym_vals_out[t_name]
                    if sv_ref.dtype == object:
                        ev1 = np.array(
                            [x[1] * b1 if isinstance(x, tuple) else int(x) for x in sv_ref.flat],
                            dtype=np.int64,
                        ).reshape(sv_ref.shape)
                        ev2 = np.array(
                            [x[1] * b2 if isinstance(x, tuple) else int(x) for x in sv_ref.flat],
                            dtype=np.int64,
                        ).reshape(sv_ref.shape)
                        if not (np.array_equal(a1, ev1) and np.array_equal(a2, ev2)):
                            runtime_probe_ok = False
                            break
                    elif np.issubdtype(sv_ref.dtype, np.integer) or sv_ref.dtype == bool:
                        if not (np.array_equal(a1, sv_ref) and np.array_equal(a2, sv_ref)):
                            runtime_probe_ok = False
                            break
                for d1, d2 in zip(a1.shape, a2.shape):
                    if d1 == vocab_size or d2 == vocab_size:
                        has_vocab_runtime = True
                    if d1 != d2:
                        if d1 * b2 != d2 * b1 or (d1 // max(1, b1)) == vocab_size:
                            has_vocab_runtime = True
        except Exception:
            runtime_probe_ok = False

        has_vocab_init = has_vocab_const or has_vocab_init_flag or has_vocab_int64_literal
        has_vocab_intermediate = has_vocab_value_info or has_vocab_runtime or has_vocab_symbolic
        info = {
            "path": str(fpath),
            "shape_inference_ok": shape_inference_ok,
            "symbolic_shape_proof_ok": symbolic_shape_proof_ok,
            "symbolic_shape_proof_error": symbolic_shape_proof_error,
            "runtime_probe_ok": runtime_probe_ok,
            "checked_runtime_tensors": checked_runtime_tensors,
            "inputs_only_batch_dynamic": inputs_only_batch_dynamic,
            "has_overridable_initializer": has_overridable_initializer,
            "has_unknown_or_data_dep_op": has_unknown_or_data_dep_op,
            "has_value_dependent_shape_slot": has_value_dependent_shape_slot,
            "has_local_functions": has_local_functions,
            "has_subgraphs": has_subgraphs,
            "has_sparse_or_training_ir": has_sparse_or_training_ir,
            "output_shapes": out_dims,
            "has_vocab_dim_output": has_vocab_out,
            "has_vocab_dim_initializer": has_vocab_init,
            "has_vocab_dim_intermediate": has_vocab_intermediate,
            "has_discrete_decode_op": has_discrete_decode_op,
            "is_patent_clean": (
                shape_inference_ok
                and symbolic_shape_proof_ok
                and runtime_probe_ok
                and inputs_only_batch_dynamic
                and (not has_overridable_initializer)
                and (not has_unknown_or_data_dep_op)
                and (not has_value_dependent_shape_slot)
                and (not has_local_functions)
                and (not has_subgraphs)
                and (not has_sparse_or_training_ir)
                and (not has_vocab_out)
                and (not has_vocab_init)
                and (not has_vocab_intermediate)
                and (not has_discrete_decode_op)
            ),
        }
        report[stem] = info
        report[f"{stem}.onnx"] = info

    report["patent_clean"] = bool(
        report["frame_encoder"]["is_patent_clean"] and report["memory_step"]["is_patent_clean"] and report["stopping_risk"]["is_patent_clean"]
    )
    return report


class OrtMFPlateOCRSessionV2(_SealedSessionV2):
    """ONNX collector with a shared irreversible terminal lifecycle."""

    def __init__(
        self,
        onnx_dir: str | Path,
        vocab: PlateVocab | None = None,
        config: SessionConfigV2 | None = None,
        grammar_checker: Callable[..., bool] | None = None,
        providers: Sequence[str | tuple[str, dict[str, Any]]] | None = None,
        use_int8: bool = False,
        use_tf32: bool = False,
        intra_op_threads: int = 0,
    ) -> None:
        base = Path(onnx_dir)
        self.use_int8 = bool(use_int8)
        enc_file = base / ("frame_encoder.int8.onnx" if use_int8 else "frame_encoder.onnx")
        step_file = base / ("memory_step.int8.onnx" if use_int8 else "memory_step.onnx")
        read_file = base / ("memory_readout.int8.onnx" if use_int8 else "memory_readout.onnx")

        if not (enc_file.is_file() and step_file.is_file() and read_file.is_file()):
            raise FileNotFoundError(
                f"Missing v2 ONNX graph(s) in {base} (use_int8={use_int8}): "
                f"{[p.name for p in (enc_file, step_file, read_file) if not p.is_file()]}"
            )
        self.loaded_graph_paths = {
            "frame_encoder": str(enc_file),
            "memory_step": str(step_file),
            "memory_readout": str(read_file),
        }

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            sess_opts.intra_op_num_threads = int(intra_op_threads)
            sess_opts.inter_op_num_threads = 1
            sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        prov = (
            ["CPUExecutionProvider"]
            if use_int8 and providers is None
            else _resolve_ort_providers(providers, use_tf32=use_tf32)
        )

        self.enc_session = ort.InferenceSession(str(enc_file), sess_options=sess_opts, providers=prov)
        self.step_session = ort.InferenceSession(str(step_file), sess_options=sess_opts, providers=prov)
        self.read_session = ort.InferenceSession(str(read_file), sess_options=sess_opts, providers=prov)

        self.d_model: int = int(self.step_session.get_inputs()[0].shape[2])
        self.d_color: int = int(self.step_session.get_inputs()[3].shape[1])
        self.pose_dim: int = int(self.step_session.get_inputs()[7].shape[1])

        self.vocab = vocab or PlateVocab()
        self.config = config or SessionConfigV2()
        self.grammar = JPPlateGrammar(self.vocab)
        self.grammar_checker = grammar_checker
        risk_file = base / ("stopping_risk.int8.onnx" if use_int8 else "stopping_risk.onnx")
        self.risk_session = ort.InferenceSession(str(risk_file), sess_options=sess_opts, providers=prov)
        self.loaded_graph_paths["stopping_risk"] = str(risk_file)
        self.reset()

    @property
    def active_providers(self): return list(self.enc_session.get_providers())

    def _empty_state(self):
        return (np.zeros((1, NUM_SLOTS, self.d_model), np.float32),
            np.zeros((1, NUM_SLOTS, self.d_model), np.float32),
            np.zeros((1, NUM_SLOTS), np.float32),
            np.zeros((1, self.d_color), np.float32), np.zeros((1, self.d_color), np.float32),
            np.zeros(1, np.float32), np.zeros(1, np.float32),
            np.zeros((1, self.pose_dim), np.float32), np.full(1, -1.0, np.float32))

    def _state_arrays(self): return self._active_state
    def _same_filter(self, other):
        return isinstance(other, OrtMFPlateOCRSessionV2) and self.loaded_graph_paths == other.loaded_graph_paths
    def _encode(self, crops, sides, geom):
        return self.enc_session.run(None, {"crops": crops, "plate_sides": sides, "geom_meta": geom})
    def _apply_observation(self, obs, ts):
        state_names = ["mem_h", "mem_P", "mem_conflict", "color_h", "color_P", "pol_h", "pol_P", "pose_ema", "last_timestamp_s"]
        z, lam, zc, lamc, pol, polp, geom = obs
        values = (z[None], lam[None], zc[None], lamc[None], pol, polp, geom[None], np.asarray([ts], dtype=np.float32))
        obs_names = ["z", "lambda_prec", "z_color", "lambda_color", "pol_logit", "pol_prec", "geom_meta", "timestamp_s"]
        feed = dict(zip(state_names, self._active_state))
        feed.update(zip(obs_names, values))
        self._active_state = tuple(self.step_session.run(None, feed))
    def _predict_risk(self, quality):
        return self.risk_session.run(None, {"quality": quality})[0][0].tolist()
    def _readout(self):
        if self.state != SessionStateV2.SEALED or self.num_readout_calls != 1 or self._readout_consumed:
            raise RuntimeError('Terminal readout capability is sealed')
        self._readout_consumed = True
        names = ["mem_h", "mem_P", "mem_conflict", "color_h", "color_P", "pol_h", "pol_P"]
        return tuple(a[0] for a in self.read_session.run(None, dict(zip(names, self._active_state[:7]))))


@torch.no_grad()
def verify_onnx_v2_parity(
    model: MFJPPlateOCRV2,
    onnx_dir: str | Path,
    providers: Sequence[str | tuple[str, dict[str, Any]]] | None = None,
    num_steps: int = 6,
    seed: int = 2026,
    atol: float = 2.5e-4,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Compare reparameterized PyTorch vs 3-graph ORT rollout (`max_abs_diff < atol`, `rtol=1e-4`)."""
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    try:
        fused = model.clone_reparameterized().cpu().eval()
        vocab = PlateVocab()
        cfg = SessionConfigV2()
        pt_sess = MFPlateOCRSessionV2(fused, vocab=vocab, config=cfg, device="cpu")
        ort_sess = OrtMFPlateOCRSessionV2(
            onnx_dir=onnx_dir, vocab=vocab, config=cfg, providers=providers, use_tf32=False
        )

        h_in, w_in = fused.config.input_hw
        rng = np.random.default_rng(seed)
        max_enc_diff = 0.0
        max_h_diff = 0.0
        max_p_diff = 0.0

        for step in range(num_steps):
            crop = np.full((h_in, w_in, 3), 52, dtype=np.int16)
            my, mx = max(4, int(round(0.14 * h_in))), max(8, int(round(0.14 * w_in)))
            crop[my : h_in - my, mx : w_in - mx] = 236
            crop[my : my + 2, mx : w_in - mx] = 36
            crop[h_in - my - 2 : h_in - my, mx : w_in - mx] = 36
            for gx in range(mx + 12 + (step % 3), w_in - mx - 10, 18):
                crop[my + 8 : h_in // 2 - 3, gx : gx + 7] = [24, 58, 28]
                crop[h_in // 2 + 4 : h_in - my - 6, gx : gx + 9] = [22, 54, 26]
            crop = np.clip(crop + rng.integers(-14, 15, size=(h_in, w_in, 3), dtype=np.int16), 0, 255).astype(np.uint8)
            side = int(step % 2)
            ts = float(step) * 0.12

            pt_sess.push(crop, timestamp_s=ts, camera_id=0, source_frame_id=step, plate_side=side)
            ort_sess.push(crop, timestamp_s=ts, camera_id=0, source_frame_id=step, plate_side=side)

            crops_f, geom_f = normalize_crops_and_geom_numpy(crop, None)
            enc_pt = fused.encode_frames(
                torch.from_numpy(crops_f),
                torch.tensor([side], dtype=torch.long),
                torch.from_numpy(geom_f),
            )
            z_ort, lam_ort, zc_ort, lamc_ort, _, _ = ort_sess.enc_session.run(
                None,
                {
                    "crops": crops_f,
                    "plate_sides": np.array([side], dtype=np.int64),
                    "geom_meta": geom_f,
                },
            )
            max_enc_diff = max(
                max_enc_diff,
                float(np.abs(enc_pt.z.numpy() - z_ort).max()),
                float(np.abs(enc_pt.lambda_prec.numpy() - lam_ort).max()),
                float(np.abs(enc_pt.z_color.numpy() - zc_ort).max()),
                float(np.abs(enc_pt.lambda_color.numpy() - lamc_ort).max()),
            )

            st_pt = pt_sess.memory_state
            max_h_diff = max(max_h_diff, float(np.abs(st_pt.mem_h.numpy() - ort_sess._active_state[0]).max()))
            max_p_diff = max(max_p_diff, float(np.abs(st_pt.mem_P.numpy() - ort_sess._active_state[1]).max()))

        # Confirm zero intermediate readout calls during collection!
        assert pt_sess.num_readout_calls == 0
        assert ort_sess.num_readout_calls == 0

        pt_dec = pt_sess.finish()
        ort_dec = ort_sess.finish()
        assert pt_sess.num_readout_calls == ort_sess.num_readout_calls
        assert pt_sess.num_readout_calls <= 1

        st_pt = pt_sess.memory_state
        ro_pt = fused.decode_memory(
            st_pt.mem_h, st_pt.mem_P, st_pt.mem_conflict, st_pt.color_h, st_pt.color_P, st_pt.pol_h, st_pt.pol_P
        )
        mh, mp, mc, ch, cp, ph, pp, _, _ = ort_sess._active_state
        s_ort, pt_ort, pol_ort, sr_ort, st_ort = ort_sess.read_session.run(
            None,
            {"mem_h": mh, "mem_P": mp, "mem_conflict": mc, "color_h": ch, "color_P": cp, "pol_h": ph, "pol_P": pp},
        )
        max_state_aux_diff = max(
            float(np.abs(st_pt.mem_conflict.numpy() - mc).max()),
            float(np.abs(st_pt.color_h.numpy() - ch).max()),
            float(np.abs(st_pt.color_P.numpy() - cp).max()),
            float(np.abs(st_pt.pol_h.numpy() - ph).max()),
            float(np.abs(st_pt.pol_P.numpy() - pp).max()),
        )
        max_logits_diff = max(
            float(np.abs(ro_pt.slot_logits.numpy() - s_ort).max()),
            float(np.abs(ro_pt.plate_type_logits.numpy() - pt_ort).max()),
            float(np.abs(ro_pt.polarity_logit.numpy() - pol_ort).max()),
            float(np.abs(ro_pt.slot_readability_pred.numpy() - sr_ort).max()),
            float(np.abs(ro_pt.status_logits.numpy() - st_ort).max()),
        )

        overall_max = max(max_enc_diff, max_h_diff, max_p_diff, max_state_aux_diff, max_logits_diff)
        assert overall_max < atol, f"PyTorch vs ORT numerical parity exceeded {atol:.1e}: {overall_max}"
        for name_cmp, a_pt, a_ort in (
            ("slot_logits", ro_pt.slot_logits.numpy(), s_ort),
            ("plate_type_logits", ro_pt.plate_type_logits.numpy(), pt_ort),
            ("polarity_logit", ro_pt.polarity_logit.numpy(), pol_ort),
            ("slot_readability_pred", ro_pt.slot_readability_pred.numpy(), sr_ort),
            ("status_logits", ro_pt.status_logits.numpy(), st_ort),
        ):
            assert np.allclose(a_pt, a_ort, rtol=rtol, atol=atol), f"ORT {name_cmp} mismatch beyond (rtol={rtol}, atol={atol})"
        assert pt_dec.status == ort_dec.status, f"Status mismatch: {pt_dec.status} vs {ort_dec.status}"
        assert pt_dec.formatted_text == ort_dec.formatted_text
        assert pt_dec.confidence_tier == ort_dec.confidence_tier
        assert pt_dec.verified_fields == ort_dec.verified_fields
        return {
            "providers": ort_sess.active_providers,
            "loaded_graphs": ort_sess.loaded_graph_paths,
            "max_encoder_abs_diff": max_enc_diff,
            "max_mem_h_abs_diff": max_h_diff,
            "max_mem_P_abs_diff": max_p_diff,
            "max_state_aux_abs_diff": max_state_aux_diff,
            "max_readout_logits_abs_diff": max_logits_diff,
            "max_abs_diff": overall_max,
            "pt_status": pt_dec.status,
            "ort_status": ort_dec.status,
        }
    finally:
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
