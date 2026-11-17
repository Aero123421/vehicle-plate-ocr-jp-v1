"""GPU-accelerated 2.5D Sim-to-Real sequence synthesis package (`vpocr_synth v2`)."""

from vpocr_synth.gpu.homography_fast import (
    REGIME_NAMES,
    REGIME_TO_ID,
    SENSOR_ROI_H,
    SENSOR_ROI_W,
    TrajectoryBatch,
    closed_form_rect_to_quad_coeffs,
    project_plate_ray_grid_roi,
    sample_se3_trajectories,
    warp_quad_to_canonical_gpu,
)
from vpocr_synth.gpu.scene_3d_gpu import (
    Scene3DClipConfig,
    Scene3DClipDataset,
    Scene3DClipSample,
    Scene3DFrameGT,
    Scene3DGPUEngine,
    Scene3DPlateGT,
    Scene3DVehicleGT,
)
from vpocr_synth.gpu.sensor_codec_gpu import SensorCodecGPU, SensorDegradedROIBatch
from vpocr_synth.gpu.sequence_engine import (
    GPUSynthIterableDataset,
    GPUSynthSequenceEngine,
    compute_geom_meta_gpu,
)
from vpocr_synth.gpu.shader_pbr_ir import ReliefPBRShaderGPU, ShadedROIBatch

__all__ = [
    "GPUSynthIterableDataset",
    "GPUSynthSequenceEngine",
    "REGIME_NAMES",
    "REGIME_TO_ID",
    "ReliefPBRShaderGPU",
    "SENSOR_ROI_H",
    "SENSOR_ROI_W",
    "Scene3DClipConfig",
    "Scene3DClipDataset",
    "Scene3DClipSample",
    "Scene3DFrameGT",
    "Scene3DGPUEngine",
    "Scene3DPlateGT",
    "Scene3DVehicleGT",
    "SensorCodecGPU",
    "SensorDegradedROIBatch",
    "ShadedROIBatch",
    "TrajectoryBatch",
    "closed_form_rect_to_quad_coeffs",
    "compute_geom_meta_gpu",
    "project_plate_ray_grid_roi",
    "sample_se3_trajectories",
    "warp_quad_to_canonical_gpu",
]
