"""Gen-6 measurement/search boundaries and the immutable Gen-5 rollback archive."""

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json
from pathlib import Path
import re
from typing import get_args

import numpy as np
import pytest

import vpocr_det
from vpocr_det import types
from vpocr_det.types import (
    Observation,
    ObservationActionMeta,
    ObservationProvenance,
    PlateDetection,
    SearchRequest,
    VehicleDetection,
    assert_not_observation,
)


MODELS = Path(__file__).resolve().parents[1] / "out" / "models"
BACKUP = MODELS / "backup_v050"
MODEL_FILES = (
    "temporal_plate_det_v4_base.pt",
    "temporal_plate_det_v4_nano.pt",
    "plate_quad_track_v4.pt",
    "mf_plate_ocr_v2_base.pt",
    "mf_plate_ocr_v2_nano.pt",
    "det_v4_phase4_provenance_manifest.json",
    "ocr_v4_phase1_provenance_manifest.json",
    "quad_track_v4_phase3_provenance_manifest.json",
)
ONNX_STEMS = {
    "onnx_det_v4": ("plate_quad_track_v4",),
    "onnx_temporal_det_base": ("temporal_det_base", "vehicle_cutout_base"),
    "onnx_temporal_det_nano": ("temporal_det_nano", "vehicle_cutout_nano"),
    "onnx_v2_base": ("frame_encoder", "memory_step", "memory_readout"),
    "onnx_v2_nano": ("frame_encoder", "memory_step", "memory_readout"),
}
ARCHIVE_FILES = (*MODEL_FILES, *(
    f"{directory}/{stem}{suffix}.onnx"
    for directory, stems in ONNX_STEMS.items()
    for stem in stems
    for suffix in ("", ".int8")
))


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@pytest.fixture(scope="module")
def backup_checksums():
    if not BACKUP.is_dir():
        pytest.skip("Private Gen-5 rollback archive is not distributed in the OSS release")
    manifest = BACKUP / "SHA256SUMS.txt"
    assert manifest.is_file(), "The Gen-5 rollback archive must include SHA256SUMS.txt"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    checksums = {}
    names = []
    for line in lines:
        assert re.fullmatch(r"[0-9a-f]{64}  .+", line), line
        checksum, name = line.split("  ", 1)
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert name not in checksums, f"Duplicate checksum entry: {name}"
        assert name != "SHA256SUMS.txt"
        checksums[name] = checksum
        names.append(name)
    assert names == sorted(names), "Keep the archive manifest deterministic"
    return checksums


def test_backup_manifest_covers_complete_release(backup_checksums):
    archived = {
        path.relative_to(BACKUP).as_posix()
        for path in BACKUP.rglob("*")
        if path.is_file() and path != BACKUP / "SHA256SUMS.txt"
    }
    assert set(backup_checksums) == archived
    assert set(ARCHIVE_FILES) <= archived
    for name in MODEL_FILES:
        if name.endswith(".json"):
            assert isinstance(json.loads((BACKUP / name).read_text(encoding="utf-8")), dict)


@pytest.mark.parametrize("name", ARCHIVE_FILES)
def test_backup_sha256_integrity(name, backup_checksums):
    archived = BACKUP / name
    assert archived.is_file() and not archived.is_symlink()
    assert archived.stat().st_size > 0
    assert _sha256(archived) == backup_checksums[name]
    # The rollback archive is pinned to its own release manifest. Live artifact
    # paths are reused by Gen-6 and can legitimately contain newer model bytes.


@pytest.mark.parametrize("name", (
    "Observation", "ObservationActionMeta", "ObservationProvenance",
    "SearchRequest", "assert_not_observation",
))
def test_new_contracts_exported_from_types_and_package(name):
    assert name in types.__all__
    assert name in vpocr_det.__all__
    assert getattr(vpocr_det, name) is getattr(types, name)


@pytest.mark.parametrize("participant,value", (
    (ObservationProvenance.DETECTOR_FULL, "detector_full"),
    (ObservationProvenance.DETECTOR_FAR_ZOOM, "detector_far_zoom"),
    (ObservationProvenance.SEARCH_MEASUREMENT, "search_measurement"),
    (ObservationProvenance.STANDALONE_PRECROP, "standalone_precrop"),
))
def test_provenance_is_explicit_and_serializable(participant, value):
    assert isinstance(participant, str)
    assert str(participant) == value
    assert ObservationProvenance(json.loads(json.dumps(participant))) is participant


def test_provenance_has_exactly_four_distinct_sampling_policies():
    assert len(ObservationProvenance) == 4
    with pytest.raises(ValueError):
        ObservationProvenance("0.38")


def test_no_executed_actions_by_default():
    action = ObservationActionMeta()
    assert action.action_id == ""
    assert action.detector_ran_full is False
    assert action.detector_ran_far_zoom is False
    assert action.searched_rois_xyxy == ()
    assert action.frame_key is None


@pytest.mark.parametrize("full,zoom", ((False, False), (False, True), (True, False), (True, True)))
def test_action_availability_keeps_independent_sampling_information(full, zoom):
    rois = ((1.0, 2.0, 40.0, 50.0), (80.0, 2.0, 100.0, 50.0))
    action = ObservationActionMeta("action-7", full, zoom, rois, "camera-2/frame-7")
    assert action.detector_ran_full is full
    assert action.detector_ran_far_zoom is zoom
    assert action.searched_rois_xyxy == rois
    assert action.frame_key == "camera-2/frame-7"
    with pytest.raises(FrozenInstanceError):
        action.detector_ran_full = not full


@pytest.fixture
def search_request():
    return SearchRequest(
        request_id="mount-search-1",
        roi_xyxy=(1.0, 2.0, 30.0, 40.0),
        target_entity_id=7,
        relation_hint="possible attachment",
        expected_cov_xy=(4.0, 9.0),
        priority_cost_gain=0.6,
    )


def test_search_request_is_score_free_and_disjoint_from_observations(search_request):
    assert assert_not_observation(search_request) is None
    assert not isinstance(search_request, Observation)
    assert set(get_args(Observation)) == {PlateDetection, VehicleDetection}
    assert {field.name for field in fields(search_request)} == {
        "request_id", "roi_xyxy", "target_entity_id", "relation_hint",
        "expected_cov_xy", "priority_cost_gain",
    }
    for name in ("score", "raw_det_score", "veh_log_odds", "plate_null_llr", "corners_xy"):
        assert not hasattr(search_request, name)
    with pytest.raises(FrozenInstanceError):
        search_request.priority_cost_gain = 0.9
    with pytest.raises((AttributeError, TypeError)):
        search_request.score = 0.99
    with pytest.raises(TypeError):
        replace(search_request, score=0.99)


def test_unbound_search_is_still_action_intent(search_request):
    unbound = replace(search_request, target_entity_id=None, relation_hint="unknown")
    assert unbound.target_entity_id is None
    assert assert_not_observation(unbound) is None


@pytest.mark.parametrize("score", (0.0, 0.38, 0.42, 0.99))
def test_runtime_guard_rejects_observations_regardless_of_legacy_score(score):
    observations = (
        VehicleDetection((0.0, 0.0, 100.0, 100.0), score),
        PlateDetection((1.0, 1.0, 10.0, 5.0), score, np.zeros((4, 2))),
    )
    for observation in observations:
        assert isinstance(observation, Observation)
        with pytest.raises(TypeError, match="SearchRequest"):
            assert_not_observation(observation)


@pytest.mark.parametrize("invalid", (None, {}, {"score": 0.38}, object()))
def test_runtime_guard_rejects_untyped_objects(invalid):
    with pytest.raises(TypeError, match="SearchRequest"):
        assert_not_observation(invalid)


@pytest.mark.parametrize("evidence_field", ("score", "raw_det_score", "veh_log_odds", "plate_null_llr"))
def test_runtime_guard_rejects_scored_search_subclasses(search_request, evidence_field):
    scored_request_type = type("ScoredSearch", (SearchRequest,), {evidence_field: 0.99})
    scored_request = scored_request_type(**{
        field.name: getattr(search_request, field.name) for field in fields(search_request)
    })
    with pytest.raises(TypeError, match="must not carry"):
        assert_not_observation(scored_request)


def test_legacy_vehicle_positional_construction_and_defaults():
    bbox = (1.0, 2.0, 100.0, 80.0)
    probs = np.ones(9, dtype=np.float32) / 9
    mask = np.ones((64, 64), dtype=np.float32)
    vehicle = VehicleDetection(bbox, 0.8, 3, probs, mask)
    assert vehicle.bbox_xyxy == bbox
    assert vehicle.score == 0.8 and vehicle.vehicle_subtype == 3
    assert vehicle.subtype_probs is probs and vehicle.cutout_mask_64 is mask
    assert vehicle.veh_log_odds is None
    assert vehicle.bbox_cov_4x4 is None
    assert vehicle.appearance_emb is None
    assert vehicle.subtype_unknown_prob == 0.0
    assert vehicle.provenance is ObservationProvenance.DETECTOR_FULL
    # The existing field must be extended in place, preserving positional callers.
    assert [field.name for field in fields(vehicle)].count("subtype_probs") == 1
    other = VehicleDetection(bbox, 0.7)
    vehicle.plate_track_ids.append(9)
    assert other.plate_track_ids == []


def test_legacy_plate_construction_and_defaults():
    corners = np.zeros((4, 2), dtype=np.float32)
    plate = PlateDetection((1.0, 2.0, 30.0, 12.0), 0.7, corners, True, 0.8,
                           track_id=7, raw_det_score=0.42, layout_prob=0.9)
    assert plate.corners_xy is corners
    assert plate.quad_refined is True and plate.quad_confidence == 0.8
    assert plate.track_id == 7 and plate.raw_det_score == 0.42
    assert plate.layout_prob == 0.9
    assert plate.plate_null_llr is None
    assert plate.corner_visibility_4 is None
    assert plate.homography_3x3 is None
    assert plate.support_nonconformity == 0.0
    assert plate.provenance is ObservationProvenance.DETECTOR_FULL
    assert plate.frame_key is None


def test_vehicle_first_principles_evidence_round_trips():
    covariance = np.eye(4, dtype=np.float32)
    appearance = np.ones(12, dtype=np.float32)
    subtype = np.array([0.2, 0.3], dtype=np.float32)
    vehicle = VehicleDetection(
        (0.0, 0.0, 100.0, 50.0), 0.8, veh_log_odds=-1.5,
        bbox_cov_4x4=covariance, appearance_emb=appearance, subtype_probs=subtype,
        subtype_unknown_prob=0.5, provenance=ObservationProvenance.DETECTOR_FAR_ZOOM,
    )
    assert vehicle.veh_log_odds == -1.5
    assert vehicle.bbox_cov_4x4 is covariance
    assert vehicle.appearance_emb is appearance
    assert vehicle.subtype_probs is subtype
    assert vehicle.subtype_unknown_prob == 0.5
    assert vehicle.provenance is ObservationProvenance.DETECTOR_FAR_ZOOM


@pytest.mark.parametrize("provenance", tuple(ObservationProvenance))
def test_fresh_plate_measurement_records_projective_and_null_evidence(provenance):
    visibility = np.array([1.0, 1.0, 0.0, 0.5], dtype=np.float32)
    homography = np.eye(3, dtype=np.float32)
    plate = PlateDetection(
        (0.0, 0.0, 10.0, 5.0), 0.8, np.zeros((4, 2)),
        plate_null_llr=-2.0, corner_visibility_4=visibility,
        homography_3x3=homography, support_nonconformity=0.12,
        provenance=provenance, frame_key="camera-2/frame-7",
    )
    assert plate.plate_null_llr == -2.0
    assert plate.corner_visibility_4 is visibility
    assert plate.homography_3x3 is homography
    assert plate.support_nonconformity == 0.12
    assert plate.provenance is provenance
    assert plate.frame_key == "camera-2/frame-7"
    # Recropping the same physical frame must retain its deduplication key.
    assert replace(plate, provenance=ObservationProvenance.SEARCH_MEASUREMENT).frame_key == plate.frame_key
