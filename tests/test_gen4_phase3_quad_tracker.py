"""Gen-6 tracker contracts, probability checks, CMC and corner-filter regressions.

Neural detector/quad training and pipeline integration have their own suites.
These tests exercise fresh observations and action plans as separate objects.
"""
from __future__ import annotations

from dataclasses import replace
import inspect
import itertools
import math

import cv2
import numpy as np
import pytest
import torch

import vpocr_det.tracker as tracker_module
from vpocr_det.quad_cropper import select_containment_crop_window
from vpocr_det.tracker import (
    BackgroundCameraMotionEstimator, CameraMotionEstimate, CoupledPMBMTracker,
    MultiVehicleKalmanTracker, PlateTrackStabilizer, TrackerCalibration,
    _B_HADAMARD, _B_INV_HADAMARD, _H_24, _compensate_plate,
    _vmf_log_ratio, detection_availability, feasible_assignment_posterior,
    missed_existence_probability,
)
from vpocr_det.types import (
    ObservationActionMeta, ObservationProvenance, PlateDetection,
    SearchRequest, VehicleDetection, assert_not_observation,
)


NO_LOOK = ObservationActionMeta()
FULL = ObservationActionMeta(detector_ran_full=True)


def _make_rect_quad(cx: float, cy: float, w: float = 64, h: float = 32) -> np.ndarray:
    return np.array([[cx-w/2,cy-h/2],[cx+w/2,cy-h/2],
                     [cx+w/2,cy+h/2],[cx-w/2,cy+h/2]],np.float32)


def _make_det(cx: float = 200, cy: float = 160, w: float = 64, h: float = 32,
              llr: float = 8, frame_key: str | None = None,
              embedding: np.ndarray | None = None) -> PlateDetection:
    q = _make_rect_quad(cx,cy,w,h)
    return PlateDetection(tuple(np.r_[q.min(0),q.max(0)]),.9,q,
                          quad_refined=True,quad_confidence=.95,plate_null_llr=llr,
                          corner_cov_xy=np.tile(np.eye(2)[None],(4,1,1)),
                          reid_emb=embedding,frame_key=frame_key,
                          topology_prob=.95)


def _make_vehicle(cx: float = 200, cy: float = 140, llr: float = 8) -> VehicleDetection:
    return VehicleDetection((cx-80,cy-60,cx+80,cy+60),.9,veh_log_odds=llr,
                            bbox_cov_4x4=np.eye(4)*2)


@pytest.mark.parametrize('vehicle',[False,True])
def test_confirmation_cost_suppresses_single_frame_public_birth(vehicle: bool) -> None:
    tracker = MultiVehicleKalmanTracker() if vehicle else PlateTrackStabilizer()
    observation = _make_vehicle() if vehicle else _make_det()
    first = tracker.step([observation],0,action_meta=FULL)[0]
    assert (first.vehicle_track_id if vehicle else first.track_id) is None
    t = tracker._tracks[0]
    c = tracker.calibration
    e = c.birth_odds*math.exp(8)
    assert t.r == pytest.approx(e/(e+1+c.persistent_birth_ratio*e))
    assert c.c_false*(1-t.r) >= c.c_wait*t.r
    second = tracker.step([observation],.05,action_meta=FULL)[0]
    assert (second.vehicle_track_id if vehicle else second.track_id) == 1
    assert c.c_false*(1-t.r) < c.c_wait*t.r


@pytest.mark.parametrize('vehicle',[False,True])
def test_finite_horizon_cost_confirms_supported_birth_on_frame_one(vehicle: bool) -> None:
    calibration = TrackerCalibration(c_false=1.0, c_wait=1.0,
                                     persistent_birth_ratio=.05, birth_odds=1.0)
    tracker = (MultiVehicleKalmanTracker if vehicle else PlateTrackStabilizer)(calibration=calibration)
    observation = _make_vehicle(llr=2) if vehicle else _make_det(llr=2)
    first = tracker.step([observation],0,action_meta=FULL)[0]
    assert (first.vehicle_track_id if vehicle else first.track_id) == 1
    assert tracker._tracks[0].confirmed
    assert tracker._tracks[0].r > .5


@pytest.mark.parametrize('vehicle',[False,True])
def test_not_looking_preserves_predicted_existence_and_explicit_miss_formula(vehicle: bool) -> None:
    make = _make_vehicle if vehicle else _make_det
    tracker = MultiVehicleKalmanTracker() if vehicle else PlateTrackStabilizer()
    tracker.step([make()],0)
    tracker.step([make()],.05)
    t = tracker._tracks[0]
    before = t.r
    tracker.step([],.15,action_meta=NO_LOOK)
    predicted = before*math.exp(-.1/tracker.calibration.survival_tau_s)
    assert t.r == pytest.approx(predicted)
    # The executed ROI is elsewhere; far zoom without target coverage is not a miss.
    outside = ObservationActionMeta(detector_ran_far_zoom=True,
                                    searched_rois_xyxy=((0,0,10,10),))
    tracker.step([],.25,action_meta=outside)
    assert t.r == pytest.approx(predicted*math.exp(-.1/tracker.calibration.survival_tau_s))
    before = t.r
    tracker.step([],.3,action_meta=FULL)
    predicted = before*math.exp(-.05/tracker.calibration.survival_tau_s)
    assert t.r == pytest.approx(predicted*.1/(1-predicted*.9))
    assert missed_existence_probability(.37,0) == .37


def test_detection_availability_conditions_on_executed_support() -> None:
    box = (20,30,40,50)
    assert detection_availability(box,NO_LOOK) == 0
    assert detection_availability(box,ObservationActionMeta(detector_ran_far_zoom=True)) == 0
    assert detection_availability(box,ObservationActionMeta(searched_rois_xyxy=((19,29,41,51),))) == .9
    assert detection_availability(box,ObservationActionMeta(searched_rois_xyxy=((25,35,45,55),))) == 0


def test_delivered_roi_measurement_associates_after_camera_compensation(monkeypatch) -> None:
    tracker = PlateTrackStabilizer()
    for ts in [0,.05]:
        tracker.step([_make_det()],ts,action_meta=FULL)
    state = tracker._tracks[0]
    observation = _make_det()
    action = ObservationActionMeta(searched_rois_xyxy=(observation.bbox_xyxy,))
    motion = CameraMotionEstimate(np.array([[1.,0.,1.],[0.,1.,0.]]),
                                  np.eye(6)*1e-6,np.eye(2)*.01)
    availability = []
    likelihood = tracker._pair_likelihood
    def measured_likelihood(t,z,ts):
        availability.append(detection_availability(t.bbox_xyxy,action))
        return likelihood(t,z,ts)
    monkeypatch.setattr(tracker,'_pair_likelihood',measured_likelihood)
    detected = state.detected
    pair_availability = []
    def measured_detected(z,ts,association_probability,p_D):
        pair_availability.append(p_D)
        return detected(z,ts,association_probability,p_D)
    monkeypatch.setattr(state,'detected',measured_detected)
    result = tracker.step([observation],.1,action_meta=action,camera_motion=motion)
    assert availability == [0.0]
    assert pair_availability == [tracker.calibration.detection_probability]
    assert result[0].track_id == 1
    assert len(tracker._tracks) == 1 and state.hits == 3


def test_survival_uses_elapsed_time_and_ignores_legacy_missed_frame_limit() -> None:
    calibration = TrackerCalibration(survival_tau_s=2)
    trackers = [PlateTrackStabilizer(calibration=calibration,max_missed=0) for _ in range(2)]
    for t in trackers:
        t.step([_make_det()],0)
        t.step([_make_det()],.05)
    for ts in np.linspace(.06,1.05,100):
        trackers[0].step([],float(ts),action_meta=NO_LOOK)
    trackers[1].step([],1.05,action_meta=NO_LOOK)
    a = trackers[0]._dormant_tracks[0]
    b = trackers[1]._dormant_tracks[0]
    assert a.r == pytest.approx(b.r)
    assert a.missed == 100 and b.missed == 1
    assert not trackers[0].last_expired_track_ids


def test_birth_policy_prior_and_search_measurement_can_birth_without_parent() -> None:
    c = TrackerCalibration(policy_birth_odds={'search_measurement':.2})
    t = PlateTrackStabilizer(calibration=c)
    z = replace(_make_det(llr=2),provenance=ObservationProvenance.SEARCH_MEASUREMENT)
    t.step([z],0,action_meta=ObservationActionMeta(searched_rois_xyxy=((0,0,400,300),)))
    e = .2*math.exp(2)
    assert t._tracks[0].r == pytest.approx(e/(1+1.25*e))
    assert t._tracks[0].parent_vehicle_track_id is None


def test_persistent_nonplate_competes_for_one_hundred_identical_views() -> None:
    t = PlateTrackStabilizer()
    emb = np.eye(1,32,3,dtype=np.float32)[0]
    for k in range(100):
        output = t.step([_make_det(llr=.4,frame_key=f'camera:{k}',embedding=emb)],k*.05)
        assert all(z.track_id is None for z in output)
    assert len(t._tracks) == 1
    state = t._tracks[0]
    assert state.hits == 100
    assert state.class_log_odds == pytest.approx(.4)
    assert state.r <= 1/(1+math.exp(-.4))
    assert state.persistent_nonplate_probability > .4
    assert not t.active_track_ids


def test_near_duplicate_innovation_damps_marks_and_repeated_frame_key() -> None:
    a = np.array([1.,0.,0.])
    b = np.array([1.,.001,0.]); b /= np.linalg.norm(b)
    t = PlateTrackStabilizer()
    t.step([_make_det(llr=.4,frame_key='cam:1',embedding=a)],0)
    state = t._tracks[0]
    factor = state.innovation(_make_det(llr=5,frame_key='cam:2',embedding=b),.05)
    assert 0 <= factor < 1e-5
    assert state.innovation(_make_det(frame_key='cam:1',embedding=-a),10) == 0
    t.step([_make_det(llr=.4,frame_key='cam:1',embedding=a)],.05,action_meta=NO_LOOK)
    assert state.hits == 1


def test_normalized_vmf_likelihood_has_no_free_cosine_nats_intercept() -> None:
    # In S^2, E_uniform exp(k*cos(theta)) = sinh(k)/k exactly.
    k = 3.
    a,b = np.array([1.,0,0]),np.array([0.,1.,0])
    assert _vmf_log_ratio(a,b,k) == pytest.approx(-math.log(math.sinh(k)/k))
    assert _vmf_log_ratio(a,a,k)-_vmf_log_ratio(a,b,k) == pytest.approx(k)
    assert _vmf_log_ratio(None,a,k) == 0


def test_hadamard_timestamp_guards_and_irregular_fps_spd() -> None:
    np.testing.assert_allclose(_B_HADAMARD@_B_INV_HADAMARD,np.eye(4))
    q = np.array([[120,90],[180,92],[185,124],[118,120]],float)
    np.testing.assert_allclose(_B_HADAMARD@(_B_INV_HADAMARD@q),q)
    t = PlateTrackStabilizer()
    assert t.step([_make_det()],1)[0].track_id is None
    state = t._tracks[0]
    x,P = state.x.copy(),state.P.copy()
    for ts in [1,.9]:
        t.step([_make_det(240)],ts)
        assert t.last_stale_rejected
        np.testing.assert_array_equal(state.x,x)
        np.testing.assert_array_equal(state.P,P)
    ts = 1.
    for dt in [.033,.12,.025,.085,.04,.11,.03,.05]:
        ts += dt
        cx = 200+100*(ts-1)
        output = t.step([_make_det(cx)],ts)
        assert len(output) == 1 and output[0].track_id == 1
        assert np.linalg.eigvalsh(state.P).min() > 0
        assert abs(state.cx-cx) < 5


def test_geometry_reference_is_posterior_span_without_reference_ema() -> None:
    t = PlateTrackStabilizer()
    t.step([_make_det()],0)
    t.step([_make_det(w=70,h=36)],.05)
    state = t._tracks[0]
    assert state.ref_w == pytest.approx(state._quad_span_wh((_H_24@state.x).reshape(4,2))[0])
    assert state.ref_h == pytest.approx(state._quad_span_wh((_H_24@state.x).reshape(4,2))[1])


def test_no_parent_numeric_penalty_side_or_size_veto_and_zero_character_coupling() -> None:
    t = PlateTrackStabilizer()
    t.step([_make_det()],0)
    d = t._tracks[0]
    z = replace(_make_det(),parent_vehicle_track_id=11)
    candidate = t._new_track(2,z,.2)
    d.parent_vehicle_track_id = 11
    matched = t._dormant_pair_log_odds(d,candidate,.2)
    candidate.parent_vehicle_track_id = 999
    candidate.plate_side = 1
    candidate.ref_w *= 10
    assert t._dormant_pair_log_odds(d,candidate,.2) == matched
    for name in ['_credible_set_indices','_ocr_reid_log_bayes_or_veto','_latent_natural_reid_log_bayes_or_veto']:
        assert not hasattr(tracker_module,name)
    assert all('ocr' not in name for name in inspect.signature(t.step).parameters)

    class MustNotRead:
        def __getattribute__(self,name):
            raise AssertionError('Character state was inspected')
    out = t.step([_make_det()],.05,ocr_slot_probs=MustNotRead(),
                 ocr_slot_masses=MustNotRead(),ocr_natural_posteriors=MustNotRead())
    assert out[0].track_id == 1


def test_global_feasible_assignment_matches_bruteforce_including_null_and_birth() -> None:
    pair = np.array([[2.,.3],[.4,1.]])
    miss = np.log([.2,.5])
    birth = np.log([1.3,1.7])
    result = feasible_assignment_posterior(pair,miss,birth,100)
    expected = {}
    for choices in itertools.product([-1,0,1],repeat=2):
        used = [j for j in choices if j >= 0]
        if len(set(used)) != len(used):
            continue
        pairs = tuple((i,j) for i,j in enumerate(choices) if j>=0)
        weight = sum(pair[i,j] if j>=0 else miss[i] for i,j in enumerate(choices))
        weight += sum(birth[j] for j in range(2) if j not in used)
        expected[pairs] = weight
    norm = np.logaddexp.reduce(list(expected.values()))
    assert len(result.hypotheses) == 7
    for h in result.hypotheses:
        assert h.log_weight == pytest.approx(expected[h.pairs])
        assert h.probability == pytest.approx(math.exp(expected[h.pairs]-norm))
    assert result.discarded_mass_upper_bound == 0
    assert sum(h.probability for h in result.hypotheses) == pytest.approx(1)


def test_hypothesis_budget_accounts_for_discarded_mass_and_allows_abstention() -> None:
    result = feasible_assignment_posterior(np.zeros((5,5)),np.zeros(5),np.zeros(5),1,1000)
    assert len(result.hypotheses) == 1
    assert result.discarded_mass_upper_bound > .9
    assert sum(h.probability for h in result.hypotheses)+result.discarded_mass_upper_bound == pytest.approx(1)
    assert result.hypotheses[0].probability < .1


def test_busy_diagonal_assignment_has_no_phantom_discarded_mass() -> None:
    n,m = 18,16
    pair = np.full((n,m),-500.)
    pair[np.arange(m),np.arange(m)] = 10.
    result = feasible_assignment_posterior(pair,np.zeros(n),np.zeros(m))
    assert result.hypotheses[0].pairs == tuple((i,i) for i in range(m))
    assert result.hypotheses[0].probability > .9
    assert 0 <= result.discarded_mass_upper_bound < .05
    assert sum(h.probability for h in result.hypotheses)+result.discarded_mass_upper_bound == pytest.approx(1)


def test_terminal_mixed_logit_scale_calibrates_long_dwell_status() -> None:
    from vpocr_ocr.session_v2 import (
        SessionConfigV2, TerminalCalibrationV2, build_decision_v2_from_arrays,
    )
    from vpocr_ocr.vocab import JPPlateGrammar, PlateVocab
    from vpocr_synth.plate.models import PlateText

    with pytest.raises(ValueError,match='nonnegative'):
        TerminalCalibrationV2(mixed_logit_scale=-.1)
    assert TerminalCalibrationV2().mixed_logit_scale == 1.0
    vocab = PlateVocab()
    logits = np.full((12,vocab.size),-30.)
    logits[np.arange(12),vocab.encode(PlateText('品川','330','さ',1234))] = 30.
    args = dict(slot_logits=logits,plate_type_logits=np.array([30.,-30.,-30.,-30.,-30.]),
                polarity_logit=0.,slot_readability_pred=np.ones(12),status_logits=np.array([8.,0.,12.]),
                mem_P=np.ones((12,8)),mem_conflict=np.full(12,.01),color_P=np.ones(4),pol_P=0.,
                num_frames_seen=96,vocab=vocab,grammar=JPPlateGrammar(vocab))
    assert build_decision_v2_from_arrays(**args,config=SessionConfigV2()).status == 'TRACK_MIXED'
    cal = TerminalCalibrationV2(mixed_logit_scale=.25,mixed_bias=-8.,conflict_weight=2.)
    decision = build_decision_v2_from_arrays(**args,config=SessionConfigV2(calibration=cal))
    mixed = math.exp(-4.98)/(math.exp(8.)+1+math.exp(-4.98))
    assert decision.readable_probability == pytest.approx((1-mixed)/(1+math.exp(-8.)))
    assert decision.status == 'ACCEPTED'


def test_symmetric_active_assignment_keeps_unmerged_candidates() -> None:
    t = PlateTrackStabilizer()
    for ts in [0,.05]:
        t.step([_make_det(160,160),_make_det(240,160)],ts)
    output = t.step([_make_det(200,120),_make_det(200,200)],.1)
    assert all(z.track_id is None for z in output)
    assert not t.last_relinked_pairs
    assert len(t._tracks) == 4


def test_camera_irls_translation_and_high_residual_covariance_inflation() -> None:
    rng = np.random.default_rng(62)
    source = rng.uniform([0,0],[640,360],(120,2))
    destination = source+[4,-3]
    estimator = BackgroundCameraMotionEstimator()
    clean = estimator.fit_correspondences(source,destination+rng.normal(0,.05,source.shape))
    noisy = estimator.fit_correspondences(source,destination+rng.normal(0,9,source.shape))
    np.testing.assert_allclose(clean.g_t[:,2],[4,-3],atol=.1)
    assert clean.Sigma_g.shape == (6,6)
    assert np.trace(noisy.Sigma_g) > np.trace(clean.Sigma_g)*50
    assert np.trace(noisy.residual_cov_2x2) > np.trace(clean.residual_cov_2x2)*50
    states = []
    # Hold the mean warp fixed to isolate residual/parameter uncertainty.
    for motion in [clean,replace(noisy,affine_2x3=clean.affine_2x3)]:
        t = PlateTrackStabilizer()
        t.step([_make_det()],0)
        t.step([],.05,action_meta=NO_LOOK,camera_motion=motion)
        states.append(t._tracks[0])
    assert np.trace(states[1].P) > np.trace(states[0].P)
    assert np.linalg.eigvalsh(states[1].P).min() > 0


def test_cmc_full_jacobian_and_vehicle_covariance_propagation() -> None:
    motion = CameraMotionEstimate(np.array([[1.,0,7],[0,1.,-2]]),np.eye(6)*.01,np.eye(2)*4)
    t = PlateTrackStabilizer()
    t.step([_make_det()],0)
    state = t._tracks[0]
    state.x[2:4] = [30,10]
    old = state.x.copy()
    oldP = state.P.copy()
    _compensate_plate(state,motion)
    np.testing.assert_allclose(state.x[:2],old[:2]+[7,-2])
    np.testing.assert_allclose(state.x[2:],old[2:])
    assert state.P[2,2] > oldP[2,2]  # uncertain affine changes velocity covariance too
    v = MultiVehicleKalmanTracker()
    v.step([_make_vehicle()],0)
    before = v._tracks[0].P.copy()
    v._tracks[0].compensate(motion)
    assert np.trace(v._tracks[0].P) > np.trace(before)


def test_sparse_klt_outside_vehicle_masks_and_unavailable_background() -> None:
    rng = np.random.default_rng(8)
    frame = rng.integers(0,255,(128,192,3),dtype=np.uint8)
    warped = cv2.warpAffine(frame,np.array([[1.,0,3],[0,1.,2]]),(192,128))
    frame[40:90,60:110] = 0
    warped[40:90,60:110] = 255
    e = BackgroundCameraMotionEstimator()
    e.estimate(frame,[(60,40,110,90)])
    motion = e.estimate(warped,[(60,40,110,90)])
    assert motion.n_correspondences >= 6
    np.testing.assert_allclose(motion.g_t[:,2],[3,2],atol=.35)
    e.reset()
    blank = np.zeros_like(frame)
    e.estimate(blank)
    uncertain = e.estimate(blank)
    assert uncertain.n_correspondences == 0
    assert np.trace(uncertain.residual_cov_2x2) > 100


def test_containment_plans_delegate_to_cropper_without_gamma_cap() -> None:
    t = PlateTrackStabilizer()
    t.step([_make_det()],0)
    requests = t.get_search_requests(.1,(720,1280))
    assert len(requests) == 1
    state = t._tracks[0]
    x,P = state.peek_predict(.1)
    q = (_H_24@x).reshape(4,2)
    C = _H_24@P@_H_24.T
    cov = np.stack([C[2*k:2*k+2,2*k:2*k+2] for k in range(4)])+np.eye(2)[None]*1e-9
    expected = select_containment_crop_window(q,cov,(720,1280))
    np.testing.assert_allclose(requests[0].roi_xyxy,expected.window_xyxy)
    far = t.get_predicted_roi_proposals(2)[0]
    assert far.gamma > 4
    assert far.needs_detector_reacquire


def test_search_request_is_score_free_and_cannot_enter_observation_updates() -> None:
    t = PlateTrackStabilizer()
    t.step([_make_det()],0)
    request = t.get_search_requests(.05,(480,640))[0]
    assert_not_observation(request)
    for name in ['score','raw_det_score','plate_null_llr','veh_log_odds','corners_xy']:
        assert not hasattr(request,name)
    with pytest.raises(TypeError,match='fresh observations'):
        t.step([request],.05)
    with pytest.raises(TypeError):
        assert_not_observation(_make_det())
    v = MultiVehicleKalmanTracker()
    for ts in [0,.05]:
        current = v.step([_make_vehicle()],ts)
    requests = v.build_unplated_bumper_proposals(current,set(),(480,640))
    assert len(requests) == 1 and isinstance(requests[0],SearchRequest)
    assert_not_observation(requests[0])
    assert not v.predict_vehicles_at(.1)


@pytest.mark.parametrize('mount_box, reason', [
    (None, 'component_search'),
    ((180., 160., 220., 180.), 'vehicle_mount'),
])
def test_vehicle_search_reason_matches_mount_support(mount_box, reason) -> None:
    tracker = MultiVehicleKalmanTracker()
    for ts in [0, .05]:
        tracker.step([replace(_make_vehicle(), plate_mount_box_xyxy=mount_box)], ts)
    requests = tracker.get_search_requests(.1, img_hw=(480, 640))
    assert len(requests) == 1
    assert requests[0].relation_hint == reason
    assert requests[0].target_entity_id == tracker._tracks[0].vehicle_track_id


@pytest.mark.parametrize('accepts_covered', [False, True])
def test_stage2_search_filters_component_and_low_voi_requests(accepts_covered) -> None:
    from types import SimpleNamespace
    from vpocr_det.pipeline import _search_requests

    reasons = ('plate_containment', 'vehicle_mount', 'component_search')
    requests = [SearchRequest(f'search:{i}', (20.*i, 0., 20.*i+10., 10.), None,
                              reason, (1., 1.), gain)
                for i, (reason, gain) in enumerate(itertools.product(reasons, (1.5, 2.0)))]

    def plate_search(ts, *, img_hw, include_dormant):
        assert ts == .1 and img_hw == (480, 640) and not include_dormant
        return requests[:2]

    def vehicle_search(ts, *, img_hw):
        assert ts == .1 and img_hw == (480, 640)
        return requests[2:]

    def vehicle_search_covered(ts, *, img_hw, covered_vehicle_ids):
        assert covered_vehicle_ids == {7}
        return vehicle_search(ts, img_hw=img_hw)

    owner = SimpleNamespace(
        tracker=SimpleNamespace(get_search_requests=plate_search,
            calibration=TrackerCalibration(c_wait=2.0),
            _tracks=[SimpleNamespace(confirmed=True, parent_vehicle_track_id=7)]),
        vehicle_tracker=SimpleNamespace(get_search_requests=(
            vehicle_search_covered if accepts_covered else vehicle_search)))
    assert _search_requests(owner, [], .1, (480, 640)) == [requests[1], requests[3]]


def test_measurement_deduplication_uses_frame_geometry_not_score_sentinel() -> None:
    t = PlateTrackStabilizer()
    a = _make_det(frame_key='camera:1')
    b = replace(a,raw_det_score=0,provenance=ObservationProvenance.SEARCH_MEASUREMENT)
    assert len(t.step([a,b],0)) == 1
    assert len(t._tracks) == 1
    assert t._tracks[0].hits == 1


def test_uncertain_attachment_normalizes_unbound_and_allows_multiple_plate_sides() -> None:
    vt = MultiVehicleKalmanTracker()
    for ts in [0,.05]:
        vehicles = vt.step([_make_vehicle()],ts)
    plates = [replace(_make_det(200,145),track_id=1),
              replace(_make_det(205,150),track_id=2),
              replace(_make_det(2000,1500),track_id=3)]
    vt.bind_plates_and_sync(vehicles,plates,img_hw=(1080,1920))
    for p in plates:
        states = vt.attachment_posteriors[p.track_id]
        assert sum(s.probability for s in states.values()) == pytest.approx(1)
        assert None in states
        assert all(s.covariance_xy.shape == (2,2) for s in states.values())
    assert vt.attachment_posteriors[1][1].probability > .9
    assert vt.attachment_posteriors[2][1].probability > .9
    assert plates[2].parent_vehicle_track_id is None
    assert vt.attachment_posteriors[3][None].probability == pytest.approx(1)


def test_attachment_numeric_vehicle_id_change_has_zero_likelihood_penalty() -> None:
    trackers = [MultiVehicleKalmanTracker(),MultiVehicleKalmanTracker()]
    for tracker in trackers:
        p = replace(_make_det(),track_id=1)
        v = replace(_make_vehicle(),vehicle_track_id=10)
        tracker.bind_plates_and_sync([v],[p],img_hw=(720,1280))
    for tracker,vid in zip(trackers,[10,999]):
        tracker.bind_plates_and_sync([replace(_make_vehicle(),vehicle_track_id=vid)],
                                     [replace(_make_det(),track_id=1)],img_hw=(720,1280))
    assert trackers[0].attachment_posteriors[1][10].probability == pytest.approx(
        trackers[1].attachment_posteriors[1][999].probability)


def test_gen6_attachment_underflow_has_no_prior_offset_mixture() -> None:
    tracker, reference = MultiVehicleKalmanTracker(), MultiVehicleKalmanTracker()
    vehicle = replace(_make_vehicle(), vehicle_track_id=10)
    far_plate = replace(_make_det(1e6, 1e6), track_id=1)
    tracker.bind_plates_and_sync([vehicle], [far_plate], img_hw=(720, 1280))
    previous = tracker.attachment_posteriors[1]
    assert previous[None].probability == 1.0
    assert previous[10].probability == 0.0
    for t in (tracker, reference):
        t.bind_plates_and_sync([vehicle], [replace(_make_det(), track_id=1)],
                               img_hw=(720, 1280))
    assert tracker.attachment_posteriors[1][10].probability == reference.attachment_posteriors[1][10].probability


@pytest.mark.parametrize('invalid_kind', ['shared_key', 'starts_before', 'starts_at'])
@pytest.mark.parametrize('with_valid_candidate', [False, True])
def test_gen6_relink_requires_disjoint_keys_and_intervals(
        monkeypatch, invalid_kind, with_valid_candidate) -> None:
    tracker = PlateTrackStabilizer()
    dormant = tracker._new_track(1, _make_det(frame_key='old'), 0.0)
    assert dormant.first_obs_ts == 0.0
    dormant.last_obs_ts = 1.0
    first_ts = {'shared_key': 1.5, 'starts_before': .5, 'starts_at': 1.0}[invalid_kind]
    key = 'old' if invalid_kind == 'shared_key' else 'invalid'
    invalid = tracker._new_track(2, _make_det(frame_key=key), first_ts)
    invalid.last_obs_ts = 2.0  # The latest timestamp alone would allow this merge.
    invalid.hits = tracker.relink_min_hits
    tracker._dormant_tracks = [dormant]
    tracker._tracks = [invalid]
    updates = {2: invalid.last_observation}
    likelihood_calls = []
    def strong_match(d, t, ts):
        likelihood_calls.append(t.track_id)
        return 100.0
    monkeypatch.setattr(tracker, '_dormant_pair_log_odds', strong_match)
    if with_valid_candidate:
        valid = tracker._new_track(3, _make_det(frame_key='valid'), 1.5)
        valid.hits = tracker.relink_min_hits
        tracker._tracks.append(valid)
        updates[3] = valid.last_observation
    tracker._resolve_disjoint_merges(2.0, updates)
    assert invalid in tracker._tracks and 2 in updates
    assert likelihood_calls == ([3] if with_valid_candidate else [])
    assert tracker.last_relinked_pairs == ([(3, 1)] if with_valid_candidate else [])
    assert dormant.evidence_keys == ({'old', 'valid'} if with_valid_candidate else {'old'})
    assert dormant.first_obs_ts == 0.0


def test_disjoint_relink_notification_preserves_evidence_despite_parent_id_change() -> None:
    t = PlateTrackStabilizer()
    emb = np.eye(1,32,0)[0]
    for ts in [0,.05]:
        t.step([replace(_make_det(embedding=emb,frame_key=f'old:{ts}'),parent_vehicle_track_id=10)],ts)
    t.step([],.4,action_meta=NO_LOOK)
    assert t.dormant_track_ids == [1]
    first = t.step([replace(_make_det(embedding=emb,frame_key='new:1'),parent_vehicle_track_id=999)],.5)
    assert first[0].track_id is None
    output = t.step([replace(_make_det(embedding=emb,frame_key='new:2'),parent_vehicle_track_id=999)],.55)
    assert output[0].track_id == 1
    assert t.last_relinked_pairs == [(2,1)]
    state = t._tracks[0]
    assert state.merge_evidence
    assert state.evidence_keys == {'old:0','old:0.05','new:1','new:2'}


def test_coupled_facade_shares_camera_covariance_and_only_emits_fresh_measurements() -> None:
    tracker = CoupledPMBMTracker()
    motion = CameraMotionEstimate(np.eye(2,3),np.eye(6)*.001,np.eye(2))
    for ts in [0,.05]:
        p,v = tracker.step([_make_det() ],[_make_vehicle()],ts,FULL,camera_motion=motion)
    assert p[0].track_id == 1 and v[0].vehicle_track_id == 1
    assert tracker.last_camera_motion is motion
    p,v = tracker.step([],[],.1,NO_LOOK,camera_motion=motion)
    assert p == [] and v == []
    requests = tracker.get_search_requests(.15,(480,640))
    assert requests
    for request in requests:
        assert_not_observation(request)


def test_coherent_heave_mixture_preserves_identity_without_velocity_runaway() -> None:
    tracker = PlateTrackStabilizer()
    for k in range(5):
        tracker.step([_make_det(150+k*8,220)],k*.05)
    output = tracker.step([_make_det(190,246)],.25)
    state = tracker._tracks[0]
    assert output[0].track_id == 1
    assert state.last_shock_triggered
    assert abs(state.cy-246) < 4
    assert abs(state.vy) < 120
    assert np.linalg.eigvalsh(state.P).min() > 0


def test_ambiguous_dormant_reappearance_does_not_force_identity_merge() -> None:
    tracker = PlateTrackStabilizer()
    for ts in [0,.05]:
        tracker.step([_make_det(160,160),_make_det(240,160)],ts)
    tracker.step([],.4,action_meta=NO_LOOK)
    assert tracker.dormant_track_ids == [1,2]
    tracker.step([_make_det(200,160)],.5)
    output = tracker.step([_make_det(200,160)],.55)
    assert output[0].track_id == 3
    assert tracker.last_relinked_pairs == []
    assert tracker.dormant_track_ids == [1,2]


def test_fresh_crop_ownership_is_not_replaced_by_filtered_or_coasted_pixels() -> None:
    tracker = PlateTrackStabilizer()
    crop1 = np.full((80,160,3),60,np.uint8)
    crop2 = np.full((80,160,3),195,np.uint8)
    tracker.step([replace(_make_det(200),ocr_crop_rgb=crop1)],0)
    fresh = replace(_make_det(228),ocr_crop_rgb=crop2)
    fresh.raw_corners_xy = fresh.corners_xy.copy()
    output = tracker.step([fresh],.05)[0]
    assert output.track_id == 1
    assert output.ocr_crop_rgb is crop2
    np.testing.assert_array_equal(output.raw_corners_xy,fresh.raw_corners_xy)
    assert tracker.step([],.1,action_meta=NO_LOOK) == []
    assert tracker.get_search_requests(.15,(480,640))


def test_attachment_uses_uncertain_co_motion_and_preserves_unbound_alternative() -> None:
    vehicle_tracker = MultiVehicleKalmanTracker()
    plate_tracker = PlateTrackStabilizer()
    for ts in [0,.05]:
        vehicles = vehicle_tracker.step([_make_vehicle(180),_make_vehicle(220)],ts)
        plates = plate_tracker.step([_make_det(200)],ts)
    # Both components overlap the plate equally; only their velocity differs.
    plate_tracker._tracks[0].x[2:4] = [0,0]
    vehicle_tracker._tracks[0].x[4:6] = [0,0]
    vehicle_tracker._tracks[1].x[4:6] = [1000,1000]
    vehicle_tracker.bind_plates_and_sync(vehicles,plates,img_hw=(720,1280),plate_tracker=plate_tracker)
    B = vehicle_tracker.attachment_posteriors[1]
    assert B[1].probability > B[2].probability
    assert B[None].probability > 0
    assert plate_tracker._tracks[0].attachments is B


class _IntegrationQuad(torch.nn.Module):
    """A deterministic neural evaluation with explicit plate/null evidence."""
    is_dual_track_net = True

    def __init__(self):
        super().__init__()
        self.evaluations = 0

    def forward(self, fovea, surround):
        self.evaluations += len(fovea)
        n = len(fovea)
        return dict(corners_norm=torch.tensor([[[.35,.35],[.65,.35],[.65,.65],[.35,.65]]]).repeat(n,1,1),
                    is_plate_logit=torch.full((n,),8.), plate_null_llr=torch.full((n,),8.),
                    side_logits=torch.zeros(n,3), quality_logit=torch.zeros(n),
                    corner_vis_logits=torch.full((n,4),4.), support_nonconformity=torch.zeros(n))


def _integration_pipeline(monkeypatch, cascade=False, **kwargs):
    import vpocr_det.pipeline as module
    from vpocr_det.types import FrameDetectionResult
    cls = module.MultiRateEdgeCascadePipeline if cascade else module.VehiclePlateVideoPipeline
    quad = _IntegrationQuad()
    pipe = cls(torch.nn.Identity(), quad, **kwargs)
    monkeypatch.setattr(module, 'decode_plate_det_predictions',
                        lambda *a, **kw: FrameDetectionResult())
    return pipe, quad


@pytest.mark.parametrize('cascade', [False, True])
def test_pipeline_executes_score_free_search_before_pmbm(monkeypatch, cascade):
    from vpocr_det.pipeline import _search_inputs
    pipe, quad = _integration_pipeline(monkeypatch, cascade)
    request = SearchRequest('search:0', (120.,80.,280.,200.), None,
                            'plate_containment', (4.,4.), 1.)
    assert_not_observation(request)
    carriers = _search_inputs([request], '0:0')
    assert carriers[0].score == carriers[0].raw_det_score == 0
    monkeypatch.setattr(pipe.tracker, 'get_search_requests', lambda *a, **kw: [request])
    calls = []
    step = pipe.tracker.step
    def measured_step(observations, *args, **kwargs):
        calls.append(kwargs)
        assert all(isinstance(o, PlateDetection) and o.quad_refined for o in observations)
        assert all(o.plate_null_llr == 8 and o.raw_det_score == 0 for o in observations)
        return step(observations, *args, **kwargs)
    monkeypatch.setattr(pipe.tracker, 'step', measured_step)
    result = pipe.process_frame(np.full((320,480,3),90,np.uint8), 0)
    assert quad.evaluations > 0 and result.plates
    for observation in result.plates:
        assert observation.provenance == ObservationProvenance.SEARCH_MEASUREMENT
        assert observation.raw_det_score == 0
        assert observation.score == pytest.approx(1/(1+math.exp(-8)))
        assert observation.score not in (.38,.42)
        assert observation.frame_key == '0:0'
    action = calls[0]['action_meta']
    assert action.detector_ran_full and action.searched_rois_xyxy == (request.roi_xyxy,)
    assert calls[0]['frame_rgb'].shape == (320,480,3)


@pytest.mark.parametrize('execute_elsewhere', [False, True])
def test_pipeline_skipped_roi_has_zero_detection_availability(monkeypatch, execute_elsewhere):
    pipe, quad = _integration_pipeline(monkeypatch, cascade=True)
    for ts in [0,.05]:
        pipe.tracker.step([_make_det()], ts, action_meta=FULL)
    state = pipe.tracker._tracks[0]
    before = state.r
    pipe._frame_idx, pipe._frames_since_det = 2, 0
    from types import SimpleNamespace
    proposal = SimpleNamespace(is_dormant=False, needs_detector_reacquire=False,
                               predicted_bbox_xyxy=state.bbox_xyxy)
    monkeypatch.setattr(pipe.tracker, 'get_predicted_roi_proposals', lambda **kw: [proposal])
    requests = ([SearchRequest('elsewhere',(0.,0.,10.,10.),None,'plate_containment',(1.,1.),1.)]
                if execute_elsewhere else [])
    monkeypatch.setattr(pipe.tracker, 'get_search_requests', lambda *a, **kw: requests)
    monkeypatch.setattr(pipe.scout, 'step', lambda *a, **kw: dict(
        unmonitored_rois_xyxy=[], motion_rois_xyxy=[], has_border_entry=False, jitter_xy=(0,0)))
    actions = []
    step = pipe.tracker.step
    def capture(observations, *args, **kwargs):
        actions.append(kwargs['action_meta'])
        return step(observations, *args, **kwargs)
    monkeypatch.setattr(pipe.tracker, 'step', capture)
    result = pipe.process_frame(np.full((320,480,3),90,np.uint8), .1)
    assert not result.metadata['ran_full_detector']
    assert not actions[0].detector_ran_full
    assert detection_availability(state.bbox_xyxy, actions[0]) == 0
    assert state.r == pytest.approx(before*math.exp(-.05/pipe.tracker.calibration.survival_tau_s))
    assert bool(quad.evaluations) == execute_elsewhere


@pytest.mark.parametrize('cascade', [False, True])
def test_pipeline_relink_merges_factors_and_finalize_is_idempotent(monkeypatch, cascade):
    from vpocr_ocr.model_v2 import MFJPPlateOCRV2, ModelConfigV2
    from vpocr_ocr.session_v2 import MFPlateOCRSessionV2, SessionStateV2
    from vpocr_ocr.vocab import PlateVocab
    vocab = PlateVocab()
    model = MFJPPlateOCRV2(ModelConfigV2.from_preset('nano', vocab_size=vocab.size)).eval()
    pipe, _ = _integration_pipeline(monkeypatch, cascade, ocr_model=model, ocr_vocab=vocab)
    target = MFPlateOCRSessionV2(model, vocab=vocab)
    source = MFPlateOCRSessionV2(model, vocab=vocab)
    crop = np.full((80,160,3),100,np.uint8)
    target.push(crop, 0, source_frame_id=0, frame_key=(0,0))
    source.push(crop, .05, source_frame_id=1, frame_key=(0,1))
    pipe._ocr_sessions.update({1:target, 2:source})
    pipe.tracker.last_relinked_pairs = [(2,1)]
    monkeypatch.setattr(pipe.tracker, 'step', lambda *a, **kw: [])
    merged = []
    merge = target.merge_disjoint_evidence
    def capture(other):
        merged.append(other)
        return merge(other)
    monkeypatch.setattr(target, 'merge_disjoint_evidence', capture)
    pipe.process_frame(np.zeros((320,480,3),np.uint8), .1)
    assert merged == [source]
    assert pipe._ocr_sessions == {1:target}
    assert target.frame_keys == {(0,0),(0,1)}
    assert target.state == source.state == SessionStateV2.COLLECTING
    assert target.num_readout_calls == source.num_readout_calls == 0
    first = pipe.finalize_tracks()
    second = pipe.finalize_tracks()
    assert first[1] is second[1]
    assert target.state == SessionStateV2.FINALIZED
    assert target.num_readout_calls <= 1 and source.num_readout_calls == 0
    # Retirement reuses the same cached terminal capability.
    pipe.tracker.last_relinked_pairs = []
    pipe.tracker.last_expired_track_ids = [1]
    retired = pipe.process_frame(np.zeros((320,480,3),np.uint8), .15)
    assert retired.metadata['retired_track_decisions'][1] is first[1]
    assert pipe.finalize_tracks()[1] is first[1]
    assert target.num_readout_calls <= 1


@pytest.mark.parametrize('cascade', [False, True])
def test_pipeline_does_not_push_after_automatic_ocr_seal(monkeypatch, cascade):
    import vpocr_det.pipeline as module
    from vpocr_ocr.model_v2 import MFJPPlateOCRV2, ModelConfigV2
    from vpocr_ocr.session_v2 import MFPlateOCRSessionV2, SessionStateV2
    from vpocr_ocr.vocab import PlateVocab
    vocab = PlateVocab()
    model = MFJPPlateOCRV2(ModelConfigV2.from_preset('nano', vocab_size=vocab.size)).eval()
    pipe, _ = _integration_pipeline(monkeypatch, cascade, ocr_model=model, ocr_vocab=vocab)
    sess = MFPlateOCRSessionV2(model, vocab=vocab)
    pipe._ocr_sessions[1] = sess
    monkeypatch.setattr(sess, 'stopping_action', lambda **kw: 'seal_and_read_once')
    observed = replace(_make_det(), track_id=1,
        ocr_crop_rgb=np.full((80,160,3),80,np.uint8), raw_corners_xy=_make_det().corners_xy)
    monkeypatch.setattr(module, 'refine_and_crop_detections', lambda *a, **kw: [observed])
    monkeypatch.setattr(pipe.tracker, 'step', lambda *a, **kw: [observed])
    frame = np.full((320,480,3),90,np.uint8)
    first = pipe.process_frame(frame, 0)
    assert sess.state == SessionStateV2.FINALIZED and sess.num_readout_calls == 1
    monkeypatch.setattr(sess, 'push', lambda *a, **kw: pytest.fail('push after seal'))
    second = pipe.process_frame(frame, .05)
    assert first.ocr_decisions[1] is second.ocr_decisions[1]
    assert sess.num_frames_seen == 1 and sess.num_readout_calls == 1
    assert pipe.finalize_tracks()[1] is first.ocr_decisions[1]


@pytest.mark.parametrize('cascade', [False, True])
def test_pipeline_crop_ownership_and_coasted_ocr_exclusion(monkeypatch, cascade):
    import vpocr_det.pipeline as module
    pipe, _ = _integration_pipeline(monkeypatch, cascade)
    raw = _make_det().corners_xy
    crop = np.full((80,160,3),193,np.uint8)
    tight = np.full((64,128,3),125,np.uint8)
    observed = replace(_make_det(), track_id=1, corners_xy=raw+10,
                       raw_corners_xy=raw.copy(), ocr_crop_rgb=crop, tight_crop_rgb=tight)
    monkeypatch.setattr(module, 'refine_and_crop_detections', lambda *a, **kw: [observed])
    monkeypatch.setattr(pipe.tracker, 'step', lambda *a, **kw: [observed])
    frame = np.full((320,480,3),90,np.uint8)
    result = pipe.process_frame(frame, 0)
    assert result.plates[0].ocr_crop_rgb is crop
    assert result.plates[0].tight_crop_rgb is tight
    expected_geom = module.extract_quad_geometry_meta(raw, crop)
    np.testing.assert_array_equal(result.plates[0].geom_meta, expected_geom)
    coasted = replace(observed, is_coasted=True)
    monkeypatch.setattr(pipe.tracker, 'step', lambda *a, **kw: [coasted])
    result = pipe.process_frame(frame, .05)
    assert result.plates[0].ocr_crop_rgb is None
    assert result.plates[0].tight_crop_rgb is None
    assert result.plates[0].geom_meta is None
