"""Equal-cost detection decisions precede one-to-one validation association."""
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('accepted', [True, False])
def test_rejected_duplicate_cannot_consume_accepted_detection(monkeypatch, accepted):
    path = Path(__file__).resolve().parents[1] / 'tools/train_det_quad.py'
    spec = importlib.util.spec_from_file_location('gen6_training_validation', path)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)

    class Pool:
        domain_split = 'val_holdout'
        items = [object()]

        def ingest_clips(self, clips):
            pass

        def sample_batch(self, *args):
            truth = {
                'labels': torch.tensor([training.CLS_VEHICLE]),
                'boxes': torch.tensor([[2., 1., 8., 9.]]),
                'amodal_boxes': torch.tensor([[0., 0., 10., 10.]]),
                'cutout_masks': torch.ones(1, 1, 64, 64),
            }
            return torch.zeros(1, 1, 3, 8, 8), [[truth]]

    class Model:
        config = type('Config', (), {'num_vehicle_queries': 2})()

        def eval(self):
            return self

        def forward_step(self, *args):
            # The rejected duplicate has perfect IoU. The accepted observation
            # has IoU .81 and must remain an eligible true positive.
            return {
                'h_next': None,
                'pred_boxes': torch.tensor([[[0., 0., 10., 10.], [0., 0., 9., 9.]]]),
                'veh_log_odds': torch.tensor([[-10., 10. if accepted else -10.]]),
                'p2_feat': torch.zeros(1), 'aux_edge_3ch': torch.zeros(1),
                'query_condition': torch.zeros(1, 2, 1),
            }

        def cutout_head(self, feature, edge, rois, condition):
            # The mask diagnostic must still cover GT when no query is accepted.
            assert rois.shape[0] == 1
            torch.testing.assert_close(rois[0, 1:], torch.tensor([2., 1., 8., 9.]))
            return torch.zeros(1, 1, 64, 64)

    monkeypatch.setattr(training, 'build_stage1_validation_pool', lambda **kwargs: Pool())
    metrics = training.evaluate_stage1(Model(), seed=1, device='cpu', clips=1, dry_run=True)
    assert metrics['vehicle_recall_iou50_equal_cost'] == float(accepted)
    assert metrics['num_vehicles'] == 1
    assert metrics['soft_cutout_iou'] == pytest.approx(.5)
