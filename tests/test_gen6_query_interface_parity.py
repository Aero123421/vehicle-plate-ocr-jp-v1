"""Internal feature parity is checked in calibrated consumer output units."""
import pytest
import torch
from vpocr_det.trainer import _verify_query_condition_cutout_parity


class Consumer:
    def cutout_head(self, features, edges, rois, query):
        return query[:, :1, None, None].expand(-1, 1, 2, 2)

    predict_vehicle_cutouts = cutout_head


def test_query_interface_null_direction_preserves_likelihood():
    consumer = Consumer()
    reference = torch.zeros(3, 2)
    actual = reference.clone()
    actual[:, 1] = 10.  # Consumer-independent coordinates have no pixel likelihood effect.
    empty = torch.zeros(1, 1, 2, 2)
    error = _verify_query_condition_cutout_parity(
        consumer, consumer, empty, empty, empty, empty,
        torch.zeros(3, 5), reference, actual, atol=2.5e-4, rtol=2e-4)
    assert error == 0


def test_query_interface_rejects_changed_pixel_likelihood():
    consumer = Consumer()
    reference = torch.zeros(3, 2)
    actual = reference.clone()
    actual[:, 0] = .1
    empty = torch.zeros(1, 1, 2, 2)
    with pytest.raises(AssertionError):
        _verify_query_condition_cutout_parity(
            consumer, consumer, empty, empty, empty, empty,
            torch.zeros(3, 5), reference, actual, atol=2.5e-4, rtol=2e-4)
