from src.utils.safety import prediction_is_reliable


def test_uncertain_or_ood_predictions_are_not_reliable():
    assert not prediction_is_reliable(0.95, 0.50, True)
    assert not prediction_is_reliable(0.69, 0.50, False)
    assert not prediction_is_reliable(0.90, 0.10, False)


def test_clear_in_domain_prediction_passes_conservative_gate():
    assert prediction_is_reliable(0.90, 0.30, False)
