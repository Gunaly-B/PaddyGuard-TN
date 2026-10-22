def prediction_is_reliable(confidence: float, top_two_margin: float, is_ood: bool,
                           confidence_floor: float = 0.70,
                           margin_floor: float = 0.15) -> bool:
    """Conservative advisory gate; it cannot guarantee detection of all OOD inputs."""
    return (not is_ood and confidence >= confidence_floor and top_two_margin >= margin_floor)
