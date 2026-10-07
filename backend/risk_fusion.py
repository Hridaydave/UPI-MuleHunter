def clamp(x):
    return max(0.0, min(1.0, float(x)))


def fuse(markov=0.0, rf=0.0, graph=0.0, anomaly=0.0, ring=0.0):
    """
    Transparent research fusion.

    Default weights are intentionally conservative and configurable. They are
    not claimed to be optimal and must be tuned/validated on held-out data
    before being reported as a paper result.
    """
    score = (
        0.25 * clamp(markov)
        + 0.35 * clamp(rf)
        + 0.25 * clamp(graph)
        + 0.10 * clamp(anomaly)
        + 0.05 * clamp(ring)
    )
    return clamp(score)


def risk_band(score):
    score = clamp(score)
    if score >= 0.80:
        return "CRITICAL"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    return "LOW"
