"""Tests rapides : python tests/test_basic.py (ou pytest)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ouint import simulate_pair, lrt_interaction


def test_parameter_recovery():
    _, X = simulate_pair(40, 1000, theta=0.2, gamma=0.1, sigma=0.2, rng=1)
    est = np.array([[r["theta"], r["gamma"], r["sigma"]]
                    for r in (lrt_interaction(x, 0.5) for x in X)])
    m = est.mean(axis=0)
    assert abs(m[0] - 0.2) < 0.04 and abs(m[1] - 0.1) < 0.03 and abs(m[2] - 0.2) < 0.02, m


def test_null_pvalues_not_extreme():
    _, X = simulate_pair(100, 700, theta=0.2, gamma=0.0, sigma=0.2, rng=2)
    p = np.array([lrt_interaction(x, 0.5)["p"] for x in X])
    assert (p < 0.05).mean() < 0.12, (p < 0.05).mean()


if __name__ == "__main__":
    test_parameter_recovery(); test_null_pvalues_not_extreme(); print("OK")
