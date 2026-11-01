"""ouint : interactions entre deux animaux à partir de trajectoires d'Ornstein-Uhlenbeck."""
from .simulate import simulate_pair, daily_mean
from .classical import classical_tests
from .inference import fit_h0, fit_h1, lrt_interaction

__all__ = ["simulate_pair", "daily_mean", "classical_tests",
           "fit_h0", "fit_h1", "lrt_interaction"]
