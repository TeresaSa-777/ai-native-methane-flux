"""methane_ai: AI-native surrogates and Bayesian inversion for methane flux.

The physics forward model lives in ``methane_ai.forward``. The learned
surrogates (MLP, FNO, DeepONet), the physics-constrained training, the
uncertainty quantification, and the Bayesian inverse problem are all developed
in the tutorial notebook so they can be read top-to-bottom.
"""

from . import forward

# Canonical problem configuration shared by the notebook and the tests.
CONFIG = {
    "Nx": 48,          # downwind stations
    "Nz": 64,          # vertical cells (ground-anchored)
    "x_max": 60.0,     # fetch to the sensor [m]
    "z_max": 10.0,     # domain top [m]
    "beam_z1": 1.0,    # lower open-path beam height [m]
    "beam_z2": 3.0,    # upper open-path beam height [m]
}

__all__ = ["forward", "CONFIG"]
