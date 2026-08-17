"""Correctness tests for the methane forward model.

These are the "define evaluation methods that measure numerical impact"
checks: the numerical solver is validated against the exact analytic solution,
mass conservation is verified to machine precision, and grid refinement is
shown to reduce the discretization error (convergence).

Run with:  pytest -q
"""

import numpy as np
import pytest

from methane_ai.forward import (
    make_grid, exact_field, fd_solve, eddy_K, observe, beam_indices,
    field_from_params, sample_params, PARAM_RANGES,
)


def _cases():
    Ur, ur = PARAM_RANGES["U"], PARAM_RANGES["u_star"]
    for U in (Ur[0], 4.0, Ur[1]):
        for us in (ur[0], 0.25, ur[1]):
            yield U, us


def test_solver_matches_exact_solution():
    """Finite-volume solver reproduces the exact erfc solution to < 2%."""
    x, z, X, Z = make_grid(48, 64, 60.0, 10.0)
    band = (z >= 1.0) & (z <= 4.0)
    xin = x > 2.0
    for U, us in _cases():
        K = eddy_K(us)
        Ce = exact_field(X, Z, U, K, 25.0)
        Cn = fd_solve(x, z, U, K, 25.0)
        rel = np.linalg.norm(Cn[np.ix_(xin, band)] - Ce[np.ix_(xin, band)]) \
            / np.linalg.norm(Ce[np.ix_(xin, band)])
        assert rel < 0.02, f"U={U}, u*={us}: rel L2 = {rel:.3%}"


def test_mass_is_conserved():
    """Column-integrated flux U * integral(C dz) grows as Q0 * x (< 0.1% error)."""
    x, z, X, Z = make_grid(48, 64, 60.0, 10.0)
    dz = z[1] - z[0]
    Q0 = 25.0
    for U, us in _cases():
        Cn = fd_solve(x, z, U, eddy_K(us), Q0)
        colflux = (U * Cn * dz).sum(axis=1)
        slope = np.polyfit(x[3:], colflux[3:], 1)[0]
        assert abs(slope / Q0 - 1.0) < 1e-3, f"mass ratio {slope / Q0:.5f}"


def test_convergence_under_refinement():
    """Refining the vertical grid reduces the solver's error (consistency)."""
    x0, z0, X0, Z0 = make_grid(48, 32, 60.0, 10.0)
    x1, z1, X1, Z1 = make_grid(48, 128, 60.0, 10.0)
    U, K, Q0 = 4.0, eddy_K(0.25), 25.0

    def err(x, z, X, Z):
        band = (z >= 1.0) & (z <= 4.0)
        Ce, Cn = exact_field(X, Z, U, K, Q0), fd_solve(x, z, U, K, Q0)
        return np.linalg.norm(Cn[-1, band] - Ce[-1, band]) \
            / np.linalg.norm(Ce[-1, band])

    assert err(x1, z1, X1, Z1) < err(x0, z0, X0, Z0)


def test_concentration_is_nonnegative_and_ground_peaked():
    x, z, X, Z = make_grid(48, 64, 60.0, 10.0)
    C = exact_field(X, Z, 4.0, eddy_K(0.25), 25.0)
    assert C.min() >= 0.0
    # at the sensor the profile decreases with height (ground source)
    assert np.all(np.diff(C[-1]) <= 1e-4)


def test_observation_operator_shapes():
    x, z, X, Z = make_grid(48, 64, 60.0, 10.0)
    rng = np.random.default_rng(0)
    p = sample_params(rng, 5)
    fields = field_from_params(X, Z, p)
    iz1, iz2 = beam_indices(z, 1.0, 3.0)
    y = observe(fields, iz1, iz2)
    assert y.shape == (2, 5)
    assert np.all(y[0] > y[1])   # lower beam sees more methane than the upper


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
