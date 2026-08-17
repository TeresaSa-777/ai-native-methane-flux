"""
methane_ai.forward
==================

The *physics* half of this project: a fast, self-contained forward model for
methane dispersion over a homogeneous rice paddy, plus its exact analytic
solution. Everything the learned surrogates imitate lives here.

The governing equation
----------------------
For methane (a passive scalar) emitted at the surface of a rice paddy and
carried downwind by the mean wind, Reynolds averaging + a K-theory (eddy
diffusivity) closure + the surface-layer assumptions (steady, crosswind
homogeneous, advection >> along-wind diffusion) collapse the 3-D transport
equation to a single 1-D "master equation" in which downwind distance ``x``
plays the role of time:

        U dC/dx = d/dz ( K dC/dz )                                       (master)

Here ``C(x, z)`` is the mean methane concentration, ``U`` the mean wind speed,
and ``K`` the turbulent eddy diffusivity. Boundary conditions: a prescribed,
spatially-uniform surface flux  ``-K dC/dz = Q0``  at the ground z = 0 (the rice
emission), zero flux at the top of the domain, and clean inflow ``C = 0`` at the
upwind edge x = 0. Because ``U dC/dx`` acts like a time derivative, this is a
1-D diffusion (heat) equation marched downwind.

Two ways to solve it
--------------------
1. **Exact** (``exact_field``): for constant K the master equation with a
   constant surface flux has the classical closed-form solution

        C(x, z) = (Q0/K) [ sqrt(4*K*tau/pi) * exp(-z^2/(4*K*tau))
                            - z * erfc( z / sqrt(4*K*tau) ) ],   tau = x/U.

   It is smooth, ground-peaked, decays with height, and conserves mass exactly
   ( integral of U*C dz = Q0*x ). This is our ground truth.

2. **Numerical** (``fd_solve``): a conservative finite-volume discretization in
   z, marched implicitly in x (one tridiagonal solve per downwind step). This
   is the "classical numerics" baseline the neural surrogates are compared
   against on both accuracy and wall-clock cost, and it is what a real
   height-varying / non-neutral K(z) would require in practice (no closed form).

K and the turbulence
--------------------
K is tied to the friction velocity through surface-layer scaling,
``K = kappa * u_star * z_ref`` with a fixed reference height, so the three things
an experiment does not know a priori -- emission ``Q0``, wind ``U``, turbulence
``u_star`` -- are exactly the surrogate's inputs and the inverse problem's
unknowns.

Reference: ``Methane_Flux_Derivations`` gives the full step-by-step derivation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc
from scipy.linalg import solve_banded

KAPPA = 0.4          # von Karman constant
Z_REF = 2.0          # reference height [m] setting the constant-K scale


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def make_grid(Nx: int = 48, Nz: int = 64, x_max: float = 60.0,
              z_max: float = 10.0):
    """Return (x, z, X, Z).

    x : downwind stations, x[0] = 0 is the clean-air inflow edge of the field.
    z : *ground-anchored* finite-volume cell centers. The Nz cells tile
        [0, z_max] with width dz = z_max/Nz, so centers sit at (j+0.5)*dz, the
        bottom cell face is exactly the ground z = 0 (where the surface flux is
        injected), and the top face is z_max (zero flux). Ground-anchoring makes
        the finite-volume solver conserve the *same* emitted mass as the exact
        semi-infinite solution.
    X, Z : meshgrids with indexing='ij'  ->  shape (Nx, Nz).
    """
    x = np.linspace(0.0, x_max, Nx)
    dz = z_max / Nz
    z = (np.arange(Nz) + 0.5) * dz
    X, Z = np.meshgrid(x, z, indexing="ij")
    return x, z, X, Z


def eddy_K(u_star: float) -> float:
    """Constant eddy diffusivity K = kappa * u_star * z_ref  [m^2/s]."""
    return KAPPA * u_star * Z_REF


# ---------------------------------------------------------------------------
# 1. Exact analytic solution  (ground truth)
# ---------------------------------------------------------------------------
def exact_field(X: np.ndarray, Z: np.ndarray, U: float, K: float,
                Q0: float) -> np.ndarray:
    """Exact concentration field C(x, z) for constant K and constant surface flux.

    C = (Q0/K)[ sqrt(4*K*tau/pi) exp(-z^2/(4*K*tau)) - z*erfc(z/sqrt(4*K*tau)) ],
    with tau = x/U. Clean inflow (C = 0) at x = 0 is the tau -> 0 limit.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = X / U
        s = 4.0 * K * tau                                # = (2 sigma_z)^2
        C = (Q0 / K) * (np.sqrt(s / np.pi) * np.exp(-Z**2 / s)
                        - Z * erfc(Z / np.sqrt(s)))
    C[X == 0.0] = 0.0
    return C.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Finite-volume numerical solver  (classical-numerics baseline)
# ---------------------------------------------------------------------------
def fd_solve(x: np.ndarray, z: np.ndarray, U: float, K: float,
             Q0: float) -> np.ndarray:
    """Conservative finite-volume solve of the master equation for constant K.

    Fully implicit (backward) march in x -> unconditionally stable, one
    tridiagonal (Thomas-algorithm) solve per downwind step. Conservative by
    construction: the surface flux Q0 enters the bottom cell, the top face is
    closed, and column-integrated flux grows exactly linearly with fetch.
    Returns the field on the (Nx, Nz) center grid.
    """
    Nx, Nz = len(x), len(z)
    dz = z[1] - z[0]
    C = np.zeros((Nx, Nz), dtype=np.float64)
    for n in range(1, Nx):
        dx = x[n] - x[n - 1]
        r = K / dz**2
        lower = np.full(Nz, -r)
        upper = np.full(Nz, -r)
        diag = np.full(Nz, U / dx + 2 * r)
        # bottom cell: closed face below (surface flux injected via RHS)
        diag[0] = U / dx + r
        lower[0] = 0.0
        # top cell: zero-flux face above
        diag[-1] = U / dx + r
        upper[-1] = 0.0
        rhs = U * C[n - 1] / dx
        rhs[0] += Q0 / dz                               # surface emission
        ab = np.zeros((3, Nz))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]
        C[n] = solve_banded((1, 1), ab, rhs)
    return C.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Parameter sampling  (defines the surrogate's operating envelope)
# ---------------------------------------------------------------------------
# Physical ranges loosely matched to a summer Sacramento-Valley rice paddy.
PARAM_RANGES = {
    "Q0":     (10.0, 40.0),     # surface methane flux  [ug m^-2 s^-1]
    "U":      (2.0, 6.0),       # mean wind speed       [m s^-1]
    "u_star": (0.15, 0.35),     # friction velocity     [m s^-1]
}
PARAM_NAMES = ("Q0", "U", "u_star")


def sample_params(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return an (n, 3) array of [Q0, U, u_star] drawn uniformly in range."""
    lo = np.array([PARAM_RANGES[k][0] for k in PARAM_NAMES])
    hi = np.array([PARAM_RANGES[k][1] for k in PARAM_NAMES])
    return rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)


def field_from_params(X, Z, params: np.ndarray) -> np.ndarray:
    """Vectorized exact fields for a batch of [Q0, U, u_star] rows.

    Returns an (n, Nx, Nz) stack of ground-truth concentration fields.
    """
    fields = np.empty((len(params), *X.shape), dtype=np.float32)
    for i, (Q0, U, u_star) in enumerate(params):
        fields[i] = exact_field(X, Z, U, eddy_K(u_star), Q0)
    return fields


# ---------------------------------------------------------------------------
# 4. Observation operator  (what the open-path laser beams measure)
# ---------------------------------------------------------------------------
def beam_indices(z: np.ndarray, z1: float, z2: float):
    """Grid indices of the two open-path beam heights."""
    return int(np.argmin(np.abs(z - z1))), int(np.argmin(np.abs(z - z2)))


def observe(field: np.ndarray, iz1: int, iz2: int) -> np.ndarray:
    """Concentrations the two beams read at the sensor (last downwind column)."""
    return np.array([field[..., -1, iz1], field[..., -1, iz2]])
