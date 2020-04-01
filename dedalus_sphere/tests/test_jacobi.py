"""Test Jacobi transforms and operators."""


import pytest
import numpy as np
from dedalus_sphere import jacobi128


N_range = [1,2,3,4,8,16]
ab_range = [-0.5, 0, 0.5, 1]


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_forward_backward_loop(N, a, b):
    """Test round-trip transforms from grid space."""
    # Setup
    grid, weights = jacobi128.quadrature(N, a, b)
    envelope = jacobi128.envelope(a, b, a, b, grid)
    polynomials = jacobi128.recursion(N, a, b, grid, envelope)
    # Build transform matrices
    forward = weights * polynomials
    backward = polynomials.T.copy()
    assert np.allclose(backward @ forward, np.identity(N+1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_backward_forward_loop(N, a, b):
    """Test round-trip transforms from coeff space."""
    # Setup
    grid, weights = jacobi128.quadrature(N, a, b)
    envelope = jacobi128.envelope(a, b, a, b, grid)
    polynomials = jacobi128.recursion(N, a, b, grid, envelope)
    # Build transform matrices
    forward = weights * polynomials
    backward = polynomials.T.copy()
    assert np.allclose(forward @ backward, np.identity(N+1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_Ap_loop(N, a, b):
    """Test round-trip transforms from grid space with conversion."""
    # Setup
    grid0, weights0 = jacobi128.quadrature(N, a, b)
    envelope0 = jacobi128.envelope(a, b, a, b, grid0)
    polynomials0 = jacobi128.recursion(N, a, b, grid0, envelope0)
    envelope1 = jacobi128.envelope(a+1, b, a+1, b, grid0)
    polynomials1 = jacobi128.recursion(N, a+1, b, grid0, envelope1)
    # Build matrices
    forward = weights0 * polynomials0
    conversion = jacobi128.operator('A+', N, a, b)
    backward = polynomials1.T.copy()
    assert np.allclose(backward @ conversion @ forward, np.identity(N+1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_Bp_loop(N, a, b):
    """Test round-trip transforms from grid space with conversion."""
    # Setup
    grid0, weights0 = jacobi128.quadrature(N, a, b)
    envelope0 = jacobi128.envelope(a, b, a, b, grid0)
    polynomials0 = jacobi128.recursion(N, a, b, grid0, envelope0)
    envelope1 = jacobi128.envelope(a, b+1, a, b+1, grid0)
    polynomials1 = jacobi128.recursion(N, a, b+1, grid0, envelope1)
    # Build matrices
    forward = weights0 * polynomials0
    conversion = jacobi128.operator('B+', N, a, b)
    backward = polynomials1.T.copy()
    assert np.allclose(backward @ conversion @ forward, np.identity(N+1))

