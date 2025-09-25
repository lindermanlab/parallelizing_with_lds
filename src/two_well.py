"""
two_well.py
helper functions for the two_well potential
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from jax.scipy.special import logsumexp

import equinox as eqx


def gaussian_diag(x, mu, diag_cov):
    arg = jnp.sum(0.5 * (x - mu) ** 2 / diag_cov)
    return 1.0 / jnp.sqrt(jnp.prod(2.0 * jnp.pi * diag_cov)) * jnp.exp(-arg)


class TwoWell(eqx.Module):
    p1: float
    p2: float
    mu1: jnp.ndarray
    diag1: jnp.ndarray
    mu2: jnp.ndarray
    diag2: jnp.ndarray
    epsilon: float
    D: int

    def __init__(
        self,
        p1=0.49,
        mu1=jnp.array([-1.4, 0.0]),
        mu2=jnp.array([1.6, 0.0]),
        diag1=jnp.array([0.45, 0.45]),
        diag2=jnp.array([0.54, 0.45]),
        epsilon=0.01,
    ):
        """
        Args:
            mu1: jnp.array, mean of the first Gaussian
            mu2: jnp.array, mean of the second Gaussian
            diag1: jnp.array, diagonal covariance of the first Gaussian
            diag2: jnp.array, diagonal covariance of the second Gaussian
            epsilon: float, step size
        """
        self.p1 = p1
        self.p2 = 1.0 - p1
        self.mu1 = mu1
        self.mu2 = mu2
        self.diag1 = diag1
        self.diag2 = diag2
        self.epsilon = epsilon  # step size
        self.D = len(mu1)

    @staticmethod
    def log_gaussian_diag(x, mu, diag_cov):
        """Computes the log probability of a Gaussian with diagonal covariance."""
        log_det = 0.5 * jnp.sum(
            jnp.log(2.0 * jnp.pi * diag_cov)
        )  # Log determinant term
        quad_form = 0.5 * jnp.sum((x - mu) ** 2 / diag_cov)  # Quadratic form
        return -log_det - quad_form

    def logp(self, x):
        """Computes log probability of the Gaussian mixture in a numerically stable way."""
        log_p1 = self.log_gaussian_diag(x, self.mu1, self.diag1) + jnp.log(self.p1)
        log_p2 = self.log_gaussian_diag(x, self.mu2, self.diag2) + jnp.log(self.p2)
        return logsumexp(jnp.array([log_p1, log_p2]))  # Stable log-sum-exp computation

    def grad_logp(self, x):
        return jax.grad(self.logp)(x)

    def deer_fxn(self, state, input):
        new_state = state + self.epsilon * self.grad_logp(state)
        new_state = new_state + jnp.sqrt(2.0 * self.epsilon) * input
        return new_state

    def scan_fxn(self, state, input):
        """
        Gives the dynamics of the system
        """
        new_state = self.deer_fxn(state, input)
        return new_state, new_state

    def sample(self, key, T, dtype=jnp.float64):
        """
        Draw a length T sample from the marginal
        """
        k1, k2 = jr.split(key, 2)
        idxs = jr.bernoulli(k1, p=self.p1, shape=(T,))
        idxs = idxs[:, None]
        ws = jax.random.normal(k2, shape=(T, self.D))
        mu1 = self.mu1[None, :]
        mu2 = self.mu2[None, :]
        mu = jnp.where(idxs, mu1, mu2)
        diag1 = self.diag1[None, :]
        diag2 = self.diag2[None, :]
        vars = jnp.where(idxs, diag1, diag2)
        return mu.astype(dtype)

class TwoWellAnisotropic(eqx.Module):
    """
    Two well but with precision matrices (anistropy)
    """
    p1: float
    p2: float
    mu1: jnp.ndarray
    prec1: jnp.ndarray
    mu2: jnp.ndarray
    prec2: jnp.ndarray
    epsilon: float
    D: int

    def __init__(
        self,
        p1=0.5,
        mu1=jnp.array([-1.4, 0.0]),
        mu2=jnp.array([1.6, 0.0]),
        prec1=jnp.eye(2),
        prec2=jnp.eye(2),
        epsilon=0.01,
    ):
        """
        Args:
            mu1: jnp.array, mean of the first Gaussian
            mu2: jnp.array, mean of the second Gaussian
            prec1: jnp.array, first precision matrix
            prec2: jnp.array, second precision matrix
            epsilon: float, step size
        """
        self.p1 = p1
        self.p2 = 1.0 - p1
        self.mu1 = mu1
        self.mu2 = mu2
        self.prec1 = prec1
        self.prec2 = prec2
        self.epsilon = epsilon  # step size
        self.D = len(mu1)

    @staticmethod
    def log_gaussian(x, mu, prec):
        diff = x - mu
        quad_form = jnp.einsum('...i, ...ij, ...j -> ...', diff, prec, diff)
        return -quad_form

    def logp(self, x):
        """Computes log probability of the Gaussian mixture in a numerically stable way."""
        log_p1 = self.log_gaussian(x, self.mu1, self.prec1) + jnp.log(self.p1)
        log_p2 = self.log_gaussian(x, self.mu2, self.prec2) + jnp.log(self.p2)
        return logsumexp(jnp.array([log_p1, log_p2]))  # Stable log-sum-exp computation

    def grad_logp(self, x):
        return jax.grad(self.logp)(x)

    def deer_fxn(self, state, input):
        new_state = state + self.epsilon * self.grad_logp(state)
        new_state = new_state + jnp.sqrt(2.0 * self.epsilon) * input
        return new_state

    def scan_fxn(self, state, input):
        """
        Gives the dynamics of the system
        """
        new_state = self.deer_fxn(state, input)
        return new_state, new_state


class KWellsAnisotropic(eqx.Module):
    ps: jnp.ndarray  # (K,)
    mus: jnp.ndarray  # (K, D)
    precs: jnp.ndarray  # (K, D, D)
    epsilon: float
    D: int
    K: int

    def __init__(self, ps, mus, precs, epsilon=0.01):
        """
        Args:
            ps: jnp.ndarray of shape (K,) - unnormalized mixture weights
            mus: jnp.ndarray of shape (K, D) - means of the components
            precs: jnp.ndarray of shape (K, D, D) - precision matrices
            epsilon: float - step size
        """
        ps = ps / jnp.sum(ps)
        self.ps = ps
        self.mus = mus
        self.precs = precs
        self.epsilon = epsilon
        self.K = ps.shape[0]
        self.D = mus.shape[1]

    def log_gaussian(self, x):
        """
        Computes the log of the weighted Gaussian mixture density at x.
        x: shape (..., D)
        Returns: shape (...,)
        """
        x = jnp.atleast_1d(x)
        diff = x - self.mus[:, None, :]  # (K, ..., D)
        quad = jnp.einsum("k...i,kij,k...j->k...", diff, self.precs, diff)  # (K, ...)
        log_weighted = -quad + jnp.log(self.ps)[:, None]  # (K, ...)
        return logsumexp(log_weighted, axis=0).squeeze() 

    def logp(self, x):
        return self.log_gaussian(x)

    def grad_logp(self, x):
        return jax.grad(self.logp)(x)

    def deer_fxn(self, state, input):
        drift = self.grad_logp(state)
        noise = jnp.sqrt(2.0 * self.epsilon) * input
        return state + self.epsilon * drift + noise

    def scan_fxn(self, state, input):
        new_state = self.deer_fxn(state, input)
        return new_state, new_state


def rand_precision_wishart(key, D, df=None):
    """Return Λ ~ Wishart(df, I_D).  Λ itself is a precision matrix."""
    df = D + 1 if df is None else df          # df > D guarantees PD
    Z  = jr.normal(key, (df, D))              # (df × D) standard normals
    return Z.T @ Z                            # Λ = ZᵀZ  (SPD by construction)
