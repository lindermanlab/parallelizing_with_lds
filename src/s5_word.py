"""
state.py
helper functions for the state tracking demo in jax
"""
import jax.numpy as jnp

import equinox as eqx

from itertools import permutations

class S5_word(eqx.Module):
    perm_mats: jnp.ndarray

    def __init__(self):
        # construct all permutation matrices
        perms = list(permutations(range(5)))  # all 120 permutations
        matrices = []

        for perm in perms:
            mat = jnp.zeros((5, 5), dtype=int)
            mat = mat.at[jnp.arange(5), jnp.array(perm)].set(1)
            matrices.append(mat)

        self.perm_mats = jnp.stack(matrices) # shape (120, 5, 5)

    def get_transition_matrix(self, input):
        """
        Args:
            input: (,) u_t
        Returns:
            out: (D*D,D*D) A_t

        Get S5 transition matrix from input
        """
        return self.perm_mats[input]

    def deer_fxn(self, state, input):
        """
        Arg:
            state: (D*D,) x_t
            input: (,) u_t
        Returns:
            out: (D*D,) x_{t+1}
        """
        transition_matrix = self.get_transition_matrix(input)
        return transition_matrix @ state

    def scan_fxn(self, state, input):
        out = self.deer_fxn(state, input)
        return out, out
    

