"""
gru.py
helper function for the gru
"""
import jax.random as jr

import equinox as eqx
from equinox.nn import GRUCell

class GRU(eqx.Module):
    D: int
    gru_cell: eqx.Module

    def __init__(self, D: int, key: jr.PRNGKey):
        self.D = D
        self.gru_cell = GRUCell(D, D, key=key)

    def deer_fxn(self, state, input):
        """
        Arg:
            state: (D,) x_t
            input: (d_input,) u_t
        Returns:
            oug: (D,) x_{t+1}
        """
        out = self.gru_cell(state, input)
        return out

    def scan_fxn(self, state, input):
        out = self.deer_fxn(state, input)
        return out, out