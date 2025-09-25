"""
super lightweight implementation of Picard and Newton iterations
"""

import jax

from jax import vmap
from jax.lax import scan
import jax.numpy as jnp
import jax.random as jr

@jax.vmap
def full_mat_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a full Jacobian matrix A
    Args:
        q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
        q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j @ A_i, A_j @ b_i + b_j


@jax.vmap
def diag_mat_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a DIAGONAL Jacobian matrix A
    Args:
        q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
        q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

@jax.vmap
def add_operator(q_i, q_j):
    return q_i + q_j

def get_residual(f, initial_state, states, drivers):
    fs = vmap(f)(states[:-1], drivers[1:])  # length T-1
    fs = jnp.concatenate([jnp.array([f(initial_state, drivers[0])]), fs])
    r = states - fs
    return r


def merit_fxn(f, initial_state, states, drivers, Ts=None):
    """
    Helper function to compute the merit function
    Note that this assumes that the initial state (say s0) is combined with the initial noise (drivers[0]) to make s1 (the first state)
    Args:
        f: a forward fxn that takes in a full state and a driver, and outputs the next full state.
        initial_state: packed_state, jax.Array (DIM,)
        states, jax.Array, (T, DIM)
        drivers, jax.Array, (T,N_noise)
        Ts: optional
            if None, then just return the merit function
            ow an array of ints, and returns the merit function for each T in Ts
    """
    r = get_residual(f, initial_state, states, drivers)  # (T,D)
    Ls = 0.5 * jnp.sum(
        jnp.cumsum(r**2, axis=0), axis=1
    )  # make sure to sum over time (T,)
    Ls = jnp.where(jnp.isnan(Ls), jnp.inf, Ls)
    if Ts is not None:
        return Ls[Ts - 1]
    else:
        return Ls[-1]


def quasi_diag_estimator(state, inputs, deer_jvp, key, num_samples=1):
    z_rad = jr.rademacher(key, (num_samples, state.shape[0])).astype(float)
    vmap_jvp = jax.vmap(deer_jvp, in_axes=(None, None, 0))
    jac_diag = jnp.mean(z_rad * vmap_jvp(state, inputs, z_rad), axis=0)
    return jac_diag


def picard_alg(f,
               initial_state,
               states_guess,
               drivers,
               num_iters,
               reset=True,
               tol=5e-4):
    """
    Picard iteration
    """
    @jax.jit
    def _step(carry, args):
        """
        Args:
            carry: tuple of (states, is_nan, iter_num)
            args: None
        """
        states, is_nan = carry
        fs = vmap(f)(states[:-1], drivers[1:])    # (T-1, D)
        bs = fs - states[:-1]
        b0 = f(initial_state, drivers[0])  # h1=f(h0, e1)
        b = jnp.concatenate([b0[jnp.newaxis, :], bs])  # (T, D)
        binary_op = add_operator
        new_states = jax.lax.associative_scan(
            binary_op, b
        )  
        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)
        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, is_nan), (new_states, nan_mask, mf_val)

    def body_func_single(iter_inp):
        """
        Body func when we only use one T
        iter_inp: tuple of (iter_idx, states, err, is_nan, extra)
            iter_idx: int, the current iteration
            states: (T,D), the current states
            err: current value of merit function
            is_nan: bool, tracker of whether there has been a nan in the deer evaluation trace
            there is an extra input
        """
        iter_idx, states, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_err = merit_fxn(f, initial_state, states, drivers)
        return iter_idx + 1, new_states, new_err, is_nan, None

    def cond_func(iter_inp):
        """
        iter_inp: tuple of (iter_idx, states, err)
            iter_idx: int, the current iteration
            states: (T,D), the current states
            err: current value of merit function
            is_nan: bool, tracker of whether there has been a nan in the deer evaluation trace
        """
        iter_idx, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    newton_steps, final_state, _, is_nan, iters_below = jax.lax.while_loop(
        cond_func,
        body_func_single,
        (
            0,
            states_guess,
            merit_fxn(f, initial_state, states_guess, drivers),
            False,
            None,
        ),
    )
    all_states = None
    return (all_states, final_state, newton_steps, is_nan, None, None, None)

def deer_alg(
    f,
    initial_state,
    states_guess,
    drivers,
    num_iters,
    quasi=False,
    diagonal_func=None,
    qmem_efficient=False,
    k=0,
    clip=False,
    full_trace=False,
    Ts=None,
    reset=False,
    tol=5e-4,
    key=jr.PRNGKey(0),
):
    """
    Lightweight implementation of DEER
    Args:
      f: a forward fxn that takes in a full state and a driver, and outputs the next full state.
          In the context of a GRU, f is a GRU cell, the full state is the hidden state, and the driver is the input
      initial_state: packed_state, jax.Array (DIM,)
      states_guess, jax.Array, (T, DIM)
      drivers, jax.Array, (T,N_noise)
      num_iters: number of iterations to run
      quasi: bool, whether to use quasi-newton or not
      qmem_efficient: bool, whether to use the Hutchinson estimator
      k: amount of damping, should be between 0 and 1. 0 is no damping, 1 is max damping.
      clip: bool, whether or not the Jacobian should be clipped to have eigenvalues in range [-1,1]
      full_trace: bool, whether or not to return the full trace of the DEER iterations
        if True, uses scan
        if False, uses while loop
      Ts: optional
        if None, then just return the number of deer iterations
        ow return the number of deer iterations for each T in Ts
      reset: bool, whether or not to reset the states to the initial guess if they are NaN
      tol: float, the tolerance for convergence (merit function)
      key: jax.random.PRNGKey, the key for the Hutchinson estimator
    Notes:
    - The initial_state is NOT the same as the initial mean we give to dynamax
    - The initial_mean is something on which we do inference
    - The initial_state is the fixed starting point.

    The structure looks like the following.
    Let h0 be the initial_state (fixed), h[1:T] be the states, and e[0:T-1] be the drivers

    Then our graph looks like

    h0 -----> h1 ---> h2 ---> ..... h_{T-1} ----> h_{T}
              |       |                   |          |
              e1      e2       ..... e_{T-1}      e_{T}
    """
    DIM = len(initial_state)
    L = len(drivers)

    if qmem_efficient:
        def deer_jvp(state, input, z_rad):
            return jax.jvp(lambda state: f(state, input), (state,), (z_rad,))[1]
        keys = jr.split(key, L-1)

    @jax.jit
    def _step(carry, args):
        """
        Args:
            carry: tuple of (states, is_nan, iter_num)
            args: None
        """
        states, is_nan = carry
        # Evaluate f and its Jacobian in parallel across timesteps 1,..,T-1
        fs = vmap(f)(
            states[:-1], drivers[1:]
        )  # get the next. Note that states[0] is h1, and drivers[1] is e2. h2=f(h1, e2)
        nan_jac_idx = False
        # Compute the As and bs from fs and Jfs
        if quasi:
            if qmem_efficient:
                As = vmap(quasi_diag_estimator, in_axes=(0, 0, None, 0))(states[:-1], drivers[1:], deer_jvp, keys)
            elif diagonal_func:
                As = vmap(diagonal_func)(states[:-1], drivers[1:])
            else:
                # Jfs are the Jacobians (what is going with the tuples rn)
                Jfs = vmap(jax.jacrev(f, argnums=0))(states[:-1], drivers[1:])
                As = vmap(lambda Jf: jnp.diag(Jf))(Jfs)
            As = (1 - k) * As  # damping
            if clip:
                As = jnp.clip(As, -1, 1)
            bs = fs - As * states[:-1]
        else:
            # Jfs are the Jacobians (what is going with the tuples rn)
            Jfs = vmap(jax.jacrev(f, argnums=0))(states[:-1], drivers[1:])  # (T, D, D)
            if reset:
                nan_mask_jacobians = jnp.isnan(Jfs).any(
                    axis=(1, 2), keepdims=True
                )  # Shape (T, 1, 1)
                nan_jac_idx = jnp.isnan(Jfs).any()
                I = jnp.eye(Jfs.shape[1])
                Jfs = jnp.where(nan_mask_jacobians, I, Jfs)
            As = Jfs
            As = (1 - k) * As  # damping
            bs = fs - jnp.einsum("tij,tj->ti", As, states[:-1])

        # initial_state is h0
        b0 = f(initial_state, drivers[0])  # h1=f(h0, e1)
        A0 = jnp.zeros_like(As[0])
        A = jnp.concatenate(
            [A0[jnp.newaxis, :], As]
        )  # (T, D, D) [or (T, D) for quasi]
        b = jnp.concatenate([b0[jnp.newaxis, :], bs])  # (T, D)
        if quasi:
            binary_op = diag_mat_operator
        else:
            binary_op = full_mat_operator
        # run appropriate parallel alg
        _, new_states = jax.lax.associative_scan(
            binary_op, (A, b)
        )  # a forward pass, but uses linearized dynamics
        is_nan = jnp.logical_or(
            jnp.logical_or(jnp.isnan(new_states).any(), is_nan), nan_jac_idx
        )
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)
        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, is_nan), (new_states, nan_mask, mf_val)

    def cond_func(iter_inp):
        """
        iter_inp: tuple of (iter_idx, states, err)
            iter_idx: int, the current iteration
            states: (T,D), the current states
            err: current value of merit function
            is_nan: bool, tracker of whether there has been a nan in the deer evaluation trace
        """
        iter_idx, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        """
        Body func when we only use one T
        iter_inp: tuple of (iter_idx, states, err)
        iter_idx: int, the current iteration
        states: (T,D), the current states
        err: current value of merit function
        is_nan: bool, tracker of whether there has been a nan in the deer evaluation trace
        """
        iter_idx, states, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_err = merit_fxn(f, initial_state, states, drivers)
        return iter_idx + 1, new_states, new_err, is_nan, None

    def body_func_multiple(iter_inp):
        """
        Body func when we use multiple Ts
        iter_inp: tuple of (iter_idx, states, err)
        iter_idx: int, the current iteration
        states: (T,D), the current states
        err: current value of merit function
        is_nan: bool, tracker of whether there has been a nan in the deer evaluation trace
        iters_below: array of ints, for each T, the number of iterations that have been below tol for the merit function
        """
        iter_idx, states, _, is_nan_, iters_below = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_errs = merit_fxn(f, initial_state, states, drivers, Ts=Ts)
        iters_below = iters_below + (new_errs < tol).astype(int)
        return iter_idx + 1, new_states, new_errs[-1], is_nan, iters_below

    if full_trace:
        last_output, all_outputs = scan(
            _step, (states_guess, False), None, length=num_iters
        )
        final_state, is_nan = last_output
        all_states, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    elif Ts is not None:
        newton_steps, final_state, _, is_nan, iters_below = jax.lax.while_loop(
            cond_func,
            body_func_multiple,
            (
                0,
                states_guess,
                merit_fxn(f, initial_state, states_guess, drivers),
                False,
                jnp.zeros_like(Ts),
            ),
        )
        all_states, nan_mask, mf_val = None, None, None
    else:
        newton_steps, final_state, _, is_nan, iters_below = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (
                0,
                states_guess,
                merit_fxn(f, initial_state, states_guess, drivers),
                False,
                None,
            ),
        )
        all_states, nan_mask, mf_val = None, None, None
    if iters_below is not None:
        all_newtons = newton_steps - iters_below + 1  # the last input is below the tol
    else:
        all_newtons = None
    return (
        all_states,
        final_state,
        newton_steps,
        is_nan,
        nan_mask,
        all_newtons,
        mf_val,
    )
