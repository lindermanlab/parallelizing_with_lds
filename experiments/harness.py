"""
harness.py
Runs experiments using hydra

Options for the experiment are "well" or "gru"
Options for alg are:
- deer
- quasi
- picard
- seq
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
import time


from two_well import TwoWell, TwoWellAnisotropic, KWellsAnisotropic, rand_precision_wishart
from gru import GRU
from s5_word import S5_word
from deer import deer_alg, picard_alg

@hydra.main(config_path="configs", config_name="harness")
def main(cfg: DictConfig):
    seed = cfg.seed
    exp = cfg.exp
    T = cfg.T
    D = cfg.D
    epsilon = cfg.epsilon
    batch_size = cfg.batch_size
    mode = cfg.mode
    alg = cfg.alg
    nwarmups = cfg.nwarmups
    nreps = cfg.nreps
    K = cfg.K

    logger = WandbLogger(project="lds", mode=mode)
    logger.log_hyperparams(OmegaConf.to_container(cfg))

    # set the dynamics
    k1, k2, k3, k4, k5 = jr.split(jr.PRNGKey(seed), 5)
    if exp=="well":
        experiment = TwoWellAnisotropic(
            mu1=jnp.ones((D,)),
            mu2=jnp.zeros((D,)),
            prec1=rand_precision_wishart(k4, D),
            prec2=rand_precision_wishart(k5, D),
            epsilon=epsilon,
        )
    elif exp=="kwell":
        experiment = KWellsAnisotropic(
            ps=jnp.ones((K,)) / K,
            mus=jr.normal(k4, (K, D)),
            precs=jax.vmap(lambda k: rand_precision_wishart(k, D))(jr.split(k5, K)),
            epsilon=epsilon,
        )
    elif exp=="gru":
        experiment = GRU(D, k3)
    elif exp=="s5_word":
        experiment = S5_word()
    else:
        raise ValueError(f"Invalid experiment: {exp}")
    
    if exp != "s5_word":
        initial_state = jnp.zeros((D,))
        inputs = jr.normal(k1, (batch_size,T,D))
        states_guess = jr.normal(k2, (T,D))
    else:
        initial_state = jnp.arange(1, D+1)
        inputs = jr.randint(k1, (batch_size, T), minval=0, maxval=120)
        states_guess = jr.normal(k2, (T, D))

    # set the algorithm
    # sequential
    def seq_eval(inputs):
        _, true_states = jax.lax.scan(
        lambda c, a: experiment.scan_fxn(c, a), initial_state, inputs)
        return true_states[-1]
    # picard
    def picard_eval(inputs):
        num_iters = inputs.shape[0]
        picard_states, final_picard_state, picard_iters, picard_is_nan, picard_nan_mask, *_ = picard_alg(
            experiment.deer_fxn,
            initial_state,
            states_guess,
            inputs,
            num_iters)
        return picard_iters
    # deer
    def deer_eval(inputs):
        num_iters = inputs.shape[0]
        deer_states, final_deer_state, newton_steps, deer_is_nan, deer_nan_mask, *_ = (
            deer_alg(
                experiment.deer_fxn,
                initial_state,
                states_guess,
                inputs,
                num_iters,
                full_trace=False,
            )
        )
        return newton_steps
    def quasi_eval(inputs):
        num_iters = inputs.shape[0]
        deer_states, final_deer_state, newton_steps, deer_is_nan, deer_nan_mask, *_ = (
            deer_alg(
                experiment.deer_fxn,
                initial_state,
                states_guess,
                inputs,
                num_iters,
                quasi=True,
                qmem_efficient=True,
                full_trace=False,
            )
        )
        return newton_steps
    if alg=='seq':
        fxn = jax.jit(jax.vmap(seq_eval))
        fxn.lower(inputs).compile()
    elif alg=='picard':
        fxn = jax.jit(jax.vmap(picard_eval))
        fxn.lower(inputs).compile()
    elif alg=='quasi':
        with jax.default_matmul_precision("highest"):
            fxn = jax.jit(jax.vmap(quasi_eval))
            fxn.lower(inputs).compile()
    elif alg=='deer':
        with jax.default_matmul_precision("highest"):
            fxn = jax.jit(jax.vmap(deer_eval))
            fxn.lower(inputs).compile()
    else:
        raise ValueError(f"Invalid algorithm: {alg}")

    # also have to run deer with highest precision
    if alg == "deer":
        with jax.default_matmul_precision("highest"):
            for _ in range(nwarmups):
                x1 = fxn(inputs)
                jax.block_until_ready(x1)
            t0 = time.time()
            for _ in range(nreps):
                x1 = fxn(inputs)
                jax.block_until_ready(x1)
            t1 = time.time()
            time1_tots = (t1 - t0) / nreps
            print(f"{alg} time: {time1_tots:.3e} s")
    else:
        for _ in range(nwarmups):
            x1 = fxn(inputs)
            jax.block_until_ready(x1)
        t0 = time.time()
        for _ in range(nreps):
            x1 = fxn(inputs)
            jax.block_until_ready(x1)
        t1 = time.time()
        time1_tots = (t1 - t0) / nreps
        print(f"{alg} time: {time1_tots:.3e} s")
    if alg == "seq":
        n_iters = T
    else:
        n_iters = jnp.mean(x1)
    print(f"{alg} n_iters: {n_iters}")

    results = {
        "time": time1_tots,
        "n_iters": n_iters,
    }

    logger.log_metrics(results)
    wandb.finish()

if __name__ == "__main__":
    main()
