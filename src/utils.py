"""
utils.py
"""
import jax
import time

def many_function_benchmark(
    func_dict,
    args,
    with_jit: bool = True,
    nwarmups: int = 5,
    nreps: int = 5,
):
    """
    Helper function to report the timing of multiple functions
    that all takes args

    Args:
        func_dict: dictionary of name and function
    """
    results = {}

    if with_jit:
        for name, func in func_dict.items():
            if name == "deer":
                with jax.default_matmul_precision("highest"):
                    jit_f = jax.jit(func)
                    jit_f.lower(*args).compile()
                    func_dict[name] = jit_f
            else:
                func_dict[name] = jax.jit(func)

    for key in func_dict.keys():
        func1 = func_dict[key]
        # warmup
        for _ in range(nwarmups):
            x1 = func1(*args)
            jax.block_until_ready(x1)

        # benchmark func1
        t0 = time.time()
        for _ in range(nreps):
            x1 = func1(*args)
            jax.block_until_ready(x1)
        t1 = time.time()
        time1_tots = (t1 - t0) / nreps
        print(f"{key} time: {time1_tots:.3e} s")
        results[key] = time1_tots

    return results
