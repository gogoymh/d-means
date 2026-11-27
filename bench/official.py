"""Official E-means and O-means baselines, so a weak reimplementation cannot inflate D-means' lead.

E-means: faiss.contrib.e_means.NumpyEmeans (Meta, MIT), run unchanged from the faiss-cpu wheel.
O-means: owlgebra-ai/faiss contrib/o_means (MIT), vendored in bench/third_party/o_means. Its
training loop and schedule are the official code; only its Triton GPU kernels are swapped for
torch CPU code with the same arithmetic (cpu_kernels.py).

faiss and torch each bundle their own OpenMP runtime, and on macOS the two crash in one process,
so each trainer runs in its own worker process (spawned once, reused). The caller imports neither.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dmeans import _log

log = _log.get("bench.official")
_pools: dict[str, ProcessPoolExecutor] = {}


def _run(lib: str, fn, *args):
    if lib not in _pools:
        _pools[lib] = ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"))
        log.info("started the %s worker process", lib)
    return _pools[lib].submit(fn, *args).result()


def emeans(x: np.ndarray, k: int, nepochs: float, seed: int) -> tuple[np.ndarray, float]:
    """Official E-means (faiss); return (centroids, passes over the data it made)."""
    return _run("faiss", _emeans, x, k, nepochs, seed)


def omeans(x: np.ndarray, k: int, nepochs: float, seed: int) -> tuple[np.ndarray, float]:
    """Official O-means on the CPU (torch); return (centroids, passes over the data it made)."""
    return _run("torch", _omeans, x, k, nepochs, seed)


def _emeans(x: np.ndarray, k: int, nepochs: float, seed: int) -> tuple[np.ndarray, float]:
    """Train the official NumPy E-means; return (centroids, passes over the data it made)."""
    from faiss.contrib.e_means import NumpyEmeans

    x = np.ascontiguousarray(x, dtype=np.float32)
    model = NumpyEmeans(x.shape[1], k, nepochs=nepochs, seed=seed)
    c, _ = model.train(x)
    passes = sum(model.recipe["epoch_rows"]) / len(x)
    log.debug("official e-means n=%d k=%d nepochs=%s batch=%d alpha0=%.4f passes=%.4f", len(x), k, nepochs, model.recipe["batch_size"], model.recipe["alpha0"], passes)
    return c, passes


def _omeans(x: np.ndarray, k: int, nepochs: float, seed: int) -> tuple[np.ndarray, float]:
    """Train the official O-means on the CPU; return (centroids, passes over the data it made)."""
    import torch

    from .third_party.o_means.o_means import Omeans

    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    model = Omeans(xt.shape[1], k, nepochs=nepochs, seed=seed, capture_assignments=False)
    c, _ = model.train(xt)
    passes = sum(model.recipe["epoch_rows"]) / len(x)
    log.debug("official o-means n=%d k=%d nepochs=%s batch=%d nlist=%d nprobe=%d passes=%.4f", len(x), k, nepochs, model.recipe["batch_size"], model.recipe["nlist"], model.recipe["nprobe"], passes)
    return c.numpy(), passes
