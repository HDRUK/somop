import numpy as np


def _sample_ages(
    size: int,
    *,
    dist: str,
    p1: float,
    p2: float,
    min_age: float,
    max_age: float,
    rng=None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    def draw(n):
        if dist == "normal":
            return rng.normal(loc=p1, scale=p2, size=n)
        elif dist == "lognormal":
            return rng.lognormal(mean=p1, sigma=p2, size=n)
        elif dist == "uniform":
            return rng.uniform(low=p1, high=p2, size=n)
        else:
            raise ValueError(f"Unsupported dist: {dist}")

    ages = draw(size)
    mask = (ages < min_age) | (ages > max_age)
    while np.any(mask):
        n_bad = int(mask.sum())
        ages[mask] = draw(n_bad)
        mask = (ages < min_age) | (ages > max_age)
    return ages
