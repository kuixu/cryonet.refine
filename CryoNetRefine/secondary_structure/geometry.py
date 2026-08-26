from __future__ import annotations

import math
import numpy as np


def vec(xyz: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(xyz, dtype=float)


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray | None:
    n = norm(v)
    if n < 1e-10:
        return None
    return v / n


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return norm(vec(a) - vec(b))


def angle_degrees_from_cos(cos_value: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_value))))


def angle_between_abs(u: np.ndarray, v: np.ndarray) -> float | None:
    un = unit(u)
    vn = unit(v)
    if un is None or vn is None:
        return None
    return angle_degrees_from_cos(abs(float(np.dot(un, vn))))
