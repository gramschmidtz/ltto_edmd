# ltto/controls.py
import numpy as np
from dynamics.config import Units

A_MAX_KMS2 = 1e-7                  # km/s^2 (논문 값)

def a_rt_profile(t_days: float):
    """3개의 100일 thrust arc / 나머지 0. Returns (a_r, a_t) in nondim."""
    on = (
        (0.0 <= t_days <= 100.0) or
        (400.0 <= t_days <= 500.0) or
        (700.0 <= t_days <= 800.0)
    )
    if on:
        return (0.7*A_MAX_KMS2, 0.7141*A_MAX_KMS2)
    return (0.0, 0.0)