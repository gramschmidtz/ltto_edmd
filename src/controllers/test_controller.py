# src/controllers/test_controller.py
import numpy as np

A_MAX_KMS2 = 1e-7                  # km/s^2 (논문 값)

def a_rt_profile(t_days: float):
    """
    0-100일, 400-500일, 700-800일 구간에서 (ar, at) = (0.7e-7, 0.7141e-7) km/s^2 인가\n
    그 외 구간에서는 (ar, at) = (0, 0) km/s^2 인가\n
    하는 테스트 제어 프로필
    
    Args
    ----
    t_days: days

    Return
    ------
    np.ndarray: shape (2,1) 배열로 반환
    """
    on = (
        (0.0 <= t_days <= 100.0) or
        (400.0 <= t_days <= 500.0) or
        (700.0 <= t_days <= 800.0)
    )
    if on:
        return np.array([[0.7 * A_MAX_KMS2],
                         [0.7141 * A_MAX_KMS2]], dtype=float)
    return np.zeros((2, 1), dtype=float)