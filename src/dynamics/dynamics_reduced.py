# src/dynamics/dynamics_reduced.py
import numpy as np
from dynamics.config import Units
from controllers.test_controller import a_rt_profile

def drift_vec(Λ: float, η: float, κ: float) -> np.ndarray:
    """
    무섭동 벡터장 p = [-η, Λ, 0]
    """
    Λ = np.asarray(Λ)
    η = np.asarray(η)
    zero = np.zeros_like(Λ)
    return np.stack([-η, Λ, zero], axis=-1)

def control_fields(Λ: float, η: float, κ: float):
    """
    제어 벡터장 (b_r, b_t) 반환. 특이점(|κ|<thresh or |κ+Λ|<thresh)이면 (None, None).
      b_r = [0, (1/(κ(κ+Λ)^2))*(LU^2/μ), 0]
      b_t = [((2κ+Λ)/(κ(κ+Λ)^3))*(LU^2/μ), 0, -(1/(κ+Λ)^3)*(LU^2/μ)]
    """
    kpL = κ + Λ

    br2 = (1.0 / (κ * (kpL**2))) * Units.LU2_over_MU
    bt1 = ((2.0 * κ + Λ) / (κ * (kpL**3))) * Units.LU2_over_MU
    bt3 = - (1.0 / (kpL**3)) * Units.LU2_over_MU

    b_r = np.array([0.0, br2, 0.0], dtype=float)
    b_t = np.array([bt1, 0.0, bt3], dtype=float)
    return b_r, b_t

def b_r(Λ: float, η: float, κ: float):
    kpL = κ + Λ

    br2 = (1.0 / (κ * (kpL**2))) * Units.LU2_over_MU

    b_r = np.array([0.0, br2, 0.0], dtype=float)
    return b_r

def b_t(Λ: float, η: float, κ: float):
    kpL = κ + Λ

    bt1 = ((2.0 * κ + Λ) / (κ * (kpL**3))) * Units.LU2_over_MU
    bt3 = - (1.0 / (kpL**3)) * Units.LU2_over_MU

    b_t = np.array([bt1, 0.0, bt3], dtype=float)
    return b_t

def sundman_days_per_tau(Λ: float, η: float, κ: float) -> float:
    """
    t'(τ) [days/τ] = ( √(LU^3/μ) / ( κ(κ+Λ)^2 ) ) / DAY
    """
    kpL = κ + Λ
    return (Units.sqrt_LU3_over_MU / (κ * (kpL**2))) / Units.DAY

# Real Dynamics (Eq.36)
def f_over_tau(x, a_rt_func=a_rt_profile):
    """
    Reduced real dynamics in τ-domain for [Λ, η, κ] + Sundman time (Eq. 36).
    State x = [Λ, η, κ, t_days]
      - Λ, η, κ : already in [-1,1]
      - t_days  : physical time in days

    Returns:
      [Λ', η', κ', t_days']  where (·)' := d(·)/dτ

    Notes:
      - a_r, a_t are km/s^2 (no unit conversion).
      - Drift/Control/Sundman은 위 공용 함수로부터 사용.
    """
    Λ, η, κ, t_days = x
    kpL = κ + Λ

    # thrust profile (km/s^2)
    ar, at = a_rt_func(t_days)

    # vector fields
    p  = drift_vec(Λ, η, κ)
    br, bt = control_fields(Λ, η, κ)

    # state derivative in τ
    x_tau_vec = p + br * ar + bt * at  # [Λ', η', κ']
    t_tau     = sundman_days_per_tau(Λ, η, κ)

    return np.array([x_tau_vec[0], x_tau_vec[1], x_tau_vec[2], t_tau], dtype=float)