# dynamics/discrete_dynamics.py
import numpy as np
import matplotlib.pyplot as plt
from dynamics.config import Units, DT_TAU
from controllers.test_controller import a_rt_profile

def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5*dt*k1, u)
    k3 = f(x + 0.5*dt*k2, u)
    k4 = f(x + dt*k3, u)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

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
    특이점이면 ValueError 발생.
    """
    THRESH = 1e-10
    kpL = κ + Λ
    if abs(κ) < THRESH or abs(kpL) < THRESH:
        raise ValueError(
            f"Sundman singular: |κ|={abs(κ):.3e}, |κ+Λ|={abs(kpL):.3e} < {THRESH:.1e}"
        )
    return (Units.sqrt_LU3_over_MU / (κ * (kpL**2))) / Units.DAY

# Continuous Dynamics (Eq.36)
def f_over_tau(x, u):
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
    Λ, η, κ = x

    # thrust profile (km/s^2)
    ar, at = u

    # vector fields
    p  = drift_vec(Λ, η, κ)
    br = b_r(Λ, η, κ)
    bt = b_t(Λ, η, κ)

    # state derivative in τ
    x_tau_vec = p + br * ar + bt * at  # [Λ', η', κ']

    return np.array([x_tau_vec[0], x_tau_vec[1], x_tau_vec[2]], dtype=float)

def discrete_dynamics(x, u, dt=DT_TAU):
    x_next = rk4_step(f_over_tau, x, u, dt)
    return np.array([x_next[0],x_next[1],x_next[2]], dtype=float)

# if __name__ == "__main__":
#     x0 = np.array([0.02330563, 0.00867989, 0.9391078], dtype=float)

#     # 스텝 수와 시간간격(τ 축)
#     N = 18_000
#     dt = DT_TAU  # dynamics.config에서 가져온 τ-스텝

#     # ====== 롤아웃 버퍼 ======
#     taus = np.zeros(N+1)                     # τ 시간축
#     X = np.zeros((N+1, 3), dtype=float)      # [Λ, η, κ]
#     U = np.zeros((N, 2), dtype=float)        # [a_r, a_t]

#     X[0] = x0
#     taus[0] = 0.0
#     U = #제어기

#     # ====== 시뮬레이션 루프 (RK4-이산화 동역학 사용) ======
#     for k in range(N):
#         Λ, η, κ = X[k]
#         tau_k = taus[k]

#         x_next = discrete_dynamics(X[k], U[k], dt)

#         X[k+1] = x_next
#         taus[k+1] = tau_k + dt

#     # ====== 플롯 ======
#     a_mag = np.sqrt(U[:, 0]**2 + U[:, 1]**2)

#     # ====== Subplot (4행 1열) ======
#     fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

#     axs[0].plot(taus[:N+1], X[:N+1, 0], color="C0")
#     axs[0].set_ylabel("Λ(τ)")
#     axs[0].grid(True, alpha=0.3)

#     axs[1].plot(taus[:N+1], X[:N+1, 1], color="C1")
#     axs[1].set_ylabel("η(τ)")
#     axs[1].grid(True, alpha=0.3)

#     axs[2].plot(taus[:N+1], X[:N+1, 2], color="C2")
#     axs[2].set_ylabel("κ(τ)")
#     axs[2].grid(True, alpha=0.3)

#     axs[3].plot(taus[:N], a_mag, color="C3")
#     axs[3].set_xlabel("τ")
#     axs[3].set_ylabel("‖a‖ (km/s²)")
#     axs[3].grid(True, alpha=0.3)

#     fig.suptitle("Reduced States and Control Magnitude over τ (RK4 rollout)")
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()

if __name__ == "__main__":
    # ====== 초기 상태: [Λ, η, κ, t_days] ======
    x0 = np.array([0.02, 0.0, 0.9391078], dtype=float)
    t0_days = 0.0
    x_ext = np.array([x0[0], x0[1], x0[2], t0_days], dtype=float)

    # ====== 적분 설정 ======
    N_max = 18000           # τ-스텝 상한 (부족하면 늘리면 됨)
    dt_tau = DT_TAU         # τ-스텝
    T_GOAL_DAYS = 890.0     # 목표 물리시간

    # ====== 확장 동역학: f_over_tau_ext(x) -> [Λ', η', κ', t_days'] ======
    def f_over_tau_ext(x_ext_vec: np.ndarray) -> np.ndarray:
        Λ, η, κ, t_days = x_ext_vec
        # t_days에 의존하는 제어 입력 (km/s^2)
        a_r, a_t = a_rt_profile(t_days)

        # 상태 미분 (τ-도메인)
        p  = drift_vec(Λ, η, κ)               # [Λ', η', κ'] (drift)
        br = b_r(Λ, η, κ)                     # control field for a_r
        bt = b_t(Λ, η, κ)                     # control field for a_t
        x_tau = p + br * a_r + bt * a_t       # [Λ', η', κ']

        # Sundman: t'(τ) [days/τ]
        t_tau = sundman_days_per_tau(Λ, η, κ)

        return np.array([x_tau[0], x_tau[1], x_tau[2], t_tau], dtype=float)

    # ====== RK4 (τ-도메인) ======
    def rk4_step_tau(f, x, dt):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ====== 버퍼 ======
    taus    = np.zeros(N_max + 1)
    t_days  = np.zeros(N_max + 1)
    X       = np.zeros((N_max + 1, 3), dtype=float)   # [Λ, η, κ]
    U       = np.zeros((N_max, 2), dtype=float)       # [a_r, a_t]
    X[0] = x0
    t_days[0] = t0_days
    taus[0] = 0.0

    # ====== 롤아웃 ======
    hit_step = None
    for k in range(N_max):
        Λ, η, κ, t_d = x_ext

        # 기록용 제어 (현재 t_days에서의 a_r, a_t)
        a_r, a_t = a_rt_profile(t_d)
        U[k] = (a_r, a_t)

        # 한 스텝 적분 (τ-도메인 RK4, 내부에서 t_days 변화 반영)
        try:
            x_ext_next = rk4_step_tau(f_over_tau_ext, x_ext, dt_tau)
        except ValueError as e:
            print(f"[WARN] τ={taus[k]:.6f}에서 특이점 감지로 중단: {e}")
            hit_step = k
            break

        # 기록
        X[k+1]      = x_ext_next[:3]
        t_days[k+1] = x_ext_next[3]
        taus[k+1]   = taus[k] + dt_tau
        x_ext       = x_ext_next

        # 목표 물리시간 도달 시 중단
        if t_days[k+1] >= T_GOAL_DAYS:
            hit_step = k + 1
            break

    # 유효 길이 결정
    if hit_step is None:
        hit_step = N_max

    # ====== 제어 크기 a(t) ======
    a_mag = np.sqrt(U[:hit_step, 0]**2 + U[:hit_step, 1]**2)

    # ====== 플롯 (x축 = t_days) ======
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(t_days[:hit_step+1], X[:hit_step+1, 0])
    axs[0].set_ylabel("Λ")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_days[:hit_step+1], X[:hit_step+1, 1])
    axs[1].set_ylabel("η")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t_days[:hit_step+1], X[:hit_step+1, 2])
    axs[2].set_ylabel("κ")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(t_days[:hit_step], a_mag)
    axs[3].set_xlabel("t [days]")
    axs[3].set_ylabel("‖a‖ (km/s²)")
    axs[3].grid(True, alpha=0.3)

    fig.suptitle("Λ, η, κ, ‖a‖ vs physical time t (Sundman-transformed RK4 in τ)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
