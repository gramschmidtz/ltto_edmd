# scripts/ground_truth.py
"""
실제 다이내믹스를 적분해서 ground_truth값을 얻고 plot한다.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from src.controllers.test_controller import a_rt_profile, random_profile, zero_profile
from src.dynamics.config import DT_TAU, T_END_DAYS
from src.dynamics.dynamics_reduced import sundman_days_per_tau, f_over_tau_without_sundman
from src.dynamics.discrete_dynamics import rk4_step

def main():
    dimension = 3
    
    # 초기 조건
    x0 = np.array([0.02330563, 0.00867989, 0.9391078], dtype=float)
    # x0 = np.array([0.3, 0.3, 0.9391078], dtype=float)
    
    taus = [0.0]
    t_days = [0.0]
    X = [x0]
    U_applied = []
    a_mag = []

    k = 0
    while t_days[-1] < T_END_DAYS:
        xk = X[-1]
        t_k = t_days[-1]

        # 제어 입력
        # u_k = random_profile(t_k)  # shape (2,1) # 완전 랜덤
        u_k = a_rt_profile(t_k)    # shape (2,1) # 논문 테스트 프로필
        # u_k = zero_profile(t_k)    # shape (2,1) # 무추력

        U_applied.append(u_k.flatten())

        # 다음 상태 (rk4_step 사용)
        x_next = rk4_step(f_over_tau_without_sundman, xk, u_k.flatten(), DT_TAU)
        X.append(x_next)

        # τ 적분
        taus.append(taus[-1] + DT_TAU)

        # t'(τ) 계산 후 시간 적분 (Sundman transform)
        Λ, η, κ = xk
        tprime_days_per_tau = sundman_days_per_tau(Λ, η, κ)
        if not np.isfinite(tprime_days_per_tau):
            tprime_days_per_tau = 0.0
        t_next = t_k + tprime_days_per_tau * DT_TAU
        t_days.append(t_next)

        # 제어 크기
        a_mag.append(float(np.sqrt(u_k[0, 0]**2 + u_k[1, 0]**2)))
        k += 1

    # np.array 변환
    taus = np.array(taus)
    t_days = np.array(t_days)
    X = np.array(X).T  # (3, steps)
    U_applied = np.array(U_applied).T  # (2, steps)
    a_mag = np.array(a_mag)

    print(f"Rollout 완료! 총 {k} 스텝, 총 {t_days[-1]:.2f} days")

    # 플롯
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(t_days, X[0, :len(t_days)], color="C0")
    axs[0].set_ylabel("Λ(t)")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_days, X[1, :len(t_days)], color="C1")
    axs[1].set_ylabel("η(t)")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t_days, X[2, :len(t_days)], color="C2")
    axs[2].set_ylabel("κ(t)")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(t_days[:-1], a_mag, color="C3")
    axs[3].set_xlabel("t [days]")
    axs[3].set_ylabel("‖a‖ (km/s²)")
    axs[3].grid(True, alpha=0.3)

    fig.suptitle("Ground Truth Rollout")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_dir = os.path.join(os.path.dirname(__file__), "..", "fig")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ground_truth_rollout.png")
    fig.savefig(save_path, dpi=300)
    print(f"Figure saved to: {save_path}")
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()