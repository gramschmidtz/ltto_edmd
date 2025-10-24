# scripts/DMD.py
import os
import numpy as np
import matplotlib.pyplot as plt

from src.edmd.make_dataset import build_dataset
from src.controllers.test_controller import a_rt_profile
from src.dynamics.config import DT_TAU, step_num, traj_num, T_END_DAYS
from src.dynamics.dynamics_reduced import sundman_days_per_tau

def main():
    dimension = 3
    
    X1, X2, U = build_dataset(
    dimension = dimension,
    N = step_num,
    traj_num = traj_num,
    a_range = (-0.3,0.3),
    b_range = (-0.3,0.3),
    c_range = (0.9,1.0)
    )
    print("데이터셋 생성 완료!")

    Omega = np.vstack((X1,U))
    pinvOmega = np.linalg.pinv(Omega)
    G = X2 @ pinvOmega
    A = G[:,0:3]
    B = G[:,3:]

    taus = [0.0]
    t_days = [0.0]
    X = [np.array([0.02330563, 0.00867989, 0.9391078], dtype=float)]
    U_applied = []
    a_mag = []

    k = 0
    while t_days[-1] < T_END_DAYS:
        xk = X[-1].reshape(dimension, 1)
        t_k = t_days[-1]

        # 제어 입력
        u_k = a_rt_profile(t_k)  # shape (2,1)
        U_applied.append(u_k.flatten())

        # 다음 상태
        x_next = (A @ xk) + (B @ u_k)
        X.append(x_next.flatten())

        # τ 적분
        taus.append(taus[-1] + DT_TAU)

        # t'(τ) 계산 후 시간 적분
        Λ, η, κ = xk.flatten()
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

    # 4) 플롯
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

    fig.suptitle("Reduced States and Control Magnitude over Real Time (days) via DMD")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    save_dir = os.path.join(os.path.dirname(__file__), "..", "fig")
    os.makedirs(save_dir, exist_ok=True)   # 폴더 없으면 자동 생성
    save_path = os.path.join(save_dir, "DMD_result.png")
    fig.savefig(save_path, dpi=300)
    print(f"✅ Figure saved to: {save_path}")
    plt.close(fig)  # 메모리 절약 (선택사항)

if __name__ == "__main__":
    main()