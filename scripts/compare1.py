# scripts/compare1.py
"""
Ground Truth vs. DMD vs. EDMD_custom 비교 스크립트
- 동일 학습 데이터(X1, X2, U)로 DMD/EDMD의 A,B를 추정
- 같은 초기조건과 제어 프로파일(a_rt_profile)로 τ-도메인 롤아웃
- Sundman 변환(t'(τ))으로 실제 시간 축 적분
- Ground Truth(RK4)와의 오차(RMSE/MAE, 시계열 |Δ|, 전체 L2) 계산/플롯
- 결과 이미지는 fig/compare1.png 로 저장
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.edmd.make_dataset import build_dataset
from src.edmd.observables import custom_observ_1
from src.controllers.test_controller import a_rt_profile
from src.dynamics.config import DT_TAU, step_num, traj_num, T_END_DAYS
from src.dynamics.dynamics_reduced import sundman_days_per_tau, f_over_tau_without_sundman
from src.dynamics.discrete_dynamics import rk4_step


# -------------------- 유틸 --------------------
def _as_col(x, length=None, dtype=float):
    """
    입력을 (n,1) 칼럼 벡터로 강제 변환.
    length가 주어지면 길이 검사.
    """
    arr = np.asarray(x, dtype=dtype).reshape(-1, 1)
    if length is not None and arr.shape != (length, 1):
        raise ValueError(f"Expected shape ({length},1) but got {arr.shape}")
    return arr

def _ensure_u_col(u_like):
    """
    a_rt_profile이 (2,1) 또는 (2,) 또는 list/tuple로 올 수 있으니
    항상 (2,1) 칼럼 벡터로 통일.
    """
    u = np.asarray(u_like, dtype=float).reshape(-1)
    if u.size != 2:
        raise ValueError(f"Control must have 2 components, got {u.size}")
    return u.reshape(2, 1)

# -------------------- 모델 적합 --------------------
def fit_dmd(X1, X2, U):
    """
    DMD: X2 @ pinv([X1; U])
    X1: (3,N), X2:(3,N), U:(2,N)
    return: A(3,3), B(3,2)
    """
    Omega = np.vstack((X1, U))         # (5, N)
    pinvOmega = np.linalg.pinv(Omega)  # (N, 5)
    G = X2 @ pinvOmega                 # (3, 5)
    A = G[:, 0:3]
    B = G[:, 3:]
    return A, B

def fit_edmd_custom(X1, X2, U, dimension=3):
    """
    EDMD_custom: PsiX2 @ pinv([PsiX1; U])
    PsiX* = custom_observ_1(dimension, X*), shape (m, N)
    return: A(m,m), B(m,2), idx_state=(Λ,η,κ)의 관측자 인덱스, m
    """
    PsiX1 = custom_observ_1(dimension, X1)   # (m,N)
    PsiX2 = custom_observ_1(dimension, X2)   # (m,N)
    Omega = np.vstack((PsiX1, U))            # (m+2, N)
    pinvOmega = np.linalg.pinv(Omega)        # (N, m+2)
    G = PsiX2 @ pinvOmega                    # (m, m+2)
    m = G.shape[0]
    A = G[:, :m]
    B = G[:, m:]
    # custom_observ_1이 관측자 첫 3개 요소로 상태(Λ,η,κ)를 그대로 포함한다고 가정
    idx_state = (0, 1, 2)
    return A, B, idx_state, m

# -------------------- 롤아웃 --------------------
def rollout_ground_truth(x0_col, t_end_days):
    """
    Ground Truth: x_{k+1} = RK4(f_over_tau_without_sundman, x_k, u_k, DT_TAU)
    return: t_days (K+1,), X(3,K+1), U_hist(2,K), a_mag(K,), steps K
    """
    x0 = x0_col.flatten()
    t_days = [0.0]
    taus = [0.0]
    X = [x0]
    U_hist = []
    a_mag = []

    steps = 0
    while t_days[-1] < t_end_days:
        xk = X[-1]
        tk = t_days[-1]

        u_k = _ensure_u_col(a_rt_profile(tk))
        U_hist.append(u_k.flatten())

        # RK4 in τ-domain
        x_next = rk4_step(f_over_tau_without_sundman, xk, u_k.flatten(), DT_TAU)
        X.append(x_next)

        # τ 적분
        taus.append(taus[-1] + DT_TAU)

        # t 적분 (Sundman)
        Λ, η, κ = xk
        tprime = sundman_days_per_tau(Λ, η, κ)
        if not np.isfinite(tprime):
            tprime = 0.0
        t_days.append(tk + tprime * DT_TAU)

        a_mag.append(float(np.linalg.norm(u_k)))
        steps += 1

    X = np.array(X).T             # (3, K+1)
    U_hist = np.array(U_hist).T   # (2, K)
    t_days = np.array(t_days)     # (K+1,)
    a_mag = np.array(a_mag)       # (K,)
    return t_days, X, U_hist, a_mag, steps

def rollout_dmd(A, B, x0_col, t_end_days, dimension=3):
    """
    DMD: x_{k+1} = A x_k + B u_k
    """
    t_days = [0.0]
    taus = [0.0]
    X = [x0_col.flatten()]
    U_hist = []
    a_mag = []

    steps = 0
    while t_days[-1] < t_end_days and steps < 200_000:
        xk = _as_col(X[-1], length=dimension)
        tk = t_days[-1]

        u_k = _ensure_u_col(a_rt_profile(tk))
        U_hist.append(u_k.flatten())

        x_next = (A @ xk + B @ u_k).flatten()
        X.append(x_next)

        # τ 적분
        taus.append(taus[-1] + DT_TAU)

        # t 적분
        Λ, η, κ = xk.flatten()
        tprime = sundman_days_per_tau(Λ, η, κ)
        if not np.isfinite(tprime):
            tprime = 0.0
        t_days.append(tk + tprime * DT_TAU)

        a_mag.append(float(np.linalg.norm(u_k)))
        steps += 1

    X = np.array(X).T
    U_hist = np.array(U_hist).T
    t_days = np.array(t_days)
    a_mag = np.array(a_mag)
    return t_days, X, U_hist, a_mag, steps

def rollout_edmd(A, B, x0_col, idx_state, t_end_days, dimension=3):
    """
    EDMD_custom: psi_{k+1} = A psi_k + B u_k, x_k는 관측자에서 추출
    """
    t_days = [0.0]
    taus = [0.0]
    psi0 = custom_observ_1(dimension, x0_col)   # (m,1)
    psi_hist = [psi0]
    idx_lam, idx_eta, idx_kap = idx_state

    lam0 = float(psi0[idx_lam])
    eta0 = float(psi0[idx_eta])
    kap0 = float(psi0[idx_kap])
    X = [np.array([lam0, eta0, kap0], dtype=float)]
    U_hist = []
    a_mag = []

    steps = 0
    while t_days[-1] < t_end_days and steps < 200_000:
        xk = _as_col(X[-1], length=dimension)
        tk = t_days[-1]
        psik = psi_hist[-1]

        u_k = _ensure_u_col(a_rt_profile(tk))
        U_hist.append(u_k.flatten())

        psi_next = A @ psik + B @ u_k
        psi_hist.append(psi_next)

        lam = float(psi_next[idx_lam])
        eta = float(psi_next[idx_eta])
        kap = float(psi_next[idx_kap])
        x_next = np.array([lam, eta, kap], dtype=float)
        X.append(x_next)

        # τ 적분
        taus.append(taus[-1] + DT_TAU)

        # t 적분
        Λ, η, κ = xk.flatten()
        tprime = sundman_days_per_tau(Λ, η, κ)
        if not np.isfinite(tprime):
            tprime = 0.0
        t_days.append(tk + tprime * DT_TAU)

        a_mag.append(float(np.linalg.norm(u_k)))
        steps += 1

    X = np.array(X).T
    U_hist = np.array(U_hist).T
    t_days = np.array(t_days)
    a_mag = np.array(a_mag)
    return t_days, X, U_hist, a_mag, steps

# -------------------- 오차 계산 --------------------
def compute_errors_vs_gt(X_est, X_gt):
    """
    X_est, X_gt: (3, K) 동일 길이
    return: per_state dict(RMSE/MAE), abs_err(3,K), total_norm(K,)
    """
    assert X_est.shape == X_gt.shape
    E = X_est - X_gt
    rmse = np.sqrt(np.mean(E**2, axis=1))
    mae  = np.mean(np.abs(E), axis=1)
    per_state = {
        'Λ': {'RMSE': rmse[0], 'MAE': mae[0]},
        'η': {'RMSE': rmse[1], 'MAE': mae[1]},
        'κ': {'RMSE': rmse[2], 'MAE': mae[2]},
    }
    abs_err = np.abs(E)
    total_norm = np.linalg.norm(E, axis=0)
    return per_state, abs_err, total_norm


def main():
    np.random.seed(42)  # 재현성

    # 1) 데이터셋 생성 (두 방법 공정 비교용)
    X1, X2, U = build_dataset(
        dimension=3,
        N=step_num,
        traj_num=traj_num,
        a_range=(-0.3, 0.3),
        b_range=(-0.3, 0.3),
        c_range=(0.9, 1.0)
    )
    print("데이터셋 생성 완료!")

    # 2) A,B 추정
    A_dmd,  B_dmd  = fit_dmd(X1, X2, U)
    A_edmd, B_edmd, idx_state, m = fit_edmd_custom(X1, X2, U, dimension=3)
    print(f"[DMD] A:{A_dmd.shape}, B:{B_dmd.shape}")
    print(f"[EDMD_custom] A:{A_edmd.shape}, B:{B_edmd.shape}, m={m}")

    # 3) 동일 초기조건
    x0 = _as_col([0.02330563, 0.00867989, 0.9391078], length=3)

    # 4) 각 방식 롤아웃
    t_gt,   X_gt,   U_gt,   a_gt,   k_gt   = rollout_ground_truth(x0, T_END_DAYS)
    t_dmd,  X_dmd,  U_dmd,  a_dmd,  k_dmd  = rollout_dmd(A_dmd,  B_dmd,  x0, T_END_DAYS)
    t_edmd, X_edmd, U_edmd, a_edmd, k_edmd = rollout_edmd(A_edmd, B_edmd, x0, idx_state, T_END_DAYS)

    # 5) 공통 길이로 자르기 (시간축은 GT 기준 사용)
    K = min(X_gt.shape[1], X_dmd.shape[1], X_edmd.shape[1])
    X_gt   = X_gt[:,   :K]
    X_dmd  = X_dmd[:,  :K]
    X_edmd = X_edmd[:, :K]
    t_axis = t_gt[:K]

    # 6) 오차 계산 (각각 GT 기준)
    per_state_dmd, abs_dmd, norm_dmd   = compute_errors_vs_gt(X_dmd,  X_gt)
    per_state_edmd, abs_edmd, norm_edmd = compute_errors_vs_gt(X_edmd, X_gt)

    print("\n=== Ground Truth 대비 오차 지표 ===")
    for name, metrics in [('DMD', per_state_dmd), ('EDMD_custom', per_state_edmd)]:
        print(f"[{name}]")
        for s in ['Λ', 'η', 'κ']:
            print(f"  {s}: RMSE={metrics[s]['RMSE']:.6e}, MAE={metrics[s]['MAE']:.6e}")

    # 7) 플롯
    fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

    labels = [r"$\Lambda$", r"$\eta$", r"$\kappa$"]

    # 상태 비교 (GT vs DMD vs EDMD)
    for i in range(3):
        axs[i].plot(t_axis, X_gt[i, :],   linewidth=2.0, label="Ground Truth")
        axs[i].plot(t_axis, X_dmd[i, :],  linestyle="--", label="DMD")
        axs[i].plot(t_axis, X_edmd[i, :], linestyle="-.", label="EDMD_custom")
        axs[i].set_ylabel(f"{labels[i]}(t)")
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc="best")

    # |Δ| (DMD vs GT)
    axs[3].plot(t_axis, abs_dmd[0, :], label="|ΔΛ| (DMD)")
    axs[3].plot(t_axis, abs_dmd[1, :], label="|Δη| (DMD)")
    axs[3].plot(t_axis, abs_dmd[2, :], label="|Δκ| (DMD)")
    axs[3].set_ylabel("|Δ| DMD")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="best")

    # |Δ| (EDMD vs GT)
    axs[4].plot(t_axis, abs_edmd[0, :], label="|ΔΛ| (EDMD)")
    axs[4].plot(t_axis, abs_edmd[1, :], label="|Δη| (EDMD)")
    axs[4].plot(t_axis, abs_edmd[2, :], label="|Δκ| (EDMD)")
    axs[4].set_ylabel("|Δ| EDMD")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc="best")

    # 전체 L2 오차 (두 방법 비교)
    axs[5].plot(t_axis, norm_dmd,  label=r"$\|X_{DMD}-X_{GT}\|_2$")
    axs[5].plot(t_axis, norm_edmd, label=r"$\|X_{EDMD}-X_{GT}\|_2$")
    axs[5].set_xlabel("t [days] (GT 기준)")
    axs[5].set_ylabel("L2 error")
    axs[5].grid(True, alpha=0.3)
    axs[5].legend(loc="best")

    fig.suptitle(f"Ground Truth vs. DMD vs. EDMD_custom (m={A_edmd.shape[0]}), K={K} steps", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()
    # 저장
    save_dir = os.path.join(os.path.dirname(__file__), "..", "fig")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "compare2.png")
    fig.savefig(save_path, dpi=300)
    print(f"\nFigure saved to: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()