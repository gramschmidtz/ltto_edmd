# scripts/EDMD_legendrepoly.py
"""
EDMD로 A, B fitting하고 rollout해서 그래프까지 그린다.
observables로는 르장드르 텐서곱, 등등
학습 데이터는 src/dynamics/config.py와 src/edmd/make_dataset.py에서 수정 
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.edmd.make_dataset import build_dataset
from src.edmd.observables import legendre_polynomial_basis, flat_index_from_multi, normalization_grid, n_basis_from_order
from src.controllers.test_controller import a_rt_profile
from src.dynamics.config import DT_TAU, step_num, traj_num, T_END_DAYS
from src.dynamics.dynamics_reduced import sundman_days_per_tau

def main():
    order = 1
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

    PsiX1 = legendre_polynomial_basis(order, dimension, X1)
    PsiX2 = legendre_polynomial_basis(order, dimension, X2)

    Omega = np.vstack((PsiX1,U))
    pinvOmega = np.linalg.pinv(Omega)
    G = PsiX2 @ pinvOmega
    
    A = G[:,0:G.shape[0]]
    B = G[:,G.shape[0]:]

    idx_lam = flat_index_from_multi((1,0,0), order, dimension)
    idx_eta = flat_index_from_multi((0,1,0), order, dimension)
    idx_kap = flat_index_from_multi((0,0,1), order, dimension)
    m = n_basis_from_order(order, dimension)
    norm_vec = normalization_grid(order, dimension)

    X = [np.array([[0.02330563], [0.00867989], [0.9391078]])]
    Psi = [legendre_polynomial_basis(order, dimension, X[0])]
    U_hist = []
    t_days = [0.0]
    tau_hist = [0.0]

    step = 0
    while t_days[-1] < T_END_DAYS and step < 100_000:
        # xk = X[-1].flatten()
        psik = Psi[-1]
        tk = t_days[-1]

        u_k = a_rt_profile(tk)
        U_hist.append(u_k.flatten())

        psi_next = A @ psik + B @ u_k
        Psi.append(psi_next)

        lam = (psi_next[idx_lam] / norm_vec[idx_lam]).item()
        eta = (psi_next[idx_eta] / norm_vec[idx_eta]).item()
        kap = (psi_next[idx_kap] / norm_vec[idx_kap]).item()
        X.append(np.array([[lam], [eta], [kap]]))

        tprime = sundman_days_per_tau(lam, eta, kap)
        if not np.isfinite(tprime):
            tprime = 0.0
        t_next = tk + tprime * DT_TAU
        t_days.append(t_next.item())
        tau_hist.append(tau_hist[-1] + DT_TAU)

        step += 1
    
    print(f"Rollout complete: {len(t_days)} steps, final time = {t_days[-1]:.2f} days")

    X = np.hstack(X)        # (3, N)
    U_rt = np.array(U_hist).T  # (2, N-1)
    t_days = np.array(t_days)
    a_mag = np.linalg.norm(U_rt, axis=0)

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(t_days, X[0, :], color="C0")
    axs[0].set_ylabel("Λ(t)")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_days, X[1, :], color="C1")
    axs[1].set_ylabel("η(t)")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t_days, X[2, :], color="C2")
    axs[2].set_ylabel("κ(t)")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(t_days[:-1], a_mag, color="C3")
    axs[3].set_xlabel("t [days]")
    axs[3].set_ylabel("‖a‖ (km/s²)")
    axs[3].grid(True, alpha=0.3)

    fig.suptitle("Reduced States and Control Magnitude up to 890 days (EDMD + Sundman transform)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # save_dir = os.path.join(os.path.dirname(__file__), "..", "fig")
    # os.makedirs(save_dir, exist_ok=True)   # 폴더 없으면 자동 생성
    # save_path = os.path.join(save_dir, "EDMD_result_legendre.png")
    # fig.savefig(save_path, dpi=300)
    # print(f"Figure saved to: {save_path}")
    # plt.close(fig)

def taus():
    order = 1
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

    PsiX1 = legendre_polynomial_basis(order, dimension, X1)
    PsiX2 = legendre_polynomial_basis(order, dimension, X2)

    Omega = np.vstack((PsiX1,U))
    pinvOmega = np.linalg.pinv(Omega)
    G = PsiX2 @ pinvOmega
    
    A = G[:,0:G.shape[0]]
    B = G[:,G.shape[0]:]

    idx_lam = flat_index_from_multi((1,0,0), order, dimension)
    idx_eta = flat_index_from_multi((0,1,0), order, dimension)
    idx_kap = flat_index_from_multi((0,0,1), order, dimension)
    m = n_basis_from_order(order, dimension)

    N = 18_000

    taus = np.linspace(0, (N-1) * DT_TAU, N)
    X = np.zeros((3,N))
    U = np.zeros((2,N))
    a = U[0,:]**2 + U[1,:]**2
    PsiX = np.zeros((m,N))

    X[:,0:1] = np.array([[0.02330563, 0.00867989, 0.9391078]]).T
    PsiX[:,0:1] = legendre_polynomial_basis(order, dimension, X[:,0:1])
    
    for k in range(X.shape[1]-1):
        PsiX[:,k+1:k+2] = (A @ PsiX[:,k:k+1] + B @ U[:,k:k+1])

    X[0,:] = PsiX[idx_lam,:] / normalization_grid(order, dimension)[idx_lam]
    X[1,:] = PsiX[idx_eta,:] / normalization_grid(order, dimension)[idx_eta]
    X[2,:] = PsiX[idx_kap,:] / normalization_grid(order, dimension)[idx_kap]

    # 4) 플롯
    fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    axs[0].plot(taus, X[0,:], color="C0")
    axs[0].set_ylabel("Λ(t)")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(taus, X[1,:], color="C1")
    axs[1].set_ylabel("η(t)")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(taus, X[2,:], color="C2")
    axs[2].set_ylabel("κ(t)")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(taus, a, color="C3")
    axs[3].set_xlabel("tau")
    axs[3].set_ylabel("‖a‖ (km/s²)")
    axs[3].grid(True, alpha=0.3)

    fig.suptitle("Reduced States and Control Magnitude over tau via EDMD")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    # save_dir = os.path.join(os.path.dirname(__file__), "..", "fig")
    # os.makedirs(save_dir, exist_ok=True)   # 폴더 없으면 자동 생성
    # save_path = os.path.join(save_dir, "EDMD_result_legendre.png")
    # fig.savefig(save_path, dpi=300)
    # print(f"Figure saved to: {save_path}")
    # plt.close(fig)

if __name__ == "__main__":
    taus()