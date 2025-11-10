# scripts/pseudoinverse_encoder_plot.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.nndmd.network import Encoder
from src.edmd.make_dataset import build_dataset
from src.controllers.test_controller import a_rt_profile
from src.dynamics.dynamics_reduced import sundman_days_per_tau
from src.dynamics.config import DT_TAU, step_num, traj_num, T_END_DAYS

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dimension = 3
    input_dimension = 2
    L = 16

    # 1) 인코더 로드
    encoder = Encoder(
        input_dim=state_dimension,
        hidden_dim=256,
        output_dim=L,
        num_layers=4
    ).to(device)
    checkpoint_path = "saved_models/encoder_best_epoch0.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval()
    print(f"Encoder loaded from {checkpoint_path}")

    # 2) 데이터셋 로드 (Torch 텐서, device로)
    X1, X2, U = build_dataset(
        dimension=state_dimension,
        N=step_num,
        traj_num=traj_num,
        a_range=(-0.3, 0.3),
        b_range=(-0.3, 0.3),
        c_range=(0.9, 1.0)
    )
    X1_t = torch.tensor(X1.T, dtype=torch.float32, device=device)  # (N, 3)
    X2_t = torch.tensor(X2.T, dtype=torch.float32, device=device)  # (N, 3)
    U_t  = torch.tensor(U.T,  dtype=torch.float32, device=device)  # (N, 2)

    # 3) Lifted 데이터와 G, A, B 계산
    with torch.no_grad():
        PsiX1 = encoder(X1_t)                 # (N, L)
        PsiX2 = encoder(X2_t)                 # (N, L)
        Omega = torch.cat((PsiX1.T, U_t.T), dim=0)    # (L+u, N)
        pinvOmega = torch.linalg.pinv(Omega)          # (N, L+u)
        G = PsiX2.T @ pinvOmega                       # (L, L+u)
        A = G[:, :L]                                  # (L, L)
        B = G[:, L:]                                  # (L, u=2)

    # 4) 단일 롤아웃 (torch만 사용)
    with torch.no_grad():
        # 초기 상태 x0: (1,3) row → Ψ0: (L,1) column
        x0 = torch.tensor([[0.02330563, 0.00867989, 0.9391078]], dtype=torch.float32, device=device)
        psi_k = encoder(x0).T    # (L,1)

        X_list = [x0.T]  # (3,1) column
        U_hist = []
        t_days = [0.0]
        tau_hist = [0.0]

        step = 0
        while t_days[-1] < T_END_DAYS and step < 100_000:
            tk = t_days[-1]

            # u_k: (2,1) column torch
            u_np = a_rt_profile(tk)                            # np or list
            u_k  = torch.tensor(u_np, dtype=torch.float32, device=device).reshape(2,1)
            U_hist.append(u_k.squeeze(1).detach().cpu().numpy())  # (2,)

            # Ψ_{k+1} = A Ψ_k + B u_k
            psi_next = A @ psi_k + B @ u_k                    # (L,1)
            psi_k = psi_next

            lam = psi_next[0, 0].item()
            eta = psi_next[1, 0].item()
            kap = psi_next[2, 0].item()
            X_list.append(torch.tensor([lam, eta, kap], device=device).reshape(3,1))

            tprime = sundman_days_per_tau(lam, eta, kap)
            if not np.isfinite(tprime):
                tprime = 0.0
            t_next = tk + tprime * DT_TAU
            t_days.append(float(t_next))
            tau_hist.append(tau_hist[-1] + DT_TAU)

            step += 1

    print(f"Rollout complete: {len(t_days)} steps, final time = {t_days[-1]:.2f} days")

    # 5) 플롯을 위해 numpy로 변환
    X_mat = torch.cat(X_list, dim=1).detach().cpu().numpy()
    U_rt  = np.vstack(U_hist).T                             # (2, N-1)
    t_arr = np.array(t_days)
    a_mag = np.linalg.norm(U_rt, axis=0)

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(t_arr, X_mat[0, :]); axs[0].set_ylabel("Λ(t)"); axs[0].grid(True, alpha=0.3)
    axs[1].plot(t_arr, X_mat[1, :]); axs[1].set_ylabel("η(t)"); axs[1].grid(True, alpha=0.3)
    axs[2].plot(t_arr, X_mat[2, :]); axs[2].set_ylabel("κ(t)"); axs[2].grid(True, alpha=0.3)
    axs[3].plot(t_arr[:-1], a_mag);  axs[3].set_xlabel("t [days]"); axs[3].set_ylabel("‖a‖ (km/s²)"); axs[3].grid(True, alpha=0.3)
    fig.suptitle(f"EDMD + Sundman transform, observers = {L}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()