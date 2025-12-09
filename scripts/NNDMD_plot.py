# scripts/NNDMD_plot.py
"""
저장된 모델로 그래프를 그리는 코드
"""
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.nndmd.network import Encoder, Decoder
from src.controllers.test_controller import a_rt_profile, random_profile, zero_profile
from src.dynamics.config import DT_TAU, T_END_DAYS
from src.dynamics.dynamics_reduced import sundman_days_per_tau

def main(model_name):
    # 1. 설정 및 디바이스 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 파라미터 설정 (학습 코드와 동일하게 맞춰야 함)
    state_dimension = 3
    input_dimension = 2
    L = 16  # 학습 코드의 L 값

    # 2. 모델 초기화
    encoder = Encoder(input_dim=state_dimension, output_dim=L).to(device)
    decoder = Decoder(input_dim=L, output_dim=state_dimension).to(device)
    
    # A와 B는 학습된 파라미터이므로 텐서로 빈 공간을 만들고 나중에 덮어씌웁니다.
    # (학습 코드에서 nn.Parameter로 저장됨)
    
    # 3. 저장된 모델 불러오기
    save_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", model_name)
    
    if not os.path.exists(save_path):
        print(f"Error: Model file not found at {save_path}")
        return

    checkpoint = torch.load(save_path, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    A = checkpoint['A'].to(device)
    B = checkpoint['B'].to(device)
    
    # 추론 모드 전환
    encoder.eval()
    decoder.eval()

    print("Model loaded successfully.")

    # 4. 시뮬레이션 초기화
    # 초기 상태 (EDMD 코드와 동일)
    x_init = np.array([[0.02330563], [0.00867989], [0.9391078]], dtype=np.float32)
    
    # 결과 저장을 위한 리스트
    X_hist = [x_init]
    U_hist = []
    t_days = [0.0]
    tau_hist = [0.0]
    
    # 초기 Latent State 계산 (Encoder 통과)
    # 네트워크는 (Batch, Dim) 입력을 기대하므로 Transpose 후 입력
    x_tensor = torch.from_numpy(x_init.T).to(device)  # Shape: (1, 3)
    
    with torch.no_grad():
        psi_curr = encoder(x_tensor).T  # Shape: (L, 1) 로 다시 변환하여 행렬 연산 준비

    step = 0
    max_steps = 100_000

    print("Starting simulation...")
    
    # 5. 메인 루프 (Latent Space Dynamics + Sundman Transform)
    while t_days[-1] < T_END_DAYS and step < max_steps:
        tk = t_days[-1]

        # 5-1. 제어 입력 계산 (Thrust)
        # a_rt_profile은 numpy (2,1) 반환한다고 가정
        # u_k_np = a_rt_profile(tk).astype(np.float32)
        # u_k_np = a_rt_profile(tk).astype(np.float32)
        u_k_np = zero_profile(tk).astype(np.float32)

        U_hist.append(u_k_np.flatten())
        
        u_k_tensor = torch.from_numpy(u_k_np).to(device) # Shape: (2, 1)

        # 5-2. Latent Space Linear Evolution: psi_{k+1} = A * psi_k + B * u_k
        with torch.no_grad():
            psi_next = torch.mm(A, psi_curr) + torch.mm(B, u_k_tensor) # (L,1) = (L,L)@(L,1) + (L,2)@(2,1)
            
            # 5-3. Physical State 복원 (Decoder)
            # Decoder 입력은 (Batch, L) 형태여야 함 -> Transpose
            x_next_tensor = decoder(psi_next.T) # Shape: (1, 3)
            
        # Numpy 변환 (Sundman 계산 및 저장을 위해)
        x_next_np = x_next_tensor.cpu().numpy().T # Shape: (3, 1)
        
        # 5-4. 데이터 저장
        # 학습된 모델이 예측한 다음 상태 저장
        X_hist.append(x_next_np)

        # 상태 변수 추출 for Sundman
        lam = x_next_np[0].item()
        eta = x_next_np[1].item()
        kap = x_next_np[2].item()

        # 5-5. Sundman Transformation (시간 업데이트)
        tprime = sundman_days_per_tau(lam, eta, kap)
        if not np.isfinite(tprime):
            tprime = 0.0
            
        t_next = tk + tprime * DT_TAU
        
        # 업데이트
        t_days.append(t_next)
        tau_hist.append(tau_hist[-1] + DT_TAU)
        psi_curr = psi_next
        step += 1

    print(f"Rollout complete: {len(t_days)} steps, final time = {t_days[-1]:.2f} days")

    # 6. 데이터 전처리 (Plotting용)
    X_hist_np = np.hstack(X_hist)         # Shape: (3, N)
    U_rt_np = np.array(U_hist).T          # Shape: (2, N-1)
    t_days_np = np.array(t_days)
    
    # 추력 크기 계산 ||a|| = sqrt(ar^2 + at^2)
    a_mag = np.linalg.norm(U_rt_np, axis=0)

    # 7. 그래프 그리기
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Lambda
    axs[0].plot(t_days_np, X_hist_np[0, :], color="C0", label=r"$\Lambda$")
    axs[0].set_ylabel(r"$\Lambda(t)$")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right")

    # Eta
    axs[1].plot(t_days_np, X_hist_np[1, :], color="C1", label=r"$\eta$")
    axs[1].set_ylabel(r"$\eta(t)$")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")

    # Kappa
    axs[2].plot(t_days_np, X_hist_np[2, :], color="C2", label=r"$\kappa$")
    axs[2].set_ylabel(r"$\kappa(t)$")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper right")

    # Control Input Magnitude
    # t_days는 N개, a_mag는 N-1개이므로 t_days[:-1] 사용
    axs[3].plot(t_days_np[:-1], a_mag, color="C3", label=r"$\|\mathbf{a}\|$")
    axs[3].set_xlabel("Time [days]")
    axs[3].set_ylabel(r"$\|\mathbf{a}\|$ ($km/s^2$)")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right")

    fig.suptitle(f"NNDMD Simulation Results (L={L})\nPrediction via Latent Linear Evolution", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main("nndmd_model.pt")