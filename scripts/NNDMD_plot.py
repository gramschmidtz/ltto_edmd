# scripts/NNDMD_plot.py
"""
저장된 모델로 그래프를 그리는 코드
"""
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.nndmd.network import NNDMD_Model
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
    model = NNDMD_Model(
        state_dim=state_dimension, 
        input_dim=input_dimension, 
        latent_dim=L
    ).to(device)
    
    # 3. 저장된 모델 불러오기
    save_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", model_name)
    
    if not os.path.exists(save_path):
        print(f"Error: Model file not found at {save_path}")
        return

    checkpoint = torch.load(save_path, map_location=device)
    
    new_state_dict = {}
    for key, value in checkpoint.items():
        # "_orig_mod." 가 있으면 제거, 없으면 그대로 사용
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    
    # 수정된 state_dict로 로드
    model.load_state_dict(new_state_dict)
    model.eval() # 추론 모드

    print("✅ Model loaded successfully (Included: Encoder, Decoder, A, B, u_scale).")
    print(f"   -> Loaded u_scale: {model.u_scale.item():.1e}") # 스케일링 값 확인

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
        psi_curr = model.get_latent(x_tensor) # Shape: (1, L) 유지

    step = 0
    max_steps = 100_000

    print("Starting simulation...")
    
    # 5. 메인 루프 (Latent Space Dynamics + Sundman Transform)
    while t_days[-1] < T_END_DAYS and step < max_steps:
        tk = t_days[-1]

        # 5-1. 제어 입력 계산 (Thrust) # 입력 프로필은 numpy (2,1) 반환

        u_k_np = a_rt_profile(tk).astype(np.float32)
        # u_k_np = random_profile(tk).astype(np.float32)
        # u_k_np = zero_profile(tk).astype(np.float32)

        U_hist.append(u_k_np.flatten())
        
        u_k_tensor = torch.from_numpy(u_k_np.T).to(device) # Shape: (2, 1)

        # 5-2. Latent Space Linear Evolution: psi_{k+1} = A * psi_k + B * u_k
        with torch.no_grad():
            psi_next = model.predict_next(psi_curr, u_k_tensor) # Shape: (1, L)
            
            # 5-3. Physical State 복원
            x_next_tensor = model.decode(psi_next) # Shape: (1, 3)
            
        # Numpy 변환 (Sundman 계산 및 저장을 위해)
        x_next_np = x_next_tensor.cpu().numpy().flatten() # Shape: (3, 1)
        
        # 5-4. 데이터 저장
        # 학습된 모델이 예측한 다음 상태 저장
        X_hist.append(x_next_np.reshape(-1, 1)) # Shape: (3, 1)

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
    main("nndmd_model_normalization.pt")