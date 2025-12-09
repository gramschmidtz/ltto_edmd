# scripts/verify_data.py
"""
"""
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.nndmd.make_dataset_for_nndmd import AllWindowsDataset, build_episode_bank
from src.dynamics.config import DT_TAU, step_num, traj_num, T_END_DAYS

def verify_data():
    print("=== 1. Raw Data Generation Check ===")
    # 데이터 생성
    X_full, U_full = build_episode_bank(step_num=step_num, traj_num=traj_num, seed=0)
    
    # 1-1. Shape 확인
    # 예상: X_full이 (traj_num, step_num, state_dim) 혹은 (total_steps, state_dim) 인지 확인 필요
    if isinstance(X_full, list):
        X_full = np.array(X_full)
    if isinstance(U_full, list):
        U_full = np.array(U_full)
        
    print(f"X_full Shape: {X_full.shape}")
    print(f"U_full Shape: {U_full.shape}")
    
    # 1-2. NaN/Inf 값 확인 (물리 시뮬레이션 발산 여부)
    if np.isnan(X_full).any() or np.isinf(X_full).any():
        print("❌ 경고: State 데이터에 NaN 또는 Inf가 포함되어 있습니다!")
    else:
        print("✅ State 데이터에 NaN/Inf 없음.")

    # 1-3. 초기 궤적 시각화 (물리적 거동이 말이 되는지 확인)
    # 데이터가 (Traj, Time, Dim) 형태라고 가정하고 첫 번째 궤적만 그립니다.
    # 만약 (Time, Dim) 형태라면 슬라이싱을 조절해야 합니다.
    plt.figure(figsize=(10, 6))
    if len(X_full.shape) == 3: # (Traj, Time, Dim)
        traj_to_plot = X_full[0]
        control_to_plot = U_full[0]
    else: # (Total_Time, Dim) -> 단순히 앞부분 1000스텝만
        traj_to_plot = X_full[:1000]
        control_to_plot = U_full[:1000]

    plt.subplot(2, 1, 1)
    plt.plot(traj_to_plot)
    plt.title("Sample Trajectory (State)")
    plt.legend([f"State {i}" for i in range(traj_to_plot.shape[1])])
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(control_to_plot)
    plt.title("Sample Control Input")
    plt.legend([f"Input {i}" for i in range(control_to_plot.shape[1])])
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() # 그래프가 뜨면 궤적이 부드럽게 이어지는지 확인하세요.

    print("\n=== 2. Dataset & DataLoader Window Check ===")
    p = 40 # NNDMD.py에서 사용한 윈도우 크기
    dataset = AllWindowsDataset(X_full, U_full, p)
    loader = DataLoader(dataset, batch_size=4, shuffle=True) # 셔플 True로 랜덤 샘플 확인

    # 배치 하나를 꺼내서 확인
    X_batch, U_batch = next(iter(loader))
    
    print(f"Batch X Shape: {X_batch.shape}") # 예상: (Batch, p, state_dim) 혹은 (Batch, state_dim) -> Dataset 구현에 따라 다름
    print(f"Batch U Shape: {U_batch.shape}")
    
    # 2-1. 윈도우 연속성 확인 (가장 중요)
    # NNDMD 학습을 위해서는 배치의 각 샘플이 연속된 시간(t, t+1, ...)이어야 합니다.
    # Dataset이 (Batch, Window_Size, Dim)을 반환한다고 가정합니다.
    if len(X_batch.shape) == 3: 
        sample_idx = 0
        print(f"\n[Sample {sample_idx} in Batch] First 5 steps comparison:")
        for i in range(4):
            # 시각적으로 값이 부드럽게 변하는지 확인 (갑자기 0으로 튀거나 끊기지 않는지)
            print(f"t={i}: {X_batch[sample_idx, i].numpy()}")
    else:
        print("Dataset이 Window 전체를 반환하지 않고 단일 스텝만 반환하는 구조일 수 있습니다. 확인이 필요합니다.")

if __name__ == "__main__":
    verify_data()