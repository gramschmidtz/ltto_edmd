# scripts/NNDMD_pseudoinverse.py
"""
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from src.nndmd.network import Encoder
from src.edmd.make_dataset import build_dataset
from src.controllers.test_controller import a_rt_profile
from src.dynamics.config import DT_TAU, step_num, traj_num, T_END_DAYS
from src.dynamics.dynamics_reduced import sundman_days_per_tau

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dimension = 3
    input_dimension = 2
    L = 16
    
    X1, X2, U = build_dataset( # X1 shape (state_dimension, step_num*traj_num), X2 shape (state_dimension, step_num*traj_num), U shape (input_dimension, step_num*traj_num)
    dimension = state_dimension,
    N = step_num,
    traj_num = traj_num,
    a_range = (-0.3,0.3),
    b_range = (-0.3,0.3),
    c_range = (0.9,1.0)
    )

    X1 = torch.tensor(X1.T, dtype=torch.float32, device=device) # shape (step_num*traj_num, state_dimension)
    X2 = torch.tensor(X2.T, dtype=torch.float32, device=device) # shape (step_num*traj_num, state_dimension)
    U = torch.tensor(U.T, dtype=torch.float32, device=device)   # shape (step_num*traj_num, input_dimension)
    
    print("데이터셋 생성 완료!")

    X1_train = X1[:(step_num-2)*traj_num,:]
    X2_train = X2[:(step_num-2)*traj_num,:]
    U_train = U[:(step_num-2)*traj_num,:]

    X1_val = X1[(step_num-2)*traj_num:(step_num-1)*traj_num,:]
    X2_val = X2[(step_num-2)*traj_num:(step_num-1)*traj_num,:]
    U_val = U[(step_num-2)*traj_num:(step_num-1)*traj_num,:]

    X1_test = X1[(step_num-1)*traj_num:,:]
    X2_test = X2[(step_num-1)*traj_num:,:]
    U_test = U[(step_num-1)*traj_num:,:]    

    encoder = Encoder(
        input_dim=state_dimension,
        hidden_dim=256,
        output_dim= L, # 앞에 3개는 원래 상태 변수
        num_layers=4
        ).to(device) # output shape (batch size, L)
    
    optimizer = torch.optim.Adam(list(encoder.parameters()), lr=1e-4)
    
    plt.ion()
    fig1, ax1 = plt.subplots()
    loss_list = []
    epochs = 1000

    pbar = tqdm(range(epochs), desc="Training", ncols=100)

    batch_size = 8192  # GPU 메모리에 맞게 조절 (4096~32768 사이 탐색 추천)

    train_ds = TensorDataset(X1_train, X2_train, U_train)
    val_ds   = TensorDataset(X1_val,   X2_val,   U_val)
    test_ds  = TensorDataset(X1_test,  X2_test,  U_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    mse = nn.MSELoss()
    lambda_reg = 1e-6  # Tikhonov 정규화 (수치 안정성)

    best_val = float("inf")
    best_state = None

    for epoch in pbar:
        # ===== 1) G 추정 (no-grad + streaming) =====
        encoder.eval()
        with torch.no_grad():
            # 누적 행렬 초기화 (작은 행렬이라 GPU/CPU 아무 데나 OK)
            S_oo = torch.zeros(L + input_dimension, L + input_dimension, device=device)
            S_po = torch.zeros(L, L + input_dimension, device=device)

            for x1_b, x2_b, u_b in train_loader:
                psi_x1_b = encoder(x1_b).detach()   # (B, L)
                psi_x2_b = encoder(x2_b).detach()   # (B, L)
                omega_b  = torch.cat([psi_x1_b, u_b], dim=1)  # (B, L+u)

                # 정상방정식 누적
                S_oo += omega_b.T @ omega_b                     # (L+u, L+u)
                S_po += psi_x2_b.T @ omega_b                    # (L,   L+u)

            # 정규화 추가 후 역행렬
            S_oo = S_oo + lambda_reg * torch.eye(L + input_dimension, device=device)
            G = S_po @ torch.linalg.inv(S_oo)   # (L, L+u)

        # ===== 2) encoder 업데이트 (G 고정) =====
        encoder.train()
        epoch_train_loss = 0.0
        for x1_b, x2_b, u_b in train_loader:
            psi_x1_b = encoder(x1_b)                          # (B, L)
            omega_b  = torch.cat([psi_x1_b, u_b], dim=1)      # (B, L+u)
            psi_x2_pred_b = (G @ omega_b.T).T                 # (B, L)

            psi_x2_b = encoder(x2_b)                          # (B, L)
            loss = mse(psi_x2_pred_b, psi_x2_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x1_b.size(0)

        epoch_train_loss /= len(train_ds)

        # ===== 3) Validation (G 고정, no-grad) =====
        encoder.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x1_b, x2_b, u_b in val_loader:
                psi_x1_b = encoder(x1_b)
                omega_b  = torch.cat([psi_x1_b, u_b], dim=1)
                psi_x2_pred_b = (G @ omega_b.T).T
                x2_pred_b = psi_x2_pred_b[:, :state_dimension]
                epoch_val_loss += mse(x2_pred_b, x2_b).item() * x1_b.size(0)

        epoch_val_loss /= len(val_ds)

        pbar.set_postfix({'train': f'{epoch_train_loss:.3e}', 'val': f'{epoch_val_loss:.3e}'})
        loss_list.append(epoch_train_loss)
        if epoch % 50 == 0:
            ax1.clear()
            ax1.plot(loss_list)
            ax1.set_yscale('log')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Train Loss')
            plt.pause(0.01)

        if epoch_val_loss < best_val:
            best_val = epoch_val_loss
            best_state = {'encoder': encoder.state_dict(), 'epoch': epoch, 'val_loss': best_val}

    # ===== 4) Test =====
    encoder.load_state_dict(best_state['encoder'])
    encoder.eval()

    # best encoder로 G 재추정 (no-grad + streaming)
    with torch.no_grad():
        S_oo = torch.zeros(L + input_dimension, L + input_dimension, device=device)
        S_po = torch.zeros(L, L + input_dimension, device=device)
        for x1_b, x2_b, u_b in train_loader:
            psi_x1_b = encoder(x1_b)
            psi_x2_b = encoder(x2_b)
            omega_b  = torch.cat([psi_x1_b, u_b], dim=1)
            S_oo += omega_b.T @ omega_b
            S_po += psi_x2_b.T @ omega_b
        S_oo = S_oo + lambda_reg * torch.eye(L + input_dimension, device=device)
        G_best = S_po @ torch.linalg.inv(S_oo)

    # Test 예측
    test_mse = test_mae = 0.0
    with torch.no_grad():
        for x1_b, x2_b, u_b in test_loader:
            psi_x1_b = encoder(x1_b)
            omega_b  = torch.cat([psi_x1_b, u_b], dim=1)
            psi_x2_pred_b = (G_best @ omega_b.T).T
            x2_pred_b = psi_x2_pred_b[:, :state_dimension]

            test_mse += torch.mean((x2_pred_b - x2_b) ** 2, dim=0).sum().item() * x1_b.size(0) / state_dimension
            test_mae += torch.mean(torch.abs(x2_pred_b - x2_b), dim=0).sum().item() * x1_b.size(0) / state_dimension

    test_mse /= len(test_ds)
    test_rmse = test_mse ** 0.5
    test_mae /= len(test_ds)

    print(f"[TEST] MSE={test_mse:.3e} | RMSE={test_rmse:.3e} | MAE={test_mae:.3e}")
    
if __name__ == "__main__":
    main()