# scripts/NNDMD_pseudoinverse_fast.py
"""
NNDMD Fast:
- 손실을 상태공간(MSE(X2_pred, X2))으로 정의 → encoder(x2) 제거(배치당 forward 1회)
- G 재추정은 k epoch마다 1회(Streaming Normal Equations + Tikhonov)
- DataLoader 최적화: num_workers, pin_memory, persistent_workers, prefetch_factor
- AMP 최신 API(torch.amp), torch.compile로 encoder JIT 최적화
- Validation 간격 축소, 플롯 기본 비활성(원하면 켜기)
"""
import os
import time
import matplotlib
matplotlib.use("Agg")  # GUI 렌더링 비활성(Windows CLI에서 느려지는 것 방지)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from src.nndmd.network import Encoder
from src.edmd.make_dataset import build_dataset
from src.dynamics.config import DT_TAU, step_num, traj_num, T_END_DAYS

def main():
    # ===================== 기본 설정 =====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    state_dimension = 3
    input_dimension = 2
    L = 16

    batch_size = 16384          # 여유 있으면 ↑, OOM 나면 8192/4096로 ↓
    num_workers = 4             # Windows에서 2~4 권장, 문제시 0
    pin_memory = True
    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers > 0 else None

    REESTIMATE_G_EVERY = 10     # G 재추정 주기(크게 잡을수록 빠름)
    VAL_EVERY = 10              # 검증 주기(크게 잡을수록 빠름)
    PLOT_EVERY = 100            # 플롯 저장 주기(느리면 크게)
    lambda_reg = 1e-5           # Tikhonov(데이터 스케일 따라 1e-6~1e-3 탐색)
    lr = 2e-4                   # 상태공간 loss로 바꾸면 약간 키워도 됨
    epochs = 1000

    # ===================== 데이터 생성 (CPU 텐서) =====================
    X1, X2, U = build_dataset(
        dimension=state_dimension,
        N=step_num,
        traj_num=traj_num,
        a_range=(-0.3, 0.3),
        b_range=(-0.3, 0.3),
        c_range=(0.9, 1.0),
    )
    X1 = torch.tensor(X1.T, dtype=torch.float32)  # (Ntot, 3)
    X2 = torch.tensor(X2.T, dtype=torch.float32)  # (Ntot, 3)
    U  = torch.tensor(U.T,  dtype=torch.float32)  # (Ntot, 2)
    print("데이터셋 생성 완료!")

    # ===================== Split =====================
    X1_train = X1[: (step_num - 2) * traj_num, :]
    X2_train = X2[: (step_num - 2) * traj_num, :]
    U_train  = U[:  (step_num - 2) * traj_num, :]

    X1_val = X1[(step_num - 2) * traj_num : (step_num - 1) * traj_num, :]
    X2_val = X2[(step_num - 2) * traj_num : (step_num - 1) * traj_num, :]
    U_val  = U[(step_num - 2) * traj_num : (step_num - 1) * traj_num, :]

    X1_test = X1[(step_num - 1) * traj_num :, :]
    X2_test = X2[(step_num - 1) * traj_num :, :]
    U_test  = U[(step_num - 1) * traj_num :, :]

    train_ds = TensorDataset(X1_train, X2_train, U_train)
    val_ds   = TensorDataset(X1_val,   X2_val,   U_val)
    test_ds  = TensorDataset(X1_test,  X2_test,  U_test)

    dl_kwargs = dict(batch_size=batch_size, pin_memory=pin_memory,
                     num_workers=num_workers, persistent_workers=persistent_workers)
    if prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **dl_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    # ===================== 모델/최적화 =====================
    encoder = Encoder(
        input_dim=state_dimension,
        hidden_dim=256,
        output_dim=L,
        num_layers=4,
    ).to(device)

    USE_COMPILE = False  # ← False로 유지

    if USE_COMPILE:
        try:
            encoder = torch.compile(encoder, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            print("[INFO] torch.compile 비활성화 (사유:", type(e).__name__, ")")

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    mse = nn.MSELoss()

    # 누적행렬/항등행렬(디바이스 상주)
    S_oo = torch.zeros(L + input_dimension, L + input_dimension, device=device)
    S_po = torch.zeros(L, L + input_dimension, device=device)
    I_oo = torch.eye(L + input_dimension, device=device)

    best_val = float("inf")
    best_state = None
    loss_hist = []

    # ===================== 학습 루프 =====================
    pbar = tqdm(range(epochs), desc="Training", ncols=100)
    for epoch in pbar:
        t0 = time.perf_counter()

        # ---- (1) G 재추정 (주기적으로만) ----
        if epoch % REESTIMATE_G_EVERY == 0:
            encoder.eval()
            S_oo.zero_(); S_po.zero_()
            with torch.no_grad():
                for x1_b, x2_b, u_b in train_loader:
                    x1_b = x1_b.to(device, non_blocking=True)
                    x2_b = x2_b.to(device, non_blocking=True)
                    u_b  = u_b.to(device,  non_blocking=True)
                    psi_x1_b = encoder(x1_b)                       # (B, L)
                    omega_b  = torch.cat([psi_x1_b, u_b], dim=1)   # (B, L+u)
                    S_oo += omega_b.T @ omega_b
                    # 상태공간 loss이므로 S_po는 'lifted → state'로 맞춰야 함
                    # X2 = (PsiX2_pred[:, :3])을 목표로 두므로,
                    # PsiX2_pred = G @ [PsiX1; U], 그 중 앞 3행(= C @ PsiX2_pred)과 X2 비교
                    # 여기서는 단순화를 위해 G만 구하고, 학습시에 앞 3개를 잘라 비교한다.
                    # S_po는 PsiX2^T @ Omega 대신, 여기선 그대로 PsiX2^T를 쓰지 않고
                    # G를 PsiX2 기준으로 학습하던 공식을 유지한다면 psi_x2_b가 필요하지만
                    # 상태공간 loss에서는 G 추정은 원래 방식(PsiX2 기반)으로 해도 무방.
                    psi_x2_b = encoder(x2_b)                       # (B, L)
                    S_po += psi_x2_b.T @ omega_b                   # (L, L+u)
                G = S_po @ torch.linalg.inv(S_oo + lambda_reg * I_oo)   # (L, L+u)

        t1 = time.perf_counter()

        # ---- (2) Encoder 업데이트 (G 고정) ----
        encoder.train()
        epoch_train_loss = 0.0
        for x1_b, x2_b, u_b in train_loader:
            x1_b = x1_b.to(device, non_blocking=True)
            x2_b = x2_b.to(device, non_blocking=True)
            u_b  = u_b.to(device,  non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                psi_x1_b = encoder(x1_b)                       # (B, L)
                omega_b  = torch.cat([psi_x1_b, u_b], dim=1)   # (B, L+u)
                psi_x2_pred_b = (G @ omega_b.T).T              # (B, L)
                # 상태공간 손실: lifted의 앞 3개를 원래 상태와 비교
                x2_pred_b = psi_x2_pred_b[:, :state_dimension]
                loss = mse(x2_pred_b, x2_b)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item() * x1_b.size(0)

        epoch_train_loss /= len(train_ds)
        t2 = time.perf_counter()

        # ---- (3) Validation (주기적으로만) ----
        if epoch % VAL_EVERY == 0:
            encoder.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for x1_b, x2_b, u_b in val_loader:
                    x1_b = x1_b.to(device, non_blocking=True)
                    x2_b = x2_b.to(device, non_blocking=True)
                    u_b  = u_b.to(device,  non_blocking=True)
                    psi_x1_b = encoder(x1_b)
                    omega_b  = torch.cat([psi_x1_b, u_b], dim=1)
                    psi_x2_pred_b = (G @ omega_b.T).T
                    x2_pred_b = psi_x2_pred_b[:, :state_dimension]
                    epoch_val_loss += mse(x2_pred_b, x2_b).item() * x1_b.size(0)
            epoch_val_loss /= len(val_ds)
            if epoch_val_loss < best_val:
                best_val = epoch_val_loss
                best_state = {
                    "encoder": encoder.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                }
        else:
            epoch_val_loss = float("nan")

        t3 = time.perf_counter()

        loss_hist.append(epoch_train_loss)
        pbar.set_postfix({
            "G": f"{(t1-t0):.2f}s",
            "train": f"{(t2-t1):.2f}s",
            "val": f"{(t3-t2):.2f}s" if epoch % VAL_EVERY == 0 else "-",
            "vLoss": f"{epoch_val_loss:.3e}" if epoch % VAL_EVERY == 0 else "-",
        })

        # 플롯 저장(느리면 주기 크게)
        if epoch % PLOT_EVERY == 0 and epoch > 0:
            plt.figure(figsize=(6,4))
            plt.plot(loss_hist)
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Train Loss (state MSE)")
            plt.tight_layout()
            plt.savefig(f"loss_epoch{epoch}.png", dpi=150)
            plt.close()

    # ===================== Test (best encoder 로드 후 G 재추정) =====================
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
    encoder.eval()

    # Train 전 구간으로 G_best 재추정
    S_oo.zero_(); S_po.zero_()
    with torch.no_grad():
        for x1_b, x2_b, u_b in train_loader:
            x1_b = x1_b.to(device, non_blocking=True)
            x2_b = x2_b.to(device, non_blocking=True)
            u_b  = u_b.to(device,  non_blocking=True)
            psi_x1_b = encoder(x1_b)
            omega_b  = torch.cat([psi_x1_b, u_b], dim=1)
            S_oo += omega_b.T @ omega_b
            psi_x2_b = encoder(x2_b)
            S_po += psi_x2_b.T @ omega_b
        G_best = S_po @ torch.linalg.inv(S_oo + lambda_reg * I_oo)

    # Test 예측
    test_mse = test_mae = 0.0
    with torch.no_grad():
        for x1_b, x2_b, u_b in test_loader:
            x1_b = x1_b.to(device, non_blocking=True)
            x2_b = x2_b.to(device, non_blocking=True)
            u_b  = u_b.to(device,  non_blocking=True)
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

    if best_state is not None:
        save_dir = "./saved_models"
        os.makedirs(save_dir, exist_ok=True)

        encoder_path = os.path.join(save_dir, f"encoder_best_epoch{best_state['epoch']}.pt")
        torch.save({
            "epoch": best_state["epoch"],
            "val_loss": best_state["val_loss"],
            "state_dict": best_state["encoder"],
        }, encoder_path)

        print(f"[INFO] Encoder saved to: {encoder_path}")

if __name__ == "__main__":
    main()