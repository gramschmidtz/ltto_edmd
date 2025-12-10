# scripts/NNDMD.py
"""
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.nndmd.make_dataset_for_nndmd import AllWindowsDataset, build_episode_bank
from src.nndmd.network import NNDMD_Model, Encoder, Decoder
from src.nndmd.loss import L_o_x, L_x_x, L_x_o, L_inf, network_L2norm
from src.dynamics.config import DT_TAU, step_num, traj_num, T_END_DAYS

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dimension = 3
    input_dimension = 2
    L = 16
    
    model = NNDMD_Model(
        state_dim=state_dimension,
        input_dim=input_dimension,
        latent_dim = L,
        u_scale=1e+7
        ).to(device) # output shape (batch size, L)

    model = torch.compile(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    X_full, U_full = build_episode_bank(step_num=step_num, traj_num=traj_num, input_mode='zero')

    p = 50
    bs = 8096
    dataset = AllWindowsDataset(X_full, U_full, p)
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4
    )
    plt.ion()
    fig1, ax1 = plt.subplots()
    loss_list = []
    epochs = 1000

    alpha = {"alpha1":1.0, "alpha2":1.0, "alpha3":0.3, "alpha4":1e-9, "alpha5":1e-9, "alpha6":1e-9}

    pbar = tqdm(range(epochs), desc="Training", ncols=100)

    for epoch in pbar:
        model.train()
        
        epoch_loss = 0.0
        for X_batch, U_batch in loader:
            X_batch = X_batch.to(device=device, dtype=torch.float32)
            U_batch = U_batch.to(device=device, dtype=torch.float32)
        
            loss = alpha["alpha1"] * L_o_x(X_batch,          model.encoder, model.decoder) + \
                   alpha["alpha2"] * L_x_x(X_batch, U_batch, model.encoder, model.decoder, model.A, model.B) + \
                   alpha["alpha3"] * L_x_o(X_batch, U_batch, model.encoder,                model.A, model.B)  + \
                   alpha["alpha4"] * L_inf(X_batch, U_batch, model.encoder, model.decoder, model.A, model.B)  + \
                   alpha["alpha5"] * network_L2norm(         model.encoder) + \
                   alpha["alpha6"] * network_L2norm(                        model.decoder)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(loader)
        pbar.set_postfix({"loss": f"{mean_loss:.5e}", "epoch": f"{epoch+1}/{epochs}"})
        loss_list.append(mean_loss)

        if (epoch % 5) == 0:
            ax1.clear()
            ax1.plot(np.arange(len(loss_list)), loss_list, label="Loss")
            ax1.set_title("Training Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.grid(True)
            ax1.legend()
            plt.pause(0.0001)
    
    plt.ioff()
    plt.show(block=False)

    save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "nndmd_model_normalization.pt"))

    print("✅ 모델이 저장되었습니다 → 'saved_models/nndmd_model_normalization.pt'")

def additional_train(model_name):
    print("Decoder와 B 행렬만 추가 학습(Fine-tuning)을 시작합니다...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dimension = 3
    input_dimension = 2
    L = 16
    
    # Fine-tuning 시 u_scale을 정의해줍니다 (예: 1e7)
    model = NNDMD_Model(
        state_dim=state_dimension,
        input_dim=input_dimension,
        latent_dim=L,
        u_scale=1e7
    ).to(device)

    # [변경 2] 기존 모델 로드 (구버전 체크포인트 호환)
    # 구버전 .pt 파일은 {'encoder': ..., 'A': ...} 형태의 딕셔너리이므로,
    # 이를 풀어서 NNDMD_Model의 하위 모듈에 넣어줍니다.
    load_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", model_name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"모델 없음: {load_path}")
        
    checkpoint = torch.load(load_path, map_location=device)
    
    # 딕셔너리 키 매핑
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.A.data = checkpoint['A'].to(device)
    model.B.data = checkpoint['B'].to(device) # 학습 전 B값 (보통 Random)
    
    print("✅ 기존 모델 파라미터 로드 완료 into NNDMD_Model")

    # ================= [수정 1] 고정할 파라미터 얼리기 (Freezing) ================= #
    # 통합 모델 내부의 하위 모듈에 접근하여 설정
    
    # 1. Encoder와 A 고정
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.A.requires_grad = False
    
    # 2. Decoder와 B는 학습 활성화
    for param in model.decoder.parameters():
        param.requires_grad = True
    model.B.requires_grad = True
    # ========================================================================= #

    # 데이터 생성
    print("🔄 새로운 입력 패턴으로 데이터 생성 중...")
    X_full, U_full = build_episode_bank(step_num=step_num, traj_num=traj_num, input_mode='holding')
    
    # [Tip] 속도를 위해 GPU로 미리 올리기 (선택사항, 4060 이상 추천)
    X_full = X_full.to(device)
    U_full = U_full.to(device)

    p = 50
    bs = 8096 # 배치 사이즈 최적화
    dataset = AllWindowsDataset(X_full, U_full, p)
    
    # GPU 텐서를 사용하므로 num_workers=0
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)

    # Optimizer (Wrapper 모델의 하위 모듈 전달)
    fine_tune_lr = 5e-5
    optimizer = torch.optim.Adam(
        list(model.decoder.parameters()) + [model.B], 
        lr=fine_tune_lr
    )

    epochs = 500
    loss_list = []
    
    alpha = {"alpha1":1.0, "alpha2":1.0, "alpha3":0.3, "alpha4":1e-9, "alpha6":1e-9} # alpha5 제거 (인코더 파라미터는 고정하므로)

    pbar = tqdm(range(epochs), desc="Training (Decoder & B)", ncols=100)
    
    for epoch in pbar:
        # ================= [수정 2] 모드 설정 ================= #
        model.encoder.eval()   # 고정
        model.decoder.train()  # 학습
        # ==================================================== #
        
        epoch_loss = 0.0
        for X_batch, U_batch in loader:
            # X_batch, U_batch는 이미 GPU에 있음 (위에서 .to(device) 했다면)
            
            # [변경 3] 입력 스케일링
            # Loss 함수는 원본 행렬 A, B를 사용하므로, 입력 U를 스케일링해서 전달해야 함
            U_scaled = U_batch * model.u_scale

            # Loss 계산 (Wrapper의 하위 모듈들을 꺼내서 전달)
            loss = alpha["alpha1"] * L_o_x(X_batch,           model.encoder, model.decoder) + \
                   alpha["alpha2"] * L_x_x(X_batch, U_scaled, model.encoder, model.decoder, model.A, model.B) + \
                   alpha["alpha3"] * L_x_o(X_batch, U_scaled, model.encoder,                model.A, model.B) + \
                   alpha["alpha4"] * L_inf(X_batch, U_scaled, model.encoder, model.decoder, model.A, model.B) + \
                   alpha["alpha6"] * network_L2norm(model.decoder) # 인코더 파라미터는 고정하므로 제외

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(loader)
        pbar.set_postfix({"loss": f"{mean_loss:.5e}"})
        loss_list.append(mean_loss)

    # 저장
    save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    save_path = os.path.join(save_dir, "nndmd_model_finetuned.pt")
    
    # [변경 4] 이제 모델 통째로 저장 (state_dict 하나에 encoder, decoder, A, B, u_scale 다 들어감)
    torch.save(model.state_dict(), save_path)

    print(f"✅ 학습 완료! 저장됨: {save_path}")
    
    plt.figure()
    plt.plot(loss_list)
    plt.title("Fine-tuning Loss (Decoder & B)")
    plt.show()

if __name__ == "__main__":
    main() # 기존 학습
    additional_train("nndmd_model_normalization.pt") # 추가 학습 실행