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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    X_full, U_full = build_episode_bank(step_num=step_num, traj_num=traj_num)

    p = 50
    bs = 256
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

    print("✅ 모델이 저장되었습니다 → 'saved_models/nndmd_model.pt'")

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     state_dimension = 3
#     input_dimension = 2
#     L = 16
    
#     encoder = Encoder(
#         input_dim=state_dimension,
#         output_dim= L
#         ).to(device) # output shape (batch size, L)
    
#     A = nn.Parameter(torch.eye(L, device=device)) # shape (L,L)
#     B = nn.Parameter(torch.rand(L,input_dimension, device=device))

#     decoder = Decoder(
#         input_dim=L,
#         output_dim=state_dimension,
#         ).to(device)

#     optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + [A] + [B], lr=1e-4)

#     X_full, U_full = build_episode_bank(step_num=step_num, traj_num=traj_num, seed=0)

#     p = 50
#     bs = 128
#     dataset = AllWindowsDataset(X_full, U_full, p)
#     loader = DataLoader(
#         dataset,
#         batch_size=bs,
#         shuffle=True,
#         drop_last=True,
#         pin_memory=True,
#         num_workers=4
#     )
#     plt.ion()
#     fig1, ax1 = plt.subplots()
#     loss_list = []
#     epochs = 1000

#     alpha = {"alpha1":1.0, "alpha2":1.0, "alpha3":0.3, "alpha4":1e-9, "alpha5":1e-9, "alpha6":1e-9}

#     pbar = tqdm(range(epochs), desc="Training", ncols=100)

#     for epoch in pbar:
#         encoder.train()
#         decoder.train()
        
#         epoch_loss = 0.0
#         for X_batch, U_batch in loader:
#             X_batch = X_batch.to(device=device, dtype=torch.float32)
#             U_batch = U_batch.to(device=device, dtype=torch.float32)
        

#             loss = alpha["alpha1"] * L_o_x(X_batch,          encoder, decoder) + \
#                    alpha["alpha2"] * L_x_x(X_batch, U_batch, encoder, decoder, A, B) + \
#                    alpha["alpha3"] * L_x_o(X_batch, U_batch, encoder,          A, B)  + \
#                    alpha["alpha4"] * L_inf(X_batch, U_batch, encoder, decoder, A, B)  + \
#                    alpha["alpha5"] * network_L2norm(encoder) + \
#                    alpha["alpha6"] * network_L2norm(decoder)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#         mean_loss = epoch_loss / len(loader)
#         pbar.set_postfix({"loss": f"{mean_loss:.5e}", "epoch": f"{epoch+1}/{epochs}"})
#         loss_list.append(mean_loss)

#         if (epoch % 5) == 0:
#             ax1.clear()
#             ax1.plot(np.arange(len(loss_list)), loss_list, label="Loss")
#             ax1.set_title("Training Loss")
#             ax1.set_xlabel("Epoch")
#             ax1.set_ylabel("Loss")
#             ax1.grid(True)
#             ax1.legend()
#             plt.pause(0.001)
    
#     plt.ioff()
#     plt.show(block=False)

#     save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
#     os.makedirs(save_dir, exist_ok=True)

#     torch.save({
#         "encoder": encoder.state_dict(),
#         "decoder": decoder.state_dict(),
#         "A": A.detach().cpu(),
#         "B": B.detach().cpu()
#     }, os.path.join(save_dir, "nndmd_model.pt"))

#     print("✅ 모델이 저장되었습니다 → 'saved_models/nndmd_model.pt'")

def additional_train(model_name):
    print("🚀 Decoder와 B 행렬만 추가 학습(Fine-tuning)을 시작합니다...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dimension = 3
    input_dimension = 2
    L = 16
    
    # 모델 생성
    encoder = Encoder(input_dim=state_dimension, output_dim=L).to(device)
    decoder = Decoder(input_dim=L, output_dim=state_dimension).to(device)
    A = nn.Parameter(torch.eye(L, device=device))
    B = nn.Parameter(torch.zeros(L, input_dimension, device=device)) 

    # 모델 로드
    load_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", model_name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"모델 없음: {load_path}")
        
    checkpoint = torch.load(load_path, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    A.data = checkpoint['A'].to(device)
    B.data = checkpoint['B'].to(device) 
    
    print("✅ 기존 모델 로드 완료")

    # ================= [수정 1] 고정할 파라미터 얼리기 (Freezing) ================= #
    # Encoder와 A는 학습되지 않도록 미분 계산을 끕니다.
    for param in encoder.parameters():
        param.requires_grad = False
    A.requires_grad = False
    
    # Decoder와 B는 학습해야 하므로 True (기본값이지만 명시)
    for param in decoder.parameters():
        param.requires_grad = True
    B.requires_grad = True
    # ========================================================================= #

    # 데이터 생성
    print("🔄 새로운 입력 패턴으로 데이터 생성 중...")
    X_full, U_full = build_episode_bank(step_num=step_num, traj_num=traj_num)

    p = 50
    bs = 256
    dataset = AllWindowsDataset(X_full, U_full, p)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

    # Optimizer (Decoder와 B만 포함)
    fine_tune_lr = 5e-5
    optimizer = torch.optim.Adam(
        list(decoder.parameters()) + [B], 
        lr=fine_tune_lr
    )

    epochs = 500
    loss_list = []
    
    # Encoder L2는 고정값이므로 alpha5 제거 가능 (계산 낭비 방지)
    alpha = {"alpha1":1.0, "alpha2":1.0, "alpha3":0.3, "alpha4":1e-9, "alpha6":1e-9}

    pbar = tqdm(range(epochs), desc="Training (Decoder & B only)", ncols=100)
    
    for epoch in pbar:
        # ================= [수정 2] 모드 설정 중요 ================= #
        encoder.eval()   # Encoder는 절대 변하면 안 됨 (Dropout/BatchNorm 고정)
        decoder.train()  # Decoder는 학습해야 함
        # ======================================================== #
        
        epoch_loss = 0.0
        for X_batch, U_batch in loader:
            X_batch = X_batch.to(device=device, dtype=torch.float32)
            U_batch = U_batch.to(device=device, dtype=torch.float32)
            
            # Loss 계산
            # L_o_x: 고정된 Encoder 결과에 맞춰 Decoder가 학습됨
            # L_x_x, L_x_o: 고정된 A와 Encoder 사이에서, B가 입력을 설명하도록 학습됨
            loss = alpha["alpha1"] * L_o_x(X_batch,          encoder, decoder) + \
                   alpha["alpha2"] * L_x_x(X_batch, U_batch, encoder, decoder, A, B) + \
                   alpha["alpha3"] * L_x_o(X_batch, U_batch, encoder,          A, B) + \
                   alpha["alpha4"] * L_inf(X_batch, U_batch, encoder, decoder, A, B) + \
                   alpha["alpha6"] * network_L2norm(decoder) 
                   # alpha5(encoder norm)는 상수이므로 제외

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(loader)
        pbar.set_postfix({"loss": f"{mean_loss:.5e}", "epoch": f"{epoch+1}/{epochs}"})
        loss_list.append(mean_loss)

    # 저장
    save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    save_path = os.path.join(save_dir, "nndmd_model_finetuned.pt")
    
    # 저장할 때는 Encoder와 A도 같이 저장해야 나중에 씁니다 (값은 변하지 않았음)
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "A": A.detach().cpu(),
        "B": B.detach().cpu()
    }, save_path)

    print(f"✅ 학습 완료! 저장됨: {save_path}")
    
    plt.figure()
    plt.plot(loss_list)
    plt.title("Fine-tuning Loss (Decoder & B)")
    plt.show()

if __name__ == "__main__":
    # main() # 기존 학습
    additional_train("nndmd_model.pt") # 추가 학습 실행