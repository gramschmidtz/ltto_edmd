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
from src.nndmd.network import Encoder, Decoder
from src.nndmd.loss import L_o_x, L_x_x, L_x_o, L_inf, network_L2norm
from src.controllers.test_controller import a_rt_profile
from src.dynamics.config import DT_TAU, step_num, traj_num, T_END_DAYS
from src.dynamics.dynamics_reduced import sundman_days_per_tau

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dimension = 3
    input_dimension = 2
    L = 16
    
    encoder = Encoder(
        input_dim=state_dimension,
        output_dim= L
        ).to(device) # output shape (batch size, L)
    
    A = nn.Parameter(torch.eye(L, device=device)) # shape (L,L)
    B = nn.Parameter(torch.rand(L,input_dimension, device=device))

    decoder = Decoder(
        input_dim=L,
        output_dim=state_dimension,
        ).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + [A] + [B], lr=1e-4)

    X_full, U_full = build_episode_bank(step_num=step_num, traj_num=traj_num, seed=0)

    p = 100
    bs = 128
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
        encoder.train()
        decoder.train()
        
        epoch_loss = 0.0
        for X_batch, U_batch in loader:
            X_batch = X_batch.to(device=device, dtype=torch.float32)
            U_batch = U_batch.to(device=device, dtype=torch.float32)
        

            loss = alpha["alpha1"] * L_o_x(X_batch,          encoder, decoder) + \
                   alpha["alpha2"] * L_x_x(X_batch, U_batch, encoder, decoder, A, B) + \
                   alpha["alpha3"] * L_x_o(X_batch, U_batch, encoder,          A, B)  + \
                   alpha["alpha4"] * L_inf(X_batch, U_batch, encoder, decoder, A, B)  + \
                   alpha["alpha5"] * network_L2norm(encoder) + \
                   alpha["alpha6"] * network_L2norm(decoder)

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
            plt.pause(0.001)
    
    plt.ioff()
    plt.show(block=False)

    save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "A": A.detach().cpu(),
        "B": B.detach().cpu()
    }, os.path.join(save_dir, "nndmd_model.pt"))

    print("✅ 모델이 저장되었습니다 → 'saved_models/nndmd_model.pt'")
    
if __name__ == "__main__":
    main()