# src/nndmd/loss.py
import torch
import torch.nn as nn

mse = nn.MSELoss(reduction="mean")

def rollout_all_steps(x0, U_seq, encoder, A, B):
    """
    식(14)
    K^[p]Psi_t 로 1...p 스텝 리프트 예측 전부 계산

    Args
    ----
    x0: 시퀀스 시작 상태
        (batch_size, state_dim)
    U_seq: 시퀀스 입력
        (batch_size, p, input_dim)
    encoder: 인코더 네트워크
        (batch_size, state_dim) -> (batch_size, L)
    A: Koopman 연산자 행렬
        (L,L)
    B: 입력 행렬
        (L, input_dim)
        
    Returns
    -------
    phi_seq: 1...p 스텝의 리프트 예측 전부
        (batch_size, p, L)
    """
    device = A.device
    dtype = A.dtype
    
    x0    = x0.to(device=device, dtype=dtype)
    U_seq = U_seq.to(device=device, dtype=dtype)

    batch_size = U_seq.shape[0]
    p = U_seq.shape[1]
    input_dim = U_seq.shape[2]
    L = A.shape[0]

    phi = encoder(x0).to(dtype)  # (batch_size, L)
    phi_seq = torch.empty((batch_size, p, L), device=device, dtype=dtype)

    for i in range(p):
        u_i = U_seq[:,i,:]       # (batch_size, input_dim)
        phi = torch.einsum('ij,bj->bi',A,phi) + torch.einsum('ij,bj->bi',B,u_i)  # (batch_size, L)
        phi_seq[:, i, :] = phi # (batch_size, L)

    return phi_seq # (batch_size, p, L)

def L_o_x(X_batch, encoder, decoder):
    """
    식(17)의 L_o_x 계산
    원복 에러

    Args
    ----
    X_batch: (batch_size, p+1, state_dim)
    """
    device = next(encoder.parameters()).device
    dtype = next(encoder.parameters()).dtype
    state_dim = X_batch.shape[2]

    X = X_batch.to(device=device, dtype=dtype).reshape(-1,state_dim)
    X_recon = decoder(encoder(X))
    loss = mse(X, X_recon)

    return loss

def L_x_x(X_batch, U_batch, encoder, decoder, A, B):
    """
    식(15)

    Args
    ----
    X_batch:
        (batch_size, p+1, state_dim)
    U_batch:
        (batch_size, p,   input_dim)
    encoder: 인코더 네트워크
        (batch_size, state_dim) -> (batch_size, L)
    decoder: 디코더 네트워크
        (batch_size, L) -> (batch_size, state_dim)
    A: Koopman 연산자 행렬
        (L,L)
    B: 입력 행렬
        (L, input_dim)
    p: 예측 스텝 수
    """
    device = A.device
    dtype =  A.dtype
    X_batch = X_batch.to(device=device, dtype=dtype)
    U_batch = U_batch.to(device=device, dtype=dtype)

    batch_size  = X_batch.shape[0]
    p  = X_batch.shape[1] - 1
    state_dim  = X_batch.shape[2]

    x0 = X_batch[:,0,:]   # (batch_size, state_dim)

    phi_seq_pred = rollout_all_steps(x0, U_batch, encoder, A, B)   # (batch_size, p, L)
    x_seq_pred = decoder(phi_seq_pred.reshape(-1, A.shape[0]))     # (batch_size*p, state_dim)
    x_seq_pred = x_seq_pred.view(batch_size, p, state_dim)

    x_seq_true = X_batch[:, 1:, :]                                 # (batch_size, p, state_dim)
    return mse(x_seq_pred, x_seq_true)

def L_x_o(X_batch, U_batch, encoder, A, B):
    """
    식(16)

    Args
    ----
    X_batch:
        (batch_size, p+1, state_dim)
    U_batch:
        (batch_size, p,   input_dim)
    encoder: 인코더 네트워크
        (batch_size, state_dim) -> (batch_size, L)
    A: Koopman 연산자 행렬
        (L,L)
    B: 입력 행렬
        (L, input_dim)
    p: 예측 스텝 수
    """
    device = A.device
    dtype =  A.dtype
    X_batch = X_batch.to(device=device, dtype=dtype)
    U_batch = U_batch.to(device=device, dtype=dtype)
    
    batch_size = X_batch.shape[0]
    p  = X_batch.shape[1] - 1
    state_dim = X_batch.shape[2]

    x0 = X_batch[:,0,:]   # (batch_size, state_dim)

    phi_seq_pred = rollout_all_steps(x0, U_batch, encoder, A, B)     # (batch_size, p, L)
    phi_seq_true = encoder(X_batch[:, 1:, :].reshape(-1, state_dim)) # (batch_size*p, L)
    phi_seq_true = phi_seq_true.view(batch_size, p, -1)

    return mse(phi_seq_pred, phi_seq_true)

def L_inf(X_batch, U_batch, encoder, decoder, A, B):
    """
    논문 식(18): L_infty
    X_batch: (batch_size, p+1, state_dim)
    U_batch: (batch_size, p,   input_dim)
    """
    device = A.device
    dtype = A.dtype
    X_batch = X_batch.to(device=device, dtype=dtype)
    U_batch = U_batch.to(device=device, dtype=dtype)

    batch_size = X_batch.shape[0]
    p = X_batch.shape[1] - 1
    state_dim = X_batch.shape[2]

    # 대상 시퀀스: i=1..p 구간의 GT
    x_true = X_batch[:, 1:, :]  # (B, p, state_dim)

    # (1) 재구성 항  ||x_i - decoder(encoder(x_i))||_inf
    x_recon = decoder(encoder(x_true.reshape(-1, state_dim))).reshape(batch_size, p, state_dim)
    err_recon = torch.abs(x_true - x_recon)                     # (B,p,dim)
    inf_recon = torch.amax(err_recon, dim=-1)                   # (B,p)
    term_recon = inf_recon.mean()                               # 배치/시간 평균

    # (2) p-스텝 예측 항  ||x_i - decoder(K^[i] Psi_0)||_inf
    x0 = X_batch[:, 0, :]                                       # (B, state_dim)
    phi_seq_pred = rollout_all_steps(x0, U_batch, encoder, A, B)  # (B,p,L)
    x_pred = decoder(phi_seq_pred.reshape(-1, A.shape[0])).reshape(batch_size, p, state_dim)
    err_pred = torch.abs(x_true - x_pred)                       # (B,p,dim)
    inf_pred = torch.amax(err_pred, dim=-1)                     # (B,p)
    term_pred = inf_pred.mean()                                 # 배치/시간 평균

    return term_recon + term_pred

def network_L2norm(module: nn.Module, include_bias: bool=False):
    """
    module 내부 가중치의 L2 제곱합(∑ w^2) 반환.
    기본적으로 bias는 제외.
    """
    total = torch.zeros((), device=next(module.parameters()).device)
    for name, p in module.named_parameters():
        if not include_bias and name.endswith("bias"):
            continue
        # 보통 weight만 포함하고 싶다면 아래처럼 더 좁게 걸러도 됨:
        # if not name.endswith("weight"): continue
        total = total + p.pow(2).sum()
    return total