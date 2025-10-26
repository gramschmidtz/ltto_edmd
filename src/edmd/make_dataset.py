# src/edmd/make_dataset.py
import numpy as np
from tqdm import tqdm
from dynamics.discrete_dynamics import discrete_dynamics
from dynamics.config import DT_TAU, step_num, traj_num
import matplotlib.pyplot as plt

def make_single_dataset(x0, N):
    """
    x0는 벡터로 넣으세요
    N은 데이터셋 길이입니다.
    출력은 (dimension, N) 행렬로 나옴
    """

    taus = np.zeros(N+1)                     # τ 시간축
    X = np.zeros((N+1, 3), dtype=float)      # [Λ, η, κ]
    U = np.random.rand(N, 2) * 1e-7          # [a_r, a_t] 완전 랜덤
    # U = np.zeros((N, 2), dtype=float)

    X[0] = x0
    taus[0] = 0.0

    for k in range(N):
        Λ, η, κ = X[k]
        tau_k = taus[k]

        x_next = discrete_dynamics(X[k], U[k], DT_TAU)

        X[k+1] = x_next
        taus[k+1] = tau_k + DT_TAU

    X_1 = X[0:N,:]
    X_2 = X[1:N+1,:]

    X_1 = X_1.T
    X_2 = X_2.T
    U = U.T

    return X_1, X_2, U

def build_dataset(
    dimension: int,
    N: int,
    traj_num: int,
    a_range: tuple[float, float]= (-0.3,0.3),
    b_range: tuple[float, float] = (-0.3,0.3),
    c_range: tuple[float, float] = (0.9,1.0)
):
    """
    여러 랜덤 초기조건으로부터 전체 데이터셋(X1, X2, U)을 생성한다.
    
    Args
    ----
    dimension : 상태 차원 (예: 3)
    N         : 각 traj의 길이
    traj_num  : traj 개수
    a_range   : a 초기범위 (low, high)
    b_range   : b 초기범위 (low, high)
    c_range   : c 초기범위 (low, high)

    Returns
    -------
    X1_all, X2_all, U_all
      X1_all, X2_all : (dimension, N * traj_num)
      U_all          : (2,         N * traj_num)
    """
    # shape 초기화
    X1_all = np.zeros((dimension, N * traj_num), dtype=float)
    X2_all = np.zeros((dimension, N * traj_num), dtype=float)
    U_all  = np.zeros((2, N * traj_num), dtype=float)

    out_of_range_cnt = 0

    with tqdm(range(traj_num), desc="Building dataset", total=traj_num) as pbar:
            for i in pbar:
                # 랜덤 초기조건
                a = np.random.uniform(*a_range)
                b = np.random.uniform(*b_range)
                c = np.random.uniform(*c_range)
                x0 = np.array([a, b, c], dtype=float)

                # 데이터셋 생성
                X1, X2, U = make_single_dataset(x0, N)

                # 범위 체크
                if (np.any(X1 < -1) or np.any(X1 > 1) or
                    np.any(X2 < -1) or np.any(X2 > 1)):
                    out_of_range_cnt += 1
                    tqdm.write(f"[Warning] 데이터 범위 초과! x0 = {x0}")
                    pbar.set_postfix({"out_of_range": out_of_range_cnt})

                # 가로 이어붙이기
                start = i * N
                end   = (i + 1) * N
                X1_all[:, start:end] = X1
                X2_all[:, start:end] = X2
                U_all[:,  start:end] = U

    tqdm.write(f"완료: 범위 초과 traj = {out_of_range_cnt} / {traj_num}")

    # X1_all, X2_all, U_all = X1_all.T, X2_all.T, U_all.T

    return X1_all, X2_all, U_all

if __name__ == "__main__":
    # 초기조건 유효성 테스트용
    x0_1 = np.array([0.02330563, 0.00867989, 0.9391078], dtype=float)
    x0_3 = np.array([0.4, -0.4, 0.98], dtype=float)
    X1, X2, U = make_single_dataset(x0_3, step_num)
    
    fig, axs = plt.subplots(3,1, figsize=(10,10), sharex=True)
    
    axs[0].plot(X1[0,:], color="C0")
    axs[0].set_ylabel("Λ(τ)")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(X1[1,:], color="C1")
    axs[1].set_ylabel("η(τ)")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(X1[2,:], color="C2")
    axs[2].set_ylabel("κ(τ)")
    axs[2].grid(True, alpha=0.3)

    plt.show()