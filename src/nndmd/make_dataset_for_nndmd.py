# src/nndmd/make_dataset_for_nndmd.py
import numpy as np
import torch
from typing import Tuple, List
from tqdm import tqdm
from torch.utils.data import Dataset

from src.dynamics.discrete_dynamics import discrete_dynamics
from src.dynamics.config import DT_TAU

def simulate_episode(
    x0: np.ndarray, # (state_dim,)
    step_num: int,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    하나의 에피소드(trajectory)를 시뮬레이션한다.

    Args
    ----
    x0: 초기상태벡터
        (state_dim,)
    step_num: 데이터셋 길이

    Returns
    -------
    taus: τ 시간축
        (step_num+1,) 
    X: 상태 시퀀스
        (step_num+1,state_dim)
    U: 입력 시퀀스
        (step_num,input_dim)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    state_dim = x0.shape[0]
    input_dim = 2

    taus = np.zeros(step_num+1, dtype=float)             # shape (step_num+1,)
    X = np.zeros((step_num+1, state_dim), dtype=float)   # shape (step_num+1,state_dim)
    U = np.zeros((step_num, input_dim), dtype=float)     # shape (step_num,input_dim)

    X[0] = np.asarray(x0, dtype=float)
    taus[0] = 0.0

    U[:,0] = rng.uniform(-1.0,1.0,size=step_num) * 1e-7
    U[:,1] = rng.uniform(-1.0,1.0,size=step_num) * 1e-7

    for k in range(step_num):
        X[k+1] = discrete_dynamics(X[k], U[k], DT_TAU)
        taus[k+1] = taus[k] + DT_TAU

    return taus, X, U # taus:(step_num+1,) # X:(step_num+1,state_dim) # U:(step_num,input_dim)

def episode_to_sequences(
    X: np.ndarray, # (step_num+1,state_dim)
    U: np.ndarray, # (step_num,input_dim)
    p: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    에피소드 하나에서 슬라이딩 윈도우로 시퀀스를 뽑는다

    Args
    ---- 
    X: 상태 시퀀스
        (step_num+1,state_dim)
    U: 입력 시퀀스
        (step_num,input_dim)
    p: 멀티스텝 윈도우 크기
    
    Returns
    -------
    X_seq: (step_num-p+1,p+1,state_dim)
    U_seq: (step_num-p  ,p,  input_dim)
    """
    step_num = X.shape[0]-1
    state_dim = 3
    input_dim = 2
    S = max(0,step_num-p+1)

    X_seq = np.empty((S,p+1,state_dim), dtype=np.float32)
    U_seq = np.empty((S,p  ,input_dim), dtype=np.float32)

    for s in range(S):
        X_seq[s] = X[s:s+p+1]
        U_seq[s] = U[s:s+p]

    return X_seq, U_seq

def build_sequence_dataset(
    step_num: int,
    traj_num: int,
    p: int,
    a_range: tuple[float, float]= (-0.3,0.3),
    b_range: tuple[float, float] = (-0.3,0.3),
    c_range: tuple[float, float] = (0.9,1.0),
    seed: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    전처리가 완벽한 시퀀스 학습 데이터셋 생성.

    여러 랜덤 초기조건으로부터 데이터셋을 생성하고,
    p 윈도우를 이동하며 시퀀스를 뽑아내고,
    이를 모두 텐서로 합쳐서 반환
    
    Args
    ----
    step_num  : 각 traj의 길이
    traj_num  : traj 개수
    p         : 멀티스텝 윈도우 크기
    a_range   : a 초기범위 (low, high)
    b_range   : b 초기범위 (low, high)
    c_range   : c 초기범위 (low, high)
    seed      : 랜덤시드

    Returns
    -------
    X_all: 학습에 사용할 상태 시퀀스 텐서
        torch.float32 텐서
        (total_batch_size, p+1, state_dim) 
    U_all: 학습에 사용할 입력 시퀀스 텐서
        torch.float32 텐서
        (total_batch_size, p,   input_dim)
    """
    rng = np.random.default_rng(seed)

    # shape 초기화
    X_blocks: List[np.ndarray] = []
    U_blocks: List[np.ndarray] = []
    
    out_of_range_cnt = 0

    with tqdm(range(traj_num), desc="Simulating trajectories") as pbar:
        for _ in pbar:
            # 랜덤 초기조건
            a = rng.uniform(*a_range)
            b = rng.uniform(*b_range)
            c = rng.uniform(*c_range)
            x0 = np.array([a, b, c], dtype=float)
            
            # 에피소드 뽑기
            taus, X, U = simulate_episode(x0, step_num, rng)

            # 범위 체크
            if (np.any(X < -1) or np.any(X > 1)):
                out_of_range_cnt += 1
                tqdm.write(f"[Warning] 데이터 범위 초과! x0 = {x0}")
                pbar.set_postfix({"out_of_range": out_of_range_cnt})

            # 에피소드에서 시퀀스 뽑기
            X_seq, U_seq = episode_to_sequences(X, U, p)
            if X_seq.shape[0] == 0:
                continue

            X_blocks.append(X_seq) # (batch_size,p+1,state_dim)
            U_blocks.append(U_seq) # (batch_size,p  ,input_dim)

    if len(X_blocks) == 0:
        raise ValueError("생성된 시퀀스가 없습니다. traj_num 또는 step_num을 늘리세요.")

    tqdm.write(f"완료: 범위 초과 traj = {out_of_range_cnt} / {traj_num}")

    X_np = np.vstack(X_blocks).astype(np.float32, copy=False)
    U_np = np.vstack(U_blocks).astype(np.float32, copy=False)
    X_all = torch.from_numpy(X_np)
    U_all = torch.from_numpy(U_np)

    return X_all, U_all

def build_episode_bank(
    step_num: int,
    traj_num: int,
    a_range=(-0.3,0.3),
    b_range=(-0.3,0.3),
    c_range=(0.9,1.0),
    seed: int | None = None,
):
    """
    여러 랜덤 초기조건으로부터 에피소드 뱅크 생성

    Args
    ----
    step_num: 각 traj 길이
    traj_num: traj 개수
    a_range: a 범위
    b_range: b 범위
    c_range: c 범위
    seed: 랜덤시드

    Returns
    -------
    X_full: 전체 상태 시퀀스 텐서
        (traj_num, step_num+1, state_dim)
    U_full: 전체 입력 시퀀스 텐서
        (traj_num, step_num, input_dim)
    """

    rng = np.random.default_rng(seed)
    state_dim = 3
    input_dim = 2

    X_full = np.empty((traj_num, step_num+1, state_dim), dtype=np.float32)
    U_full = np.empty((traj_num, step_num,   input_dim), dtype=np.float32)

    out_of_range_cnt = 0
    for i in tqdm(range(traj_num), desc="Simulating trajectories (episodes)"):
        a = rng.uniform(*a_range)
        b = rng.uniform(*b_range)
        c = rng.uniform(*c_range)
        x0 = np.array([a, b, c], dtype=float)
        taus, X, U = simulate_episode(x0, step_num, rng)

        if (np.any(X < -1) or np.any(X > 1)):
            out_of_range_cnt += 1
            tqdm.write(f"[Warning] 데이터 범위 초과! x0 = {x0}")

        X_full[i] = X
        U_full[i] = U

    tqdm.write(f"완료: 범위 초과 traj = {out_of_range_cnt} / {traj_num}")
    X_full = torch.tensor(X_full, dtype=torch.float32)
    U_full = torch.tensor(U_full, dtype=torch.float32)
    return X_full, U_full  # (traj_num, step_num+1, state_dim), (traj_num, step_num, input_dim)

class AllWindowsDataset(Dataset):
    """
    궤적 내부의 모든 시작 인덱스를 샘플로 전개

    Args
    ----
    X_full: 전체 상태 시퀀스 텐서
        (traj_num, step_num+1, state_dim)
    U_full: 전체 입력 시퀀스 텐서
        (traj_num, step_num, input_dim)
    p: 멀티스텝 윈도우 크기
    stride: 윈도우 이동 간격
    return_idx: 인덱스도 반환할지 여부
    """
    def __init__(self, X_full, U_full, p, stride=1, return_idx=False):
        self.X_full = X_full
        self.U_full = U_full
        self.p = p
        self.traj_num = X_full.shape[0]
        self.step_num = X_full.shape[1] - 1
        self.stride = stride
        self.return_idx = return_idx

        starts_per_traj = max(0, (self.step_num - p) // self.stride + 1)
        self.index = []
        for traj_idx in range(self.traj_num):
            for k in range(starts_per_traj):
                start = k * self.stride
                self.index.append((traj_idx, start))
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        traj_idx, start = self.index[idx]
        X_seq = self.X_full[traj_idx, start:start+self.p+1,:]
        U_seq = self.U_full[traj_idx, start:start+self.p  ,:]
        X_seq = torch.as_tensor(X_seq, dtype=torch.float32)
        U_seq = torch.as_tensor(U_seq, dtype=torch.float32)
        if self.return_idx:
            return X_seq, U_seq, traj_idx, start
        return X_seq, U_seq