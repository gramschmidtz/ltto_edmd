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
    input_mode: str = 'holding' # ['zero', 'random', 'holding'] 중 선택
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    하나의 에피소드(trajectory)를 시뮬레이션한다.

    Args
    ----
    x0: 초기상태벡터
        (state_dim,)
    step_num: 데이터셋 길이
    rng: 난수 생성기
    input_mode: 입력 생성 모드 ('zero', 'random', 'holding')

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

    # ==================== 입력 패턴 선택 로직 ==================== #
    if input_mode == 'zero':
        # 0만 입력 패턴
        U[:] = 0.0

    elif input_mode == 'random':
        # 완전 랜덤 패턴 (White Noise)
        U[:,0] = rng.uniform(0.0, 1.0, size=step_num) * 1e-7
        U[:,1] = rng.uniform(0.0, 1.0, size=step_num) * 1e-7

    elif input_mode == 'holding':
        # 어느정도 유지되는 패턴 (PRBS 유사)
        patternA = np.array([0.0, 0.0])
        patternB = np.array([0.7, 0.7141]) * 1e-7

        current = patternA.copy()       # 시작 패턴
        min_hold = 100                  # 최소 유지 step
        switch_prob = 0.01              # 최소 유지 이후 패턴 전환 확률

        hold_count = 0                  # 현재 패턴 유지 기간

        for k in range(step_num):
            U[k] = current
            hold_count += 1

            # 최소 유지 시간 못 넘었으면 무조건 유지
            if hold_count < min_hold:
                continue
            
            # 최소 유지 시간 지났으면 일정 확률로 패턴 변경
            if rng.random() < switch_prob:
                # 패턴 전환
                if np.allclose(current, patternA):
                    current = patternB
                else:
                    current = patternA

                hold_count = 0  # 카운트 리셋
    
    else:
        raise ValueError(f"지원하지 않는 input_mode입니다: {input_mode}. ('zero', 'random', 'holding' 중 선택)")
    # ========================================================== #

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
    seed: int | None = None,
    input_mode: str = 'holding' # 인자 추가됨
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    전처리가 완벽한 시퀀스 학습 데이터셋 생성.
    """
    rng = np.random.default_rng(seed)

    # shape 초기화
    X_blocks: List[np.ndarray] = []
    U_blocks: List[np.ndarray] = []
    
    out_of_range_cnt = 0

    with tqdm(range(traj_num), desc=f"Simulating trajectories ({input_mode})") as pbar:
        for _ in pbar:
            # 랜덤 초기조건
            a = rng.uniform(*a_range)
            b = rng.uniform(*b_range)
            c = rng.uniform(*c_range)
            x0 = np.array([a, b, c], dtype=float)
            
            # 에피소드 뽑기 (input_mode 전달)
            taus, X, U = simulate_episode(x0, step_num, rng, input_mode=input_mode)

            # 범위 체크
            if (np.any(X < -1) or np.any(X > 1)):
                out_of_range_cnt += 1
                # tqdm.write(f"[Warning] 데이터 범위 초과! x0 = {x0}") # 너무 자주 뜨면 주석 처리
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
    input_mode: str = 'holding' # 인자 추가됨
):
    """
    여러 랜덤 초기조건으로부터 에피소드 뱅크 생성
    """

    rng = np.random.default_rng(seed)
    state_dim = 3
    input_dim = 2

    X_full = np.empty((traj_num, step_num+1, state_dim), dtype=np.float32)
    U_full = np.empty((traj_num, step_num,   input_dim), dtype=np.float32)

    out_of_range_cnt = 0
    for i in tqdm(range(traj_num), desc=f"Simulating episodes ({input_mode})"):
        a = rng.uniform(*a_range)
        b = rng.uniform(*b_range)
        c = rng.uniform(*c_range)
        x0 = np.array([a, b, c], dtype=float)
        
        # input_mode 전달
        taus, X, U = simulate_episode(x0, step_num, rng, input_mode=input_mode)

        if (np.any(X < -1) or np.any(X > 1)):
            out_of_range_cnt += 1
            # tqdm.write(f"[Warning] 데이터 범위 초과! x0 = {x0}")

        X_full[i] = X
        U_full[i] = U

    tqdm.write(f"완료: 범위 초과 traj = {out_of_range_cnt} / {traj_num}")
    X_full = torch.tensor(X_full, dtype=torch.float32)
    U_full = torch.tensor(U_full, dtype=torch.float32)
    return X_full, U_full

class AllWindowsDataset(Dataset):
    """
    궤적 내부의 모든 시작 인덱스를 샘플로 전개
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