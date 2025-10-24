# src/dynamics/config.py
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Units:
    MU = 1.32712e+11                           # km^3/s^2  (Sun)
    AU = 149_597_870.7                         # km
    LU = 0.9 * AU                              # km
    VU = np.sqrt(MU / LU)                      # km/s
    TU = LU / VU                               # s
    DAY = 86400.0
    
    LU2_over_MU = (LU**2) / MU                 # LU^2 / μ
    sqrt_LU3_over_MU  = np.sqrt((LU**3) / MU)  # √(LU^3/μ) = TU [s]

T_END_DAYS = 890.0
DT_TAU = 1e-3                 # tau 적분스텝
step_num = 10_000             # 학습 스텝 개수
traj_num = 40                 # 학습 궤적 개수