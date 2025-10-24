import numpy as np
from src.edmd.observables import basis, n_basis_from_order, flat_index_from_multi, normalization_grid

def bilinear_dynamics(order,dimension,A,B,x,u):
    """
    Input
    -----
    x: (dimension,1)짜리 어레이로 넣어라
    u: (num_input,1)짜리 어레이로 넣어라

    Retrun
    ------
    x다음번째: (dimension,1)짜리 어레이가 나올 것이다.
    """
    m = n_basis_from_order(order, dimension)
    phi_1 = basis(order,dimension,x.reshape(1,dimension)).T
    
    phi_2 = A @ phi_1 + B @ u
    phi_2 = phi_2 / normalization_grid(order,dimension).reshape(m,1)

    idx_lam = flat_index_from_multi((1,0,0),order,dimension)
    idx_eta = flat_index_from_multi((0,1,0),order,dimension)
    idx_kap = flat_index_from_multi((0,0,1),order,dimension)

    return np.array([[phi_2[idx_lam,0],phi_2[idx_eta,0],phi_2[idx_kap,0]]]).T