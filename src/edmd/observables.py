# src/edmd/observables.py
import numpy as np
from typing import Tuple, Sequence

def _legendre_table(order: int, x: np.ndarray) -> np.ndarray:
    """
    x: (N,)에서 0..order 르장드르 값을 모두 계산 → (N, order+1)
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    o1 = order + 1
    T = np.empty((N, o1), dtype=float)
    T[:, 0] = 1.0
    if order == 0:
        return T
    T[:, 1] = x
    for n in range(1, order):
        # (n+1)P_{n+1} = (2n+1) x P_n - n P_{n-1}
        T[:, n+1] = ((2*n + 1) * x * T[:, n] - n * T[:, n-1]) / (n + 1)
    return T

def _legendre_table_deriv(order: int, x: np.ndarray) -> np.ndarray:
    """
    x: (N,)에서 0..order 르장드르 도함수 값을 모두 계산 → (N, order+1)
    P0' = 0, P1' = 1, 그리고
    Pn'(x) = n/(x^2-1) * (x Pn(x) - P_{n-1}(x))  for n>=2
    (x=±1에서의 특수처리는 여기선 생략: 일반적으로 [-1,1] 내부에서 사용)
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    o1 = order + 1
    T = _legendre_table(order, x)
    dT = np.empty((N, o1), dtype=float)
    dT[:, 0] = 0.0
    if order == 0:
        return dT
    dT[:, 1] = 1.0
    if order >= 2:
        denom = (x**2 - 1.0)
        # 수치적으로 x=±1에 매우 근접하면 불안정할 수 있음 → 필요시 epsilon 처리 권장
        for n in range(2, order+1):
            dT[:, n] = n/denom * (x * T[:, n] - T[:, n-1])
    return dT

def normalization_grid(order: int, dimension: int) -> np.ndarray:
    """
    정규화 계수 그리드 sqrt( ∏_k (2*n_k+1)/2 ), shape = (o1, o1, ..., o1) (d개 축)
    를 (m,)로 펼쳐 반환. [-1,1]^d (가중치 1)에서 orthonormal.
    """
    o1 = order + 1
    # 인덱스 그리드 만들기: inds.shape = (d, o1, ..., o1)
    inds = np.indices([o1] * dimension, dtype=int)
    # 각 축에 대해 (2*n_k+1)/2 곱한 뒤 sqrt
    coeff = np.ones([o1] * dimension, dtype=float)
    for ax in range(dimension):
        coeff *= (2*inds[ax] + 1) / 2.0
    coeff = np.sqrt(coeff)
    return coeff.reshape(-1)  # (m,)

def legendre_polynomial_basis(order: int, dimension: int, pts: np.ndarray) -> np.ndarray:
    """
    여러 점에서의 정규화된 텐서곱 르장드르 기저값.
    Pts: (dimesion, N)  →  Returns: (m, N),  m = (order+1)^dimension
    순서는 마지막 축이 가장 빠르게 변함.

    Inputs
    ------
        Pts: (dimesion, N) 배열, 각 열이 d차원 점, 한 점을 넣고 싶다면 (d,1) matrix 형태로 넣어야 함
    
    Returns
    -------
        L: (m,N) 배열, 각 열이 해당 점에서의 m개 기저함수 값, 한 점을 받았다면 (m,1) matrix
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] != dimension:
        raise ValueError(f"Pts must have shape ({dimension},N), got {pts.shape}")
    N = pts.shape[1]
    m = n_basis_from_order(order, dimension)

    # 각 축별 (N, o1) 테이블
    tables = [_legendre_table(order, pts[k, :]) for k in range(dimension)]

    # 브로드캐스팅 형태로 곱을 누적 → (N, o1, ..., o1)
    shape_nd = (N,) + (order + 1,)*dimension
    prod_nd = np.ones(shape_nd, dtype=float)
    for k in range(dimension):
        # (N, 1, 1, ..., o1(여기), ..., 1)
        shape = [1]*(1+dimension)
        shape[0] = N
        shape[1+k] = order + 1
        prod_nd *= tables[k].reshape(shape)

    # (m,N)로 펼치기
    L = prod_nd.reshape(m,N)

    # 정규화 계수 곱하기
    norm = normalization_grid(order, dimension)  # (m,)
    L *= norm[:, None]

    return L  # (N, m) # N은 점 개수

def grad_basis(order: int, dimension: int, Pts: np.ndarray) -> np.ndarray:
    """
    여러 점에서의 (정규화된) 텐서곱 르장드르 기저 그래디언트.
    Pts: (N, d) → Returns: (N, m, d)
    순서는 basis_many와 동일 (마지막 축 빠름).
    """
    Pts = np.asarray(Pts, dtype=float)
    if Pts.ndim != 2 or Pts.shape[1] != dimension:
        raise ValueError(f"Pts must have shape (N, {dimension}), got {Pts.shape}")
    N = Pts.shape[0]
    o1 = order + 1
    m = o1 ** dimension

    # 값 테이블, 도함수 테이블 (각각 리스트 길이 d, 원소는 (N, o1))
    T  = [_legendre_table(order,      Pts[:, k]) for k in range(dimension)]
    dT = [_legendre_table_deriv(order, Pts[:, k]) for k in range(dimension)]

    norm = normalization_grid(order, dimension)  # (m,)
    grad = np.empty((N, m, dimension), dtype=float)

    # 각 축 r에 대한 편미분: r축만 dT, 나머지는 T
    for r in range(dimension):
        # 누적 곱 (N, o1, ..., o1)
        shape_nd = (N,) + (o1,)*dimension
        prod_nd = np.ones(shape_nd, dtype=float)
        for k in range(dimension):
            table_k = dT[k] if k == r else T[k]
            shape = [1]*(1+dimension)
            shape[0] = N
            shape[1+k] = o1
            prod_nd *= table_k.reshape(shape)

        Gr = prod_nd.reshape(N, m)
        Gr *= norm[None, :]
        grad[:, :, r] = Gr

    return grad  # (N, m, d) # N은 점 개수

def flat_index_from_multi(indices: Sequence[int], order: int, dimension: int) -> int:
    """
    다차원 인덱스 (i,j,...)를 1차원 인덱스 idx로 변환.
    
    Input
    -----
        indices: 길이 dimension의 튜플/리스트 (i1,i2,...,id)
        order: 최대 차수 n
        dimension: 차원 수 d
    
    Returns
    -------
        idx: 1차원 인덱스 [0, (order+1)^dimension - 1]
    """
    if len(indices) != dimension:
        raise ValueError(f"indices 길이는 {dimension}이어야 합니다. got {len(indices)}")
    o1 = order + 1
    idx = 0
    for d in range(dimension):
        if indices[d] < 0 or indices[d] > order:
            raise ValueError(f"각 인덱스는 [0, {order}] 범위여야 합니다. got {indices[d]} at dim {d}")
        idx = idx * o1 + indices[d]
    return idx

def multi_index_from_flat(idx: int, order: int, dimension: int) -> Tuple[int, ...]:
    """
    1차원 인덱스 idx를 다차원 튜플 인덱스 (i,j,...)로 변환.
    
    Input
    -----
        idx: 1차원 인덱스 [0, (order+1)^dimension - 1]
        order: 최대 차수 n
        dimension: 차원 수 d
    
    Returns
    -------
        indices: 길이 dimension의 튜플 (i1,i2,...,id)
    """
    o1 = order + 1
    m = o1 ** dimension
    if idx < 0 or idx >= m:
        raise ValueError(f"idx는 [0, {m-1}] 범위여야 합니다. got {idx}")
    indices = []
    for d in range(dimension):
        indices.append(idx % o1)
        idx //= o1
    indices.reverse()
    return tuple(indices)

def n_basis_from_order(order: int, dimension: int) -> int:
    """
    주어진 차수와 차원에서의 기저 함수 개수 m = (order+1)^dimension 반환.
    """
    return (order + 1) ** dimension

def custom_observ_1(dimension: int, pts: np.ndarray) -> np.ndarray:
    """
    Pts: (dimesion, N)  →  Returns: (m, N),  m = 관측함수 개수

    Inputs
    ------
        Pts: (dimesion, N) 배열, 각 열이 d차원 점, 한 점을 넣고 싶다면 (d,1) matrix 형태로 넣어야 함
    
    Returns
    -------
        L: (m,N) 배열, 각 열이 해당 점에서의 m개 기저함수 값, 한 점을 받았다면 (m,1) matrix
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] != dimension:
        raise ValueError(f"Pts must have shape ({dimension},N), got {pts.shape}")

    one = np.ones((1,pts.shape[1]), dtype=float)
    P110 = pts[0,:] * pts[1,:]
    P011 = pts[1,:] * pts[2,:]
    P101 = pts[2,:] * pts[0,:]
    P111 = pts[0,:] * pts[1,:] * pts[2,:]
    L = np.vstack((pts,P110,P011,P101,P111))

    P200 = 0.5*(3*pts[0,:]**2 - 1)
    P020 = 0.5*(3*pts[1,:]**2 - 1)
    P002 = 0.5*(3*pts[2,:]**2 - 1)

    L = np.vstack((L, P200*P020, P020*P002, P002*P200, P200*P020*P002))

    # P300 = 0.5*(5*pts[0,:]**3 - 3*pts[0,:])
    # P030 = 0.5*(5*pts[1,:]**3 - 3*pts[1,:])
    # P003 = 0.5*(5*pts[2,:]**3 - 3*pts[2,:])

    # L = np.vstack((L, P300*P030, P030*P003, P003*P300, P300*P030*P003))

    # sin1 = np.sin(pts[0,:])
    # cos1 = np.cos(pts[0,:])
    # sin2 = np.sin(pts[1,:])
    # cos2 = np.cos(pts[1,:])
    # sin3 = np.sin(pts[2,:])
    # cos3 = np.cos(pts[2,:])

    # L = np.vstack((L,
    #     sin1*sin1, sin1*sin2, sin1*sin3, sin1*cos1, sin1*cos2, sin1*cos3,
    #     sin2*sin2, sin2*sin3, sin2*cos1, sin2*cos2, sin2*cos3,
    #     sin3*sin3, sin3*cos1, sin3*cos2, sin3*cos3,
    #     cos1*cos1, cos1*cos2, cos1*cos3,
    #     cos2*cos2, cos2*cos3,
    #     cos3*cos3,
    #     sin1,
    #     cos1,
    #     sin2,
    #     cos2,
    #     sin3,
    #     cos3
    # ))

    return L  # (m, N) # N은 점 개수