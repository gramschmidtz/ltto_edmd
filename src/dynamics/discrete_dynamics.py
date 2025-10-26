# src/dynamics/discrete_dynamics.py
"""
src/dynamics/dynamics_reduced.py기반으로 discrete_system으로 변환 (rk4사용)
"""
import numpy as np

from dynamics.config import DT_TAU
from src.dynamics.dynamics_reduced import f_over_tau_without_sundman

def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5*dt*k1, u)
    k3 = f(x + 0.5*dt*k2, u)
    k4 = f(x + dt*k3, u)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def discrete_dynamics(x, u, dt=DT_TAU):
    x_next = rk4_step(f_over_tau_without_sundman, x, u, dt)
    return np.array([x_next[0],x_next[1],x_next[2]], dtype=float)