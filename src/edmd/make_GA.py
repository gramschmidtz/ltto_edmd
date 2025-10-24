# src/edmd/make_GA.py
import numpy as np
from dynamics.config import step_num
from edmd.observables import basis
from edmd.make_dataset import make_single_dataset

X1, X2 = make_single_dataset()
print(X1.shape)
x0 = np.array([[0.02330563, 0.00867989, 0.9391078]], dtype=float)
print(basis(3,3,x0).shape)
