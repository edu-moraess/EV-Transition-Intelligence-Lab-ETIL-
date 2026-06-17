import numpy as np
from models.diffusion.bass import BassDiffusion

def test_bass_fit():
    t = np.array([0, 1, 2, 3])
    S = np.array([0, 10, 30, 60])
    model = BassDiffusion()
    M, p, q = model.fit(t, S)
    assert M > 0
