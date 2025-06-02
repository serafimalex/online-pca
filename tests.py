import numpy as np
import matplotlib.pyplot as plt
import svd
from scipy.sparse.linalg import svds

np.random.seed(42)

def run_tests(num_tests=10):
    for i in range(num_tests):
        random_matrix = np.random.rand(6, 20)
        
        u, s, vt = np.linalg.svd(random_matrix)

        p = 4
        traces, ubar, x_approx = svd.approx_svd(random_matrix, p, 200)

        aux = u.transpose() @ random_matrix @ vt.transpose()
        true_energy = np.trace(aux[:p, :p])

        print(true_energy - traces[-1])
        print()
