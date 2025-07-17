import numpy as np
import matplotlib.pyplot as plt
from svd import ApproxSVD
from scipy.sparse.linalg import svds

np.random.seed(42)

def run_tests(num_tests=10, score_fn = 'svd'):
    for i in range(num_tests):
        random_matrix = np.random.rand(6, 20)
        
        u, s, vt = np.linalg.svd(random_matrix)

        p = 4
        approx_svd = ApproxSVD(200, p, score_fn, False)
        traces, ubar, x_approx = approx_svd.fit(random_matrix)

        aux = u.transpose() @ random_matrix @ vt.transpose()
        true_energy = np.trace(aux[:p, :p])

        print(true_energy - traces[-1])
        print()
