
import numpy as np
import pylab as pl
import time
import sys
import os
sys.path.append(os.path.abspath(".."))  # or "D:\"
from online_psp.online_psp.util import generate_samples, subspace_error
from online_psp.online_psp.ccipca import CCIPCA

print('Testing CCIPCA...')
def explained_variance_ratio(X, X_recon):
    error = np.linalg.norm(X - X_recon, 'fro') ** 2
    total = np.linalg.norm(X, 'fro') ** 2
    return 1 - error / total
# ----------
# Parameters
# ----------
# Number of epochs
n_epoch = 2
# Size of PCA subspace to recover
K = 50
D, N = 500, 1000
# ----------

X, U, sigma2 = generate_samples(K, N, D, method='spiked_covariance', scale_data=True)

# Initial guess
sigma2_0 = 1e-8 * np.ones(K)
Uhat0 = X[:, :K] / np.sqrt((X[:, :K] ** 2).sum(0))

errs = []
ccipca = CCIPCA(K, D, Uhat0=Uhat0, sigma2_0=sigma2_0, cython=True)
print(X.dtype)
time_1 = time.time()
for n_e in range(n_epoch):
    for x in X.T:
        ccipca.fit_next(x)
        errs.append(subspace_error(ccipca.get_components(), U[:, :K]))
        
X_reduced = ccipca.get_components().T @ X
X_recon = ccipca.get_components() @ X_reduced
print(explained_variance_ratio(X,X_recon))
time_2 = time.time() - time_1

# Plotting...
print('Elapsed time: ' + str(time_2))
print('Final subspace error: ' + str(subspace_error(ccipca.get_components(), U[:, :K])))

pl.semilogy(errs)
pl.ylabel('Relative subspace error')
pl.xlabel('Samples (t)')
pl.show()
print('Test complete!')