import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.sparse.linalg import svds
np.set_printoptions(suppress=True, precision = 4, linewidth = 200)

# we get a d x N matrix x and a 2x2 matrix m
# columns i and j of x are multiplied with columns 1 and 2 of m
# equivalent to applying a givens rotation on the matrix
def rightMatmul(x, i, j, m):
    # extract columns i and j
    column_i = copy.deepcopy(x[:,i])
    column_j = copy.deepcopy(x[:,j])
    x[:, i] = m[0][0] * column_i + m[1][0] * column_j
    x[:, j] = m[0][1] * column_i + m[1][1] * column_j

def leftMatmul(x, i, j, m):
    row_i = copy.deepcopy(x[i, :])
    row_j = copy.deepcopy(x[j, :])
    x[i, :] = m[0][0] * row_i + m[0][1] * row_j
    x[j, :] = m[1][0] * row_i + m[1][1] * row_j

# first version, using SVD for score
# trueX is d X N
def approx_svd(trueX, p, g, utrue = None):
    d = trueX.shape[0]
    n = trueX.shape[1]
    u = np.identity(n) ### should be n or d?
    x = copy.deepcopy(trueX)
    traces = np.array([])
    scores = np.zeros((p, n))
    print(d)
    print(n)
    for i in range(p):
        for j in range(i + 1, n):
            if j >= d:
                xji = 0
                xjj = 0  ### here may be xij and xji
            else:
                xji = x[j][i]
                xjj = x[j][j]
            t = np.array([[x[i][i], x[i][j]],
                          [xji, xjj]])
            _, s, _ = np.linalg.svd(t)
            scores[i][j] = s[0] - x[i][i]
    
    print(scores)
    for q in range(g):
        # get max score from matrix
        iq, jq = np.unravel_index(np.argmax(scores), scores.shape)
        if jq >= d:
            xji = 0
            xjj = 0
        else:
            xji = x[jq][iq]
            xjj = x[jq][jq]
        t = np.array([[x[iq][iq], x[iq][jq]],
                        [xji, xjj]])
        G, _, H = np.linalg.svd(t)
        # update working x and u
        rightMatmul(x, iq, jq, H.transpose()) # equivalent to x @ Ht.transpose()
        rightMatmul(u, iq, jq, G) # equivalent to u @ G
        if jq < d:
            leftMatmul(x, iq, jq, G.transpose()) # equivalent to G.transpose() @ x

        # update scores
        for s in range(iq + 1, d):
            t = np.array([[x[iq][iq], x[iq][s]],
                        [x[s][iq], x[s][s]]])
            _, sig, _ = np.linalg.svd(t)
            scores[iq][s] = sig[0] - x[iq][iq]

        # print("After row update:")
        # print(scores)
        
        if jq < p:
            for s in range(jq + 1, n):
                if s >= d:
                    xsjq = 0
                    xss = 0
                else:
                    xsjq = x[s][jq]
                    xss = x[s][s]
                t = np.array([[x[jq][jq], x[jq][s]],
                        [xsjq, xss]])
                _, sig, _ = np.linalg.svd(t)
                scores[jq][s] = sig[0] - x[jq][jq]
        
        for r in range(iq): #? maybe iq instead of iq-1
            t = np.array([[x[r][r], x[r][iq]],
                        [x[iq][r], x[iq][iq]]])
            _, sig, _ = np.linalg.svd(t)
            scores[r][iq] = sig[0] - x[r][r]
        
        for r in range(min(jq, p)):
            if jq >= d:
                x_jq_r = 0
                x_jq_jq = 0
            else:
                x_jq_r = x[jq][r]
                x_jq_jq = x[jq][jq]
            t = np.array([[x[r][r], x[r][jq]],
                        [x_jq_r, x_jq_jq]])
            _, sig, _ = np.linalg.svd(t)
            scores[r][jq] = sig[0] - x[r][r]
        
        traces = np.append(traces,  np.trace(x[:p, :p]))
    
    print(u)
    print(x)
    U, _, _ = np.linalg.svd(u)
    print(U)
    print(traces)
    return traces

np.random.seed(42)
random_matrix = np.random.rand(4, 12)
#random_matrix = np.reshape(random_matrix, (4, 12), order='F')
p = 3

print(random_matrix)
u, s, v = svds(random_matrix, k=p)
print(u)
# print(s)
# print(v)
traces = approx_svd(random_matrix, p, 200)

plt.plot(traces, label='Data')
aux = u.transpose() @ random_matrix @ v.transpose()
true_energy = np.trace(aux[:p, :p])
# Add a dotted horizontal line at y = 2.2
plt.axhline(y=true_energy, color='red', linestyle='dotted', linewidth=2, label='Real U trace')

plt.ylim(bottom=min(0, traces.min()))

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Trace')
plt.title('Trace progression vs real U trace')
plt.legend()

# Show the plot
plt.show()


            

    