import numpy as np
import copy
np.set_printoptions(suppress=True, precision = 4, linewidth = 200)
# we get a d x N matrix x and a 2x2 matrix m
# columns i and j of x are multiplied with columns 1 and 2 of m
# equivalent to applying a givens rotation on the matrix
def leftMatmul(x, i, j, m):
    # extract columns i and j
    column_i = x[:,i]
    column_j = x[:,j]
    x[:, i] = m[0][0] * column_i + m[1][0] * column_j
    x[:, j] = m[0][1] * column_i + m[1][1] * column_j

def rightMatmul(x, i, j, m):
    row_i = x[i, :]
    row_j = x[j, :]
    x[i, :] = m[0][0] * row_i + m[1][0] * row_j
    x[j, :] = m[0][1] * row_i + m[1][1] * row_j

# first version, using SVD for score
# trueX is d X N
def approx_svd(trueX, p, g, utrue = None):
    d = trueX.shape[0]
    n = trueX.shape[1]
    u = np.identity(d)
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
            # print(i, j)
            # print(s)
            
            # print()
            scores[i][j] = s[0] - x[i][i]
    
    # do sanity check here -> what does scores matrix look like
    ###
    ### scores is alright
    ###
    print(scores)
    for q in range(g):
        # get max score from matrix
        iq, jq = np.unravel_index(np.argmax(scores), scores.shape)
        if jq >= d:
            xji = 0
            xjj = 0  ### here may be xij and xji
        else:
            xji = x[jq][iq]
            xjj = x[jq][jq]
        t = np.array([[x[iq][iq], x[iq][jq]],
                        [xji, xjj]])
        G, _, Ht = np.linalg.svd(t)
        # update working x and u
        # print("\n\nChecking matrix multiplication:")
        # print(iq, jq)
        # print("Initial X:\n", x)
        # print("G:", G)
        # leftMatmul(x, iq, jq, G.transpose()) # equivalent to G.transpose() @ x
        # print("Final X:\n", x)
        # if jq < d:
        #     print("Initial X:\n", x)
        #     print("Gt:", Ht.transpose())
        #     rightMatmul(x, iq, jq, Ht.transpose()) # equivalent to x @ Ht.transpose()
        #     print("Final X:\n", x)
        #     rightMatmul(u, iq, jq, G) # equivalent to u @ G

        print("Initial X:\n", x)
        print("Gt:", Ht.transpose())
        leftMatmul(x, iq, jq, Ht.transpose()) # equivalent to x @ Ht.transpose()
        print("Final X:\n", x)
        leftMatmul(u, iq, jq, Ht.transpose()) # equivalent to u @ G
        if jq < d:
            print("\n\nChecking matrix multiplication:")
            print(iq, jq)
            print("Initial X:\n", x)
            print("G:", G)
            rightMatmul(x, iq, jq, G) # equivalent to G.transpose() @ x
            print("Final X:\n", x)

        # update scores
        for s in range(iq + 1, d):
            t = np.array([[x[iq][iq], x[iq][s]], # do we need to check if s doesnt go out of bounds?
                        [x[s][iq], x[s][s]]])
            _, sig, _ = np.linalg.svd(t)
            scores[iq][s] = sig[0] - x[iq][iq]
        
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
        
        for r in range(iq-1): #? maybe iq instead of iq-1
            t = np.array([[x[r][r], x[r][iq]], # do we need to check if r doesnt go out of bounds?
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
            t = np.array([[x[r][r], x[r][jq]], # do we need to check if r doesnt go out of bounds?
                        [x_jq_r, x_jq_jq]])
            _, sig, _ = np.linalg.svd(t)
            scores[r][jq] = sig[0] - x[r][r]
        
        traces = np.append(traces, np.trace(x))
    
    print(u)
    print(x)
    traces = [f"{n:.5f}".rstrip('0').rstrip('.') for n in traces]
    U, _, _ = np.linalg.svd(u)
    print(U)
    print(traces)

np.random.seed(42)
random_matrix = np.random.rand(48)
random_matrix = np.reshape(random_matrix, (4, 12), order='F')
givens_matrix = np.array([
    [np.cos(np.pi/3.), np.sin(np.pi/3.)],
    [-np.sin(np.pi/3.), np.cos(np.pi/3.)]
])
    
print(random_matrix)
# print(givens_matrix)
# print(leftMatmul(random_matrix, 1, 2, givens_matrix))
#print(random_matrix)
u, s, v = np.linalg.svd(random_matrix)
print(u)
print(s)
print(v)
approx_svd(random_matrix, 3, 5)




            

    