import numpy as np
import scipy as sp

def dct_matrix(M):
    return np.array([
        (np.sqrt(2 / M) if k != 0 else np.sqrt(1 / M)) * 
            np.cos(k * np.pi * (2 * np.arange(M) + 1) / (2 * M))
        for k in range(M)
    ]).T

def dct2d_matrix(p):
    return np.stack([
        sp.fftpack.idct(sp.fftpack.idct(np.eye(p**2)[k].reshape(p, p).T, norm='ortho').T, norm='ortho').ravel()
        for k in range(p**2)
    ]).T

def psnr(x, y):
    return -10 * np.log10(np.mean(np.square(x - y)))

def hard_th(x, tau):
    return np.where(np.abs(x) > tau, x, 0)

def soft_th(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)

def Lsparse(size, L):
    x = np.random.normal(size=size)
    x[np.random.choice(size, size - L, replace=False)] = 0
    return x

def img_to_patches(img, p, stride=1):
    return np.stack([
        img[i:i+p, j:j+p].ravel()
        for i in range(0, img.shape[0] - p, stride)
        for j in range(0, img.shape[1] - p, stride)
    ], axis=1)

def LSOMP(s, D, L, tau):
    M, N = D.shape
    x_LSOMP = np.zeros(N)
    r = s
    omega = []
    resNorm = np.linalg.norm(r)
    while np.count_nonzero(x_LSOMP) < min(L, M) and resNorm > tau:
        tmp = {}
        for j in [k for k in range(N) if k not in omega]:
            omega_j = omega + [j]
            z, e, _, _ = np.linalg.lstsq(D[:, omega_j], s, rcond=None)
            tmp[j] = (e, z)
        e = np.array([np.inf] * N)
        for j in tmp:
            e[j] = tmp[j][0]
        jStar = np.argmin(e)
        omega.append(jStar)
        x = tmp[jStar][1]
        x_LSOMP = np.zeros(N)
        x_LSOMP[omega] = x

        r = s - D[:, omega] @ x_LSOMP[omega]
        resNorm = np.linalg.norm(r)
    return x_LSOMP

def OMP(s, D, L, tau, normalize_dict=True):
    if normalize_dict:
        D = D / np.linalg.norm(D, axis=0)
    M, N = D.shape
    r = s
    x = np.zeros(N)
    omega = []
    while np.count_nonzero(x) < L and np.linalg.norm(r) > tau:
        j = np.argmax(np.abs(np.dot(D.T, r)))
        omega.append(j)
        x_omega = np.linalg.lstsq(D[:, omega], s, rcond=None)[0]
        x = np.zeros(N)
        x[omega] = x_omega
        r = s - np.dot(D[:, omega], x_omega)
    return x

def OMP_good(s, D, L, min_res_norm=0.1, verbose=True):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for sparse signal recovery.

    Parameters:
    - s: numpy array, the input signal to be recovered
    - D: numpy array, the redundant dictionary matrix
    - L: int, the desired sparsity level
    - min_res_norm: float, optional, the minimum residual norm to stop the algorithm (default is 0.1)
    - verbose: bool, optional, whether to print information logging during the algorithm (default is True)

    Returns:
    - x_OMP: numpy array, the recovered sparse coefficients
    """

    # Initialization
    x_OMP = np.zeros(D.shape[1])    # coefficients
    r = s                           # residual vector
    omega = np.empty(0, dtype=int)  # support set
    res_norm = np.linalg.norm(r)    # norm of the residual vector

    # Main loop
    while np.count_nonzero(x_OMP) < L and res_norm > min_res_norm:
        # Sweep step
        e = np.zeros(D.shape[1])
        for j in range(D.shape[1]):
            e[j] = (res_norm ** 2) - (r.T @ D[:, j]) ** 2

        # Find the column of D that best matches the residual vector
        j_star = np.argmin(e)

        # Update the support set with the j_star coefficient
        omega = np.append(omega, j_star)

        # Update the coefficients by solving the least square problem argmin(||s - D_omega @ x_omega||)
        x_OMP = np.zeros(D.shape[1])
        x_OMP[omega] = np.linalg.lstsq(D[:, omega], s, rcond=None)[0]
        # Or, alternatively:
        # np.linalg.inv(D[:, omega].T @ D[:, omega]) @ D[:, omega].T @ s

        # Update the residual
        r = s - D[:, omega] @ x_OMP[omega]

        # Update the residual norm
        res_norm = np.linalg.norm(r)

        # Information logging
        if verbose:
            print(f'Round {np.count_nonzero(x_OMP) + 1}: j_star = {j_star} with e[j_star] = {e[j_star]}')

    return x_OMP

def FISTA(A, b, lmbda, gamma=1e-3, tol=1e-6, max_iter=1000):
    x = np.zeros((A.shape[1], b.shape[1]))
    alpha = 1
    y = x.copy()
    ATb = A.T @ b
    for _ in range(max_iter):
        x_new = soft_th(y - gamma * A.T @ (A @ y) - ATb, lmbda * gamma)
        alpha_new = (1 + np.sqrt(1 + 4 * alpha**2)) / 2
        y = x_new + (alpha - 1) / alpha_new * (x_new - x)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        alpha = alpha_new
    return x

def IRLS(s, D, lmbda, x0=None):
    if x0 is None:
        x0 = np.zeros(D.shape[1])
    delta = 1e-6
    max_iter = 20
    distanceX = 1e10
    toll_x = 1e-3

    x = x0

    cnt = 0
    while cnt < max_iter or distanceX > toll_x:
        W = np.diag(1 / (np.abs(x) + delta))
        # solve the weighted regularized LS system
        x_new = np.linalg.solve((2 * lmbda * W + D.T @ D), D.T @ s)
        distanceX = np.linalg.norm(x - x_new, ord=2)
        x = x_new
        cnt = cnt + 1
    return x_new

def MOD(S, lmbda, max_iter, verbose=True):
    D = np.random.normal
    for iter in range(max_iter):
        if verbose:
            print(f'Iteration {iter+1}/{max_iter}')
        # perform the sparse coding for all the patches in S
        # for n in range(npatch):
        #     s = S[:, n]
        #     # x =
        #     X[:, n] = x
        X = np.stack([
            IRLS(si, D, lmbda)
            for si in S.T
        ], axis=1)

        # MOD update
        D = np.linalg.lstsq(X.T, S.T, rcond=None)[0].T
        # Or, alternatively:
        # D = S @ X.T @ np.linalg.inv(X @ X.T)

        # normalize the column
        D = D / np.linalg.norm(D, axis=0)
    return D