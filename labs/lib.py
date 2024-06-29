import numpy as np
import scipy as sp
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import convolve

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

def power_method(A, rtol=5e-3):
    """
    Power method for computing the largest eigenvalue and eigenvector of a matrix.

    Parameters:
    - A: numpy array, the matrix
    - rtol: float, the relative tolerance for convergence (default is 5e-3, which is 0.5%)

    Returns:
    - x_new: numpy array, the largest eigenvector
    - np.linalg.norm(A @ x_new): float, the largest eigenvalue
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('Matrix must be square')
    x = np.random.randn(A.shape[0])
    x = x / np.linalg.norm(x)
    while True:
        x_new = A @ x
        x_new = x_new / np.linalg.norm(x_new)
        if np.linalg.norm(x_new - x) < rtol:
            break
    return x_new, np.linalg.norm(A @ x_new)


class MatchingPursuit:
    """
    Matching Pursuit algorithm for sparse signal recovery.
    """

    class Mode:
        VANILLA = 0
        ORTHOGONAL = 1
        LEAST_SQUARES = 2

    def __init__(
            self, 
            sparsity: int, 
            mode: Mode = Mode.VANILLA, 
            max_iter: int = 100,
            atol: float = None
        ):
        """
        Parameters
        ----------
        mode : int
            Algorithm mode. One of `Mode.VANILLA`, `Mode.ORTHOGONAL`, or `Mode.LEAST_SQUARES`.
        sparsity : int
            Maximum number of non-zero elements in the solution.
        max_iter : int
            Maximum number of iterations, default = 100.
        atol : float
            Absolute tolerance for stopping criterion.
        """
        if mode not in [self.Mode.VANILLA, self.Mode.ORTHOGONAL, self.Mode.LEAST_SQUARES]:
            raise ValueError("Invalid mode")
        self.mode = mode
        self.sparsity = sparsity
        self.max_iter = max_iter
        self.atol = atol

    def fit(self, D: np.ndarray, S: np.ndarray, n_jobs: int = 1):
        """
        Run the algorithm.

        Parameters
        ----------
        D : np.ndarray
            Dictionary matrix (M, N).
        S : np.ndarray
            2D Signal matrix (M, W), or 1D signal vector (M,).
        n_jobs : int
            Number of threads to use for parallel computation (default is single threaded).

        Returns
        -------
        np.ndarray
            Sparse representation of the signal.
        """
        if len(D.shape) != 2:
            raise ValueError("Invalid dictionary matrix shape")
        if D.shape[0] != S.shape[0]:
            raise ValueError("Dictionary and signal matrix must have the same number of rows")
        if n_jobs < 1:
            raise ValueError("Invalid number of threads")
        if len(S.shape) == 1:
            return self.__compute(D, S)
        if len(S.shape) != 2:
            raise ValueError("Invalid signal matrix shape")
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            batch_size = S.shape[1] // n_jobs
            results = []
            for i in range(0, S.shape[1], batch_size):
                results.append(
                    executor.submit(self.__compute_multiple, D, S[:, i:i + batch_size])
                )
            results = [r.result() for r in results]
            return np.hstack(results)
        
    def __compute_multiple(self, D: np.ndarray, S: np.ndarray):
        return np.stack([self.__compute(D, S[:, idx]) for idx in range(S.shape[1])]).T

    def __compute(self, D: np.ndarray, s: np.ndarray):
        """
        Compute the sparse solution.
        """
        self.__preliminar_computations(D)
        if self.mode == self.Mode.VANILLA:
            return self.__compute_vanilla(D, s)
        if self.mode == self.Mode.ORTHOGONAL:
            return self.__compute_orthogonal(D, s)
        if self.mode == self.Mode.LEAST_SQUARES:
            return self.__compute_least_squares(D, s)
        raise ValueError("Invalid mode")

    def __stopping_criterion(self, x: np.ndarray, r: np.ndarray, n_it: int):
        """
        Check if the stopping criterion is satisfied.
        """
        if n_it >= self.max_iter:
            return True
        if np.count_nonzero(x) >= self.sparsity:
            return True
        if self.atol is not None:
            return np.linalg.norm(r) <= self.atol
        return False
    
    def __preliminar_computations(self, D: np.ndarray):
        D /= np.linalg.norm(D, axis=0)
        
    def __compute_vanilla(self, D: np.ndarray, s: np.ndarray):
        """
        Vanilla Matching Pursuit algorithm.
        """
        M, N = D.shape
        x = np.zeros(N)
        r = s.copy()
        n_it = 0
        while not self.__stopping_criterion(x, r, n_it):
            n_it += 1
            sim = np.dot(D.T, r)
            j_star = np.argmax(np.abs(sim))
            x[j_star] += sim[j_star]
            r = s - np.dot(D, x)
        return x
    
    def __compute_orthogonal(self, D: np.ndarray, s: np.ndarray):
        """
        Orthogonal Matching Pursuit algorithm.
        """
        M, N = D.shape
        x = np.zeros(N)
        r = s.copy()
        n_it = 0
        omega = np.zeros(N, dtype=bool)
        while not self.__stopping_criterion(x, r, n_it):
            n_it += 1
            sim = np.dot(D.T, r)
            sim[omega] = 0
            j_star = np.argmax(np.abs(sim))
            omega[j_star] = True
            x_omega = np.linalg.lstsq(D[:, omega], s, rcond=None)[0]
            x[omega] = x_omega
            r = s - np.dot(D, x)
        return x
    
    def __compute_least_squares(self, D: np.ndarray, s: np.ndarray):
        """
        Least Squares Matching Pursuit algorithm.
        """
        M, N = D.shape
        x = np.zeros(N)
        r = s.copy()
        n_it = 0
        omega = []
        while not self.__stopping_criterion(x, r, n_it):
            n_it += 1
            tmp = {}
            for j in [k for k in range(N) if k not in omega]:
                omega_j = omega + [j]
                z, e, _, _ = np.linalg.lstsq(D[:, omega_j], s, rcond=None)
                tmp[j] = (e, z)
            e = np.array([np.inf] * N)
            for j in tmp:
                if len(tmp[j][0]) > 0:
                    e[j] = tmp[j][0].item()
            jStar = np.argmin(e)
            omega.append(jStar)
            x_omega = tmp[jStar][1]
            x = np.zeros(N)
            x[omega] = x_omega
            r = s - D[:, omega] @ x[omega]
        return x
    
class kSVD:
    def __init__(
        self,
        sparsity: int,
        N: int,
        epsilon: float = 1e-3,
        max_iter: int = 10,
        omp_mode: MatchingPursuit.Mode = MatchingPursuit.Mode.VANILLA,
        n_jobs: int = 1
    ):
        """
        Parameters
        ----------
        sparsity : int
            Maximum number of non-zero elements in the solution.
        N : int
            Number of atoms in the dictionary.
        epsilon : float
            Tolerance for stopping criterion.
        max_iter : int
            Maximum number of iterations.
        omp_mode : MatchingPursuit.Mode
            Matching Pursuit algorithm mode.
        n_jobs : int
            Number of threads to use for parallel computation.
        """
        self.sparsity = sparsity
        self.N = N
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.omp_mode = omp_mode
        self.n_jobs = n_jobs
        
    def fit(self, S: np.ndarray):
        """
        Run the algorithm.

        Parameters
        ----------
        S : np.ndarray
            2D Signal matrix (M, W).

        Returns
        -------
        np.ndarray
            Dictionary matrix (M, N).
        np.ndarray
            Sparse representation of the signal (N, W).
        """
        if len(S.shape) != 2:
            raise ValueError("Invalid signal matrix shape")
        return self.__compute(S)
    
    def __compute(self, S: np.ndarray):
        D = S[:, np.random.choice(S.shape[1], self.N)] # Random initialization
        D /= np.linalg.norm(D, axis=0)
        omp = MatchingPursuit(self.sparsity, self.omp_mode, max_iter = 500)
        it = 0
        while True:
            it += 1
            print(f'Iteration {it}')
            X = omp.fit(D, S, n_jobs=self.n_jobs)
            E = S - np.dot(D, X)
            for j in range(self.N):
                if (X[j] == 0).all():
                    continue
                E_j = E + np.outer(D[:, j], X[j])
                u, _, _ = np.linalg.svd(E_j[:, X[j] != 0], full_matrices=False)
                D[:, j] = u[:, 0]
            if it >= self.max_iter or np.linalg.norm(S - np.dot(D, X)) < self.epsilon:
                break
        return D, X

def FISTA(A, b, lmbda, gamma=1e-3, tol=1e-2, max_iter=1000):
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

def IRLS(s, D, lmbda, toll_x=1e-2, max_iter=50):
    x0 = np.zeros(D.shape[1])
    delta = 1e-6
    distanceX = 1e10
    toll_x = 1e-3

    x = x0

    cnt = 0
    while cnt < max_iter and distanceX > toll_x:
        W = np.diag(1 / (np.abs(x) + delta))
        # solve the weighted regularized LS system
        x_new = np.linalg.solve((2 * lmbda * W + D.T @ D), D.T @ s)
        distanceX = np.linalg.norm(x - x_new)
        x = x_new
        cnt = cnt + 1
    return x_new

def MOD(S, N, lmbda, max_iter, verbose=True):
    D = np.random.normal(size=(S.shape[0], N))
    D = D / np.linalg.norm(D, axis=0)
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

def generate_lpa_filter(M: int, N: int, weights: np.ndarray | str = 'uniform'):
    """
    Generate a LPA filter of order N to fit M points.

    Parameters
    ----------
    M : int
        Number of points to fit.
    N : int
        Order of the filter.
    weights : np.ndarray or str
        Weights for the fitting. 
        May also be one of ['uniform', 'left', 'right', 'center'].

    Returns
    -------
    numpy array, the LPA filter
    numpy array, the weights
    """
    if M < N:
        raise ValueError('M must be greater than N')
    if type(weights) == np.ndarray:
        w = weights
    elif weights == 'uniform':
        w = np.ones(M)
    elif weights in ['left', 'right', 'center']:
        if M in [1, 3]:
            w = np.ones(M)
        else:
            # half filter size
            HFS = int((M-1)/2)

            # set the weights. Here weights simply define the window size
            w = np.zeros(M)

            # centered kernel
            wc = w.copy()
            wc[int(HFS/2):-int(HFS/2)] = 1

            # left kernel
            wl = w.copy()
            wl[:HFS+1] = 1

            # right kernel
            wr = w.copy()
            wr[-HFS-1:] = 1
            if weights == 'center':
                w = wc
            elif weights == 'left':
                w = wl
            elif weights == 'right':
                w = wr
    else:
        raise NotImplementedError()
    T = np.vander(1 + np.arange(M), N+1, increasing=True)
    Q, _ = np.linalg.qr(np.diag(w) @ T)
    filter = np.flip((Q @ Q[M // 2]) * w / w[M // 2])
    return filter, w

class LPA_ICI:
    """
    LPA-ICI algorithm for signal denoising.
    """
    def __init__(
        self,
        hmax: int,
        max_degree: int,
        sigma: float,
        Gamma: float = 2.0
    ):
        """
        Parameters
        ----------
        hmax : int
            Maximum window size.
        max_degree : int
            Maximum degree of the polynomial.
        sigma : float
            Standard deviation of the noise.
        Gamma : float
            Regularization parameter. Default is 2.0.
        """
        self.hmax = hmax
        self.max_degree = max_degree
        self.sigma = sigma
        self.Gamma = Gamma

    def fit(self, s: np.ndarray, weights: np.ndarray | str = 'center'):
        """
        Run the algorithm.

        Parameters
        ----------
        s : np.ndarray
            1D signal vector.
        weights : np.ndarray or str
            Weights for the fitting. 
            May also be one of ['left', 'right', 'center'].

        Returns
        -------
        np.ndarray
            Denoised signal.
        np.ndarray
            Best window size for each point.
        """
        # Initialization
        LENGTH = len(s)
        yhat = np.zeros(LENGTH)
        best_scale = np.zeros(LENGTH)
        lower_bounds = - np.ones(LENGTH) * np.inf
        upper_bounds = np.ones(LENGTH) * np.inf
        estimated = np.zeros(LENGTH, dtype=bool)

        # Algorithm
        all_h = np.arange(1, self.hmax + 1)
        for i, h in enumerate(all_h):
            M = 2 * h - 1
            n = M-1 if M < self.max_degree else self.max_degree
            g, _ = generate_lpa_filter(M, n, weights=weights)
            yhat_h = convolve(s, g, mode='same')
            iwidth = self.Gamma * self.sigma * np.linalg.norm(g)
            lbh = yhat_h - iwidth
            ubh = yhat_h + iwidth
            lower_bounds[~estimated] = np.maximum(lower_bounds[~estimated], lbh[~estimated])
            upper_bounds[~estimated] = np.minimum(upper_bounds[~estimated], ubh[~estimated])
            yhat[~estimated] = yhat_h[~estimated]
            best_scale[~estimated] = h
            estimated = lower_bounds >= upper_bounds
        return yhat, best_scale

class RANSAC:
    def __init__(self, eps: float, model, cost, min_samples: int, p_outlier: float = 0.1, p_success: float = 0.99):
        """
        Parameters
        ----------
        eps : float
            Inlier threshold.
        model : Model class
            Has a fit method and a predict method.
        cost : Cost function or one of ['standard', 'MSAC']
            Cost function to evaluate the model.
        min_samples : int
            Minimum number of samples to fit the model.
        p_outlier : float
            Probability of an outlier.
        p_success : float
            Probability of success.
        """
        if eps < 0:
            raise ValueError('eps must be non-negative')
        if min_samples < 1:
            raise ValueError('min_samples must be greater than 0')
        if p_outlier < 0 or p_outlier > 1:
            raise ValueError('p_outlier must be in [0, 1]')
        if p_success < 0 or p_success > 1:
            raise ValueError('p_success must be in [0, 1]')
        self.eps = eps
        self.model = model
        if cost == 'standard':
            self.cost = lambda res: -np.sum(np.abs(res) < eps)
        elif cost == 'MSAC':
            self.cost = lambda res: np.sum(np.minimum(np.abs(res), eps))
        else:
            self.cost = cost
        self.min_samples = min_samples
        self.p_outlier = p_outlier
        self.p_success = p_success

    def fit(self, X, y):
        """
        Run the algorithm.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (n_samples, n_features).
        y : np.ndarray
            1D array of shape (n_samples,).

        Returns
        -------
        Model
            Best fitted model.
        """
        if len(X) < self.min_samples:
            raise ValueError('Not enough samples')
        best_model = None
        best_cost = np.inf
        n_iter = np.log(1 - self.p_success) / np.log(1 - (1 - self.p_outlier) ** self.min_samples)
        for it in range((int)(n_iter)):
            indices = np.random.choice(len(X), self.min_samples, replace=False)
            model = self.model()
            model.fit(X[indices, :], y[indices])
            residuals = y - model.predict(X)
            cost = self.cost(residuals)
            if cost < best_cost:
                best_model = model
                best_cost = cost
        inliers = np.abs(y - best_model.predict(X)) < self.eps
        best_model.fit(X[inliers], y[inliers])
        return best_model, inliers
        
class DLT:
    """
    Direct Linear Transform algorithm.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Run the algorithm.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (n_samples, n_features).
        y : np.ndarray
            1D array of shape (n_samples,).
        """
        A = np.c_[X, np.ones(len(X)), y]
        _, _, VT = np.linalg.svd(A)
        self.coef_ = VT[-1]
    
    def predict(self, X):
        """
        Predict the output.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        return np.dot(np.c_[X, np.ones(len(X))], self.coef_[:-1])


    
class LS:
    """
    Least Squares algorithm.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Run the algorithm.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (n_samples, n_features).
        y : np.ndarray
            1D array of shape (n_samples,).
        """
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
    
    def predict(self, X):
        """
        Predict the output.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        return X @ self.coef_