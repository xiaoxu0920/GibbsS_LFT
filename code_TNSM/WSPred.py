import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import time


# 设置matplotlib中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Bitstream Vera Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


@jit(nopython=True, fastmath=True)
def calculate_average_qos_numba(Y, mask):
    m, n, c = Y.shape
    Y_avg = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            sum_val = 0.0
            count = 0
            for k in range(c):
                if mask[i, j, k] == 1:
                    sum_val += Y[i, j, k]
                    count += 1
            if count > 0:
                Y_avg[i, j] = sum_val / count
    return Y_avg

@jit(nopython=True, fastmath=True)
def update_latent_features_numba(U, S, T, Y, mask, Y_avg, lambda1, lambda2, lambda3, eta, learning_rate):
    m, n, c = Y.shape
    l = U.shape[1]
    for i in range(m):
        for j in range(n):
            for k in range(c):
                if mask[i, j, k] == 1:
                    Y_pred = 0.0
                    for f in range(l):
                        Y_pred += U[i, f] * S[j, f] * T[k, f]

                    error = Y_pred - Y[i, j, k]
                    avg_error = Y_pred - Y_avg[i, j]

                    for f in range(l):
                        grad_U = (error * S[j, f] * T[k, f] + 
                                 lambda1 * U[i, f] +
                                 eta * avg_error * S[j, f] * T[k, f])
                        U[i, f] -= learning_rate * grad_U

                        grad_S = (error * U[i, f] * T[k, f] + 
                                 lambda2 * S[j, f] +
                                 eta * avg_error * U[i, f] * T[k, f])
                        S[j, f] -= learning_rate * grad_S

                        grad_T = (error * U[i, f] * S[j, f] + 
                                 lambda3 * T[k, f] +
                                 eta * avg_error * U[i, f] * S[j, f])
                        T[k, f] -= learning_rate * grad_T
    return U, S, T

@jit(nopython=True, fastmath=True)
def calculate_loss_numba(U, S, T, Y, mask, Y_avg, lambda1, lambda2, lambda3, eta):
    m, n, c = Y.shape
    l = U.shape[1]
    total_loss = 0.0
    
    for i in range(m):
        for j in range(n):
            for k in range(c):
                if mask[i, j, k] == 1:
                    Y_pred = 0.0
                    for f in range(l):
                        Y_pred += U[i, f] * S[j, f] * T[k, f]
                    total_loss += 0.5 * (Y[i, j, k] - Y_pred) ** 2

    for i in range(m):
        for f in range(l):
            total_loss += 0.5 * lambda1 * U[i, f] ** 2
    
    for j in range(n):
        for f in range(l):
            total_loss += 0.5 * lambda2 * S[j, f] ** 2
    
    for k in range(c):
        for f in range(l):
            total_loss += 0.5 * lambda3 * T[k, f] ** 2
    
    for i in range(m):
        for j in range(n):
            for k in range(c):
                if mask[i, j, k] == 1:
                    Y_pred = 0.0
                    for f in range(l):
                        Y_pred += U[i, f] * S[j, f] * T[k, f]
                    total_loss += 0.5 * eta * (Y_pred - Y_avg[i, j]) ** 2
    
    return total_loss

@jit(nopython=True, fastmath=True, parallel=True)
def predict_all_numba(U, S, T):
    m, l = U.shape
    n = S.shape[0]
    c = T.shape[0]
    Y_pred = np.zeros((m, n, c))
    
    for i in prange(m):
        for j in range(n):
            for k in range(c):
                pred = 0.0
                for f in range(l):
                    pred += U[i, f] * S[j, f] * T[k, f]
                Y_pred[i, j, k] = pred
    return Y_pred

class WSPredJIT:
    def __init__(self, l=20, lambda1=0.001, lambda2=0.001, lambda3=0.001, eta=0.001, max_iter=100, tol=1e-4):
        self.l = l
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, Y, mask):
        self.Y = Y.astype(np.float64)
        self.mask = mask.astype(np.float64)
        
        m, n, c = Y.shape
        self.m, self.n, self.c = m, n, c

        np.random.seed(42)
        self.U = np.random.rand(m, self.l).astype(np.float64) * 0.01
        self.S = np.random.rand(n, self.l).astype(np.float64) * 0.01
        self.T = np.random.rand(c, self.l).astype(np.float64) * 0.01
        
        self.Y_avg = calculate_average_qos_numba(self.Y, self.mask)
        
        prev_loss = float('inf')
        learning_rate = 0.001
        
        self.loss_history = []
        
        for iteration in range(self.max_iter):
            self.U, self.S, self.T = update_latent_features_numba(
                self.U, self.S, self.T, self.Y, self.mask, self.Y_avg,
                self.lambda1, self.lambda2, self.lambda3, self.eta, learning_rate
            )
            
            current_loss = calculate_loss_numba(
                self.U, self.S, self.T, self.Y, self.mask, self.Y_avg,
                self.lambda1, self.lambda2, self.lambda3, self.eta
            )
            
            self.loss_history.append(current_loss)
            
            if abs(prev_loss - current_loss) < self.tol:
                print(f"Converged at iteration {iteration}")
                break
                
            prev_loss = current_loss
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {current_loss:.6f}")
  
        if len(self.loss_history) <= 1:
            print("Warning: Model converged too quickly. Consider adjusting learning rate or tolerance.")
    
    def predict(self, i, j, k):
        pred = 0.0
        for f in range(self.l):
            pred += self.U[i, f] * self.S[j, f] * self.T[k, f]
        return pred
    
    def predict_all(self):
        return predict_all_numba(self.U, self.S, self.T)

def create_large_sample_data(size_factor=1):
    np.random.seed(42)
    m, n, c = 10 * size_factor, 8 * size_factor, 6 * size_factor

    U_true = np.random.rand(m, 3).astype(np.float64)
    S_true = np.random.rand(n, 3).astype(np.float64)
    T_true = np.random.rand(c, 3).astype(np.float64)

    Y_true = np.zeros((m, n, c), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            for k in range(c):
                Y_true[i, j, k] = np.sum(U_true[i, :] * S_true[j, :] * T_true[k, :])

    Y_true += np.random.normal(0, 0.1, Y_true.shape).astype(np.float64)
    mask = np.random.choice([0, 1], size=(m, n, c), p=[0.7, 0.3]).astype(np.float64)
    
    for i in range(m):
        for j in range(n):
            if np.sum(mask[i, j, :]) == 0:
                k = np.random.randint(0, c)
                mask[i, j, k] = 1
    
    Y_obs = Y_true * mask
    return Y_obs, Y_true, mask

def evaluate_performance(Y_pred, Y_true, mask):
    observed_mask = mask == 1
    mae = np.mean(np.abs(Y_pred[observed_mask] - Y_true[observed_mask]))
    rmse = np.sqrt(np.mean((Y_pred[observed_mask] - Y_true[observed_mask]) ** 2))
    
    return mae, rmse


