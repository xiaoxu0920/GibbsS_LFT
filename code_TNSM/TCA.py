import numpy as np
import time
from numba import jit
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import random

@jit(nopython=True)
def khatri_rao_two(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n1, r1 = A.shape
    n2, r2 = B.shape
    if r1 != r2:
        raise ValueError("两个矩阵的列数必须相同")
    result = np.zeros((n1 * n2, r1))
    for i in range(n1):
        for j in range(n2):
            for k in range(r1):
                result[i * n2 + j, k] = A[i, k] * B[j, k]
    return result

@jit(nopython=True)
def cp_reconstruction_simple(A_s: np.ndarray, A_p: np.ndarray, A_c: np.ndarray, A_t: np.ndarray) -> np.ndarray:
    R = A_s.shape[1]
    S, P, C, T = A_s.shape[0], A_p.shape[0], A_c.shape[0], A_t.shape[0]
    H_pred = np.zeros((S, P, C, T))

    for s in range(S):
        for p in range(P):
            for c in range(C):
                for t in range(T):
                    for r in range(R):
                        H_pred[s, p, c, t] += A_s[s, r] * A_p[p, r] * A_c[c, r] * A_t[t, r]
    
    return H_pred

@jit(nopython=True)
def compute_gradient_simple(H: np.ndarray, H_pred: np.ndarray, W: np.ndarray, 
                           A_s: np.ndarray, A_p: np.ndarray, A_c: np.ndarray, A_t: np.ndarray,
                           mode: int) -> np.ndarray:
    S, P, C, T = H.shape
    R = A_s.shape[1]
    
    if mode == 0: 
        gradient = np.zeros_like(A_s)
        for s in range(S):
            for r in range(R):
                grad_val = 0.0
                for p in range(P):
                    for c in range(C):
                        for t in range(T):
                            if W[s, p, c, t] > 0:
                                error = H[s, p, c, t] - H_pred[s, p, c, t]
                                grad_val += error * A_p[p, r] * A_c[c, r] * A_t[t, r]
                gradient[s, r] = grad_val
                
    elif mode == 1: 
        gradient = np.zeros_like(A_p)
        for p in range(P):
            for r in range(R):
                grad_val = 0.0
                for s in range(S):
                    for c in range(C):
                        for t in range(T):
                            if W[s, p, c, t] > 0:
                                error = H[s, p, c, t] - H_pred[s, p, c, t]
                                grad_val += error * A_s[s, r] * A_c[c, r] * A_t[t, r]
                gradient[p, r] = grad_val
                
    elif mode == 2: 
        gradient = np.zeros_like(A_c)
        for c in range(C):
            for r in range(R):
                grad_val = 0.0
                for s in range(S):
                    for p in range(P):
                        for t in range(T):
                            if W[s, p, c, t] > 0:
                                error = H[s, p, c, t] - H_pred[s, p, c, t]
                                grad_val += error * A_s[s, r] * A_p[p, r] * A_t[t, r]
                gradient[c, r] = grad_val
                
    elif mode == 3:  
        gradient = np.zeros_like(A_t)
        for t in range(T):
            for r in range(R):
                grad_val = 0.0
                for s in range(S):
                    for p in range(P):
                        for c in range(C):
                            if W[s, p, c, t] > 0:
                                error = H[s, p, c, t] - H_pred[s, p, c, t]
                                grad_val += error * A_s[s, r] * A_p[p, r] * A_c[c, r]
                gradient[t, r] = grad_val
                
    else:
        raise ValueError("Invalid mode")
    
    return gradient

@jit(nopython=True)
def calculate_scale_factor(H: np.ndarray, W: np.ndarray) -> float:
    sum_H = 0.0
    count = 0
    S, P, C, T = H.shape
    
    for s in range(S):
        for p in range(P):
            for c in range(C):
                for t in range(T):
                    if W[s, p, c, t] > 0:
                        sum_H += H[s, p, c, t]
                        count += 1
    
    if count > 0:
        return sum_H / count  
    else:
        return 0.5

@jit(nopython=True)
def tensor_completion_simple(H: np.ndarray, W: np.ndarray, rank: int = 3, 
                            learning_rate: float = 0.01, 
                            reg_param: float = 0.01, 
                            error_threshold: float = 1e-5, 
                            max_iter: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    S, P, C, T = H.shape
    
    scale_factor = calculate_scale_factor(H, W)
    A_s = np.random.rand(S, rank) * scale_factor + scale_factor
    A_p = np.random.rand(P, rank) * scale_factor + scale_factor
    A_c = np.random.rand(C, rank) * scale_factor + scale_factor
    A_t = np.random.rand(T, rank) * scale_factor + scale_factor
    
    errors = np.zeros(max_iter)
    prev_loss = float('inf')
    
    current_lr = learning_rate
    
    for iteration in range(max_iter):

        H_pred = cp_reconstruction_simple(A_s, A_p, A_c, A_t)
        

        loss = 0.0
        for s in range(S):
            for p in range(P):
                for c in range(C):
                    for t in range(T):
                        if W[s, p, c, t] > 0:
                            error = H[s, p, c, t] - H_pred[s, p, c, t]
                            loss += error * error
        loss = 0.5 * loss
        
        reg_loss = 0.0
        for s in range(S):
            for r in range(rank):
                reg_loss += A_s[s, r] * A_s[s, r]
        for p in range(P):
            for r in range(rank):
                reg_loss += A_p[p, r] * A_p[p, r]
        for c in range(C):
            for r in range(rank):
                reg_loss += A_c[c, r] * A_c[c, r]
        for t in range(T):
            for r in range(rank):
                reg_loss += A_t[t, r] * A_t[t, r]
                
        loss += 0.5 * reg_param * reg_loss
        errors[iteration] = loss
        

        if iteration > 10 and abs(prev_loss - loss) < error_threshold:
            errors = errors[:iteration+1]
            break

        if iteration > 5 and prev_loss < loss:
            current_lr *= 0.95  
        
        prev_loss = loss

        grad_s = compute_gradient_simple(H, H_pred, W, A_s, A_p, A_c, A_t, 0)
        grad_p = compute_gradient_simple(H, H_pred, W, A_s, A_p, A_c, A_t, 1)
        grad_c = compute_gradient_simple(H, H_pred, W, A_s, A_p, A_c, A_t, 2)
        grad_t = compute_gradient_simple(H, H_pred, W, A_s, A_p, A_c, A_t, 3)
        
        for s in range(S):
            for r in range(rank):
                A_s[s, r] += current_lr * (grad_s[s, r] - reg_param * A_s[s, r])
                
        for p in range(P):
            for r in range(rank):
                A_p[p, r] += current_lr * (grad_p[p, r] - reg_param * A_p[p, r])
                
        for c in range(C):
            for r in range(rank):
                A_c[c, r] += current_lr * (grad_c[c, r] - reg_param * A_c[c, r])
                
        for t in range(T):
            for r in range(rank):
                A_t[t, r] += current_lr * (grad_t[t, r] - reg_param * A_t[t, r])

        for s in range(S):
            for r in range(rank):
                if A_s[s, r] < 1e-8:
                    A_s[s, r] = 1e-3  
                    
        for p in range(P):
            for r in range(rank):
                if A_p[p, r] < 1e-8:
                    A_p[p, r] = 1e-3
                    
        for c in range(C):
            for r in range(rank):
                if A_c[c, r] < 1e-8:
                    A_c[c, r] = 1e-3
                    
        for t in range(T):
            for r in range(rank):
                if A_t[t, r] < 1e-8:
                    A_t[t, r] = 1e-3
    
    H_pred = cp_reconstruction_simple(A_s, A_p, A_c, A_t)
    return H_pred, errors

def create_weight_tensor(H: np.ndarray) -> np.ndarray:
    W = np.zeros_like(H)
    S, P, C, T = H.shape
    for s in range(S):
        for p in range(P):
            for c in range(C):
                for t in range(T):
                    if H[s, p, c, t] != 0:
                        W[s, p, c, t] = 1.0
    return W

def generate_synthetic_data(num_services=4, num_providers=4, num_consumers=4, num_time_slots=4,
                           sparsity=0.4, random_seed=42) -> np.ndarray:

    np.random.seed(random_seed)
    random.seed(random_seed)
    

    H = np.zeros((num_services, num_providers, num_consumers, num_time_slots))
    

    base_pattern = np.random.rand(num_services, num_providers, num_consumers) * 0.5 + 0.3
    time_trend = np.linspace(0.8, 1.2, num_time_slots)
    

    total_entries = num_services * num_providers * num_consumers * num_time_slots
    num_nonzero = int(total_entries * sparsity)
    

    indices = [(s, p, c, t) for s in range(num_services) for p in range(num_providers)
              for c in range(num_consumers) for t in range(num_time_slots)]
    selected_indices = random.sample(indices, num_nonzero)
    
    for s, p, c, t in selected_indices:
        base_value = base_pattern[s, p, c] * time_trend[t]
        noise = np.random.normal(0, 0.05)
        value = np.clip(base_value + noise, 0.1, 0.9)
        H[s, p, c, t] = value
    
    return H

def evaluate_completion(H_original: np.ndarray, H_completed: np.ndarray, 
                       mask_known: np.ndarray) -> Dict[str, float]:
    mask_unknown = np.zeros_like(H_original, dtype=bool)
    S, P, C, T = H_original.shape
    

    for s in range(S):
        for p in range(P):
            for c in range(C):
                for t in range(T):
                    if H_original[s, p, c, t] != 0 and not mask_known[s, p, c, t]:
                        mask_unknown[s, p, c, t] = True
    
    num_unknown = np.sum(mask_unknown)
    
    if num_unknown == 0:
        return {"mae": 0, "rmse": 0, "relative_error": 0}
    
    mae_sum = 0.0
    rmse_sum = 0.0
    relative_sum = 0.0
    
    for s in range(S):
        for p in range(P):
            for c in range(C):
                for t in range(T):
                    if mask_unknown[s, p, c, t]:
                        true_val = H_original[s, p, c, t]
                        pred_val = H_completed[s, p, c, t]
                        diff = true_val - pred_val
                        mae_sum += abs(diff)
                        rmse_sum += diff ** 2
                        if true_val > 1e-8: 
                            relative_sum += abs(diff) / true_val
    
    mae = mae_sum / num_unknown
    rmse = np.sqrt(rmse_sum / num_unknown)
    relative_error = relative_sum / num_unknown
    
    return {"mae": mae, "rmse": rmse, "relative_error": relative_error}

