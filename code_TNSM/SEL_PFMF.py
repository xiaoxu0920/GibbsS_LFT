import numpy as np
import numba
from numba import jit, prange
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

@jit(nopython=True)
def calculate_statistics(R):
    m, n = R.shape
    user_avg = np.zeros(m)
    user_std = np.zeros(m)
    service_avg = np.zeros(n)
    service_std = np.zeros(n)
    
    for u in range(m):
        user_records = R[u, :]
        non_zero_indices = np.where(user_records > 0)[0]
        if len(non_zero_indices) > 0:
            user_avg[u] = np.mean(user_records[non_zero_indices])
            user_std[u] = np.std(user_records[non_zero_indices])
        else:
            user_avg[u] = 0.0
            user_std[u] = 0.0
    
    for i in range(n):
        service_records = R[:, i]
        non_zero_indices = np.where(service_records > 0)[0]
        if len(non_zero_indices) > 0:
            service_avg[i] = np.mean(service_records[non_zero_indices])
            service_std[i] = np.std(service_records[non_zero_indices])
        else:
            service_avg[i] = 0.0
            service_std[i] = 0.0
            
    return user_avg, user_std, service_avg, service_std

@jit(nopython=True)
def improved_similarity(u_records, v_records, u_std, v_std):
    common_indices = np.where((u_records > 0) & (v_records > 0))[0]
    
    if len(common_indices) < 2:
        return 0.0
    
    u_common = u_records[common_indices]
    v_common = v_records[common_indices]
    

    u_mean = np.mean(u_common)
    v_mean = np.mean(v_common)
    
    numerator = np.sum((u_common - u_mean) * (v_common - v_mean))
    denominator_u = np.sum((u_common - u_mean)**2)
    denominator_v = np.sum((v_common - v_mean)**2)
    
    if denominator_u == 0 or denominator_v == 0:
        pcc_sim = 0.0
    else:
        pcc_sim = numerator / (np.sqrt(denominator_u) * np.sqrt(denominator_v))
    
    if u_std == 0 or v_std == 0:
        stability_factor = 1.0
    else:
        stability_factor = min(u_std, v_std) / max(u_std, v_std)
    
    return stability_factor * pcc_sim

@jit(nopython=True)
def find_top_k_neighbors(target_id, records, std_values, top_k):
    n = records.shape[0]
    similarities = np.zeros(n)
    
    for i in range(n):
        if i != target_id:
            similarities[i] = improved_similarity(
                records[target_id], records[i], std_values[target_id], std_values[i]
            )

    if n <= top_k + 1:  
        valid_indices = np.where(similarities > 0)[0]
        if len(valid_indices) > 0:
            return valid_indices, similarities[valid_indices]
        else:
            empty_int = np.array([0], dtype=np.int64)
            empty_float = np.array([0.0], dtype=np.float64)
            return empty_int, empty_float
    else:
        top_k_indices = np.argsort(similarities)[-top_k:]
        top_k_similarities = similarities[top_k_indices]
        
        valid_indices = top_k_similarities > 0
        if np.sum(valid_indices) > 0:
            return top_k_indices[valid_indices], top_k_similarities[valid_indices]
        else:
            empty_int = np.array([0], dtype=np.int64)
            empty_float = np.array([0.0], dtype=np.float64)
            return empty_int, empty_float

@jit(nopython=True)
def nmf_optimization(R_pf, h, lambda_p, lambda_q, t_mf):

    m, n = R_pf.shape
    

    P = np.random.rand(m, h) * 0.5 + 0.1
    Q = np.random.rand(n, h) * 0.5 + 0.1
    
    L = np.zeros((m, n))
    for u in range(m):
        for i in range(n):
            if R_pf[u, i] > 0:
                L[u, i] = 1.0
    
 
    non_zero_count = np.sum(L)
    if non_zero_count < m * n * 0.1:  
        for u in range(m):
            for i in range(n):
                if L[u, i] == 0 and np.random.rand() < 0.1:  
                    R_pf[u, i] = np.random.rand() * 0.5 + 0.1
                    L[u, i] = 1.0
    
    for iteration in range(t_mf):
        for u in range(m):
            for k in range(h):
                numerator = 0.0
                denominator = 0.0
                
                for i in range(n):
                    if L[u, i] > 0:
                        r_hat = 0.0
                        for k2 in range(h):
                            r_hat += P[u, k2] * Q[i, k2]
                        numerator += L[u, i] * Q[i, k] * R_pf[u, i]
                        denominator += L[u, i] * Q[i, k] * r_hat
                
                numerator += 1e-9 
                denominator += lambda_p + 1e-9
                
                if denominator > 0:
                    P[u, k] = P[u, k] * numerator / denominator
        
        for i in range(n):
            for k in range(h):
                numerator = 0.0
                denominator = 0.0
                
                for u in range(m):
                    if L[u, i] > 0:
                        r_hat = 0.0
                        for k2 in range(h):
                            r_hat += P[u, k2] * Q[i, k2]
                        numerator += L[u, i] * P[u, k] * R_pf[u, i]
                        denominator += L[u, i] * P[u, k] * r_hat
                
                numerator += 1e-9
                denominator += lambda_q + 1e-9
                
                if denominator > 0:
                    Q[i, k] = Q[i, k] * numerator / denominator
    
    return P, Q

@jit(nopython=True)
def pso_optimization(predictions, R_actual, sp, omega, c1, c2, t_max):
    d, m, n = predictions.shape
    
    positions = np.random.rand(sp, d) * 0.5 + 0.1 
    velocities = np.random.rand(sp, d) * 0.1
    
    for j in range(sp):
        positions[j] = positions[j] / np.sum(positions[j])
    
    pbest_positions = positions.copy()
    pbest_errors = np.full(sp, np.inf)
    gbest_position = positions[0].copy()
    gbest_error = np.inf
    
    for t in range(t_max):
        for j in range(sp):

            weighted_pred = np.zeros((m, n))
            for k in range(d):
                weighted_pred += positions[j, k] * predictions[k]
            weighted_pred = weighted_pred / np.sum(positions[j])
            
            error = 0.0
            count = 0
            for u in range(m):
                for i in range(n):
                    if R_actual[u, i] > 0:
                        error += (weighted_pred[u, i] - R_actual[u, i]) ** 2
                        count += 1
            if count > 0:
                error = np.sqrt(error / count)
            else:
                error = np.inf
            
            if error < pbest_errors[j]:
                pbest_errors[j] = error
                pbest_positions[j] = positions[j].copy()
            
            if error < gbest_error:
                gbest_error = error
                gbest_position = positions[j].copy()
        
        for j in range(sp):
            r1, r2 = np.random.rand(2)
            velocities[j] = (omega * velocities[j] + 
                           c1 * r1 * (pbest_positions[j] - positions[j]) + 
                           c2 * r2 * (gbest_position - positions[j]))
            positions[j] = positions[j] + velocities[j]
            
            positions[j] = np.maximum(positions[j], 0.01)  
            positions[j] = positions[j] / np.sum(positions[j])
    
    return gbest_position

class SEL_PFMF:
    def __init__(self, top_k=5, lambda_param=0.1, lambda_p=30, lambda_q=30, 
                 d=10, theta=0.1, phi=0.8, sp=20, omega=0.7, c1=1.5, c2=1.5, 
                 t_max=50, h=10, t_mf=100):
        self.top_k = top_k
        self.lambda_param = lambda_param
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q
        self.d = d
        self.theta = theta
        self.phi = phi
        self.sp = sp
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.t_max = t_max
        self.h = h
        self.t_mf = t_mf

        self.user_avg = None
        self.user_std = None
        self.service_avg = None
        self.service_std = None
        self.prefilled_matrices = []
        self.prediction_matrices = []
        self.ensemble_weights = None
    
    def calculate_local_average(self, R, user_idx, service_idx, is_user=True):
        if is_user:
            user_records = R[user_idx, :]
            service_neighbors, service_sims = find_top_k_neighbors(
                service_idx, R.T, self.service_std, min(self.top_k, R.shape[1]-1)
            )
            
            if len(service_neighbors) == 0:
                return self.user_avg[user_idx] if self.user_avg[user_idx] > 0 else np.mean(R[R > 0])
            
            neighbor_records = []
            for j in service_neighbors:
                if user_records[j] > 0:
                    neighbor_records.append(user_records[j])
            
            if len(neighbor_records) > 0:
                return np.mean(neighbor_records)
            else:
                return self.user_avg[user_idx] if self.user_avg[user_idx] > 0 else np.mean(R[R > 0])
        else:
            service_records = R[:, service_idx]
            user_neighbors, user_sims = find_top_k_neighbors(
                user_idx, R, self.user_std, min(self.top_k, R.shape[0]-1)
            )
            
            if len(user_neighbors) == 0:
                return self.service_avg[service_idx] if self.service_avg[service_idx] > 0 else np.mean(R[R > 0])
            
            neighbor_records = []
            for v in user_neighbors:
                if service_records[v] > 0:
                    neighbor_records.append(service_records[v])
            
            if len(neighbor_records) > 0:
                return np.mean(neighbor_records)
            else:
                return self.service_avg[service_idx] if self.service_avg[service_idx] > 0 else np.mean(R[R > 0])
    
    def improved_upcc(self, R, u, i):
        user_neighbors, user_sims = find_top_k_neighbors(
            u, R, self.user_std, min(self.top_k, R.shape[0]-1)
        )
        
        if len(user_neighbors) == 0 or len(user_sims) == 0:
            return self.user_avg[u] if self.user_avg[u] > 0 else np.mean(R[R > 0])
        

        local_avg_u = self.calculate_local_average(R, u, i, is_user=True)
        
        numerator = 0.0
        denominator = 0.0
        
        for idx, (v, sim) in enumerate(zip(user_neighbors, user_sims)):
            if R[v, i] > 0: 
                local_avg_v = self.calculate_local_average(R, v, i, is_user=True)
                numerator += sim * (R[v, i] - local_avg_v)
                denominator += sim
        
        if denominator == 0:
            return local_avg_u
        else:
            result = local_avg_u + numerator / denominator
            return max(result, 0.1) if result > 0 else 0.1
    
    def improved_ipcc(self, R, u, i):
        service_neighbors, service_sims = find_top_k_neighbors(
            i, R.T, self.service_std, min(self.top_k, R.shape[1]-1)
        )
        
        if len(service_neighbors) == 0 or len(service_sims) == 0:
            return self.service_avg[i] if self.service_avg[i] > 0 else np.mean(R[R > 0])
        
        local_avg_i = self.calculate_local_average(R, u, i, is_user=False)
        
        numerator = 0.0
        denominator = 0.0
        
        for idx, (j, sim) in enumerate(zip(service_neighbors, service_sims)):
            if R[u, j] > 0: 
                local_avg_j = self.calculate_local_average(R, u, j, is_user=False)
                numerator += sim * (R[u, j] - local_avg_j)
                denominator += sim
        
        if denominator == 0:
            return local_avg_i
        else:
            result = local_avg_i + numerator / denominator
            return max(result, 0.1) if result > 0 else 0.1
    
    def prefilling_value(self, R, u, i):
        pf_user = self.improved_upcc(R, u, i)
        pf_service = self.improved_ipcc(R, u, i)
        
        result = self.lambda_param * pf_user + (1 - self.lambda_param) * pf_service
        return max(result, 0.1)
    
    def generate_prefilled_matrix(self, R):
        m, n = R.shape
        R_pf = R.copy()
        missing_indices = np.argwhere(R == 0)
        num_to_fill = max(1, int(len(missing_indices) * self.phi))
        selected_indices = random.sample(list(missing_indices), num_to_fill)
        
        for idx in selected_indices:
            u, i = idx
            try:
                R_pf[u, i] = self.prefilling_value(R, u, i)
            except Exception as e:
                avg_val = np.mean(R[R > 0]) if np.sum(R > 0) > 0 else 0.5
                R_pf[u, i] = avg_val
        
        if np.sum(R_pf > 0) < m * n * 0.3:  
            zero_indices = np.argwhere(R_pf == 0)
            num_to_add = min(len(zero_indices), int(m * n * 0.2))  
            add_indices = random.sample(list(zero_indices), num_to_add)
            for idx in add_indices:
                u, i = idx
                R_pf[u, i] = np.random.rand() * 0.5 + 0.1
        
        return R_pf
    
    def pfmp_prediction(self, R):
        R_pf = self.generate_prefilled_matrix(R)
        P, Q = nmf_optimization(
            R_pf, self.h, self.lambda_p, self.lambda_q, self.t_mf
        )
        

        R_pred = np.dot(P, Q.T)
        R_pred = np.maximum(R_pred, 0.01)
        
        return R_pred
    
    def fit(self, R_train):
        self.user_avg, self.user_std, self.service_avg, self.service_std = calculate_statistics(R_train)
        
        self.prefilled_matrices = []
        self.prediction_matrices = []
        
        for k in range(self.d):
            R_pred = self.pfmp_prediction(R_train)
            self.prediction_matrices.append(R_pred)
            print(f"预测矩阵 {k+1} 的非零元素比例: {np.sum(R_pred > 0.01) / (R_pred.shape[0] * R_pred.shape[1]):.3f}")
        
        predictions_array = np.array(self.prediction_matrices)
        

        self.ensemble_weights = pso_optimization(
            predictions_array, R_train, self.sp, self.omega, self.c1, self.c2, self.t_max
        )
        
        return self
    
    def predict(self, R_test):
        if self.ensemble_weights is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        final_prediction = np.zeros_like(R_test)
        
        for k in range(self.d):
            final_prediction += self.ensemble_weights[k] * self.prediction_matrices[k]
        
        final_prediction = final_prediction / np.sum(self.ensemble_weights)

        final_prediction = np.maximum(final_prediction, 0.01)
        
        return final_prediction

def evaluate_performance(y_true, y_pred, mask):
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    mae = mean_absolute_error(y_true_masked, y_pred_masked)
    rmse = np.sqrt(mean_squared_error(y_true_masked, y_pred_masked))
    
    return mae, rmse


