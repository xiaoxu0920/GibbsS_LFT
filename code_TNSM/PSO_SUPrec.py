import numpy as np
import pandas as pd
from numba import jit, prange
import math
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns


# 设置matplotlib中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Bitstream Vera Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


@jit(nopython=True)
def initialize_velocities_numba(pNum, d):
    velocities = np.empty((pNum, d))
    for i in range(pNum):
        for j in range(d):
            velocities[i, j] = np.random.uniform(-1, 1)
    return velocities

@jit(nopython=True)
def update_velocity_numba(velocity, position, pbest, gbest, w):
    c1, c2 = 2.0, 2.0  # acceleration coefficients
    new_velocity = np.empty_like(velocity)
    
    for k in range(len(velocity)):
        r1, r2 = np.random.random(), np.random.random()
        new_velocity[k] = (w * velocity[k] + 
                          c1 * r1 * (pbest[k] - position[k]) + 
                          c2 * r2 * (gbest[k] - position[k]))
        
        if new_velocity[k] > 1.0:
            new_velocity[k] = 1.0
        elif new_velocity[k] < -1.0:
            new_velocity[k] = -1.0
            
    return new_velocity

@jit(nopython=True)
def update_position_numba(position, velocity):
    """Update particle position - numba compatible version"""
    return position + velocity

@jit(nopython=True)
def smooth_outliers_numba(position, user_mean, user_std, service_means, service_stds, missing_indices, theta):
    """Smooth outliers using SPC - numba compatible version"""
    result = position.copy()
    
    for j in range(len(missing_indices)):
        service_id = missing_indices[j]
        service_mean = service_means[service_id]
        service_std = service_stds[service_id]
        
        user_lower = user_mean - theta * user_std
        user_upper = user_mean + theta * user_std
        service_lower = service_mean - theta * service_std
        service_upper = service_mean + theta * service_std
        
        if (result[j] < user_lower and result[j] < service_lower):
            result[j] = max(user_lower, service_lower)
        elif (result[j] > user_upper and result[j] > service_upper):
            result[j] = min(user_upper, service_upper)
            
    return result

@jit(nopython=True)
def calculate_stats_numba(Q):
    m, n = Q.shape
    user_means = np.zeros(m)
    user_stds = np.zeros(m)
    service_means = np.zeros(n)
    service_stds = np.zeros(n)
    

    total_sum = 0.0
    total_count = 0
    for i in range(m):
        for j in range(n):
            if not np.isnan(Q[i, j]):
                total_sum += Q[i, j]
                total_count += 1
    
    overall_mean = total_sum / total_count if total_count > 0 else 1.0
    

    total_var = 0.0
    for i in range(m):
        for j in range(n):
            if not np.isnan(Q[i, j]):
                total_var += (Q[i, j] - overall_mean) ** 2
    overall_std = np.sqrt(total_var / total_count) if total_count > 1 else 0.1


    for i in range(m):
        user_sum = 0.0
        user_count = 0
        for j in range(n):
            if not np.isnan(Q[i, j]):
                user_sum += Q[i, j]
                user_count += 1
        
        if user_count > 0:
            user_means[i] = user_sum / user_count
            

            user_var = 0.0
            for j in range(n):
                if not np.isnan(Q[i, j]):
                    user_var += (Q[i, j] - user_means[i]) ** 2
            user_stds[i] = np.sqrt(user_var / user_count) if user_count > 1 else 0.1
        else:
            user_means[i] = overall_mean
            user_stds[i] = overall_std
    

    for j in range(n):
        service_sum = 0.0
        service_count = 0
        for i in range(m):
            if not np.isnan(Q[i, j]):
                service_sum += Q[i, j]
                service_count += 1
        
        if service_count > 0:
            service_means[j] = service_sum / service_count
            

            service_var = 0.0
            for i in range(m):
                if not np.isnan(Q[i, j]):
                    service_var += (Q[i, j] - service_means[j]) ** 2
            service_stds[j] = np.sqrt(service_var / service_count) if service_count > 1 else 0.1
        else:
            service_means[j] = overall_mean
            service_stds[j] = overall_std
    
    return user_means, user_stds, service_means, service_stds

class PSOUSRec:
    def __init__(self, m, n, pNum=50, top_k=10, maxGen=50, w=0.8, theta=3, lambda_val=0.5, tau=21):
        self.m = m  
        self.n = n  
        self.pNum = pNum  
        self.top_k = top_k  
        self.maxGen = maxGen  
        self.w = w  
        self.theta = theta  
        self.lambda_val = lambda_val  
        self.tau = tau  
        
    def fit(self, Q):
        """Train the model on QoS matrix Q"""
        self.Q = Q.copy()
        self.m, self.n = Q.shape
        

        self.user_means, self.user_stds, self.service_means, self.service_stds = calculate_stats_numba(Q)
        
        return self
    
    def predict_all(self):
        Q_pred = self.Q.copy()
        
        Q_user_pred = self._predict_user_perspective()
        
        Q_service_pred = self._predict_service_perspective()
        
        for i in range(self.m):
            for j in range(self.n):
                if np.isnan(self.Q[i, j]):
                    Q_pred[i, j] = (self.lambda_val * Q_user_pred[i, j] + 
                                   (1 - self.lambda_val) * Q_service_pred[i, j])
        
        return Q_pred
    
    def _predict_user_perspective(self):
        Q_pred = self.Q.copy()
        
        for user_id in range(self.m):
            missing_indices = np.where(np.isnan(self.Q[user_id]))[0]
            if len(missing_indices) == 0:
                continue
                
            predicted_values = self._pso_urec(user_id, missing_indices)

            for idx, service_id in enumerate(missing_indices):
                Q_pred[user_id, service_id] = predicted_values[idx]
                
        return Q_pred
    
    def _predict_service_perspective(self):
        Q_pred = self.Q.copy()
        
        for service_id in range(self.n):
            missing_indices = np.where(np.isnan(self.Q[:, service_id]))[0]
            if len(missing_indices) == 0:
                continue
                
            predicted_values = self._pso_srec(service_id, missing_indices)
            
            for idx, user_id in enumerate(missing_indices):
                Q_pred[user_id, service_id] = predicted_values[idx]
                
        return Q_pred
    
    def _pso_urec(self, user_id, missing_indices):
        d = len(missing_indices) 
        if d == 0:
            return np.array([])
            
        positions = self._initialize_population_user(user_id, missing_indices, d)
        velocities = initialize_velocities_numba(self.pNum, d)
        
        pbest_positions = positions.copy()
        pbest_fitness = np.array([self._fitness_user(user_id, pos, missing_indices) 
                                for pos in positions])
        
        gbest_idx = np.argmin(pbest_fitness)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]
        

        for gen in range(self.maxGen):
            for i in range(self.pNum):
                velocities[i] = update_velocity_numba(velocities[i], positions[i], 
                                                    pbest_positions[i], gbest_position, self.w)
                positions[i] = update_position_numba(positions[i], velocities[i])
                
                positions[i] = smooth_outliers_numba(
                    positions[i], 
                    self.user_means[user_id], 
                    self.user_stds[user_id],
                    self.service_means,
                    self.service_stds,
                    missing_indices,
                    self.theta
                )
                
                fitness = self._fitness_user(user_id, positions[i], missing_indices)
                
                if fitness < pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = positions[i].copy()
                    
                    if fitness < gbest_fitness:
                        gbest_fitness = fitness
                        gbest_position = positions[i].copy()
                        
        return gbest_position
    
    def _pso_srec(self, service_id, missing_indices):
        d = len(missing_indices)  # dimension
        if d == 0:
            return np.array([])
            
        positions = self._initialize_population_service(service_id, missing_indices, d)
        velocities = initialize_velocities_numba(self.pNum, d)
        
        pbest_positions = positions.copy()
        pbest_fitness = np.array([self._fitness_service(service_id, pos, missing_indices) 
                                for pos in positions])
        
        gbest_idx = np.argmin(pbest_fitness)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]
        

        for gen in range(self.maxGen):
            for i in range(self.pNum):
                velocities[i] = update_velocity_numba(velocities[i], positions[i], 
                                                    pbest_positions[i], gbest_position, self.w)
                positions[i] = update_position_numba(positions[i], velocities[i])
                

                positions[i] = smooth_outliers_numba(
                    positions[i], 
                    self.service_means[service_id], 
                    self.service_stds[service_id],
                    self.user_means,
                    self.user_stds,
                    missing_indices,
                    self.theta
                )
                

                fitness = self._fitness_service(service_id, positions[i], missing_indices)
                

                if fitness < pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = positions[i].copy()
                    

                    if fitness < gbest_fitness:
                        gbest_fitness = fitness
                        gbest_position = positions[i].copy()
                        
        return gbest_position
    
    def _initialize_population_user(self, user_id, missing_indices, d):
        positions = np.zeros((self.pNum, d))
        
        for i in range(min(self.tau, self.pNum)):
            for j, service_id in enumerate(missing_indices):
                positions[i, j] = self._uipcc_prediction_user(user_id, service_id, 
                                                             lambda_val=i/self.tau)
        
        user_mean = self.user_means[user_id]
        user_std = self.user_stds[user_id]
        
        for i in range(self.tau, self.pNum):
            for j in range(d):
                positions[i, j] = np.random.normal(user_mean, user_std)
                positions[i, j] = max(0.1, positions[i, j])  
                
        return positions
    
    def _initialize_population_service(self, service_id, missing_indices, d):
        positions = np.zeros((self.pNum, d))
        
        for i in range(min(self.tau, self.pNum)):
            for j, user_id in enumerate(missing_indices):
                positions[i, j] = self._uipcc_prediction_service(service_id, user_id, 
                                                               lambda_val=i/self.tau)
        
        service_mean = self.service_means[service_id]
        service_std = self.service_stds[service_id]
        
        for i in range(self.tau, self.pNum):
            for j in range(d):
                positions[i, j] = np.random.normal(service_mean, service_std)
                positions[i, j] = max(0.1, positions[i, j])  
                
        return positions
    
    def _fitness_user(self, user_id, position, missing_indices):
        fitness = 0.0
        similar_users = self._find_similar_users(user_id)
        
        for j, service_id in enumerate(missing_indices):
            top_users = self._get_top_users_for_service(user_id, service_id, similar_users)
            
            if len(top_users) == 0:
                continue
                
            sum_similarity = 0.0
            weighted_sum = 0.0
            
            for v in top_users:
                similarity = self._enhanced_similarity(user_id, v)
                if similarity <= 0:
                    continue
                    
                user_mean = self.user_means[user_id]
                v_mean = self.user_means[v]
                
                weighted_sum += similarity * abs(
                    (position[j] - user_mean) - (self.Q[v, service_id] - v_mean)
                )
                sum_similarity += similarity
                
            if sum_similarity > 0:
                fitness += weighted_sum / sum_similarity
                
        return fitness
    
    def _fitness_service(self, service_id, position, missing_indices):

        fitness = 0.0
        similar_services = self._find_similar_services(service_id)
        
        for j, user_id in enumerate(missing_indices):
            top_services = self._get_top_services_for_user(service_id, user_id, similar_services)
            
            if len(top_services) == 0:
                continue
                
            sum_similarity = 0.0
            weighted_sum = 0.0
            
            for z in top_services:
                similarity = self._enhanced_similarity_service(service_id, z)
                if similarity <= 0:
                    continue
                    
                service_mean = self.service_means[service_id]
                z_mean = self.service_means[z]
                
                weighted_sum += similarity * abs(
                    (position[j] - service_mean) - (self.Q[user_id, z] - z_mean)
                )
                sum_similarity += similarity
                
            if sum_similarity > 0:
                fitness += weighted_sum / sum_similarity
                
        return fitness
    
    def _find_similar_users(self, user_id):
        similarities = []
        for v in range(self.m):
            if v != user_id:
                sim = self._enhanced_similarity(user_id, v)
                if sim > 0:
                    similarities.append((v, sim))
        

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [user for user, sim in similarities[:self.top_k]]
    
    def _find_similar_services(self, service_id):
        similarities = []
        for z in range(self.n):
            if z != service_id:
                sim = self._enhanced_similarity_service(service_id, z)
                if sim > 0:
                    similarities.append((z, sim))
        

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [service for service, sim in similarities[:self.top_k]]
    
    def _enhanced_similarity(self, u, v):
        common_services = []
        for j in range(self.n):
            if not np.isnan(self.Q[u, j]) and not np.isnan(self.Q[v, j]):
                common_services.append(j)
        
        if len(common_services) == 0:
            return 0.0
        
        u_values = np.array([self.Q[u, j] for j in common_services])
        v_values = np.array([self.Q[v, j] for j in common_services])
        
        u_mean = np.mean(u_values)
        v_mean = np.mean(v_values)
        
        numerator = np.sum((u_values - u_mean) * (v_values - v_mean))
        denominator = (np.sqrt(np.sum((u_values - u_mean)**2)) * 
                     np.sqrt(np.sum((v_values - v_mean)**2)))
        
        if denominator == 0:
            pcc = 0.0
        else:
            pcc = numerator / denominator
        
        u_invoked = np.sum(~np.isnan(self.Q[u]))
        v_invoked = np.sum(~np.isnan(self.Q[v]))
        
        if u_invoked + v_invoked == 0:
            significance = 0.0
        else:
            significance = (2 * len(common_services)) / (u_invoked + v_invoked)
        
        return significance * pcc
    
    def _enhanced_similarity_service(self, i, j):
        common_users = []
        for u in range(self.m):
            if not np.isnan(self.Q[u, i]) and not np.isnan(self.Q[u, j]):
                common_users.append(u)
        
        if len(common_users) == 0:
            return 0.0
        
        i_values = np.array([self.Q[u, i] for u in common_users])
        j_values = np.array([self.Q[u, j] for u in common_users])
        
        i_mean = np.mean(i_values)
        j_mean = np.mean(j_values)
        
        numerator = np.sum((i_values - i_mean) * (j_values - j_mean))
        denominator = (np.sqrt(np.sum((i_values - i_mean)**2)) * 
                     np.sqrt(np.sum((j_values - j_mean)**2)))
        
        if denominator == 0:
            pcc = 0.0
        else:
            pcc = numerator / denominator

        i_invoked = np.sum(~np.isnan(self.Q[:, i]))
        j_invoked = np.sum(~np.isnan(self.Q[:, j]))
        
        if i_invoked + j_invoked == 0:
            significance = 0.0
        else:
            significance = (2 * len(common_users)) / (i_invoked + j_invoked)
        
        return significance * pcc
    
    def _get_top_users_for_service(self, user_id, service_id, similar_users):
        top_users = []
        for v in similar_users:
            if not np.isnan(self.Q[v, service_id]):
                top_users.append(v)
            if len(top_users) >= self.top_k:
                break
        return top_users
    
    def _get_top_services_for_user(self, service_id, user_id, similar_services):
        top_services = []
        for z in similar_services:
            if not np.isnan(self.Q[user_id, z]):
                top_services.append(z)
            if len(top_services) >= self.top_k:
                break
        return top_services
    
    def _uipcc_prediction_user(self, user_id, service_id, lambda_val):
        user_mean = self.user_means[user_id]
        return max(0.1, user_mean * (1 + 0.1 * (lambda_val - 0.5)))
    
    def _uipcc_prediction_service(self, service_id, user_id, lambda_val):
        service_mean = self.service_means[service_id]
        return max(0.1, service_mean * (1 + 0.1 * (lambda_val - 0.5)))



