import torch
import torch.jit
import numpy as np
import time
from typing import List

class PosteriorNeighborhoodRegularizedLF:
    def __init__(self, num_users, num_services, f=20, K1=15, K2=50, 
                 alpha1=0.1, alpha2=0.4, lmbda=0.01, lr=0.01, max_iters=100):
        self.num_users = num_users
        self.num_services = num_services
        self.f = f  
        self.K1 = K1  
        self.K2 = K2  
        self.alpha1 = alpha1  
        self.alpha2 = alpha2  
        self.lmbda = lmbda   
        self.lr = lr          
        self.max_iters = max_iters  
        
        self.P = torch.randn(num_users, f, dtype=torch.float32) * 0.01
        self.Q = torch.randn(num_services, f, dtype=torch.float32) * 0.01
        self.P_prime = torch.randn(num_users, f, dtype=torch.float32) * 0.01
        self.Q_prime = torch.randn(num_services, f, dtype=torch.float32) * 0.01
        

        self.T_u = None  
        self.T_s = None  
        self.Su = None   
        self.Ss = None   
        
    def phase1_primal_lf_extraction(self, R):
        print("Phase 1: Primal LF Extraction...")
        known_indices = torch.nonzero(R, as_tuple=True)
        
        for iteration in range(self.max_iters):
            total_loss = 0
            for idx in range(len(known_indices[0])):
                u = known_indices[0][idx]
                s = known_indices[1][idx]
                r_us = R[u, s]
                

                pred = torch.dot(self.P[u], self.Q[s])
                error = r_us - pred
                

                p_u = self.P[u].clone()
                q_s = self.Q[s].clone()
                
                self.P[u] += self.lr * error * q_s
                self.Q[s] += self.lr * error * p_u
                
                total_loss += 0.5 * error ** 2
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}, Loss: {total_loss.item():.4f}")
    
    def calculate_pcc_matrix(self, P):
        n, f = P.shape
        pcc_matrix = torch.zeros(n, n, dtype=torch.float32)
        
        means = torch.mean(P, dim=1)
        
        for i in range(n):
            for j in range(i+1, n):
                cov = torch.sum((P[i] - means[i]) * (P[j] - means[j]))
                std_i = torch.sqrt(torch.sum((P[i] - means[i]) ** 2))
                std_j = torch.sqrt(torch.sum((P[j] - means[j]) ** 2))
                
                if std_i > 0 and std_j > 0:
                    pcc = cov / (std_i * std_j)
                    pcc_matrix[i, j] = pcc
                    pcc_matrix[j, i] = pcc
        
        return pcc_matrix
    
    def phase2_posterior_neighborhood_construction(self):
        print("Phase 2: Posterior Neighborhood Construction...")
        user_pcc = self.calculate_pcc_matrix(self.P)

        service_pcc = self.calculate_pcc_matrix(self.Q)

        self.T_u = []
        self.Su = []
        
        for u in range(self.num_users):
            similarities = []
            for k in range(self.num_users):
                if k != u and user_pcc[u, k] > 0:
                    similarities.append((k, user_pcc[u, k].item()))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            neighbors = similarities[:self.K1]
            
            if neighbors:
                neighbor_indices = [n[0] for n in neighbors]
                neighbor_weights = [n[1] for n in neighbors]
                
                total_weight = sum(neighbor_weights)
                normalized_weights = [w / total_weight for w in neighbor_weights]
                
                self.T_u.append(neighbor_indices)
                self.Su.append(normalized_weights)
            else:
                self.T_u.append([])
                self.Su.append([])

        self.T_s = []
        self.Ss = []
        
        for s in range(self.num_services):
            similarities = []
            for j in range(self.num_services):
                if j != s and service_pcc[s, j] > 0:
                    similarities.append((j, service_pcc[s, j].item()))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            neighbors = similarities[:self.K2]
            
            if neighbors:
                neighbor_indices = [n[0] for n in neighbors]
                neighbor_weights = [n[1] for n in neighbors]

                total_weight = sum(neighbor_weights)
                normalized_weights = [w / total_weight for w in neighbor_weights]
                
                self.T_s.append(neighbor_indices)
                self.Ss.append(normalized_weights)
            else:
                self.T_s.append([])
                self.Ss.append([])
        
        print(f"Constructed neighborhoods: {len([t for t in self.T_u if t])} users, {len([t for t in self.T_s if t])} services")
    
    def calculate_neighborhood_regularization(self, P_prime, T_u, Su, alpha, user_idx):
        if not T_u[user_idx]:
            return torch.zeros(P_prime.shape[1], dtype=torch.float32)
        
        reg_term = torch.zeros(P_prime.shape[1], dtype=torch.float32)
        neighbors = T_u[user_idx]
        weights = Su[user_idx]
        
        for i, neighbor_idx in enumerate(neighbors):
            weight = weights[i]
            reg_term += weight * (P_prime[user_idx] - P_prime[neighbor_idx])
        
        return alpha * reg_term
    
    def phase3_posterior_neighborhood_regularized_lf(self, R):
        print("Phase 3: Posterior Neighborhood Regularized LF Analysis...")
        
        self.P_prime = self.P.clone()
        self.Q_prime = self.Q.clone()
        
        known_indices = torch.nonzero(R, as_tuple=True)
        
        for iteration in range(self.max_iters):
            total_loss = 0
            
            for idx in range(len(known_indices[0])):
                u = known_indices[0][idx]
                s = known_indices[1][idx]
                r_us = R[u, s]
                
                pred = torch.dot(self.P_prime[u], self.Q_prime[s])
                error = r_us - pred
                
                user_reg = self.calculate_neighborhood_regularization(
                    self.P_prime, self.T_u, self.Su, self.alpha1, u)
                
                service_reg = self.calculate_neighborhood_regularization(
                    self.Q_prime, self.T_s, self.Ss, self.alpha2, s)
                
                p_grad = -error * self.Q_prime[s] + self.lmbda * self.P_prime[u] + user_reg
                self.P_prime[u] -= self.lr * p_grad

                q_grad = -error * self.P_prime[u] + self.lmbda * self.Q_prime[s] + service_reg
                self.Q_prime[s] -= self.lr * q_grad
                
                total_loss += 0.5 * error ** 2
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}, Loss: {total_loss.item():.4f}")
    
    def fit(self, R):
        print("Starting PLF model training...")
        start_time = time.time()
        
        self.phase1_primal_lf_extraction(R)    

        self.phase2_posterior_neighborhood_construction()
        
        self.phase3_posterior_neighborhood_regularized_lf(R)
        
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    def predict(self, u, s):
        return torch.dot(self.P_prime[u], self.Q_prime[s]).item()
    
    def predict_all(self):
        return torch.mm(self.P_prime, self.Q_prime.t())


def evaluate_model(model, R_true, mask):
    predictions = model.predict_all()

    test_indices = torch.nonzero(~mask, as_tuple=True)
    
    if len(test_indices[0]) > 0:
        mae = 0
        rmse = 0
        
        for i in range(len(test_indices[0])):
            u = test_indices[0][i]
            s = test_indices[1][i]
            true_val = R_true[u, s].item()
            pred_val = predictions[u, s].item()
            
            error = abs(true_val - pred_val)
            mae += error
            rmse += error ** 2
        
        mae /= len(test_indices[0])
        rmse = torch.sqrt(torch.tensor(rmse / len(test_indices[0])))
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        return mae, rmse
    else:
        print("No test data available")
        return 0, 0