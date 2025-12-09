import numpy as np
import random
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from numba import jit, njit, prange
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

# 设置matplotlib中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Bitstream Vera Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


@njit(fastmath=True)
def predict_score_jit(u, s, mu, b_u, b_s, U, S):
    user_service_interaction = 0.0
    for f in range(U.shape[1]):
        user_service_interaction += U[u, f] * S[s, f]
    
    return mu + b_u[u] + b_s[s] + user_service_interaction

@njit(fastmath=True)
def euclidean_loss_jit(predicted, actual):
    return 0.5 * (predicted - actual) ** 2

@njit(fastmath=True, parallel=True)
def batch_predict_jit(users, services, mu, b_u, b_s, U, S):
    n = len(users)
    predictions = np.zeros(n)
    for i in prange(n):
        predictions[i] = predict_score_jit(users[i], services[i], mu, b_u, b_s, U, S)
    return predictions

class PITF_QoS_JIT:
    def __init__(self, n_users, n_services, k=64, learning_rate=0.05, reg_param=5e-05):
        self.n_users = n_users
        self.n_services = n_services
        self.k = k
        self.alpha = learning_rate
        self.lambda_theta = reg_param
        

        self.U = np.random.normal(0, 0.01, (n_users, k)).astype(np.float64)
        self.S = np.random.normal(0, 0.01, (n_services, k)).astype(np.float64)
        self.b_u = np.random.normal(0, 0.01, n_users).astype(np.float64)
        self.b_s = np.random.normal(0, 0.01, n_services).astype(np.float64)
        self.mu = np.array([0.0], dtype=np.float64)
        
    def predict_score(self, u, s):
        return predict_score_jit(u, s, self.mu[0], self.b_u, self.b_s, self.U, self.S)
    
    def fit(self, train_data, n_epochs=50, batch_size=1000, verbose=True):
        train_data_np = np.array(train_data, dtype=np.int32)
        users = train_data_np[:, 0]
        services = train_data_np[:, 1]
        qos_values = train_data_np[:, 2].astype(np.float64)

        self.mu[0] = np.mean(qos_values)

        self.train_loss_history = []

        _ = predict_score_jit(0, 0, self.mu[0], self.b_u, self.b_s, self.U, self.S)
        _ = euclidean_loss_jit(0.0, 0.0)
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            total_loss = 0
            batch_count = 0

            indices = np.random.permutation(len(train_data_np))
            shuffled_users = users[indices]
            shuffled_services = services[indices]
            shuffled_qos = qos_values[indices]

            for batch_start in range(0, len(train_data_np), batch_size):
                batch_end = min(batch_start + batch_size, len(train_data_np))
                batch_users = shuffled_users[batch_start:batch_end]
                batch_services = shuffled_services[batch_start:batch_end]
                batch_qos = shuffled_qos[batch_start:batch_end]
                
                batch_loss = 0
                for i in range(len(batch_users)):
                    u = batch_users[i]
                    s = batch_services[i]
                    actual_qos = batch_qos[i]

                    predicted_qos = predict_score_jit(u, s, self.mu[0], self.b_u, self.b_s, self.U, self.S)
                    
                    error = predicted_qos - actual_qos
                    
                    loss = euclidean_loss_jit(predicted_qos, actual_qos)

                    self.mu[0] -= self.alpha * error

                    grad_bu = error - self.lambda_theta * self.b_u[u]
                    self.b_u[u] -= self.alpha * grad_bu

                    grad_bs = error - self.lambda_theta * self.b_s[s]
                    self.b_s[s] -= self.alpha * grad_bs

                    for f in range(self.k):
                        grad_U = error * self.S[s, f] - self.lambda_theta * self.U[u, f]
                        self.U[u, f] -= self.alpha * grad_U

                    for f in range(self.k):
                        grad_S = error * self.U[u, f] - self.lambda_theta * self.S[s, f]
                        self.S[s, f] -= self.alpha * grad_S

                    batch_loss += loss + self.lambda_theta * (
                        self.b_u[u]**2 + self.b_s[s]**2 + 
                        np.sum(self.U[u]**2) + np.sum(self.S[s]**2)
                    )
                
                total_loss += batch_loss / len(batch_users)
                batch_count += 1
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            self.train_loss_history.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
    
    def evaluate(self, test_data):
        test_data_np = np.array(test_data, dtype=np.int32)
        test_users = test_data_np[:, 0]
        test_services = test_data_np[:, 1]
        actual_qos = test_data_np[:, 2].astype(np.float64)

        predicted_qos = batch_predict_jit(test_users, test_services, self.mu[0], self.b_u, self.b_s, self.U, self.S)
        rmse = math.sqrt(mean_squared_error(actual_qos, predicted_qos))
        mae = mean_absolute_error(actual_qos, predicted_qos)
        
        return rmse, mae, actual_qos, predicted_qos
    
    def get_all_predictions(self):
        predictions = np.zeros((self.n_users, self.n_services))
        for u in range(self.n_users):
            for s in range(self.n_services):
                predictions[u, s] = self.predict_score(u, s)
        return predictions



    
