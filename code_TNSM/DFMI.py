import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import time
import math


class FeatureMappingBlock(nn.Module):
    """Module 1"""
    def __init__(self, input_channels, output_channels, num_layers=4):
        super(FeatureMappingBlock, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        

        for i in range(num_layers):
            in_ch = input_channels if i == 0 else output_channels
            self.conv_layers.append(
                nn.Conv2d(in_ch, output_channels, kernel_size=3, padding=1, stride=1)
            )
    
    def forward(self, x, return_intermediate=False):
        intermediates = []
        
        e_k = x
        for i, conv in enumerate(self.conv_layers):
            e_k = F.relu(conv(e_k))
            if return_intermediate:
                if i == 0:  
                    intermediates.append(e_k)
                elif i == self.num_layers // 2:  
                    intermediates.append(e_k)
        
        if return_intermediate:
            intermediates.append(e_k)  
            return intermediates
        else:
            return e_k

class FeatureInferenceBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=4):
        super(FeatureInferenceBlock, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList()
        

        if num_layers == 1:
            layer_dims = [input_dim, output_dim]
        else:
            layer_dims = np.linspace(input_dim, output_dim, num_layers + 1, dtype=int)
            
        for i in range(num_layers):
            self.fc_layers.append(
                nn.Linear(layer_dims[i], layer_dims[i+1])
            )
    
    def forward(self, x, return_intermediate=False):
        intermediates = []
        
        d_k = x
        for i, fc in enumerate(self.fc_layers):
            d_k = F.relu(fc(d_k))
            if return_intermediate:
                if i == 0:  
                    intermediates.append(d_k)
                elif i == self.num_layers // 2:  
                    intermediates.append(d_k)
        
        if return_intermediate:
            intermediates.append(d_k)  
            return intermediates
        else:
            return d_k

class FeatureCompensationBlock(nn.Module):
    def __init__(self, input_dim, output_shape):
        super(FeatureCompensationBlock, self).__init__()
        self.output_shape = output_shape  # (C, H, W)
        self.target_elements = output_shape[0] * output_shape[1] * output_shape[2]
        
        self.fc = nn.Linear(input_dim, self.target_elements)
        
        self.conv = nn.Conv2d(output_shape[0] * 2, output_shape[0], kernel_size=1)
    
    def forward(self, first_layer_feat, middle_layer_feat):
        batch_size = first_layer_feat.shape[0]

        d_prime = F.relu(self.fc(first_layer_feat))

        d_prime = d_prime.view(batch_size, self.output_shape[0], 
                              self.output_shape[1], self.output_shape[2])
        

        if len(middle_layer_feat.shape) == 2:
            if middle_layer_feat.shape[1] > self.target_elements:
                middle_layer_feat = middle_layer_feat[:, :self.target_elements]
            elif middle_layer_feat.shape[1] < self.target_elements:
                padding = torch.zeros(batch_size, self.target_elements - middle_layer_feat.shape[1], 
                                    device=middle_layer_feat.device)
                middle_layer_feat = torch.cat([middle_layer_feat, padding], dim=1)
            middle_layer_feat = middle_layer_feat.view(batch_size, self.output_shape[0], 
                                                     self.output_shape[1], self.output_shape[2])
        elif middle_layer_feat.shape[2:] != self.output_shape[1:]:

            middle_layer_feat = F.interpolate(middle_layer_feat, size=self.output_shape[1:], 
                                           mode='bilinear', align_corners=False)

            if middle_layer_feat.shape[1] != self.output_shape[0]:
                adjust_conv = nn.Conv2d(middle_layer_feat.shape[1], self.output_shape[0], 
                                      kernel_size=1).to(middle_layer_feat.device)
                middle_layer_feat = adjust_conv(middle_layer_feat)
        
        combined = torch.cat([d_prime, middle_layer_feat], dim=1)
        
        output = F.relu(self.conv(combined))
        
        return output

class FMINet(nn.Module):
    def __init__(self, input_shape, latent_dim, num_layers=4):
        super(FMINet, self).__init__()
        self.input_shape = input_shape  # (K, l)
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        

        self.feature_mapping = FeatureMappingBlock(
            input_channels=1, 
            output_channels=8,
            num_layers=num_layers
        )
        

        flattened_size = 8 * input_shape[0] * input_shape[1]
        self.feature_inference = FeatureInferenceBlock(
            input_dim=flattened_size,
            output_dim=latent_dim,
            num_layers=num_layers
        )
        

        compensation_shape = (8, input_shape[0], input_shape[1])
        

        mapping_input_dim = 8 * input_shape[0] * input_shape[1] 
        self.mapping_compensation = FeatureCompensationBlock(
            input_dim=mapping_input_dim,
            output_shape=compensation_shape
        )
        
        if num_layers == 1:
            inference_first_dim = flattened_size  
        else:
            if num_layers > 1:
                layer_dims = np.linspace(flattened_size, latent_dim, num_layers + 1, dtype=int)
                inference_first_dim = layer_dims[1]  
            else:
                inference_first_dim = latent_dim
        
        self.inference_compensation = FeatureCompensationBlock(
            input_dim=inference_first_dim,  
            output_shape=compensation_shape
        )
    
    def forward(self, x, stage=2):
        batch_size = x.shape[0]
        
        x_conv = x.unsqueeze(1)  
        
        if stage == 1:
            M_i = self.feature_mapping(x_conv, return_intermediate=False)
            M_i_flat = M_i.view(batch_size, -1)
            U_i_prime = self.feature_inference(M_i_flat, return_intermediate=False)
            return U_i_prime
        
        else:
            mapping_intermediates = self.feature_mapping(x_conv, return_intermediate=True)
            e1, eM, M_i = mapping_intermediates[0], mapping_intermediates[1], mapping_intermediates[2]
            

            M_i_flat = M_i.view(batch_size, -1)
            inference_intermediates = self.feature_inference(M_i_flat, return_intermediate=True)
            d1, dM, U_i_prime = inference_intermediates[0], inference_intermediates[1], inference_intermediates[2]
            
            e1_flat = e1.contiguous().view(batch_size, -1)
            E_i = self.mapping_compensation(e1_flat, eM)
            
            d1_flat = d1.contiguous().view(batch_size, -1)
            D_i = self.inference_compensation(d1_flat, dM)
            
            F_i = torch.cat([E_i, D_i, M_i], dim=1)
            
            return F_i

class QoSPredictionNetwork(nn.Module):
    def __init__(self, input_channels):
        super(QoSPredictionNetwork, self).__init__()
        
        self.input_size = input_channels * 6 * 8  
        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, user_features, service_features):
        x = torch.cat([user_features, service_features], dim=1)
        
        x_flat = x.view(x.size(0), -1)
        output = self.fc_layers(x_flat)
        
        return output.squeeze()

class DFMI(nn.Module):
    def __init__(self, num_users, num_services, latent_dim=50, K=20, num_layers=4):
        super(DFMI, self).__init__()
        self.latent_dim = latent_dim
        self.K = K
        
        input_shape = (K, latent_dim)
        self.user_fminet = FMINet(input_shape, latent_dim, num_layers)
        self.service_fminet = FMINet(input_shape, latent_dim, num_layers)
        
        total_channels = 8 * 3 * 2
        self.qos_predictor = QoSPredictionNetwork(total_channels)
    
    def forward(self, user_features, service_features):
        user_fusion = self.user_fminet(user_features, stage=2)
        service_fusion = self.service_fminet(service_features, stage=2)
        
        qos_pred = self.qos_predictor(user_fusion, service_fusion)
        
        return qos_pred

def calculate_euclidean_loss(predictions, targets):
    squared_errors = (predictions - targets) ** 2
    mse = torch.mean(squared_errors)
    euclidean_loss = torch.sqrt(mse)
    return euclidean_loss

class QoSPredictor:
    def __init__(self, num_users, num_services, latent_dim=50, K=20):
        self.num_users = num_users
        self.num_services = num_services
        self.latent_dim = latent_dim
        self.K = K
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        

        self.model = DFMI(num_users, num_services, latent_dim, K).to(self.device)
        
        self.user_features_nmf = None
        self.service_features_nmf = None
        self.user_similarity = None
        self.service_similarity = None
        
    def safe_cosine_similarity(self, matrix: np.ndarray) -> np.ndarray:
        matrix_filled = np.nan_to_num(matrix, nan=0.0)

        similarity = cosine_similarity(matrix_filled)
        
        np.fill_diagonal(similarity, 1.0)
        
        return similarity
    
    def compute_similarity_matrices(self, QoS_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print("Computing similarity matrices...")
        
        if self.user_features_nmf is None or self.service_features_nmf is None:
            U, S = self.fit_nmf(QoS_matrix, self.latent_dim)
            self.user_features_nmf = U.T  
            self.service_features_nmf = S.T  
        
        user_similarity = self.safe_cosine_similarity(self.user_features_nmf)
        service_similarity = self.safe_cosine_similarity(self.service_features_nmf)
        
        return user_similarity, service_similarity
    
    def get_similar_entities(self, similarity_matrix: np.ndarray, entity_idx: int, K: int) -> np.ndarray:
        similarities = similarity_matrix[entity_idx].copy()
        similarities[entity_idx] = -1
        similar_indices = np.argsort(similarities)[-K:][::-1]
        return similar_indices
    
    def fit_nmf(self, QoS_matrix: np.ndarray, latent_dim: int, max_iter: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        print("Fitting NMF...")
        
        mask = np.isnan(QoS_matrix)
        filled_matrix = QoS_matrix.copy()
        
        col_means = np.nanmean(QoS_matrix, axis=0)
        for j in range(QoS_matrix.shape[1]):
            if not np.isnan(col_means[j]):  
                filled_matrix[mask[:, j], j] = col_means[j]
        
        filled_matrix = np.nan_to_num(filled_matrix, nan=0.1)
        
        filled_matrix = np.maximum(filled_matrix, 0.01)
        
        nmf = NMF(n_components=latent_dim, init='random', random_state=42, max_iter=max_iter)
        W = nmf.fit_transform(filled_matrix) 
        H = nmf.components_  
        
        return W.T, H  
    
    def prepare_training_data(self, QoS_matrix: np.ndarray, max_samples: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print("Preparing training data...")
        
        start_time = time.time()

        if self.user_similarity is None or self.service_similarity is None:
            self.user_similarity, self.service_similarity = self.compute_similarity_matrices(QoS_matrix)
        

        user_similar_matrices = []
        service_similar_matrices = []
        targets = []
        

        known_positions = np.where(~np.isnan(QoS_matrix))
        num_known = len(known_positions[0])
        
        print(f"Found {num_known} known QoS values")
        

        if num_known > max_samples:
            indices = np.random.choice(num_known, max_samples, replace=False)
            user_indices = known_positions[0][indices]
            service_indices = known_positions[1][indices]
        else:
            user_indices = known_positions[0]
            service_indices = known_positions[1]
        
        processed = 0
        for user_idx, service_idx in zip(user_indices, service_indices):
            similar_users = self.get_similar_entities(self.user_similarity, user_idx, self.K)
            user_sim_matrix = self.user_features_nmf[similar_users]  
            user_similar_matrices.append(user_sim_matrix)
            

            similar_services = self.get_similar_entities(self.service_similarity, service_idx, self.K)
            service_sim_matrix = self.service_features_nmf[similar_services]  
            service_similar_matrices.append(service_sim_matrix)
            

            targets.append(QoS_matrix[user_idx, service_idx])
            
            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed}/{len(user_indices)} samples")
        
        preparation_time = time.time() - start_time
        print(f"Data preparation completed in {preparation_time:.2f} seconds")
        print(f"Final dataset size: {len(targets)} samples")
        
        return (np.array(user_similar_matrices), 
                np.array(service_similar_matrices), 
                np.array(targets))
    
    def train(self, QoS_matrix: np.ndarray, epochs: int = 20, lr: float = 0.001, batch_size: int = 16):
        user_mats, service_mats, targets = self.prepare_training_data(QoS_matrix)
        

        user_tensor = torch.FloatTensor(user_mats).to(self.device)
        service_tensor = torch.FloatTensor(service_mats).to(self.device)
        target_tensor = torch.FloatTensor(targets).to(self.device)
        

        dataset = torch.utils.data.TensorDataset(user_tensor, service_tensor, target_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        

        losses = []
        start_time = time.time()
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_user, batch_service, batch_target in dataloader:
                optimizer.zero_grad()
                
                predictions = self.model(batch_user, batch_service)
                
                loss = calculate_euclidean_loss(predictions, batch_target)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds!")
        
        return losses
    
    def predict(self, user_idx: int, service_idx: int, QoS_matrix: np.ndarray) -> float:
        self.model.eval()
        
        with torch.no_grad():
            if self.user_similarity is None or self.service_similarity is None:
                self.user_similarity, self.service_similarity = self.compute_similarity_matrices(QoS_matrix)
            

            similar_users = self.get_similar_entities(self.user_similarity, user_idx, self.K)
            user_sim_matrix = self.user_features_nmf[similar_users]  # (K, l)
            
            similar_services = self.get_similar_entities(self.service_similarity, service_idx, self.K)
            service_sim_matrix = self.service_features_nmf[similar_services]  # (K, l)
            

            user_tensor = torch.FloatTensor(user_sim_matrix).unsqueeze(0).to(self.device)
            service_tensor = torch.FloatTensor(service_sim_matrix).unsqueeze(0).to(self.device)
            
            prediction = self.model(user_tensor, service_tensor)
            
            return prediction.cpu().item()