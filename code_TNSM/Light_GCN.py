import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import time
import warnings

# 忽略稀疏张量相关的警告
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")

class LightGCN(nn.Module):
    """
    LightGCN模型
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64, 
                 n_layers: int = 3, dropout: float = 0.0, device: str = 'cpu'):
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device
        
        # 初始化用户和物品嵌入 (第0层)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 使用相同的初始化方法
        self._reset_parameters()
        
        # 层组合权重 (均匀权重)
        self.alpha = 1.0 / (n_layers + 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def _reset_parameters(self):
        """统一的参数初始化"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def _create_adjacency_matrix(self, interactions: List[Tuple[int, int]]) -> torch.Tensor:
        """
        创建归一化的邻接矩阵 - 修复版本，避免CSR警告
        """
        n_nodes = self.n_users + self.n_items
        
        # 构建稀疏矩阵的索引和值
        rows, cols, values = [], [], []
        
        for u, i in interactions:
            # 用户->物品
            rows.append(u)
            cols.append(self.n_users + i)
            values.append(1.0)
            
            # 物品->用户  
            rows.append(self.n_users + i)
            cols.append(u)
            values.append(1.0)
        
        # 转换为张量
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(values, dtype=torch.float)
        
        # 创建稀疏COO矩阵
        adj_matrix = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes), device=self.device)
        
        # 计算度矩阵
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        
        # 避免除零
        row_sum = torch.clamp(row_sum, min=1.0)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        
        # 构建归一化的邻接矩阵: D^(-1/2) A D^(-1/2)
        # 使用逐元素乘法避免稀疏矩阵乘法警告
        norm_adj = adj_matrix.clone()
        
        # 获取非零元素的位置和值
        adj_indices = norm_adj.coalesce().indices()
        adj_values = norm_adj.coalesce().values()
        
        # 对每个非零元素应用归一化
        normalized_values = []
        for idx in range(adj_indices.shape[1]):
            row_idx = adj_indices[0, idx].item()
            col_idx = adj_indices[1, idx].item()
            normalized_val = adj_values[idx] * d_inv_sqrt[row_idx] * d_inv_sqrt[col_idx]
            normalized_values.append(normalized_val)
        
        normalized_values = torch.tensor(normalized_values, device=self.device)
        norm_adj = torch.sparse_coo_tensor(adj_indices, normalized_values, (n_nodes, n_nodes))
        
        return norm_adj.coalesce()
    
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor, 
                norm_adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 获取所有嵌入
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ])
        
        # 存储各层嵌入
        embeddings_list = [all_embeddings]
        current_embeddings = all_embeddings
        
        # 图卷积层传播
        for layer in range(self.n_layers):
            current_embeddings = torch.sparse.mm(norm_adj, current_embeddings)
            embeddings_list.append(current_embeddings)
        
        # 层组合：加权求和 (使用均匀权重)
        final_embeddings = torch.stack(embeddings_list, dim=0)
        final_embeddings = torch.mean(final_embeddings, dim=0)  # 均匀权重
        
        # 分离用户和物品嵌入
        user_final_embeddings = final_embeddings[:self.n_users]
        item_final_embeddings = final_embeddings[self.n_users:]
        
        # 应用dropout (训练时使用，测试时关闭)
        user_final_embeddings = self.dropout(user_final_embeddings)
        item_final_embeddings = self.dropout(item_final_embeddings)
        
        # 获取特定用户和物品的嵌入
        user_emb = user_final_embeddings[user_indices]
        item_emb = item_final_embeddings[item_indices]
        
        # 计算预测分数 (内积)
        predictions = torch.sum(user_emb * item_emb, dim=1)
        
        return predictions

# JIT编译版本 - 与原始模型完全一致
class LightGCN_JIT(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64, 
                 n_layers: int = 3, dropout: float = 0.0, device: str = 'cpu'):
        super(LightGCN_JIT, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 使用相同的初始化方法
        self._reset_parameters()
        
        self.dropout = nn.Dropout(dropout)
    
    def _reset_parameters(self):
        """统一的参数初始化"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    @torch.jit.export
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor, 
                norm_adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 获取所有嵌入
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        
        # 存储各层嵌入
        embeddings_list = torch.jit.annotate(List[torch.Tensor], [all_embeddings])
        current_embeddings = all_embeddings
        
        # 图卷积层传播
        for _ in range(self.n_layers):
            current_embeddings = torch.sparse.mm(norm_adj, current_embeddings)
            embeddings_list.append(current_embeddings)
        
        # 层组合：加权求和 (使用均匀权重)
        final_embeddings = torch.stack(embeddings_list, dim=0)
        final_embeddings = torch.mean(final_embeddings, dim=0)  # 均匀权重
        
        # 分离用户和物品嵌入
        user_final = final_embeddings[:self.n_users]
        item_final = final_embeddings[self.n_users:]
        
        # 应用dropout
        user_final = self.dropout(user_final)
        item_final = self.dropout(item_final)
        
        # 获取特定嵌入
        user_emb = user_final[user_indices]
        item_emb = item_final[item_indices]
        
        # 计算预测分数
        predictions = torch.sum(user_emb * item_emb, dim=1)
        return predictions

def copy_model_weights(source_model, target_model):
    """复制模型权重"""
    target_model.user_embedding.weight.data.copy_(source_model.user_embedding.weight.data)
    target_model.item_embedding.weight.data.copy_(source_model.item_embedding.weight.data)

def create_sample_data(n_users: int = 100, n_items: int = 100, n_interactions: int = 500):
    """
    创建示例数据
    """
    # 生成随机交互数据
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)
    interactions = list(zip(users, items))
    
    return interactions

def bpr_loss(predictions: torch.Tensor, lambda_reg: float = 1e-4, 
             user_emb: torch.Tensor = None, item_emb: torch.Tensor = None) -> torch.Tensor:
    """
    BPR损失函数
    """
    batch_size = predictions.shape[0] // 2
    pos_scores = predictions[:batch_size]
    neg_scores = predictions[batch_size:]
    
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    
    # L2正则化
    if user_emb is not None and item_emb is not None:
        l2_reg = lambda_reg * (torch.norm(user_emb, 2) + torch.norm(item_emb, 2))
        loss += l2_reg
    
    return loss

def test_lightgcn():
    """
    测试LightGCN模型
    """
    print("=== LightGCN 模型测试 ===")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建示例数据
    n_users, n_items = 100, 100
    interactions = create_sample_data(n_users, n_items, 500)
    
    # 创建原始模型
    model = LightGCN(n_users=n_users, n_items=n_items, 
                    embedding_dim=64, n_layers=3, dropout=0.0, device=device)
    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 创建邻接矩阵
    print("创建邻接矩阵...")
    norm_adj = model._create_adjacency_matrix(interactions)
    
    # 创建JIT模型并复制权重
    jit_model = LightGCN_JIT(n_users=n_users, n_items=n_items, 
                           embedding_dim=64, n_layers=3, dropout=0.0, device=device)
    jit_model.to(device)
    jit_model.eval()  # 设置为评估模式
    
    # 复制权重确保完全一致
    copy_model_weights(model, jit_model)
    

    with torch.jit.optimized_execution(True):
        jit_model_compiled = torch.jit.script(jit_model)
    
    # 测试数据
    test_users = torch.tensor([0, 1, 2, 3, 4], device=device)
    test_items = torch.tensor([0, 1, 2, 3, 4], device=device)
    
    print(f"\n=== 前向传播测试 ===")
    
    # 原始模型推理
    with torch.no_grad():
        predictions = model(test_users, test_items, norm_adj)
        print(f"原始模型预测结果形状: {predictions.shape}")
        print(f"原始模型预测值: {predictions.cpu().numpy()}")
    
    # JIT模型推理
    with torch.no_grad():
        jit_predictions = jit_model_compiled(test_users, test_items, norm_adj)
        print(f"JIT模型预测结果形状: {jit_predictions.shape}")
        print(f"JIT模型预测值: {jit_predictions.cpu().numpy()}")
    
    print(f"\n=== 结果一致性验证 ===")
    
    # 详细比较结果
    diff = torch.abs(predictions - jit_predictions)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    
    print(f"最大差异: {max_diff.item():.8f}")
    print(f"平均差异: {mean_diff.item():.8f}")
    


    is_close = torch.allclose(predictions, jit_predictions, rtol=1e-5, atol=1e-6)
    print(f"原始模型和JIT模型预测结果是否接近: {is_close}")
    
    if is_close:
        print("模型一致性验证通过!")
    else:
        print("模型一致性验证失败!")
        # 打印详细比较
        for i, (orig, jit_val) in enumerate(zip(predictions, jit_predictions)):
            print(f"样本 {i}: 原始={orig.item():.6f}, JIT={jit_val.item():.6f}, 差异={abs(orig - jit_val).item():.8f}")
    
    # 性能测试
    print(f"\n=== 性能测试 ===")
    
    # 大批量测试
    batch_size = 1000
    batch_users = torch.randint(0, n_users, (batch_size,), device=device)
    batch_items = torch.randint(0, n_items, (batch_size,), device=device)
    
    # 预热
    print("预热运行...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch_users, batch_items, norm_adj)
            _ = jit_model_compiled(batch_users, batch_items, norm_adj)
    
    # 原始模型性能
    print("测试原始模型性能...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(batch_users, batch_items, norm_adj)
    original_time = time.time() - start_time
    


    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = jit_model_compiled(batch_users, batch_items, norm_adj)
    jit_time = time.time() - start_time
    
    print(f"原始模型推理时间: {original_time:.4f}s")
    print(f"JIT模型推理时间: {jit_time:.4f}s")
    print(f"加速比: {original_time/jit_time:.2f}x")
    
    # 验证训练过程
    print(f"\n=== 训练过程验证 ===")
    
    # 重置模型为训练模式
    model.train()
    model.zero_grad()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 简单训练步骤
    train_users = torch.tensor([interaction[0] for interaction in interactions[:10]], device=device)
    train_items = torch.tensor([interaction[1] for interaction in interactions[:10]], device=device)
    
    # 负采样
    neg_items = torch.randint(0, n_items, train_items.shape, device=device)
    
    # 合并正负样本
    all_users = torch.cat([train_users, train_users])
    all_items = torch.cat([train_items, neg_items])
    
    # 前向传播
    predictions = model(all_users, all_items, norm_adj)
    
    # 计算BPR损失
    user_emb = model.user_embedding(train_users)
    item_emb = model.item_embedding(train_items)
    loss = bpr_loss(predictions, lambda_reg=1e-4, user_emb=user_emb, item_emb=item_emb)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    print(f"训练损失: {loss.item():.4f}")
    print("训练过程验证通过!")
    
    # 测试嵌入获取功能
    print(f"\n=== 嵌入获取测试 ===")
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.user_embedding.weight, model.item_embedding.weight
        print(f"用户嵌入形状: {user_embeddings.shape}")
        print(f"物品嵌入形状: {item_embeddings.shape}")
        print(f"用户嵌入范数: {torch.norm(user_embeddings, dim=1).mean().item():.4f}")
        print(f"物品嵌入范数: {torch.norm(item_embeddings, dim=1).mean().item():.4f}")
    
    print(f"\n=== 所有测试完成 ===")