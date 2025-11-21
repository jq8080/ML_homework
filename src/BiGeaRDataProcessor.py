import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
import pandas as pd

class BiGeaRDataProcessor:
    def __init__(self, data_path=None, df=None):
        """
        数据预处理器
        Args:
            data_path: 数据文件路径
            df: 直接传入DataFrame (二选一)
        """
        self.data_path = data_path
        self.df = df
        self.n_users = 0
        self.n_items = 0
        self.norm_adj = None
        self.train_interactions = []
        self.test_interactions = []
        
    def load_and_process(self, test_ratio=0.2):
        """加载并处理数据"""
        # 1. 加载数据
        if self.df is None and self.data_path:
            self.df = pd.read_csv(self.data_path)
        elif self.df is None:
            raise ValueError("必须提供data_path或df")
        
        print("原始数据形状:", self.df.shape)
        print("数据列:", self.df.columns.tolist())
        
        # 2. 数据清洗和ID映射
        self._create_id_mappings()
        
        # 3. 构建交互图
        self._build_interaction_graph(test_ratio)
        
        # 4. 构建归一化邻接矩阵
        self._build_normalized_adjacency()
        
        return self._get_processed_data()
    
    def _create_id_mappings(self):
        """创建用户和物品的ID映射"""
        # 假设数据中有'user_id'和'item_id'列
        unique_users = self.df['user_id'].unique()
        unique_items = self.df['item_id'].unique()
        
        # 创建映射字典
        self.user_id_map = {orig_id: new_id for new_id, orig_id in enumerate(unique_users)}
        self.item_id_map = {orig_id: new_id for new_id, orig_id in enumerate(unique_items)}
        
        # 反向映射
        self.id_to_user = {v: k for k, v in self.user_id_map.items()}
        self.id_to_item = {v: k for k, v in self.item_id_map.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        print(f"用户数量: {self.n_users}, 物品数量: {self.n_items}")
        
        # 应用映射
        self.df['user_id_mapped'] = self.df['user_id'].map(self.user_id_map)
        self.df['item_id_mapped'] = self.df['item_id'].map(self.item_id_map)
    
    def _build_interaction_graph(self, test_ratio=0.2):
        """构建用户-物品交互图并划分训练测试集"""
        # 按用户分组，为每个用户保留部分测试交互
        user_groups = self.df.groupby('user_id_mapped')
        
        train_interactions = []
        test_interactions = []
        
        for user_id, group in user_groups:
            interactions = list(zip(group['user_id_mapped'], group['item_id_mapped']))
            
            if len(interactions) > 1:
                # 随机选择测试交互
                test_size = max(1, int(len(interactions) * test_ratio))
                test_indices = np.random.choice(len(interactions), test_size, replace=False)
                
                for idx, interaction in enumerate(interactions):
                    if idx in test_indices:
                        test_interactions.append(interaction)
                    else:
                        train_interactions.append(interaction)
            else:
                # 只有一个交互，放入训练集
                train_interactions.extend(interactions)
        
        self.train_interactions = train_interactions
        self.test_interactions = test_interactions
        
        print(f"训练交互数: {len(train_interactions)}, 测试交互数: {len(test_interactions)}")
    
    def _build_normalized_adjacency(self):
        """构建归一化邻接矩阵 (LightGCN核心)"""
        n_nodes = self.n_users + self.n_items
        rows, cols = [], []
        
        # 添加训练交互到邻接矩阵
        for user_id, item_id in self.train_interactions:
            # 用户-物品边
            rows.append(user_id)
            cols.append(self.n_users + item_id)
            # 物品-用户边 (无向图)
            rows.append(self.n_users + item_id)  
            cols.append(user_id)
        
        # 创建稀疏邻接矩阵
        data = np.ones(len(rows))
        adj = sp.coo_matrix((data, (rows, cols)), 
                           shape=(n_nodes, n_nodes))
        
        print(f"邻接矩阵非零元素: {adj.nnz}")
        
        # 归一化: D^(-1/2) A D^(-1/2)
        degree = np.array(adj.sum(1))
        d_inv_sqrt = np.power(degree, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        self.norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)
        
        print("归一化邻接矩阵构建完成")
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        """将scipy稀疏矩阵转换为torch稀疏张量"""
        coo = X.tocoo().astype(np.float32)
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        index = torch.stack([row, col])
        value = torch.tensor(coo.data, dtype=torch.float)
        
        return torch.sparse_coo_tensor(index, value, torch.Size(coo.shape))
    
    def _get_processed_data(self):
        """返回处理后的数据"""
        return {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'norm_adj': self.norm_adj,
            'train_interactions': self.train_interactions,
            'test_interactions': self.test_interactions,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'id_to_user': self.id_to_user,
            'id_to_item': self.id_to_item
        }

# 数据加载器
class BiGeaRDataLoader:
    def __init__(self, processed_data, batch_size=1024, neg_ratio=4):
        self.processed_data = processed_data
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.train_interactions = processed_data['train_interactions']
        
    def generate_train_batches(self):
        """生成训练批次 (BPR损失需要的正负样本对)"""
        np.random.shuffle(self.train_interactions)
        
        n_batches = len(self.train_interactions) // self.batch_size
        if len(self.train_interactions) % self.batch_size != 0:
            n_batches += 1
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(self.train_interactions))
            
            batch_interactions = self.train_interactions[start_idx:end_idx]
            
            user_ids, pos_item_ids = zip(*batch_interactions)
            user_ids = torch.tensor(user_ids, dtype=torch.long)
            pos_item_ids = torch.tensor(pos_item_ids, dtype=torch.long)
            
            # 为每个正样本生成负样本
            neg_item_ids = self._sample_negative_items(user_ids, pos_item_ids)
            
            yield user_ids, pos_item_ids, neg_item_ids
    
    def _sample_negative_items(self, user_ids, pos_item_ids):
        """为每个用户采样负样本物品"""
        n_items = self.processed_data['n_items']
        batch_size = user_ids.shape[0]
        
        neg_items = []
        for i in range(batch_size):
            user_id = user_ids[i].item()
            pos_item_id = pos_item_ids[i].item()
            
            # 获取用户的所有交互物品
            user_interactions = set()
            for u, i in self.train_interactions:
                if u == user_id:
                    user_interactions.add(i)
            
            # 采样负样本
            negative_candidates = []
            while len(negative_candidates) < self.neg_ratio:
                neg_id = np.random.randint(0, n_items)
                if neg_id not in user_interactions and neg_id != pos_item_id:
                    negative_candidates.append(neg_id)
            
            neg_items.append(negative_candidates)
        
        return torch.tensor(neg_items, dtype=torch.long)