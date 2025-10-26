# -*- coding: UTF-8 -*-
# @Author : Your Name
# @Email  : your_email@example.com

""" BiGeaR_FM
Reference:
    "Learning Binarized Graph Representations with Multi-faceted Quantization Reinforcement for Top-K Recommendation"
    Yankai Chen et al., 2022.
    + Factorization Machines (FM) base
"""

import torch
import torch.nn as nn
import numpy as np
import inspect

from models.BaseContextModel import ContextCTRModel, ContextModel

class BiGeaR_FM_Base(object):
    @staticmethod
    def parse_model_args_FM(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                           help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=3,
                           help='Number of graph convolution layers.')
        parser.add_argument('--gamma', type=float, default=10.0,
                           help='Parameter for gradient estimation in Dirac delta approximation.')
        parser.add_argument('--distill_lambda', type=float, default=1.0,
                           help='Weight for inference distillation loss.')
        parser.add_argument('--top_k_distill', type=int, default=50,
                           help='Top-K items for inference distillation.')
        parser.add_argument('--pretrain_epochs', type=int, default=10,
                           help='Number of pre-training epochs with full precision.')
        return parser

    def _define_init_params(self, args, corpus):
        self.vec_size = args.emb_size
        self.n_layers = args.n_layers
        self.gamma = args.gamma
        self.distill_lambda = args.distill_lambda
        self.top_k_distill = args.top_k_distill
        self.pretrain_epochs = args.pretrain_epochs
    
    def _define_init(self, args, corpus):
        self._define_init_params(args, corpus)
        # 确保父类初始化完成后再调用参数定义
        self._define_params_BiGeaR_FM()
        self.apply(self.init_weights)
    
    def _define_params_BiGeaR_FM(self):    
        # 检查必要的属性是否已由父类定义
        if not hasattr(self, 'context_features'):
            # 如果父类还没有定义 context_features，尝试从 corpus 获取
            if hasattr(self, 'corpus') and hasattr(self.corpus, 'context_feature_names'):
                self.context_features = self.corpus.context_feature_names
            else:
                # 如果无法获取，设置默认值
                self.context_features = []
                print("Warning: context_features not defined, using empty list")
        
        if not hasattr(self, 'feature_max'):
            self.feature_max = {}
            print("Warning: feature_max not defined, using empty dict")
        
        print(f"初始化 BiGeaR_FM 参数，上下文特征: {self.context_features}")
        print(f"特征维度: {self.feature_max}")
        
        # Layer-wise embeddings for BiGeaR
        self.context_embedding_layers = nn.ModuleList()
        self.linear_embedding_layers = nn.ModuleList()
        
        for layer in range(self.n_layers + 1):
            context_embedding = nn.ModuleDict()
            linear_embedding = nn.ModuleDict()
            
            for f in self.context_features:
                if f in self.feature_max and (f.endswith('_c') or f.endswith('_id')):
                    # 分类特征使用 Embedding
                    context_embedding[f] = nn.Embedding(self.feature_max[f], self.vec_size)
                    linear_embedding[f] = nn.Embedding(self.feature_max[f], 1)
                else:
                    # 连续特征使用 Linear
                    context_embedding[f] = nn.Linear(1, self.vec_size, bias=False)
                    linear_embedding[f] = nn.Linear(1, 1, bias=False)
            
            self.context_embedding_layers.append(context_embedding)
            self.linear_embedding_layers.append(linear_embedding)
        
        self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
        self.layer_weights = nn.Parameter(torch.ones(self.n_layers + 1))
        
        # Graph propagation weights (simplified)
        if self.n_layers > 0:
            self.propagation_weights = nn.Parameter(torch.ones(self.n_layers, self.vec_size))

    # BiGeaR Binarization Function with Dirac delta gradient estimation
    class BinarizeFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, gamma):
            ctx.save_for_backward(x)
            ctx.gamma = gamma
            return torch.sign(x)

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            gamma = ctx.gamma
            # Dirac delta approximation for gradient: 2γ/√π * exp(-(γx)²)
            grad_input = grad_output * (2 * gamma / np.sqrt(np.pi)) * torch.exp(-(gamma * x) ** 2)
            return grad_input, None

    def binarize(self, x):
        return self.BinarizeFunction.apply(x, self.gamma)

    def _get_layer_wise_embeddings(self, feed_dict, layer_idx, binarize=False):
        """Get embeddings for a specific layer with optional binarization"""
        item_ids = feed_dict['item_id']
        batch_size, item_num = item_ids.shape

        # Get base embeddings
        context_embedding = self.context_embedding_layers[layer_idx]
        linear_embedding = self.linear_embedding_layers[layer_idx]
        
        fm_vectors = []
        for f in self.context_features:
            if f in feed_dict:
                if f in context_embedding and isinstance(context_embedding[f], nn.Embedding):
                    # 分类特征
                    emb = context_embedding[f](feed_dict[f])
                else:
                    # 连续特征
                    emb = context_embedding[f](feed_dict[f].float().unsqueeze(-1))
                
                # Expand to match item_num if needed
                if len(emb.shape) == 2:  # [batch_size, emb_size]
                    emb = emb.unsqueeze(1).repeat(1, item_num, 1)  # [batch_size, item_num, emb_size]
                fm_vectors.append(emb)
        
        if not fm_vectors:
            # 如果没有上下文特征，返回零向量
            fm_vectors = [torch.zeros(batch_size, item_num, self.vec_size, device=item_ids.device)]
        
        fm_vectors = torch.stack(fm_vectors, dim=-2)  # [batch_size, item_num, feature_num, emb_size]
        
        # Apply binarization if requested
        if binarize:
            # Compute scaling factors (alpha in BiGeaR)
            scaling_factors = torch.mean(torch.abs(fm_vectors), dim=-1, keepdim=True)  # [batch_size, item_num, feature_num, 1]
            binarized_vectors = self.binarize(fm_vectors)  # [batch_size, item_num, feature_num, emb_size]
            fm_vectors = binarized_vectors * scaling_factors
        
        return fm_vectors

    def _graph_propagation(self, layer_embs):
        """Simplified graph propagation (BiGeaR style)"""
        propagated_embs = []
        
        for l in range(self.n_layers + 1):
            if l == 0:
                # Initial layer - no propagation yet
                propagated_embs.append(layer_embs[l])
                continue
            
            if l < len(self.propagation_weights) + 1:
                # Simplified propagation: weighted combination with previous layer
                current_emb = layer_embs[l]
                prev_emb = propagated_embs[l-1]
                
                # Apply propagation weights
                prop_weight = self.propagation_weights[l-1].view(1, 1, 1, -1)
                propagated_emb = current_emb + prev_emb * prop_weight
                
                propagated_embs.append(propagated_emb)
            else:
                propagated_embs.append(layer_embs[l])
        
        return propagated_embs

    def _get_embeddings_BiGeaR(self, feed_dict, binarize=False):
        """BiGeaR-style layer-wise embedding computation"""
        item_ids = feed_dict['item_id']
        batch_size, item_num = item_ids.shape

        # Get layer-wise embeddings
        layer_embs = []
        for layer_idx in range(self.n_layers + 1):
            layer_emb = self._get_layer_wise_embeddings(feed_dict, layer_idx, binarize=False)
            layer_embs.append(layer_emb)

        # Apply graph propagation
        if self.n_layers > 0 and hasattr(self, 'propagation_weights'):
            propagated_embs = self._graph_propagation(layer_embs)
        else:
            propagated_embs = layer_embs

        # Apply binarization if requested
        if binarize:
            binarized_embs = []
            scaling_factors = []
            for l in range(self.n_layers + 1):
                emb = propagated_embs[l]
                # Compute scaling factors (alpha in BiGeaR)
                alpha = torch.mean(torch.abs(emb), dim=-1, keepdim=True)  # [batch_size, item_num, feature_num, 1]
                q = self.binarize(emb)  # [batch_size, item_num, feature_num, emb_size]
                binarized_embs.append(q)
                scaling_factors.append(alpha)
            return binarized_embs, scaling_factors, propagated_embs
        else:
            return None, None, propagated_embs

    def _compute_fm_prediction(self, fm_vectors, linear_value=None):
        """Compute FM prediction from embedding vectors"""
        if len(fm_vectors.shape) == 4:  # [batch_size, item_num, feature_num, emb_size]
            # FM computation: 0.5 * [(Σv)^2 - Σ(v^2)]
            sum_vectors = fm_vectors.sum(dim=-2)  # [batch_size, item_num, emb_size]
            sum_squares = fm_vectors.pow(2).sum(dim=-2)  # [batch_size, item_num, emb_size]
            fm_interaction = 0.5 * (sum_vectors.pow(2) - sum_squares)  # [batch_size, item_num, emb_size]
            fm_score = fm_interaction.sum(dim=-1)  # [batch_size, item_num]
        else:
            fm_score = 0
        
        if linear_value is not None:
            return fm_score + linear_value
        return fm_score

    def get_prediction(self, feed_dict, binarize=False):
        """Get prediction scores with optional binarization"""
        if binarize:
            binarized_embs, scaling_factors, full_embs = self._get_embeddings_BiGeaR(feed_dict, binarize=True)
            
            # Apply layer weights
            layer_weights = torch.softmax(self.layer_weights, dim=0)
            
            scores = 0
            for l in range(self.n_layers + 1):
                w_l = layer_weights[l]
                q_emb = binarized_embs[l]  # [batch_size, item_num, feature_num, emb_size]
                alpha = scaling_factors[l]  # [batch_size, item_num, feature_num, 1]
                
                # Apply scaling
                scaled_emb = q_emb * alpha
                
                # Compute FM score for this layer
                layer_score = self._compute_fm_prediction(scaled_emb)
                scores = scores + w_l * layer_score
            
            return scores + self.overall_bias
        else:
            # Full precision prediction
            _, _, full_embs = self._get_embeddings_BiGeaR(feed_dict, binarize=False)
            
            # Weighted sum of layer predictions
            layer_weights = torch.softmax(self.layer_weights, dim=0)
            scores = 0
            
            for l in range(self.n_layers + 1):
                w_l = layer_weights[l]
                layer_emb = full_embs[l]
                layer_score = self._compute_fm_prediction(layer_emb)
                scores = scores + w_l * layer_score
            
            return scores + self.overall_bias

    def get_inference_distillation_loss(self, feed_dict):
        """Compute inference distillation loss (BiGeaR mid-stage reinforcement)"""
        u_ids = feed_dict['user_id']
        batch_size = u_ids.shape[0]
        
        # Get all item IDs
        all_i_ids = torch.arange(self.item_num).to(u_ids.device)
        
        # Create feed_dict for all items
        all_items_feed = feed_dict.copy()
        all_items_feed['item_id'] = all_i_ids.unsqueeze(0).repeat(batch_size, 1)
        
        # Get teacher (full-precision) scores
        with torch.no_grad():
            teacher_scores = self.get_prediction(all_items_feed, binarize=False)  # [batch_size, item_num]
        
        # Get student (binarized) scores
        student_scores = self.get_prediction(all_items_feed, binarize=True)  # [batch_size, item_num]
        
        # Get top-K items from teacher for each user
        k = min(self.top_k_distill, self.item_num)
        _, top_indices = torch.topk(teacher_scores, k=k, dim=1)
        
        # Compute ranking-aware weights (BiGeaR Equation 12)
        k_positions = torch.arange(1, k + 1).float().to(u_ids.device)
        weights = torch.exp(-0.1 * k_positions)  # Exponential decay
        
        # Compute distillation loss
        distill_loss = 0
        for pos in range(k):
            item_idx = top_indices[:, pos]
            student_score = student_scores[torch.arange(batch_size), item_idx]
            # Encourage student to give high scores to teacher's top items
            distill_loss = distill_loss - weights[pos] * torch.log(torch.sigmoid(student_score) + 1e-8)
        
        return distill_loss / k

    def forward(self, feed_dict):
        """Forward pass with BiGeaR multi-stage training"""
        if self.training:
            # Check if we're in pre-training phase (first few epochs)
            current_epoch = getattr(self, 'current_epoch', 0)
            
            if current_epoch < self.pretrain_epochs:
                # Pre-stage: full precision training
                prediction = self.get_prediction(feed_dict, binarize=False)
                return {'prediction': prediction}
            else:
                # Mid-stage: binarized training with distillation
                prediction = self.get_prediction(feed_dict, binarize=True)
                distill_loss = self.get_inference_distillation_loss(feed_dict)
                return {
                    'prediction': prediction,
                    'distill_loss': distill_loss
                }
        else:
            # Inference: use binarized model
            prediction = self.get_prediction(feed_dict, binarize=True)
            return {'prediction': prediction}

class BiGeaR_FM_CTR(ContextCTRModel, BiGeaR_FM_Base):
    reader, runner = 'ContextReader', 'CTRRunner'
    extra_log_args = ['emb_size', 'n_layers', 'gamma', 'distill_lambda', 'pretrain_epochs']

    @staticmethod
    def parse_model_args(parser):
        parser = BiGeaR_FM_Base.parse_model_args_FM(parser)
        return ContextCTRModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        # 先完成父类初始化
        ContextCTRModel.__init__(self, args, corpus)
        # 然后调用 BiGeaR 初始化
        self._define_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = BiGeaR_FM_Base.forward(self, feed_dict)
        out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
        out_dict['label'] = feed_dict['label'].view(-1)
        return out_dict

    def loss(self, out_dict):
        """
        CTR 版本的 loss 方法
        """
        if 'distill_loss' in out_dict:
            # Combined loss: main prediction loss + distillation loss
            # 创建只包含预测的字典给父类
            pred_dict = {'prediction': out_dict['prediction']}
            main_loss = super().loss(pred_dict)
            total_loss = main_loss + self.distill_lambda * out_dict['distill_loss']
            return total_loss
        else:
            return super().loss(out_dict)

class BiGeaR_FM_TopK(ContextModel, BiGeaR_FM_Base):
    reader, runner = 'ContextReader', 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'gamma', 'distill_lambda', 'pretrain_epochs']

    @staticmethod
    def parse_model_args(parser):
        parser = BiGeaR_FM_Base.parse_model_args_FM(parser)
        return ContextModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        # 先完成父类初始化
        ContextModel.__init__(self, args, corpus)
        # 然后调用 BiGeaR 初始化
        self._define_init(args, corpus)

    def forward(self, feed_dict):
        return BiGeaR_FM_Base.forward(self, feed_dict)

    def loss(self, out_dict):
        """
        TopK 版本的 loss 方法
        """
        if 'distill_loss' in out_dict:
            # For Top-K recommendation, use BPR loss
            predictions = out_dict['prediction']
            
            # 检查预测结果的形状，确保正确处理正负样本
            if len(predictions.shape) == 2:
                pos_pred = predictions[:, 0]  # Assuming first is positive
                neg_pred = predictions[:, 1:]  # Rest are negatives
                
                # BPR loss
                diff = pos_pred.unsqueeze(1) - neg_pred
                main_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
            else:
                # 如果形状不对，使用默认损失
                main_loss = -torch.log(torch.sigmoid(predictions) + 1e-8).mean()
            
            total_loss = main_loss + self.distill_lambda * out_dict['distill_loss']
            return total_loss
        else:
            # 检查父类 loss 方法的参数数量
            try:
                # 尝试调用父类的 loss 方法
                parent_loss = super().loss
                parent_params = inspect.signature(parent_loss).parameters
                
                if len(parent_params) == 2:  # 包括 self
                    return super().loss(out_dict)
                else:
                    # 如果父类需要更多参数，使用默认 BPR 实现
                    return self._default_bpr_loss(out_dict)
            except:
                # 如果出错，使用默认 BPR 实现
                return self._default_bpr_loss(out_dict)
    
    def _default_bpr_loss(self, out_dict):
        """
        默认的 BPR 损失实现
        """
        predictions = out_dict['prediction']
        
        if len(predictions.shape) == 2:
            pos_pred = predictions[:, 0]
            neg_pred = predictions[:, 1:]
            diff = pos_pred.unsqueeze(1) - neg_pred
            return -torch.log(torch.sigmoid(diff) + 1e-8).mean()
        else:
            return -torch.log(torch.sigmoid(predictions) + 1e-8).mean()