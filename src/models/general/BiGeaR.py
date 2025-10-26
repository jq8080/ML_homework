# -*- coding: UTF-8 -*-
# @Author  : Your Name
# @Email   : your_email@example.com

""" BiGeaR
Reference:
    "Learning Binarized Graph Representations with Multi-faceted Quantization Reinforcement for Top-K Recommendation"
    Yankai Chen et al., 2022.
CMD example:
    python main.py --model_name BiGeaR --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food' --n_layers 3 --gamma 10.0 --distill_lambda 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import GeneralModel
from models.BaseImpressionModel import ImpressionModel

class BiGeaRBase(object):
    @staticmethod
    def parse_model_args(parser):
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
        return parser

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.gamma = args.gamma
        self.distill_lambda = args.distill_lambda
        self.top_k_distill = args.top_k_distill
        
        self._base_define_params()
        self._init_graph(corpus)
        self.apply(self.init_weights)
    
    def _base_define_params(self):    
        # Full-precision embeddings for pre-training and teacher
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        
        # Layer weights for prediction
        self.layer_weights = nn.Parameter(torch.ones(self.n_layers + 1))
        
    def _init_graph(self, corpus):
        """Initialize the user-item interaction graph"""
        # This is a simplified version - in practice you'd use the actual graph structure
        # from your dataset
        self.graph = None  # Would be initialized from corpus in actual implementation
    
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
    
    def compute_layer_wise_embeddings(self, u_ids, i_ids, binarize=False):
        """Compute layer-wise embeddings with graph propagation"""
        batch_size = u_ids.shape[0]
        num_items = i_ids.shape[1] if i_ids.dim() > 1 else 1
        
        # Get initial embeddings
        u_emb0 = self.u_embeddings(u_ids)  # [batch_size, emb_size]
        if i_ids.dim() > 1:
            i_emb0 = self.i_embeddings(i_ids.view(-1)).view(batch_size, num_items, -1)  # [batch_size, num_items, emb_size]
        else:
            i_emb0 = self.i_embeddings(i_ids)  # [batch_size, emb_size]
        
        # Store layer-wise embeddings
        u_embs = [u_emb0]
        i_embs = [i_emb0]
        
        # Simplified graph propagation (in practice, use actual graph structure)
        for layer in range(self.n_layers):
            # Simplified propagation - in practice, use actual graph convolution
            u_emb = u_embs[-1]
            i_emb = i_embs[-1]
            
            # User aggregation from items (simplified)
            if i_emb.dim() > 2:  # Multiple items
                u_emb_next = i_emb.mean(dim=1)  # Average over items
            else:
                u_emb_next = i_emb
            
            # Item aggregation from users (simplified)
            if u_emb.dim() > 1:  # Multiple users (shouldn't happen in standard batch)
                i_emb_next = u_emb.mean(dim=0, keepdim=True).expand_as(i_emb)
            else:
                i_emb_next = u_emb.unsqueeze(0).expand_as(i_emb)
            
            u_embs.append(u_emb_next)
            i_embs.append(i_emb_next)
        
        if binarize:
            # Apply layer-wise quantization
            u_bin_embs = []
            u_scalers = []
            i_bin_embs = []
            i_scalers = []
            
            for l in range(self.n_layers + 1):
                # Binarize user embeddings
                u_emb = u_embs[l]
                u_alpha = torch.mean(torch.abs(u_emb), dim=1, keepdim=True)  # [batch_size, 1]
                u_q = self.binarize(u_emb)  # [batch_size, emb_size]
                
                u_bin_embs.append(u_q)
                u_scalers.append(u_alpha)
                
                # Binarize item embeddings
                i_emb = i_embs[l]
                if i_emb.dim() > 2:
                    i_alpha = torch.mean(torch.abs(i_emb), dim=2, keepdim=True)  # [batch_size, num_items, 1]
                    i_q = self.binarize(i_emb)  # [batch_size, num_items, emb_size]
                else:
                    i_alpha = torch.mean(torch.abs(i_emb), dim=1, keepdim=True)  # [batch_size, 1]
                    i_q = self.binarize(i_emb)  # [batch_size, emb_size]
                
                i_bin_embs.append(i_q)
                i_scalers.append(i_alpha)
            
            return (u_bin_embs, u_scalers, i_bin_embs, i_scalers, 
                   u_embs, i_embs)
        else:
            return (None, None, None, None, u_embs, i_embs)
    
    def get_prediction(self, u_ids, i_ids, binarize=False):
        """Get prediction scores for user-item pairs"""
        if binarize:
            (u_bin_embs, u_scalers, i_bin_embs, i_scalers, 
             u_full_embs, i_full_embs) = self.compute_layer_wise_embeddings(u_ids, i_ids, binarize=True)
            
            # Apply layer weights
            layer_weights = F.softmax(self.layer_weights, dim=0)  # [n_layers+1]
            
            scores = 0
            for l in range(self.n_layers + 1):
                w_l = layer_weights[l]
                u_q = u_bin_embs[l]  # [batch_size, emb_size]
                u_alpha = u_scalers[l]  # [batch_size, 1]
                
                if i_bin_embs[l].dim() > 2:
                    i_q = i_bin_embs[l]  # [batch_size, num_items, emb_size]
                    i_alpha = i_scalers[l]  # [batch_size, num_items, 1]
                else:
                    i_q = i_bin_embs[l]  # [batch_size, emb_size]
                    i_alpha = i_scalers[l]  # [batch_size, 1]
                
                # Convert from {-1, 1} to {0, 1} for bitwise operation approximation
                u_q_binary = (u_q + 1) / 2  # [batch_size, emb_size]
                if i_q.dim() > 2:
                    i_q_binary = (i_q + 1) / 2  # [batch_size, num_items, emb_size]
                else:
                    i_q_binary = (i_q + 1) / 2  # [batch_size, emb_size]
                
                # Simulate XNOR and Popcount operations
                if i_q.dim() > 2:
                    u_q_expanded = u_q_binary.unsqueeze(1)  # [batch_size, 1, emb_size]
                    xnor_result = 1 - torch.abs(u_q_expanded - i_q_binary)  # [batch_size, num_items, emb_size]
                    popcount = torch.sum(xnor_result, dim=2, keepdim=True)  # [batch_size, num_items, 1]
                else:
                    xnor_result = 1 - torch.abs(u_q_binary - i_q_binary)  # [batch_size, emb_size]
                    popcount = torch.sum(xnor_result, dim=1, keepdim=True)  # [batch_size, 1]
                
                # Final score for this layer
                layer_score = w_l ** 2 * u_alpha * i_alpha * (2 * popcount - self.emb_size)
                scores = scores + layer_score
            
            return scores.squeeze()
        else:
            # Full-precision prediction
            _, _, _, _, u_embs, i_embs = self.compute_layer_wise_embeddings(u_ids, i_ids, binarize=False)
            
            # Apply layer weights and sum
            layer_weights = F.softmax(self.layer_weights, dim=0)
            
            u_final = 0
            i_final = 0
            
            for l in range(self.n_layers + 1):
                w_l = layer_weights[l]
                u_final = u_final + w_l * u_embs[l]
                i_final = i_final + w_l * i_embs[l]
            
            if i_final.dim() > 2:
                scores = torch.sum(u_final.unsqueeze(1) * i_final, dim=2)  # [batch_size, num_items]
            else:
                scores = torch.sum(u_final * i_final, dim=1)  # [batch_size]
            
            return scores
    
    def get_inference_distillation_loss(self, u_ids, all_i_ids):
        """Compute inference distillation loss"""
        batch_size = u_ids.shape[0]
        num_items = all_i_ids.shape[1] if all_i_ids.dim() > 1 else all_i_ids.shape[0]
        
        # Get teacher (full-precision) scores for all items
        with torch.no_grad():
            teacher_scores = self.get_prediction(u_ids, all_i_ids, binarize=False)  # [batch_size, num_items]
        
        # Get student (binarized) scores for all items
        student_scores = self.get_prediction(u_ids, all_i_ids, binarize=True)  # [batch_size, num_items]
        
        # Get top-K items from teacher for each user
        _, top_indices = torch.topk(teacher_scores, k=min(self.top_k_distill, num_items), dim=1)
        
        # Compute ranking-aware weights
        k_positions = torch.arange(1, min(self.top_k_distill, num_items) + 1).float().to(u_ids.device)
        weights = torch.exp(-0.1 * k_positions)  # Exponential decay
        
        # Compute distillation loss
        distill_loss = 0
        for k in range(min(self.top_k_distill, num_items)):
            item_idx = top_indices[:, k]
            student_score = student_scores[torch.arange(batch_size), item_idx]
            distill_loss = distill_loss - weights[k] * torch.log(torch.sigmoid(student_score) + 1e-8)
        
        return distill_loss / min(self.top_k_distill, num_items)
    
    def forward(self, feed_dict, binarize=False):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1] or [batch_size]

        prediction = self.get_prediction(u_ids, i_ids, binarize=binarize)
        
        if binarize and self.training and hasattr(self, 'get_inference_distillation_loss'):
            # For training with distillation
            all_i_ids = torch.arange(self.item_num).to(u_ids.device)
            distill_loss = self.get_inference_distillation_loss(u_ids, all_i_ids)
            return {
                'prediction': prediction.view(feed_dict['batch_size'], -1), 
                'distill_loss': distill_loss
            }
        else:
            return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

class BiGeaR(GeneralModel, BiGeaRBase):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'gamma', 'distill_lambda']

    @staticmethod
    def parse_model_args(parser):
        parser = BiGeaRBase.parse_model_args(parser)
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        # Use full-precision for general model (pre-training phase)
        out_dict = BiGeaRBase.forward(self, feed_dict, binarize=False)
        return {'prediction': out_dict['prediction']}

class BiGeaRImpression(ImpressionModel, BiGeaRBase):
    reader = 'ImpressionReader'
    runner = 'ImpressionRunner'
    extra_log_args = ['emb_size', 'n_layers', 'gamma', 'distill_lambda']

    @staticmethod
    def parse_model_args(parser):
        parser = BiGeaRBase.parse_model_args(parser)
        return ImpressionModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        ImpressionModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        # Use binarized embeddings for impression model (quantization training phase)
        out_dict = BiGeaRBase.forward(self, feed_dict, binarize=True)
        return out_dict

    def loss(self, out_dict: dict, target=None):
        # Get the main prediction loss
        if 'distill_loss' in out_dict:
            # BPR loss with distillation
            prediction = out_dict['prediction']
            distill_loss = out_dict['distill_loss']
            
            # Compute standard BPR loss from ImpressionModel
            main_loss = super().loss({'prediction': prediction}, target)
            
            # Combine with distillation loss
            total_loss = main_loss + self.distill_lambda * distill_loss
            return total_loss
        else:
            # Standard loss without distillation
            return super().loss(out_dict, target)