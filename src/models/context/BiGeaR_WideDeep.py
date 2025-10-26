# -*- coding: UTF-8 -*-
# @Author : Your Name
# @Email  : your_email@example.com

""" BiGeaR_WideDeep
Reference:
    "Learning Binarized Graph Representations with Multi-faceted Quantization Reinforcement for Top-K Recommendation"
    Yankai Chen et al., 2022.
    + Wide & Deep base architecture
"""

import torch
import torch.nn as nn
import numpy as np

from models.BaseContextModel import ContextCTRModel, ContextModel

class BiGeaR_WideDeep_Base(object):
    @staticmethod
    def parse_model_args_WideDeep(parser):
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
        parser.add_argument('--layers', type=str, default='[64]',
                           help='Size of each deep layer.')
        return parser

    def _define_init_params(self, args, corpus):
        self.vec_size = args.emb_size
        self.n_layers = args.n_layers
        self.gamma = args.gamma
        self.distill_lambda = args.distill_lambda
        self.top_k_distill = args.top_k_distill
        self.pretrain_epochs = args.pretrain_epochs
        self.dropout = args.dropout
        
        # Parse layers
        self.layers = eval(args.layers)
        
        self._define_params_BiGeaR_WideDeep()
        self.apply(self.init_weights)
    
    def _define_params_BiGeaR_WideDeep(self):    
        # Wide part: full precision (keeps important feature interactions)
        self.wide_embedding = nn.ModuleDict()
        for f in self.context_features:
            if f.endswith('_c') or f.endswith('_id'):
                self.wide_embedding[f] = nn.Embedding(self.feature_max[f], 1)
            else:
                self.wide_embedding[f] = nn.Linear(1, 1, bias=False)
        
        # Deep part: BiGeaR layer-wise embeddings for quantization
        self.deep_embedding_layers = nn.ModuleList()
        for layer in range(self.n_layers + 1):
            deep_embedding = nn.ModuleDict()
            for f in self.context_features:
                if f.endswith('_c') or f.endswith('_id'):
                    deep_embedding[f] = nn.Embedding(self.feature_max[f], self.vec_size)
                else:
                    deep_embedding[f] = nn.Linear(1, self.vec_size, bias=False)
            self.deep_embedding_layers.append(deep_embedding)
        
        # Deep MLP layers with optional binarization
        self.deep_layers = nn.ModuleList()
        input_size = len(self.context_features) * self.vec_size
        
        for idx, layer_size in enumerate(self.layers):
            self.deep_layers.append(nn.Linear(input_size, layer_size))
            self.deep_layers.append(nn.ReLU())
            if self.dropout > 0:
                self.deep_layers.append(nn.Dropout(p=self.dropout))
            input_size = layer_size
        
        # Final prediction layer
        self.prediction = nn.Linear(input_size, 1)
        
        # BiGeaR specific parameters
        self.layer_weights = nn.Parameter(torch.ones(self.n_layers + 1))
        self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)

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

    def _get_wide_output(self, feed_dict):
        """Wide part: full precision feature interactions"""
        item_ids = feed_dict['item_id']
        batch_size, item_num = item_ids.shape
        
        wide_output = 0
        for f in self.context_features:
            if f.endswith('_c') or f.endswith('_id'):
                emb = self.wide_embedding[f](feed_dict[f])
            else:
                emb = self.wide_embedding[f](feed_dict[f].float().unsqueeze(-1))
            
            # Expand to match item_num if needed
            if len(emb.shape) == 2:  # [batch_size, 1]
                emb = emb.unsqueeze(1).repeat(1, item_num, 1)  # [batch_size, item_num, 1]
            wide_output = wide_output + emb
        
        return wide_output.squeeze(-1)  # [batch_size, item_num]

    def _get_deep_embeddings(self, feed_dict, layer_idx, binarize=False):
        """Get deep embeddings for a specific layer with optional binarization"""
        item_ids = feed_dict['item_id']
        batch_size, item_num = item_ids.shape

        deep_embeddings = []
        deep_embedding = self.deep_embedding_layers[layer_idx]
        
        for f in self.context_features:
            if f.endswith('_c') or f.endswith('_id'):
                emb = deep_embedding[f](feed_dict[f])
            else:
                emb = deep_embedding[f](feed_dict[f].float().unsqueeze(-1))
            
            # Expand to match item_num if needed
            if len(emb.shape) == 2:  # [batch_size, emb_size]
                emb = emb.unsqueeze(1).repeat(1, item_num, 1)  # [batch_size, item_num, emb_size]
            
            # Apply binarization if requested (BiGeaR quantization)
            if binarize:
                # Compute scaling factors (alpha in BiGeaR)
                scaling_factor = torch.mean(torch.abs(emb), dim=-1, keepdim=True)
                binarized_emb = self.binarize(emb)
                emb = binarized_emb * scaling_factor
            
            deep_embeddings.append(emb)
        
        # Concatenate all feature embeddings
        deep_embedding_concat = torch.cat(deep_embeddings, dim=-1)  # [batch_size, item_num, feature_num * emb_size]
        
        return deep_embedding_concat

    def _get_deep_output(self, feed_dict, binarize=False):
        """Deep part: BiGeaR-style layer-wise deep network"""
        item_ids = feed_dict['item_id']
        batch_size, item_num = item_ids.shape
        
        # Get layer-wise embeddings with optional binarization
        layer_outputs = []
        for layer_idx in range(self.n_layers + 1):
            layer_emb = self._get_deep_embeddings(feed_dict, layer_idx, binarize=binarize)
            layer_outputs.append(layer_emb)
        
        # Apply layer weights (BiGeaR layer-wise importance)
        layer_weights = torch.softmax(self.layer_weights, dim=0)
        weighted_output = 0
        
        for l in range(self.n_layers + 1):
            w_l = layer_weights[l]
            layer_emb = layer_outputs[l]
            
            # Pass through deep MLP
            current_output = layer_emb
            for layer in self.deep_layers:
                # Handle different input dimensions
                if len(current_output.shape) == 3:  # [batch_size, item_num, features]
                    original_shape = current_output.shape
                    current_output = current_output.view(-1, original_shape[-1])
                    current_output = layer(current_output)
                    current_output = current_output.view(original_shape[0], original_shape[1], -1)
                else:
                    current_output = layer(current_output)
            
            # Final prediction for this layer
            if len(current_output.shape) == 3:
                original_shape = current_output.shape
                layer_pred = self.prediction(current_output.view(-1, original_shape[-1]))
                layer_pred = layer_pred.view(original_shape[0], original_shape[1])
            else:
                layer_pred = self.prediction(current_output).squeeze(-1)
            
            weighted_output = weighted_output + w_l * layer_pred
        
        return weighted_output

    def get_prediction(self, feed_dict, binarize=False):
        """Get combined Wide & Deep prediction"""
        # Wide part: always full precision
        wide_output = self._get_wide_output(feed_dict)
        
        # Deep part: optional binarization
        deep_output = self._get_deep_output(feed_dict, binarize=binarize)
        
        # Combine wide and deep
        combined_output = wide_output + deep_output
        
        return combined_output

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


class BiGeaR_WideDeepTopK(ContextModel, BiGeaR_WideDeep_Base):
    reader, runner = 'ContextReader', 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'gamma', 'distill_lambda', 'pretrain_epochs']

    @staticmethod
    def parse_model_args(parser):
        parser = BiGeaR_WideDeep_Base.parse_model_args_WideDeep(parser)
        return ContextModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        ContextModel.__init__(self, args, corpus)
        self._define_init_params(args, corpus)

    def forward(self, feed_dict):
        return BiGeaR_WideDeep_Base.forward(self, feed_dict)

    def loss(self, out_dict):
        """
        Loss function for BiGeaR training
        Args:
            out_dict: dictionary containing 'prediction' and optionally 'distill_loss'
        """
        if 'distill_loss' in out_dict:
            # Combined loss: main prediction loss + distillation loss
            predictions = out_dict['prediction']
            
            # For Top-K recommendation with multiple items per user
            if len(predictions.shape) == 2 and predictions.shape[1] > 1:
                # BPR loss: positive item should have higher score than negative items
                pos_pred = predictions[:, 0]  # First item is positive
                neg_pred = predictions[:, 1:]  # Rest are negatives
                
                # Calculate BPR loss
                diff = pos_pred.unsqueeze(1) - neg_pred
                main_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
            else:
                # Fallback to parent class loss
                main_loss = super().loss(out_dict)
            
            # Combine main loss with distillation loss
            total_loss = main_loss + self.distill_lambda * out_dict['distill_loss']
            return total_loss
        else:
            # Standard loss without distillation (pre-training phase)
            return super().loss(out_dict)