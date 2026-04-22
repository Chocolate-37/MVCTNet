"""
MVCTNet :3D point cloud + 2D multi-view image multimodal fusion

Architecture: Encoder-Decoder (U-Net style) + multimodal fusion modules
Input:  [B, N, 6] point cloud (xyz + normals) + optional 2D image features
Output: [B, N, 3] per-point segmentation probabilities (crown / trunk / interference)

Fusion strategy:
  1. Global fusion — inject 2D global features at the l3_points encoder bottleneck
  2. Point-level fusion — inject 2D local features before the final classification head
  3. Modular design — each fusion point can be independently enabled / disabled
  4. Backward compatible — without image features behaves identically to 3D-only model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mvctnet_utils import MVCTNetSetAbstraction, MVCTNetFeaturePropagation
from time import time
import numpy as np
import math
from scipy import stats  # 

#  GUCLuncertainty weight
from .gucl_modules import GUCL_Loss as GUCL_Loss_Module

class BiologicalConstants:
    """
    Rubber tree biological constants
    biological parameters based on allometric growth theory
    """
    # organ allometric growth coefficients ()
    ORGAN_ALLOMETRIC_COEFFS = {
        'crown': {'base': 1.2, 'range': (0.8, 1.8)},    # crown: preferential growth
        'trunk': {'base': 0.8, 'range': (0.5, 1.2)},    # trunk: stable growth  
        'noise': {'base': 0.3, 'range': (0.1, 0.6)}     # interference: suppressed growth
    }
    
    # multi-scale quantile thresholds (2024 Nature Multi-scaling allometry)
    QUANTILE_THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]  # key quantile values
    
    # growth-phase dynamic coefficients
    GROWTH_PHASES = {
        'juvenile': {'crown_priority': 1.5, 'competition': 0.8},   # juvenile: crown-first
        'mature': {'crown_priority': 1.0, 'competition': 1.0},     # mature: balanced
        'stress': {'crown_priority': 0.8, 'competition': 1.5}      # stress: intense competition
    }

class HeterogeneousGraph:

    def __init__(self):
        self.nodes = {}      # 
        self.edges = {}      # 
        self.node_types = [] # 
        self.edge_types = [] # 
    
    def add_node_type(self, node_type, features):
        """"""
        self.nodes[node_type] = features
        if node_type not in self.node_types:
            self.node_types.append(node_type)
    
    def add_edge_type(self, edge_type, edge_index, edge_attr=None):
        """"""
        self.edges[edge_type] = {
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }
        if edge_type not in self.edge_types:
            self.edge_types.append(edge_type)
    
    def to(self, device):
        """"""
        for node_type in self.nodes:
            if torch.is_tensor(self.nodes[node_type]):
                self.nodes[node_type] = self.nodes[node_type].to(device)
        
        for edge_type in self.edges:
            if torch.is_tensor(self.edges[edge_type]['edge_index']):
                self.edges[edge_type]['edge_index'] = self.edges[edge_type]['edge_index'].to(device)
            if self.edges[edge_type]['edge_attr'] is not None and torch.is_tensor(self.edges[edge_type]['edge_attr']):
                self.edges[edge_type]['edge_attr'] = self.edges[edge_type]['edge_attr'].to(device)
        
        return self
class HeteroMessagePassing(nn.Module):
    def __init__(self):
        super(HeteroMessagePassing, self).__init__()
    
    def message(self, src_feat, edge_attr=None):
        """"""
        if edge_attr is not None:
            return src_feat * edge_attr.unsqueeze(-1)
        return src_feat
    
    def aggregate(self, messages, edge_index, num_nodes):
        """"""
        # scatter_add
        out = torch.zeros(num_nodes, messages.size(-1), device=messages.device)
        out.scatter_add_(0, edge_index[1].unsqueeze(-1).expand_as(messages), messages)
        return out
    
    def update(self, aggr_out, node_feat):
        """"""
        return aggr_out + node_feat


class HeteroGraphConv(HeteroMessagePassing):
    
    def __init__(self, in_channels_dict, out_channels_dict, edge_types):
        """
        Args:
            in_channels_dict: feature dim
            out_channels_dict: feature dim  
            edge_types: 
        """
        super(HeteroGraphConv, self).__init__()
        
        self.in_channels_dict = in_channels_dict
        self.out_channels_dict = out_channels_dict
        self.edge_types = edge_types
        
        self.edge_transforms = nn.ModuleDict()
        for edge_type in edge_types:
            src_type, dst_type = edge_type.split('_to_')
            in_dim = in_channels_dict[src_type]
            out_dim = out_channels_dict[dst_type]
            
            self.edge_transforms[edge_type] = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim)
            )
        
        self.node_updates = nn.ModuleDict()
        for node_type in out_channels_dict:
            self.node_updates[node_type] = nn.Sequential(
                nn.Linear(out_channels_dict[node_type], out_channels_dict[node_type]),
                nn.LayerNorm(out_channels_dict[node_type]),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, hetero_graph):
        """
        Args:
            hetero_graph: HeterogeneousGraph
        Returns:
            updated_graph: 
        """
        updated_nodes = {}
        
        for node_type in hetero_graph.node_types:
            if node_type in self.out_channels_dict:
                num_nodes = hetero_graph.nodes[node_type].size(0)
                out_dim = self.out_channels_dict[node_type]
                updated_nodes[node_type] = torch.zeros(
                    num_nodes, out_dim, 
                    device=hetero_graph.nodes[node_type].device
                )
        
        for edge_type in self.edge_types:
            if edge_type not in hetero_graph.edges:
                continue
                
            src_type, dst_type = edge_type.split('_to_')
            edge_data = hetero_graph.edges[edge_type]
            edge_index = edge_data['edge_index']
            edge_attr = edge_data.get('edge_attr', None)
            
            src_feat = hetero_graph.nodes[src_type]
            
            transformed_feat = self.edge_transforms[edge_type](src_feat)
            
            src_indices = edge_index[0]
            messages = self.message(transformed_feat[src_indices], edge_attr)
            
            num_dst_nodes = hetero_graph.nodes[dst_type].size(0)
            aggr_messages = self.aggregate(messages, edge_index, num_dst_nodes)
            
            if dst_type in updated_nodes:
                updated_nodes[dst_type] = updated_nodes[dst_type] + aggr_messages
        
        for node_type in updated_nodes:
            if node_type in hetero_graph.nodes:
                #  + 
                original_feat = hetero_graph.nodes[node_type]
                if original_feat.size(-1) == updated_nodes[node_type].size(-1):
                    updated_nodes[node_type] = updated_nodes[node_type] + original_feat
                
                updated_nodes[node_type] = self.node_updates[node_type](updated_nodes[node_type])
        
        updated_graph = HeterogeneousGraph()
        for node_type in hetero_graph.node_types:
            if node_type in updated_nodes:
                updated_graph.add_node_type(node_type, updated_nodes[node_type])
            else:
                updated_graph.add_node_type(node_type, hetero_graph.nodes[node_type])
        
        for edge_type in hetero_graph.edge_types:
            edge_data = hetero_graph.edges[edge_type]
            updated_graph.add_edge_type(edge_type, edge_data['edge_index'], edge_data['edge_attr'])
        
        return updated_graph
class BAHGHeteroConv(HeteroGraphConv):
    
    def __init__(self, in_channels_dict, out_channels_dict, edge_types, boundary_threshold=0.3):
        super(BAHGHeteroConv, self).__init__(in_channels_dict, out_channels_dict, edge_types)
        
        self.boundary_threshold = boundary_threshold
        
        self.boundary_detectors = nn.ModuleDict()
        for node_type in in_channels_dict:
            self.boundary_detectors[node_type] = nn.Sequential(
                nn.Linear(in_channels_dict[node_type], 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        self.boundary_weight_nets = nn.ModuleDict()
        for edge_type in edge_types:
            self.boundary_weight_nets[edge_type] = nn.Sequential(
                nn.Linear(2, 32),  # boundary scores + boundary scores
                nn.ReLU(inplace=True),
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
    
    def detect_node_boundaries(self, hetero_graph):
        """detect boundary probability for each node"""
        boundary_scores = {}
        
        for node_type in hetero_graph.node_types:
            if node_type in self.boundary_detectors:
                node_feat = hetero_graph.nodes[node_type]
                boundary_scores[node_type] = self.boundary_detectors[node_type](node_feat).squeeze(-1)
            else:
                # no boundary detector for this type — return neutral score 0.5
                num_nodes = hetero_graph.nodes[node_type].size(0)
                boundary_scores[node_type] = torch.ones(
                    num_nodes, device=hetero_graph.nodes[node_type].device
                ) * 0.5
        
        return boundary_scores
    
    def compute_boundary_weights(self, edge_type, edge_index, boundary_scores):
        """compute boundary-aware edge weights"""
        src_type, dst_type = edge_type.split('_to_')
        
        # boundary scores
        src_boundary = boundary_scores[src_type][edge_index[0]]  # [num_edges]
        dst_boundary = boundary_scores[dst_type][edge_index[1]]  # [num_edges]
        
        # boundary scores
        boundary_input = torch.stack([src_boundary, dst_boundary], dim=-1)  # [num_edges, 2]
        
        # boundary weights
        boundary_weights = self.boundary_weight_nets[edge_type](boundary_input).squeeze(-1)  # [num_edges]
        
        # boundary threshold
        boundary_mask = (src_boundary > self.boundary_threshold) | (dst_boundary > self.boundary_threshold)
        enhanced_weights = torch.where(boundary_mask, boundary_weights * 2.0, boundary_weights)
        
        return enhanced_weights
    
    def message(self, src_feat, edge_attr=None, boundary_weights=None):
        """boundary-aware message computation"""
        messages = src_feat
        
        if edge_attr is not None:
            messages = messages * edge_attr.unsqueeze(-1)
        
        if boundary_weights is not None:
            # boundary weights modulate message intensity
            messages = messages * boundary_weights.unsqueeze(-1)
        
        return messages
    
    def forward(self, hetero_graph):
        """BAHG forward pass"""
        
        #  Step 1: detect node boundaries
        boundary_scores = self.detect_node_boundaries(hetero_graph)
        
        updated_nodes = {}
        boundary_info = {
            'node_boundaries': boundary_scores,
            'edge_boundaries': {}
        }
        
        for node_type in hetero_graph.node_types:
            if node_type in self.out_channels_dict:
                num_nodes = hetero_graph.nodes[node_type].size(0)
                out_dim = self.out_channels_dict[node_type]
                updated_nodes[node_type] = torch.zeros(
                    num_nodes, out_dim, 
                    device=hetero_graph.nodes[node_type].device
                )
        
        #  Step 2: boundary-aware message passing
        for edge_type in self.edge_types:
            if edge_type not in hetero_graph.edges:
                continue
                
            src_type, dst_type = edge_type.split('_to_')
            edge_data = hetero_graph.edges[edge_type]
            edge_index = edge_data['edge_index']
            edge_attr = edge_data.get('edge_attr', None)
            
            #  compute boundary-aware edge weights
            boundary_weights = self.compute_boundary_weights(edge_type, edge_index, boundary_scores)
            boundary_info['edge_boundaries'][edge_type] = boundary_weights
            
            src_feat = hetero_graph.nodes[src_type]
            transformed_feat = self.edge_transforms[edge_type](src_feat)
            
            #  boundary-aware message computation
            src_indices = edge_index[0]
            messages = self.message(
                transformed_feat[src_indices], 
                edge_attr, 
                boundary_weights
            )
            
            num_dst_nodes = hetero_graph.nodes[dst_type].size(0)
            aggr_messages = self.aggregate(messages, edge_index, num_dst_nodes)
            
            if dst_type in updated_nodes:
                updated_nodes[dst_type] = updated_nodes[dst_type] + aggr_messages
        
        #  Step 3: node feature update
        for node_type in updated_nodes:
            if node_type in hetero_graph.nodes:
                original_feat = hetero_graph.nodes[node_type]
                if original_feat.size(-1) == updated_nodes[node_type].size(-1):
                    updated_nodes[node_type] = updated_nodes[node_type] + original_feat
                
                updated_nodes[node_type] = self.node_updates[node_type](updated_nodes[node_type])
        
        updated_graph = HeterogeneousGraph()
        for node_type in hetero_graph.node_types:
            if node_type in updated_nodes:
                updated_graph.add_node_type(node_type, updated_nodes[node_type])
            else:
                updated_graph.add_node_type(node_type, hetero_graph.nodes[node_type])
        
        for edge_type in hetero_graph.edge_types:
            edge_data = hetero_graph.edges[edge_type]
            updated_graph.add_edge_type(edge_type, edge_data['edge_index'], edge_data['edge_attr'])
        
        return updated_graph, boundary_info
class ALFENetwork(nn.Module):
    
    def __init__(self, 
                 input_dim,
                 target_dim,
                 num_organs=3,
                 allometric_coeffs=[1.2, 0.8, 1.0],
                 competition_strength=0.3,
                 adaptive_factor=0.1,
                 debug_mode=False,
                 verbose_level=1,
                 ):
        """
        ALFE
        Args:
            input_dim: feature dim
            target_dim: feature dim
            num_organs:  (3: //)
            allometric_coeffs: 
            competition_strength: 
            adaptive_factor: adaptive factor
        """
        super(ALFENetwork, self).__init__()
        
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.num_organs = num_organs
        self.allometric_coeffs = torch.tensor(allometric_coeffs, dtype=torch.float32)
        self.competition_strength = competition_strength
        self.adaptive_factor = adaptive_factor
        self.debug_mode = debug_mode
        self.verbose_level = verbose_level
        
        #  organ ratio estimation MLP
        self.organ_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_organs),
            nn.Softmax(dim=-1)  # output organ probability distribution
        )
        
        #  learnable inter-organ competition matrix
        self.competition_matrix = nn.Parameter(torch.tensor([
            [1.0,  -0.3, -0.1],
            [-0.2,  1.0, -0.2],
            [-0.1, -0.1,  1.0]
        ], dtype=torch.float32))
        
        #  adaptive feature projection network
        # designed to handle variable input dimensions
        if input_dim != target_dim:
            self.adaptive_projection = nn.Sequential(
                nn.Linear(input_dim, max(input_dim, target_dim)),
                nn.LayerNorm(max(input_dim, target_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(max(input_dim, target_dim), target_dim)
            )
        else:
            self.adaptive_projection = nn.Identity()
        
        #  feature reorganisation network (post-allometric modulation)
        self.feature_reorganizer = nn.Sequential(
            nn.Linear(target_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        if self.debug_mode and self.verbose_level >= 2:  # 
            print(f" ALFE: {input_dim} → {target_dim}")
            print(f"   : {allometric_coeffs}")
            print(f"   : {competition_strength}")
    
    def _debug_print(self, level, message):
        """"""
        if self.debug_mode and self.verbose_level >= level:
            print(message)
    
    def estimate_organ_ratios(self, features):
        """
        Args:
            features: [B, C, N]
        Returns:
            organ_probs: [B, num_organs] 
        """
        B, C, N = features.shape
        
        global_feat = features.mean(dim=2)  # [B, C]
        
        organ_probs = self.organ_classifier(global_feat)  # [B, num_organs]
        
        if self.debug_mode:
            avg_probs = organ_probs.mean(dim=0).detach().cpu().numpy()
            print(f"   Step 1: organ ratio estimation: ={avg_probs[0]:.3f}, ={avg_probs[1]:.3f}, ={avg_probs[2]:.3f}")
        
        return organ_probs
    
    def compute_allometric_scaling(self, organ_probs):
        """
        Args:
            organ_probs: [B, num_organs]
        Returns:
            scaling_factors: [B] 
        """
        device = organ_probs.device
        B = organ_probs.shape[0]
        
        # move allometric coefficients to correct device
        allometric_coeffs = self.allometric_coeffs.to(device)  # [num_organs]
        
        #  compute organ competition effects
        # per-sample competition matrix modulation
        competition_effects = torch.matmul(organ_probs, self.competition_matrix.to(device))  # [B, num_organs]
        
        #  competition-modulated allometric coefficients
        adjusted_coeffs = allometric_coeffs.unsqueeze(0) * (1 + self.competition_strength * competition_effects)  # [B, num_organs]
        
        #  Step 2: allometric scaling computation
        # weighted sum of allometric factors by organ probabilities
        base_scaling = torch.sum(organ_probs * adjusted_coeffs, dim=1)  # [B]
        
        #  adaptive modulation (prevents over-scaling)
        adaptive_scaling = 1.0 + self.adaptive_factor * (base_scaling - 1.0)  # [B]
        
        #  stability constraint (clamp scaling range)
        scaling_factors = torch.clamp(adaptive_scaling, 0.5, 2.0)  # [B]
        
        if self.debug_mode:
            avg_scaling = scaling_factors.mean().detach().item()
            print(f"   : {avg_scaling:.4f}")
        
        return scaling_factors
    
    def apply_allometric_evolution(self, features, scaling_factors):
        """
        Step 3: allometric feature evolution
        Args:
            features: [B, C, N] 
            scaling_factors: [B]
        Returns:
            evolved_features: [B, target_dim, N] 
        """
        B, C, N = features.shape
        
        #  dimensional projection (if required)
        if C != self.target_dim:
            # rearrange to [B, N, C] for linear transform
            features_reshaped = features.permute(0, 2, 1)  # [B, N, C]
            projected_features = self.adaptive_projection(features_reshaped)  # [B, N, target_dim]
            features = projected_features.permute(0, 2, 1)  # [B, target_dim, N]
        
        #  apply allometric modulation to feature magnitude
        # broadcast scaling factor across feature channels
        scaling_factors = scaling_factors.view(B, 1, 1)  # [B, 1, 1]
        modulated_features = features * scaling_factors  # [B, target_dim, N]
        
        #  feature reorganisation and regularisation
        modulated_features = modulated_features.permute(0, 2, 1)  # [B, N, target_dim]
        reorganized_features = self.feature_reorganizer(modulated_features)  # [B, N, target_dim]
        evolved_features = reorganized_features.permute(0, 2, 1)  # [B, target_dim, N]
        
        #  residual connection for training stability
        if features.shape == evolved_features.shape:
            evolved_features = evolved_features + features
        
        return evolved_features
    
    def forward(self, features):
        """
        ALFE -
        Args:
            features: [B, C, N]
        Returns:
            evolved_features: [B, target_dim, N] 
            evolution_info: dict 
        """
        #  1: Step 1: organ ratio estimation
        organ_probs = self.estimate_organ_ratios(features)
        
        #  2: Step 2: allometric scaling computation
        scaling_factors = self.compute_allometric_scaling(organ_probs)
        
        #  3: Step 3: allometric feature evolution
        evolved_features = self.apply_allometric_evolution(features, scaling_factors)
        
        #  record evolution metadata
        evolution_info = {
            'organ_probs': organ_probs,
            'scaling_factors': scaling_factors,
            'input_shape': features.shape,
            'output_shape': evolved_features.shape
        }
        
        if self.debug_mode and self.verbose_level >= 3:  # 
            print(f" ALFE: {features.shape} → {evolved_features.shape}")
        
        return evolved_features, evolution_info


class MultiModalFusionModule(nn.Module):
    """
    BAHG Cross-Modal Heterogeneous Graph Fusion Module
    Fuses 2D image features into the 3D point cloud feature space
    Supports: global fusion + local fusion + boundary-aware graph conv + configurable strategy
    Key innovation: boundary-aware cross-modal information exchange
    """
    
    def __init__(self, 
                 image_feature_dim=256,
                 point_global_dim=512, 
                 point_local_dim=128,
                 enable_global_fusion=True,
                 enable_local_fusion=True,
                 fusion_method='concat',
                 enable_BAHG=True,  #  
                 boundary_threshold=0.3,      #  boundary detection threshold
                 debug_mode=True):
        """
        Args:
            image_feature_dim: image feature dimension (256)
            point_global_dim: 3Dfeature dim (512) 
            point_local_dim: 3Dfeature dim (128)
            enable_global_fusion: 
            enable_local_fusion: 
            fusion_method:  ('concat', 'add', 'attention')
            debug_mode: 
        """
        super(MultiModalFusionModule, self).__init__()
        
        #  params
        self.image_feature_dim = image_feature_dim
        self.point_global_dim = point_global_dim
        self.point_local_dim = point_local_dim
        self.enable_global_fusion = enable_global_fusion
        self.enable_local_fusion = enable_local_fusion
        self.fusion_method = fusion_method
        self.enable_BAHG = enable_BAHG  #  
        self.boundary_threshold = boundary_threshold        #  boundary detection threshold
        self.debug_mode = debug_mode
        
        if self.enable_global_fusion:
            if fusion_method == 'concat':
                # concatenate then project down - LayerNorm preferred over BatchNorm1d for point cloud features
                self.global_fusion = nn.Sequential(
                    nn.Linear(point_global_dim + image_feature_dim, point_global_dim),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(point_global_dim),  # LayerNorm
                    nn.Dropout(0.1)
                )
            elif fusion_method == 'attention':
                self.global_attention = nn.MultiheadAttention(
                    embed_dim=point_global_dim, 
                    num_heads=8, 
                    dropout=0.1
                )
                self.global_proj = nn.Linear(image_feature_dim, point_global_dim)
            else:  # 'add'
                self.global_proj = nn.Linear(image_feature_dim, point_global_dim)
        
        if self.enable_local_fusion:
            if fusion_method == 'concat':
                # 2D local feature adapter
                self.local_image_adapter = nn.Sequential(
                    nn.AdaptiveAvgPool2d((8, 8)),  # [B, 256, 16, 16] → [B, 256, 8, 8]
                    nn.Flatten(2),                 # [B, 256, 64]
                    nn.Linear(64, 1),              # [B, 256, 1] → broadcast to all points
                    nn.ReLU(inplace=True)
                )
                # process concatenated features
                self.local_fusion = nn.Sequential(
                    nn.Conv1d(point_local_dim + image_feature_dim, point_local_dim, 1),
                    nn.BatchNorm1d(point_local_dim), 
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                )
            elif fusion_method == 'attention':
                # cross-modal attention
                self.local_attention = nn.MultiheadAttention(
                    embed_dim=point_local_dim,
                    num_heads=4,
                    dropout=0.1
                )
                self.local_image_proj = nn.Linear(image_feature_dim, point_local_dim)
            else:  # 'add'
                self.local_image_proj = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),  # 
                    nn.Flatten(),                  # [B, 256]
                    nn.Linear(image_feature_dim, point_local_dim)
                )

        if self.enable_BAHG:
            #  heterogeneous graph configuration
            self.hetero_node_types = ['point_3d', 'patch_2d', 'view_global']
            self.hetero_edge_types = [
                'point_3d_to_point_3d',    # 3D-3D
                'patch_2d_to_point_3d',    # 2D-3D
                'view_global_to_point_3d', # 3D global
                'point_3d_to_patch_2d',    # 3D-2D
                'patch_2d_to_patch_2d',    # 2D-2D
                'view_global_to_patch_2d'  # 2D global
            ]
            
            #  feature dim ()
            self.hetero_in_channels = {
                'point_3d': point_global_dim,    # 3D feature dim: 512
                'patch_2d': image_feature_dim,   # 2D feature dim: 256
                'view_global': image_feature_dim # feature dim: 256
            }
            
            #   ()
            unified_dim = point_global_dim  # 3D 512
            self.hetero_out_channels = {
                'point_3d': unified_dim,     # 3D feature dim: 512
                'patch_2d': unified_dim,     # 2D feature dim: 512 ()
                'view_global': unified_dim   # feature dim: 512 ()
            }
            
            if self.debug_mode:
                print(f" :")
                print(f"   : {self.hetero_in_channels}")
                print(f"   : {self.hetero_out_channels}")
            
            #  multi-layer BAHG convolution design
            self.num_hetero_layers = 2
            self.hetero_conv_layers = nn.ModuleList()
            
            for i in range(self.num_hetero_layers):
                if i == 0:
                    #  layer 0: original dims -> unified dim
                    layer_in_channels = self.hetero_in_channels.copy()    # : {point_3d: 512, patch_2d: 256, view_global: 256}
                    layer_out_channels = self.hetero_out_channels.copy()  # : {point_3d: 512, patch_2d: 512, view_global: 512}
                else:
                    #  subsequent layers: unified dim -> unified dim
                    layer_in_channels = self.hetero_out_channels.copy()   # : {point_3d: 512, patch_2d: 512, view_global: 512}
                    layer_out_channels = self.hetero_out_channels.copy()  # : {point_3d: 512, patch_2d: 512, view_global: 512}
                
                # create BAHG conv for this layer
                hetero_conv = BAHGHeteroConv(
                    in_channels_dict=layer_in_channels,
                    out_channels_dict=layer_out_channels,
                    edge_types=self.hetero_edge_types,
                    boundary_threshold=boundary_threshold
                )
                
                self.hetero_conv_layers.append(hetero_conv)
                
                if self.debug_mode:
                    print(f"  {i}: {layer_in_channels} -> {layer_out_channels}")
            
            #  heterogeneous graph output projection
            if self.enable_global_fusion:
                self.hetero_to_global = nn.Sequential(
                    nn.Linear(point_global_dim, point_global_dim),
                    nn.LayerNorm(point_global_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                )
            
            if self.enable_local_fusion:
                self.hetero_to_local = nn.Sequential(
                    nn.Conv1d(point_local_dim, point_local_dim, 1),
                    nn.BatchNorm1d(point_local_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                )
        
        if self.debug_mode:
            print(f" BAHG :")
            print(f"   image feature dimension: {image_feature_dim}")
            print(f"   : {'' if enable_global_fusion else ''}")
            print(f"   : {'' if enable_local_fusion else ''}")
            print(f"   : {'' if enable_BAHG else ''}")
            print(f"   : {fusion_method}")
            if enable_BAHG:
                print(f"   boundary threshold: {boundary_threshold}")
    
    def build_heterogeneous_graph(self, point_global_feat, image_global_feat, image_local_feat):
        """
        BAHG
        Args:
            point_global_feat: [B, point_C, N] 3D
            image_global_feat: [B, image_C] 2D  
            image_local_feat: [B, image_C, H, W] 2D
        Returns:
            hetero_graph: HeterogeneousGraph
        """
        if not self.enable_BAHG:
            return None
            
        #  get feature dimensions for each modality
        B, point_C, N = point_global_feat.shape      # 3D: [B, 512, N]
        _, image_C, H, W = image_local_feat.shape     # 2D: [B, 256, H, W]
        
        device = point_global_feat.device
        hetero_graph = HeterogeneousGraph()
        
        if self.debug_mode:
            print(f" :")
            print(f"   3D: {point_global_feat.shape} -> : {point_C}")
            print(f"   2D: {image_local_feat.shape} -> : {image_C}")
            print(f"   2D: {image_global_feat.shape}")
        
        #  Step 1: add heterogeneous nodes ()
        
        # 3D: [B*N, point_C]
        point_nodes = point_global_feat.permute(0, 2, 1).reshape(B*N, point_C)  # [B*N, 512]
        hetero_graph.add_node_type('point_3d', point_nodes)
        
        # 2D: [B*H*W, image_C] 
        patch_nodes = image_local_feat.permute(0, 2, 3, 1).reshape(B*H*W, image_C)  # [B*H*W, 256]
        hetero_graph.add_node_type('patch_2d', patch_nodes)
        
        # : [B, image_C]
        view_nodes = image_global_feat  # [B, 256]
        hetero_graph.add_node_type('view_global', view_nodes)
        
        #  Step 2: build heterogeneous edge connections
        
        # 2.1 point cloud self-edges
        point_edges = self._build_point_edges(B, N, device)
        hetero_graph.add_edge_type('point_3d_to_point_3d', point_edges)
        
        # 2.2 2D patch -> 3D point edges (spatial projection)
        patch_to_point_edges = self._build_patch_to_point_edges(B, N, H, W, device)
        hetero_graph.add_edge_type('patch_2d_to_point_3d', patch_to_point_edges)
        
        # 2.3 3D point -> 2D patch edges (back projection)
        point_to_patch_edges = self._build_point_to_patch_edges(B, N, H, W, device)
        hetero_graph.add_edge_type('point_3d_to_patch_2d', point_to_patch_edges)
        
        # 2.4 view global -> 3D point edges (view_global_to_point_3d)
        view_to_point_edges = self._build_view_to_point_edges(B, N, device)
        hetero_graph.add_edge_type('view_global_to_point_3d', view_to_point_edges)
        
        # 2.5 view global -> 2D patch edges (view_global_to_patch_2d)
        view_to_patch_edges = self._build_view_to_patch_edges(B, H, W, device)
        hetero_graph.add_edge_type('view_global_to_patch_2d', view_to_patch_edges)
        
        # 2.6 intra-patch 2D edges
        patch_edges = self._build_patch_edges(B, H, W, device)
        hetero_graph.add_edge_type('patch_2d_to_patch_2d', patch_edges)
        
        if self.debug_mode:
            print(f" :")
            print(f"   : {point_nodes.shape}")
            print(f"   : {patch_nodes.shape}")
            print(f"   : {view_nodes.shape}")
            print(f"   : {len(hetero_graph.edge_types)}")
        
        return hetero_graph.to(device)
    
    def _build_point_edges(self, B, N, device):
        """build 3D point self-edges using ring connectivity"""
        k = min(8, N-1)  # each point connected to its 8 nearest circular neighbours
        edges = []
        
        for b in range(B):
            base_idx = b * N
            # simplified ring connectivity
            for i in range(N):
                src = base_idx + i
                for j in range(1, k+1):
                    dst = base_idx + (i + j) % N
                    edges.append([src, dst])
        
        return torch.tensor(edges, device=device).T  # [2, num_edges]
    
    def _build_patch_to_point_edges(self, B, N, H, W, device):
        """build 2D patch -> 3D point projection edges"""
        edges = []
        
        # simplified projection: uniform mapping of patches to points
        points_per_patch = max(1, N // (H * W))
        
        for b in range(B):
            point_base = b * N
            patch_base = b * H * W
            
            for h in range(H):
                for w in range(W):
                    patch_idx = patch_base + h * W + w
                    
                    # each patch connected to multiple 3D points
                    for p in range(points_per_patch):
                        point_idx = point_base + (patch_idx - patch_base) * points_per_patch + p
                        if point_idx < point_base + N:
                            edges.append([patch_idx, point_idx])
        
        return torch.tensor(edges, device=device).T if edges else torch.empty(2, 0, device=device)
    
    def _build_point_to_patch_edges(self, B, N, H, W, device):
        """build 3D point -> 2D patch back-projection edges"""
        edges = []
        
        patches_per_point = max(1, (H * W) // N)
        
        for b in range(B):
            point_base = b * N  
            patch_base = b * H * W
            
            for i in range(N):
                point_idx = point_base + i
                
                # each 3D point connected to multiple patches
                for p in range(patches_per_point):
                    patch_idx = patch_base + (i * patches_per_point + p) % (H * W)
                    edges.append([point_idx, patch_idx])
        
        return torch.tensor(edges, device=device).T if edges else torch.empty(2, 0, device=device)
    
    def _build_view_to_point_edges(self, B, N, device):
        """build view-global -> 3D point edges"""
        edges = []
        
        for b in range(B):
            view_idx = b
            point_base = b * N
            
            # view-global node connects to all 3D points in its batch
            for i in range(N):
                point_idx = point_base + i
                edges.append([view_idx, point_idx])
        
        return torch.tensor(edges, device=device).T
    
    def _build_view_to_patch_edges(self, B, H, W, device):
        """build view-global -> 2D patch edges"""
        edges = []
        
        for b in range(B):
            view_idx = b
            patch_base = b * H * W
            
            # view-global node connects to all patches in its batch
            for i in range(H * W):
                patch_idx = patch_base + i
                edges.append([view_idx, patch_idx])
        
        return torch.tensor(edges, device=device).T
    
    def _build_patch_edges(self, B, H, W, device):
        """build 2D patch self-edges (4-connectivity)"""
        edges = []
        
        for b in range(B):
            patch_base = b * H * W
            
            # 4-neighbour spatial connectivity
            for h in range(H):
                for w in range(W):
                    src_idx = patch_base + h * W + w
                    
                    # right neighbour
                    if w + 1 < W:
                        dst_idx = patch_base + h * W + (w + 1)
                        edges.append([src_idx, dst_idx])
                    
                    # bottom neighbour  
                    if h + 1 < H:
                        dst_idx = patch_base + (h + 1) * W + w
                        edges.append([src_idx, dst_idx])
        
        return torch.tensor(edges, device=device).T if edges else torch.empty(2, 0, device=device)

    def fuse_global_features(self, point_global_feat, image_global_feat, image_local_feat=None):
        """
        BAHG global feature fusion
        Args:
            point_global_feat: [B, point_global_dim, num_points] 3D
            image_global_feat: [B, image_feature_dim] 2D
            image_local_feat: [B, image_feature_dim, H, W] 2D ()
        Returns:
            fused_global: [B, point_global_dim, num_points] 
            boundary_info: dict  ()
        """
        if not self.enable_global_fusion:
            return point_global_feat, {}
            
        B, C, N = point_global_feat.shape
        boundary_info = {}
        
        if self.enable_BAHG and image_local_feat is not None:
            hetero_graph = self.build_heterogeneous_graph(
                point_global_feat, image_global_feat, image_local_feat
            )
            
            if hetero_graph is not None:
                current_graph = hetero_graph
                for i, hetero_conv in enumerate(self.hetero_conv_layers):
                    current_graph, layer_boundary_info = hetero_conv(current_graph)
                    boundary_info[f'layer_{i}'] = layer_boundary_info
                    
                    if self.debug_mode:
                        print(f"  {i+1} ")
                
                #  3D
                updated_point_feat = current_graph.nodes['point_3d']  # [B*N, unified_dim]
                unified_dim = self.hetero_out_channels['point_3d']  # 512
                fused_global = updated_point_feat.reshape(B, N, unified_dim).permute(0, 2, 1)  # [B, 512, N]
                
                fused_global = fused_global.permute(0, 2, 1)  # [B, N, C]
                fused_global = self.hetero_to_global(fused_global)  # [B, N, C]
                fused_global = fused_global.permute(0, 2, 1)  # [B, C, N]
                
                if self.debug_mode:
                    print(f" : {point_global_feat.shape} + {image_global_feat.shape} → {fused_global.shape}")
                    total_boundary_edges = sum(
                        len(layer_info.get('edge_boundaries', {})) 
                        for layer_info in boundary_info.values()
                    )
                    print(f"   : {total_boundary_edges}")
            else:
                fused_global = point_global_feat
        
        else:

            if self.fusion_method == 'concat':
                img_feat_expanded = image_global_feat.unsqueeze(2).expand(-1, -1, N)  # [B, 256, N]
                concat_feat = torch.cat([point_global_feat, img_feat_expanded], dim=1)  # [B, 512+256, N]
                concat_feat = concat_feat.permute(0, 2, 1)  # [B, N, 512+256]
                fused_feat = self.global_fusion(concat_feat)  # [B, N, 512]
                fused_global = fused_feat.permute(0, 2, 1)  # [B, 512, N]
                
            elif self.fusion_method == 'attention':
                img_proj = self.global_proj(image_global_feat)  # [B, 512]
                img_query = img_proj.unsqueeze(1)  # [B, 1, 512]
                point_kv = point_global_feat.permute(2, 0, 1)  # [N, B, 512]
                
                attended_feat, _ = self.global_attention(img_query.permute(1, 0, 2), point_kv, point_kv)
                attended_feat = attended_feat.permute(1, 2, 0).expand(-1, -1, N)  # [B, 512, N]
                fused_global = point_global_feat + attended_feat
                
            else:  # 'add'
                #  additive fusion
                img_proj = self.global_proj(image_global_feat)  # [B, 512]
                img_expanded = img_proj.unsqueeze(2).expand(-1, -1, N)  # [B, 512, N]
                fused_global = point_global_feat + img_expanded
            
            if self.debug_mode:
                print(f" : {point_global_feat.shape} + {image_global_feat.shape} → {fused_global.shape}")
            
        return fused_global, boundary_info
    
    def fuse_local_features(self, point_local_feat, image_local_feat, boundary_info=None):
        """
        BAHG
        Args:
            point_local_feat: [B, point_local_dim, num_points] 3D
            image_local_feat: [B, image_feature_dim, H, W] 2D
            boundary_info: dict
        Returns:
            fused_local: [B, point_local_dim, num_points] 
        """
        if not self.enable_local_fusion:
            return point_local_feat
            
        B, C, N = point_local_feat.shape
        
        if self.enable_BAHG and boundary_info is not None:
            boundary_weights = None
            
            # boundary weights
            if boundary_info:
                all_boundary_weights = []
                for layer_key, layer_info in boundary_info.items():
                    if 'node_boundaries' in layer_info and 'point_3d' in layer_info['node_boundaries']:
                                                 # 3Dboundary scores
                        point_boundaries = layer_info['node_boundaries']['point_3d']  # [B*N_global]
                        
                        N_global = point_boundaries.shape[0] // B  #  (64)
                        
                        if self.debug_mode:
                            print(f" boundary weights: ={N_global}, ={N}")
                        
                        # N_globalN
                        if point_boundaries.shape[0] >= B * N_global:
                            point_boundaries = point_boundaries[:B*N_global].reshape(B, N_global)  # [B, N_global]
                            
                            #  ()
                            if N_global != N:
                                point_boundaries = F.interpolate(
                                    point_boundaries.unsqueeze(1),  # [B, 1, N_global]
                                    size=N,
                                    mode='linear',
                                    align_corners=False
                                ).squeeze(1)  # [B, N]
                            
                            all_boundary_weights.append(point_boundaries)
                            
                            if self.debug_mode:
                                print(f"   boundary weights: {point_boundaries.shape}")
                
                # boundary weights
                if all_boundary_weights:
                    boundary_weights = torch.stack(all_boundary_weights, dim=0).mean(dim=0)  # [B, N]
            
            # boundary weights
            if boundary_weights is not None:
                img_adapted = self.local_image_adapter(image_local_feat)  # [B, 256, 1]
                img_expanded = img_adapted.expand(-1, -1, N)  # [B, 256, N]
                
                # boundary weights
                weighted_img_feat = img_expanded * boundary_weights.unsqueeze(1)  # [B, 256, N]
                
                # 3D-2D
                concat_feat = torch.cat([point_local_feat, weighted_img_feat], dim=1)  # [B, 128+256, N]
                fused_local = self.local_fusion(concat_feat)  # [B, 128, N]
                
                fused_local = self.hetero_to_local(fused_local)  # [B, 128, N]
                
                if self.debug_mode:
                    boundary_points = torch.sum(boundary_weights > self.boundary_threshold).item()
                    print(f" : {point_local_feat.shape} + {image_local_feat.shape} → {fused_local.shape}")
                    print(f"   : {boundary_points}/{B*N}")
            else:
                fused_local = self._traditional_local_fusion(point_local_feat, image_local_feat)
        
        else:

            fused_local = self._traditional_local_fusion(point_local_feat, image_local_feat)
            
        return fused_local
    
    def _traditional_local_fusion(self, point_local_feat, image_local_feat):
        """"""
        B, C, N = point_local_feat.shape
        
        if self.fusion_method == 'concat':
            img_adapted = self.local_image_adapter(image_local_feat)  # [B, 256, 1]
            img_expanded = img_adapted.expand(-1, -1, N)  # [B, 256, N]
            concat_feat = torch.cat([point_local_feat, img_expanded], dim=1)  # [B, 128+256, N]
            fused_local = self.local_fusion(concat_feat)  # [B, 128, N]
            
        elif self.fusion_method == 'attention':
            img_flattened = image_local_feat.flatten(2)  # [B, 256, H*W]
            img_proj = self.local_image_proj(img_flattened.mean(dim=2))  # [B, 128]
            
            img_query = img_proj.unsqueeze(1)  # [B, 1, 128]
            point_kv = point_local_feat.permute(2, 0, 1)  # [N, B, 128]
            
            attended_feat, _ = self.local_attention(img_query.permute(1, 0, 2), point_kv, point_kv)
            attended_feat = attended_feat.permute(1, 2, 0).expand(-1, -1, N)  # [B, 128, N]
            fused_local = point_local_feat + attended_feat
            
        else:  # 'add'
            #  additive fusion
            img_proj = self.local_image_proj(image_local_feat)  # [B, 128]
            img_expanded = img_proj.unsqueeze(2).expand(-1, -1, N)  # [B, 128, N]
            fused_local = point_local_feat + img_expanded
        
        if self.debug_mode:
            print(f" : {point_local_feat.shape} + {image_local_feat.shape} → {fused_local.shape}")
        
        return fused_local


class MVCTNetModel(nn.Module):
    """Multi-Modal MVCTNet"""
    
    def __init__(self, 
                 num_class, 
                 normal_channel=True,
                 enable_multimodal=False,
                 image_feature_dim=256,
                 enable_global_fusion=True,
                 enable_local_fusion=True,
                 fusion_method='concat',
                 enable_BAHG=True,
                 boundary_threshold=0.3,
                 enable_ALFE=True,
                 ALFE_competition_strength=0.3,
                 debug_mode=True,
                 verbose_level=1):
        """
        MVCTNet
        Args:
            num_class:  (3: //)
            normal_channel: whether to use surface normals (True)
            enable_multimodal:  (False, )
            image_feature_dim: image feature dimension (256)
            enable_global_fusion: 
            enable_local_fusion: 
            fusion_method:  ('concat', 'add', 'attention')
            enable_BAHG:  (BAHG)
            boundary_threshold: boundary detection threshold
            enable_ALFE: enable ALFE feature evolution (ALFE)
            ALFE_competition_strength: 
            debug_mode: 
        """
        super(MVCTNetModel, self).__init__()
        
        #  params
        in_channel = 64
        self.category_num = 1      # number of tree species (complex tree = 1)  
        self.normal_channel = normal_channel  # whether to use surface normals
        
        #  multimodal configurationparams
        self.enable_multimodal = enable_multimodal
        self.image_feature_dim = image_feature_dim
        self.enable_global_fusion = enable_global_fusion
        self.enable_local_fusion = enable_local_fusion
        self.fusion_method = fusion_method
        self.enable_BAHG = enable_BAHG
        self.boundary_threshold = boundary_threshold
        
        #  ALFEparams
        self.enable_ALFE = enable_ALFE
        self.ALFE_competition_strength = ALFE_competition_strength
        self.debug_mode = debug_mode
        self.verbose_level = verbose_level
        
        if self.debug_mode and self.verbose_level >= 1:
            print(f" BAHG + ALFE :")
            print(f"   : {num_class}, : {'' if normal_channel else ''}")
            print(f"   : {'' if enable_multimodal else ''}")
            print(f"    ALFE: {'' if enable_ALFE else ''}")
            if enable_ALFE:
                print(f"    ALFE: {ALFE_competition_strength}")
        
        #   () - MVCTNet Set Abstraction Layers
        # : 2048 → 512 → 256 → 128 → 64 
        #  (RISP features)
        
        self.sa0 = MVCTNetSetAbstraction(
            npoint=512,    # 512
            radius=0.2,    #   
            nsample=8,     # 8
            in_channel=0,  # feature dim (0)
            out_channel=64, # feature dim
            group_all=False
        )
        
        self.sa1 = MVCTNetSetAbstraction(
            npoint=256,    # 256
            radius=0.4,    # 
            nsample=16,    # 
            in_channel=64, # 64  
            out_channel=128, 
            group_all=False
        )
        
        self.sa2 = MVCTNetSetAbstraction(
            npoint=128,    # 128
            radius=0.6,    
            nsample=32,    
            in_channel=128,
            out_channel=256,
            group_all=False
        )
        
        self.sa3 = MVCTNetSetAbstraction(
            npoint=64,     # 64 ()
            radius=0.8,    # 
            nsample=64,    # 
            in_channel=256,
            out_channel=512, #  ()
            group_all=False
        )
        
        #  ALFEStep 3: allometric feature evolution (SA)
        if self.enable_ALFE:
            # ALFE-0: sa0 (64 → 64, )
            self.ALFE_0 = ALFENetwork(
                input_dim=64, target_dim=64,
                competition_strength=self.ALFE_competition_strength,
                debug_mode=debug_mode,
                verbose_level=self.verbose_level
            )
            
            # ALFE-1: sa1 (128 → 128, )
            self.ALFE_1 = ALFENetwork(
                input_dim=128, target_dim=128,
                competition_strength=self.ALFE_competition_strength,
                debug_mode=debug_mode,
                verbose_level=self.verbose_level
            )
            
            # ALFE-2: sa2 (256 → 256, )
            self.ALFE_2 = ALFENetwork(
                input_dim=256, target_dim=256,
                competition_strength=self.ALFE_competition_strength,
                debug_mode=debug_mode,
                verbose_level=self.verbose_level
            )
            
            # ALFE-3: sa3 (512 → 512, )
            self.ALFE_3 = ALFENetwork(
                input_dim=512, target_dim=512,
                competition_strength=self.ALFE_competition_strength,
                debug_mode=debug_mode,
                verbose_level=self.verbose_level
            )
            
            if self.debug_mode:
                print(f" ALFE4SA")
        else:
            self.ALFE_0 = self.ALFE_1 = self.ALFE_2 = self.ALFE_3 = None
            if self.debug_mode:
                print(f" ALFE")

        #   () - RIConv Feature Propagation Layers  
        # : 64 → 128 → 256 → 512 → 2048 
        
        self.fp3 = MVCTNetFeaturePropagation(
            radius=1.5,    # 
            nsample=64,    # 
            in_channel=512+64,    #  + 
            in_channel_2=512+256, # feature dim
            out_channel=512,      # feature dim
            mlp=[512]            # MLP
        )
        
        self.fp2 = MVCTNetFeaturePropagation(
            radius=0.8, 
            nsample=32,
            in_channel=512+64,   # 512(fp3) + 64()
            in_channel_2=512+128,
            out_channel=512,
            mlp=[256]
        )
        
        self.fp1 = MVCTNetFeaturePropagation(
            radius=0.48,
            nsample=32, 
            in_channel=256+64,   # 256(fp2) + 64()
            in_channel_2=256+64,
            out_channel=256,
            mlp=[128]
        )
        
        self.fp0 = MVCTNetFeaturePropagation(
            radius=0.48,
            nsample=32,
            in_channel=128+64,   # 128(fp1) + 64() 
            in_channel_2=128+16, # 128() + 16(one-hot)
            out_channel=128,     # feature dim
            mlp=[]              # MLP
        )
        
        #   - 
        self.conv1 = nn.Conv1d(128+self.category_num, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(128, num_class, 1)

        if self.enable_multimodal:
            self.multimodal_fusion = MultiModalFusionModule(
                image_feature_dim=image_feature_dim,
                point_global_dim=512,  # l3_pointsfeature dim
                point_local_dim=128,   # feature dim
                enable_global_fusion=enable_global_fusion,
                enable_local_fusion=enable_local_fusion,
                fusion_method=fusion_method,
                enable_BAHG=enable_BAHG,  #  BAHG arguments
                boundary_threshold=boundary_threshold,        #  boundary detection threshold
                debug_mode=debug_mode
            )
            if self.debug_mode:
                fusion_type = " " if enable_BAHG else " "
                print(f"{fusion_type}")
        else:
            self.multimodal_fusion = None
            if self.debug_mode:
                print(f" pure 3D mode")

        
    def forward(self, xyz, cls_label, image_features=None):
        """
        Multi-Modal MVCTNet
        Args:
            xyz: [B, N, 6]  (xyz + )
            cls_label: [B, 1] class label integer ()
            image_features: dict, 
                - 'local': [B, 256, 16, 16] 
                - 'global': [B, 256]   
                - 'sample_id': [...] sampleID ()
            
        Returns:
            x: [B, N, 3]  (log_softmax3D+2D)
            l3_points: [B, 512, 64]  (/)
        """
        
        B, N, C = xyz.shape  # B=batch_size, N=2048, C=6
        if self.normal_channel:
            norm = xyz[:, :, 3:]  # [B, N, 3]  
            xyz = xyz[:, :, :3]   # [B, N, 3] xyz
        
        is_multimodal_input = (self.enable_multimodal and 
                              image_features is not None and 
                              'local' in image_features and 
                              'global' in image_features)
        
        #   - 
        if is_multimodal_input:
            device = xyz.device  # 
            
            image_features['local'] = image_features['local'].to(device)
            image_features['global'] = image_features['global'].to(device)
            
            if self.debug_mode:
                print(f" :  {device}")
        
        if self.debug_mode and self.verbose_level >= 2:  # 
            print(f" :")
            print(f"   : {xyz.shape}, : {'' if self.enable_multimodal else ''}")
            print(f"   : {'' if is_multimodal_input else ''}")
            if is_multimodal_input:
                print(f"   : {image_features['local'].shape}, : {image_features['global'].shape}")
            
        use_multimodal = is_multimodal_input
        
        #   -  + ALFE
        # : (, , ) + ALFE

        l0_xyz, l0_norm, l0_points = self.sa0(xyz, norm, None)
        # l0: [B, 512, 3], [B, 512, 3], [B, 64, 512] 
        # 2048 → 512, 64
        
        #  ALFE-0: sa0Step 3: allometric feature evolution
        if self.enable_ALFE and self.ALFE_0 is not None:
            if self.debug_mode and self.verbose_level >= 3:  # 
                print(" ALFE-0...")
            l0_points, ALFE_0_info = self.ALFE_0(l0_points)
            if self.debug_mode and self.verbose_level >= 3:  # 
                avg_scaling = ALFE_0_info['scaling_factors'].mean().detach().item()
                print(f"   ALFE-0: {avg_scaling:.4f}")

        l1_xyz, l1_norm, l1_points = self.sa1(l0_xyz, l0_norm, l0_points)
        # l1: [B, 256, 3], [B, 256, 3], [B, 128, 256]
        # 512 → 256, 64→128

        #  ALFE-1: sa1Step 3: allometric feature evolution
        if self.enable_ALFE and self.ALFE_1 is not None:
            if self.debug_mode and self.verbose_level >= 3:  # 
                print(" ALFE-1...")
            l1_points, ALFE_1_info = self.ALFE_1(l1_points)
            if self.debug_mode and self.verbose_level >= 3:  # 
                avg_scaling = ALFE_1_info['scaling_factors'].mean().detach().item()
                print(f"   ALFE-1: {avg_scaling:.4f}")

        l2_xyz, l2_norm, l2_points = self.sa2(l1_xyz, l1_norm, l1_points)
        # l2: [B, 128, 3], [B, 128, 3], [B, 256, 128]
        # 256 → 128, 128→256

        #  ALFE-2: sa2Step 3: allometric feature evolution
        if self.enable_ALFE and self.ALFE_2 is not None:
            if self.debug_mode and self.verbose_level >= 3:  # 
                print(" ALFE-2...")
            l2_points, ALFE_2_info = self.ALFE_2(l2_points)
            if self.debug_mode and self.verbose_level >= 3:  # 
                avg_scaling = ALFE_2_info['scaling_factors'].mean().detach().item()
                print(f"   ALFE-2: {avg_scaling:.4f}")

        l3_xyz, l3_norm, l3_points = self.sa3(l2_xyz, l2_norm, l2_points)
        # l3: [B, 64, 3], [B, 64, 3], [B, 512, 64]
        # 128 → 64, 256→512 ()

        #  ALFE-3: sa3Step 3: allometric feature evolution ()
        if self.enable_ALFE and self.ALFE_3 is not None:
            if self.debug_mode and self.verbose_level >= 2:  # 
                print(" ALFE-3 ()...")
            l3_points, ALFE_3_info = self.ALFE_3(l3_points)
            if self.debug_mode and self.verbose_level >= 2:  # 
                avg_scaling = ALFE_3_info['scaling_factors'].mean().detach().item()
                crown_prob = ALFE_3_info['organ_probs'][:, 0].mean().detach().item()
                trunk_prob = ALFE_3_info['organ_probs'][:, 1].mean().detach().item()
                noise_prob = ALFE_3_info['organ_probs'][:, 2].mean().detach().item()
                print(f"   ALFE-3: {avg_scaling:.4f}")
                print(f"   : ={crown_prob:.3f}, ={trunk_prob:.3f}, ={noise_prob:.3f}")

        #   (1: )
        boundary_info = {}  # 
        if use_multimodal and self.enable_global_fusion:
            if self.debug_mode and self.verbose_level >= 2:  # 
                fusion_type = " " if self.enable_BAHG else " "
                print(f"{fusion_type}...")
                print(f"   l3_points: {l3_points.shape}")

            #  : [B, 512, 64] + [B, 256] +  → [B, 512, 64]
            l3_points, boundary_info = self.multimodal_fusion.fuse_global_features(
                l3_points, image_features['global'], image_features['local']
            )

            if self.debug_mode and self.verbose_level >= 2:  # 
                print(f"   l3_points: {l3_points.shape}")
                if boundary_info and 'boundary_mask' in boundary_info:
                    num_boundary = torch.sum(boundary_info['boundary_mask']).item()
                    print(f"   : {num_boundary}")

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_norm, l3_norm, l2_points, l3_points)
        # fp3: 64 → 128, l3(512)l2(256) → 512
        # : [B, 512, 128]

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_norm, l2_norm, l1_points, l2_points)
        # fp2: 128 → 256, fp3(512)l1(128) → 512
        # : [B, 512, 256]

        l0_points = self.fp1(l0_xyz, l1_xyz, l0_norm, l1_norm, l0_points, l1_points)
        # fp1: 256 → 512, fp2(512)l0(64) → 256
        # : [B, 256, 512]

        cls_label_one_hot = cls_label.view(B, self.category_num, 1).repeat(1, 1, N).cuda()
        # class label integer [B, 1, N] one-hot

        l0_points = self.fp0(xyz, l0_xyz, norm, l0_norm, cls_label_one_hot, l0_points)
        # fp0: 512 → 2048 (), 
        # : [B, 128, 2048] 

        feat = F.relu(self.bn1(self.conv1(l0_points)))
        # [B, 128+1, 2048] → [B, 128, 2048] 

        if use_multimodal and self.enable_local_fusion:
            if self.debug_mode and self.verbose_level >= 2:  # 
                fusion_type = " " if self.enable_BAHG else " "
                print(f"{fusion_type}...")
                print(f"   feat: {feat.shape}")

            #  : [B, 128, 2048] + [B, 256, 16, 16] +  → [B, 128, 2048]
            feat = self.multimodal_fusion.fuse_local_features(
                feat, image_features['local'], boundary_info
            )

            if self.debug_mode and self.verbose_level >= 2:  # 
                print(f"   feat: {feat.shape}")
                if boundary_info:
                    num_layers = len([k for k in boundary_info.keys() if k.startswith('layer_')])
                    print(f"   : {num_layers}")

        x = self.drop1(feat)  # Dropout
        x = self.conv2(x)     # [B, 128, 2048] → [B, 3, 2048]
        x = F.log_softmax(x, dim=1)  # log_softmax
        x = x.permute(0, 2, 1)       # [B, 3, 2048] → [B, 2048, 3]

        if self.debug_mode and self.verbose_level >= 1:  # 
            print(f" :  {x.shape}")
            print(f"   : {l3_points.shape}")
            
            mode_parts = []
            if use_multimodal and self.enable_BAHG:
                mode_parts.append(" BAHG")
            elif use_multimodal:
                mode_parts.append(" ")
            else:
                mode_parts.append(" 3D")
                
            if self.enable_ALFE:
                mode_parts.append(" ALFE")
                
            print(f"   : {' + '.join(mode_parts)}")
            
            if use_multimodal:
                fusion_info = []
                if self.enable_global_fusion:
                    fusion_info.append("")
                if self.enable_local_fusion:
                    fusion_info.append("")
                if self.enable_BAHG:
                    fusion_info.append("")
                print(f"   : {' + '.join(fusion_info) if fusion_info else ''}")
                
            if self.enable_ALFE:
                print(f"    : 4")

        return x, l3_points


class get_loss(nn.Module):
    """
    NLL GUCL uncertainty weight
    enable_guclparams
    """
    
    def __init__(self, 
                 enable_gucl=False,
                 gucl_geometric_weight=0.5,
                 gucl_uncertainty_weight=0.5,
                 gucl_adaptive_factor=0.15,
                 debug_mode=True):
        super(get_loss, self).__init__()
        
        self.enable_gucl = enable_gucl
        
        if enable_gucl:
            #  GUCLuncertainty weight ()
            self.gucl_loss = GUCL_Loss_Module(
                num_classes=3,
                geometric_weight=gucl_geometric_weight,
                uncertainty_weight=gucl_uncertainty_weight,
                adaptive_factor=gucl_adaptive_factor,
                debug_mode=debug_mode
            )
            if debug_mode:
                print(" GUCL - uncertainty weight")
        else:
            if debug_mode:
                print(" standard NLL loss")
        
    def forward(self, pred, target, trans_feat, boundary_info=None, modal_features=None, organ_ratios=None, epoch=0):
        """
        Args:
            pred: [B*N, 3] log (log_softmax)
            target: [B*N]  (0=, 1=, 2=)
            trans_feat: [B, 512, 64]  (BAHG)
            boundary_info: dict  (BAHG, )
            modal_features: dict  (+, )
            organ_ratios: tensor  (ALFE, )
            epoch: int epoch (GUCL)
        Returns:
            total_loss: 
        """
        
        if self.enable_gucl:
            #  GUCLuncertainty weight
            # BAHGALFE
            total_loss = self.gucl_loss(
                pred, target, trans_feat, 
                boundary_info, modal_features, organ_ratios, epoch
            )
            return total_loss
        else:
            #   (log_softmax)
            # NLL Loss = -log(pred[target_class])
            total_loss = F.nll_loss(pred, target)

        return total_loss


def get_model(num_classes=3, 
               normal_channel=False,
               enable_multimodal=False, 
               image_feature_dim=256,
               enable_global_fusion=True,
               enable_local_fusion=True,
               fusion_method='concat',
               enable_BAHG=False,
               boundary_threshold=0.3,
               enable_ALFE=False,
               ALFE_competition_strength=0.3,
               enable_gucl=False,
               gucl_geometric_weight=0.5,
               gucl_uncertainty_weight=0.5,
               gucl_adaptive_factor=0.15,
               gucl_debug_mode=True,
               debug_mode=True,
               verbose_level=1):
    """
    GUCL: Geometric Uncertainty Collaborative Learning
    geometric weightuncertainty weight
    Args:
        num_classes:  (3: //)
        normal_channel: whether to use surface normals (False)
        enable_multimodal:  (False)
        image_feature_dim: image feature dimension (256)
        enable_global_fusion:  (True)
        enable_local_fusion:  (True)
        fusion_method:
        enable_BAHG:  (BAHG)
        boundary_threshold: boundary detection threshold
        enable_ALFE: enable ALFE feature evolution (ALFE)
        ALFE_competition_strength: ALFE
        enable_gucl:  GUCL
        gucl_*: GUCLparams
        debug_mode:
        verbose_level:  (0=, 1=, 2=, 3=)
    Returns:
        model: MVCTNetModel
    """
    
    if debug_mode:
        print(" MVCTNet...")
        print(f"   : {num_classes}")
        print(f"   : {'' if normal_channel else ''}")
        print(f"   : {'' if enable_multimodal else ''}")
        if enable_multimodal:
            print(f"   image feature dimension: {image_feature_dim}")
            print(f"   : {'' if enable_BAHG else ''}")
            print(f"   : {'' if enable_ALFE else ''}")
        print(f"    GUCL: {'' if enable_gucl else ''}")
    
    model = MVCTNetModel(
        num_class=num_classes,
        normal_channel=normal_channel,
        enable_multimodal=enable_multimodal,
        image_feature_dim=image_feature_dim,
        enable_global_fusion=enable_global_fusion,
        enable_local_fusion=enable_local_fusion,
        fusion_method=fusion_method,
        enable_BAHG=enable_BAHG,
        boundary_threshold=boundary_threshold,
        enable_ALFE=enable_ALFE,
        ALFE_competition_strength=ALFE_competition_strength,
        debug_mode=debug_mode,
        verbose_level=verbose_level
    )
    
    if debug_mode:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f" model arguments:")
        print(f"   total parameters: {total_params:,}")
        print(f"   training arguments: {trainable_params:,}")
        
        if enable_multimodal:
            print(f" :")
            if enable_BAHG:
                print(f"    BAHG: ")
            if enable_ALFE:
                print(f"    ALFE: ")
            if enable_gucl:
                print(f"    GUCL: uncertainty weight")
    
    return model


def get_loss_function(enable_gucl=False,
                      gucl_geometric_weight=0.5,
                      gucl_uncertainty_weight=0.5,
                      gucl_adaptive_factor=0.15,
                      debug_mode=True):
    """
    GUCL weight
    Args:
        enable_gucl: GUCL
        gucl_*: GUCLparams
        debug_mode:
    Returns:
        criterion: 
    """
    
    if debug_mode:
        print(" ...")
        if enable_gucl:
            print(" GUCLuncertainty weight")
        else:
            print(" standard NLL loss")
    
    criterion = get_loss(
        enable_gucl=enable_gucl,
        gucl_geometric_weight=gucl_geometric_weight,
        gucl_uncertainty_weight=gucl_uncertainty_weight,
        gucl_adaptive_factor=gucl_adaptive_factor,
        debug_mode=debug_mode
    )
    
    if debug_mode:
        print(" ")
    
    return criterion