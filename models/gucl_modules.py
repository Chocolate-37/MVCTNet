"""
GUCL: Geometric Uncertainty Collaborative Learning uncertainty weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GUCLConstants:
    """
     GUCLuncertainty weight
    geometric weightuncertainty weightparams
    """
    
    # geometric weight
    GEOMETRIC_ADAPTATION_RATES = {
        'curvature': 0.02,      # 
        'density': 0.015,       # density weight adjustment rate
        'normal': 0.018         # normal-consistency weight adjustment rate
    }
    
    # uncertainty weight
    UNCERTAINTY_THRESHOLDS = {
        'aleatoric_low': 0.1,   # aleatoric uncertainty low threshold
        'aleatoric_high': 0.8,  # aleatoric uncertainty high threshold
        'epistemic_low': 0.15,  # epistemic uncertainty low threshold
        'epistemic_high': 0.75  # epistemic uncertainty high threshold
    }
    
    #  ()
    GEOMETRIC_FEATURE_WEIGHTS = {
        'boundary_curvature': 2.0,  # boundary curvature enhancement
        'surface_density': 1.2,     # surface density adaptation
        'normal_consistency': 1.5   # normal consistency
    }
    
    COLLABORATIVE_BALANCE_FACTORS = {
        'early_geometric': 0.7,     # geometry weight in early training
        'late_uncertainty': 0.8,    # uncertainty weight in late training
        'convergence_balance': 0.5  # weight balance at convergence
    }


class CurvatureEstimator(nn.Module):
    
    def __init__(self, k_neighbors=16, feature_dim=512):
        super(CurvatureEstimator, self).__init__()
        self.k_neighbors = k_neighbors
        
        # curvature feature extraction MLP
        self.curvature_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # outputs curvature in [0,1]
        )
        
    def forward(self, features, points=None):
        """
        Args:
            features: [B, C, N] 
            points: [B, N, 3]  ()
            
        Returns:
            curvatures: [B, N] 
        """
        B, C, N = features.shape
        
        features_flat = features.permute(0, 2, 1).reshape(B*N, C)  # [B*N, C]
        curvatures = self.curvature_net(features_flat)  # [B*N, 1]
        curvatures = curvatures.reshape(B, N)  # [B, N]
        
        return curvatures


class DensityEstimator(nn.Module):
    
    def __init__(self, feature_dim=512, radius=0.1):
        super(DensityEstimator, self).__init__()
        self.radius = radius
        
        # density feature extraction MLP
        self.density_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # outputs density in [0,1]
        )
    
    def forward(self, features, points=None):
        """
        Args:
            features: [B, C, N] 
            points: [B, N, 3]  ()
            
        Returns:
            densities: [B, N] 
        """
        B, C, N = features.shape
        
        features_flat = features.permute(0, 2, 1).reshape(B*N, C)  # [B*N, C]
        densities = self.density_net(features_flat)  # [B*N, 1]
        densities = densities.reshape(B, N)  # [B, N]
        
        return densities


class NormalConsistencyEstimator(nn.Module):
    """
     normal consistency
    """
    
    def __init__(self, feature_dim=512):
        super(NormalConsistencyEstimator, self).__init__()
        
        # normal consistency
        self.normal_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # outputs consistency in [0,1]
        )
    
    def forward(self, features, normals=None):
        """
        normal consistency
        Args:
            features: [B, C, N] 
            normals: [B, N, 3]  ()
        Returns:
            consistency: [B, N] 
        """
        B, C, N = features.shape
        
        # normal consistency
        features_flat = features.permute(0, 2, 1).reshape(B*N, C)  # [B*N, C]
        consistency = self.normal_net(features_flat)  # [B*N, 1]
        consistency = consistency.reshape(B, N)  # [B, N]
        
        return consistency
class AleatoricUncertaintyEstimator(nn.Module):
    """
    Aleatoric Uncertainty Estimator
    uncertainty weight
    """
    
    def __init__(self, feature_dim=512):
        super(AleatoricUncertaintyEstimator, self).__init__()
        
        # aleatoric uncertainty MLP
        self.aleatoric_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1uncertainty weight
        )
    
    def forward(self, features):
        """
        uncertainty weight
        Args:
            features: [B, C, N]
        Returns:
            aleatoric_uncertainty: [B, N] uncertainty weight
        """
        B, C, N = features.shape
        
        features_flat = features.permute(0, 2, 1).reshape(B*N, C)  # [B*N, C]
        uncertainty = self.aleatoric_net(features_flat)  # [B*N, 1]
        uncertainty = uncertainty.reshape(B, N)  # [B, N]
        
        return uncertainty
class EpistemicUncertaintyEstimator(nn.Module):
    """
    Epistemic Uncertainty Estimator
    estimates model-parameter uncertainty
    """
    
    def __init__(self, feature_dim=512, num_samples=10):
        super(EpistemicUncertaintyEstimator, self).__init__()
        self.num_samples = num_samples
        
        # uncertainty weight (DropoutMonte Carlo)
        self.epistemic_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # higher Dropout rate for Monte Carlo sampling
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """
        estimates epistemic uncertainty via Monte Carlo Dropout
        Args:
            features: [B, C, N]
        Returns:
            epistemic_uncertainty: [B, N] uncertainty weight
        """
        B, C, N = features.shape
        features_flat = features.permute(0, 2, 1).reshape(B*N, C)  # [B*N, C]
        
        # Monte Carlo
        self.epistemic_net.train()  # Dropout
        predictions = []
        
        for _ in range(self.num_samples):
            pred = self.epistemic_net(features_flat)  # [B*N, 1]
            predictions.append(pred)
        
        # uncertainty weight
        predictions = torch.stack(predictions, dim=0)  # [num_samples, B*N, 1]
        uncertainty = torch.var(predictions, dim=0)  # [B*N, 1]
        uncertainty = uncertainty.reshape(B, N)  # [B, N]
        
        return uncertainty
class GeometricAwareLoss(nn.Module):
    """
    geometric weightadaptive factor
    normal consistency
    """
    
    def __init__(self, num_classes=3, curvature_weight=0.4, density_weight=0.3, normal_weight=0.3):
        super(GeometricAwareLoss, self).__init__()
        self.num_classes = num_classes
        self.curvature_weight = curvature_weight
        self.density_weight = density_weight 
        self.normal_weight = normal_weight

        self.geometric_weight_net = nn.Sequential(
            nn.Linear(512, 128),  # feature dim
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pred, target, features, boundary_info=None):
        """
        geometric weight
        Args:
            pred: [B, N, 3] 
            target: [B, N] 
            features: [B, 512, N] 
            boundary_info: dict  ()
        Returns:
            geometric_loss: geometric weight
            geometric_weights: [B, N] 
        """
        B, N, C = pred.shape
        
        features_flat = features.permute(0, 2, 1).reshape(B*N, -1)  # [B*N, 512]
        geometric_weights = self.geometric_weight_net(features_flat)  # [B*N, 1]
        geometric_weights = geometric_weights.reshape(B, N)  # [B, N]
        
        #  Focal Loss
        log_probs = F.log_softmax(pred, dim=-1)  # [B, N, 3]
        ce_loss = F.nll_loss(log_probs.reshape(-1, C), target.reshape(-1), reduction='none')  # [B*N]
        ce_loss = ce_loss.reshape(B, N)  # [B, N]
        
        weighted_loss = ce_loss * geometric_weights  # [B, N]

        if boundary_info is not None and 'node_boundaries' in boundary_info:
            for layer_info in boundary_info.values():
                if 'node_boundaries' in layer_info and 'point_3d' in layer_info['node_boundaries']:
                    point_boundaries = layer_info['node_boundaries']['point_3d']
                    if point_boundaries.shape[0] >= B * N:
                        boundary_weights = point_boundaries[:B*N].reshape(B, N)
                        weighted_loss = weighted_loss * (1 + boundary_weights)
                        break
        
        geometric_loss = weighted_loss.mean()
        
        return geometric_loss, geometric_weights
class UncertaintyAwareLoss(nn.Module):
    """
    uncertainty-aware loss component
    uncertainty weight
    """
    
    def __init__(self, num_classes=3, aleatoric_weight=0.5, epistemic_weight=0.5):
        super(UncertaintyAwareLoss, self).__init__()
        self.num_classes = num_classes
        self.aleatoric_weight = aleatoric_weight
        self.epistemic_weight = epistemic_weight
        
        # uncertainty weight
        self.uncertainty_weight_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pred, target, features):
        """
        uncertainty weight
        Args:
            pred: [B, N, 3] 
            target: [B, N] 
            features: [B, 512, N]
        Returns:
            uncertainty_loss: uncertainty weight
            uncertainty_weights: [B, N] uncertainty weight
        """
        B, N, C = pred.shape
        
        #  uncertainty weight
        features_flat = features.permute(0, 2, 1).reshape(B*N, -1)  # [B*N, 512]
        uncertainty_weights = self.uncertainty_weight_net(features_flat)  # [B*N, 1]
        uncertainty_weights = uncertainty_weights.reshape(B, N)  # [B, N]
        
        #  uncertainty weight
        log_probs = F.log_softmax(pred, dim=-1)  # [B, N, 3]
        ce_loss = F.nll_loss(log_probs.reshape(-1, C), target.reshape(-1), reduction='none')  # [B*N]
        ce_loss = ce_loss.reshape(B, N)  # [B, N]
        
        # uncertainty weight
        weighted_loss = ce_loss * (1 + uncertainty_weights)  # [B, N]
        
        #  uncertainty regularisation term
        uncertainty_reg = uncertainty_weights.mean()
        uncertainty_loss = weighted_loss.mean() + 0.1 * uncertainty_reg
        
        return uncertainty_loss, uncertainty_weights


class CollaborativeWeightingNetwork(nn.Module):
    """
    collaborative weighting network
    geometric weightuncertainty weight
    """
    
    def __init__(self, feature_dim=512):
        super(CollaborativeWeightingNetwork, self).__init__()
        
        # weight prediction MLP
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # uncertainty weight
            nn.Softmax(dim=-1)
        )
        
        # momentum-averaged weights
        self.register_buffer('weight_momentum', torch.tensor([0.5, 0.5]))
        self.momentum_factor = 0.9
    
    def forward(self, features, geometric_weights, uncertainty_weights, epoch=0):
        """
        Args:
            features: [B, 512, N] 
            geometric_weights: [B, N] 
            uncertainty_weights: [B, N] uncertainty weight
            epoch:
        Returns:
            collaborative_weights: dict 
        """
        B, C, N = features.shape
        
        global_feat = features.mean(dim=2)  # [B, C]
        batch_global_feat = global_feat.mean(dim=0)  # [C]
        
        predicted_weights = self.weight_predictor(batch_global_feat)  # [2]
        geometric_alpha = predicted_weights[0].item()
        uncertainty_alpha = predicted_weights[1].item()
        
        # adaptive factor
        stage_factor = 1.0 / (1.0 + math.exp(-0.05 * (epoch - 100)))  # sigmoid
        geometric_alpha = geometric_alpha * (1 - stage_factor * 0.3)
        uncertainty_alpha = uncertainty_alpha * (1 + stage_factor * 0.3)
        
        current_weights = torch.tensor([geometric_alpha, uncertainty_alpha], device=geometric_weights.device)
        self.weight_momentum = self.momentum_factor * self.weight_momentum + (1 - self.momentum_factor) * current_weights
        
        final_geometric_weight = self.weight_momentum[0].item()
        final_uncertainty_weight = self.weight_momentum[1].item()
        
        collaborative_weights = {
            'geometric': final_geometric_weight,
            'uncertainty': final_uncertainty_weight,
            'geometric_map': geometric_weights,
            'uncertainty_map': uncertainty_weights
        }
        
        return collaborative_weights
class GUCL_Loss(nn.Module):
    """
    GUCL: Geometric Uncertainty Collaborative Learning
    uncertainty weight
    1. geometric weightadaptive factor
    2. uncertainty weightuncertainty weightuncertainty weight  
    3. uncertainty weight
    -  uncertainty weightuncertainty weight
    -  adaptive factor
    """
    
    def __init__(self, 
                 num_classes=3,
                 geometric_weight=0.5,       # geometric weight
                 uncertainty_weight=0.5,     # uncertainty weight
                 adaptive_factor=0.15,       # adaptive factor
                 debug_mode=True):
        """
        GUCL
        Args:
            num_classes: number of classes
            geometric_weight: geometric weight
            uncertainty_weight: uncertainty weight
            adaptive_factor: adaptive factor
            debug_mode: 
        """
        super(GUCL_Loss, self).__init__()
        
        self.num_classes = num_classes
        self.geometric_weight = geometric_weight
        self.uncertainty_weight = uncertainty_weight
        self.adaptive_factor = adaptive_factor
        self.debug_mode = debug_mode
        
        #  geometry-aware loss component
        self.geometric_loss = GeometricAwareLoss(
            num_classes=num_classes,
            curvature_weight=0.4,
            density_weight=0.3,
            normal_weight=0.3
        )
        
        #  uncertainty-aware loss component
        self.uncertainty_loss = UncertaintyAwareLoss(
            num_classes=num_classes,
            aleatoric_weight=0.5,
            epistemic_weight=0.5
        )
        
        #  collaborative weighting network
        self.collaborative_weighting = CollaborativeWeightingNetwork()
        
        #  loss history buffer
        self.loss_history = {
            'geometric': [],
            'uncertainty': [],
            'total': [],
            'weights': []
        }
        
        if self.debug_mode:
            print(" GUCLuncertainty weight")
            print(f"   : {geometric_weight}, uncertainty weight: {uncertainty_weight}")
            print(f"   adaptive factor: {adaptive_factor}")
    
    def forward(self, pred, target, trans_feat, boundary_info=None, modal_features=None, organ_ratios=None, epoch=0):
        """
        GUCL
        Args:
            pred: [B, N, 3]  [B*N, 3] ()
            target: [B, N]  [B*N] ()
            trans_feat: [B, 512, N]  [B, 512, 64]  (uncertainty weight)
            boundary_info: dict  (BAHG)
            modal_features: dict  ()
            organ_ratios: tensor  (ALFE)
            epoch:
        Returns:
            total_loss: GUCL
        """
        
        if pred.dim() == 2:  # [B*N, 3] -> [B, N, 3]
            B_times_N, C = pred.shape
            # trans_featBN
            if trans_feat.dim() == 3:
                B, feat_dim, N = trans_feat.shape
                if B_times_N == B * N:
                    pred = pred.reshape(B, N, C)
                    if target.dim() == 1:
                        target = target.reshape(B, N)
                else:
                    raise ValueError(f" {pred.shape}  {trans_feat.shape} ")
            else:
                raise ValueError(f"trans_feat {trans_feat.shape} BN")
        
        B, N, C = pred.shape
        
        #  trans_feat[B, 512, 64][B, 512, N]
        if trans_feat.shape[-1] != N:
            global_feat = trans_feat  # [B, 512, 64]
            B_global, C_global, N_global = global_feat.shape
            
            if self.debug_mode:
                print(f" GUCL:  {global_feat.shape} ->  [B, {C_global}, {N}]")
            
            global_feat_reshaped = global_feat.reshape(B_global * C_global, N_global)  # [B*C, 64]
            upsampled_feat = F.interpolate(
                global_feat_reshaped.unsqueeze(0), 
                size=N, 
                mode='linear', 
                align_corners=False
            ).squeeze(0)  # [B*C, N]
            point_feat = upsampled_feat.reshape(B_global, C_global, N)  # [B, 512, N]
        else:
            point_feat = trans_feat  # 
        
        #  geometric weight
        try:
            geometric_loss_val, geometric_weights = self.geometric_loss(
                pred, target, point_feat, boundary_info
            )
        except Exception as e:
            if self.debug_mode:
                print(f" geometric weight: {e}")
                print(f"   : {pred.shape}, : {target.shape}, : {point_feat.shape}")
            geometric_loss_val = F.cross_entropy(pred.reshape(-1, C), target.reshape(-1))
            geometric_weights = torch.ones(B, N, device=pred.device)
        
        #  uncertainty weight
        try:
            uncertainty_loss_val, uncertainty_weights = self.uncertainty_loss(
                pred, target, point_feat
            )
        except Exception as e:
            if self.debug_mode:
                print(f" uncertainty weight: {e}")
            # uncertainty weight
            uncertainty_loss_val = F.cross_entropy(pred.reshape(-1, C), target.reshape(-1))
            uncertainty_weights = torch.ones(B, N, device=pred.device)
        
        try:
            collaborative_weights = self.collaborative_weighting(
                trans_feat, geometric_weights, uncertainty_weights, epoch
            )
        except Exception as e:
            if self.debug_mode:
                print(f" : {e}")
            collaborative_weights = {
                'geometric': 0.5,
                'uncertainty': 0.5,
                'geometric_map': geometric_weights,
                'uncertainty_map': uncertainty_weights
            }
        
        final_geometric_weight = collaborative_weights['geometric']
        final_uncertainty_weight = collaborative_weights['uncertainty']
        
        total_loss = (final_geometric_weight * geometric_loss_val + 
                     final_uncertainty_weight * uncertainty_loss_val)
        
        self._record_loss_components({
            'geometric': geometric_loss_val.item(),
            'uncertainty': uncertainty_loss_val.item(),
            'total': total_loss.item(),
            'weights': {
                'geometric': final_geometric_weight,
                'uncertainty': final_uncertainty_weight
            }
        }, epoch)
        
        if self.debug_mode and epoch % 10 == 0:
            self._print_debug_info(
                geometric_loss_val, uncertainty_loss_val, total_loss,
                final_geometric_weight, final_uncertainty_weight, epoch
            )
        
        return total_loss
    
    def _record_loss_components(self, loss_dict, epoch):
        """"""
        self.loss_history['geometric'].append(loss_dict['geometric'])
        self.loss_history['uncertainty'].append(loss_dict['uncertainty']) 
        self.loss_history['total'].append(loss_dict['total'])
        self.loss_history['weights'].append(loss_dict['weights'])
        
        # cap loss history at 1000 entries
        for key in self.loss_history:
            if len(self.loss_history[key]) > 1000:
                self.loss_history[key] = self.loss_history[key][-1000:]
    
    def _print_debug_info(self, geo_loss, unc_loss, total_loss, geo_weight, unc_weight, epoch):
        """"""
        print(f" GUCL (Epoch {epoch}):")
        print(f"   geometric weight: {geo_loss:.6f} (: {geo_weight:.3f})")
        print(f"   uncertainty weight: {unc_loss:.6f} (: {unc_weight:.3f})")
        print(f"   : {total_loss:.6f}")
        
        if geo_weight > 0.7:
            print(f"    : geometric weight is high — model focusing on geometry")
        elif unc_weight > 0.7:
            print(f"    : uncertainty weight is high — model focusing on prediction confidence")
        else:
            print(f"    weights are well balanced")


# backward-compatible alias
AMCL_Loss = GUCL_Loss  #  