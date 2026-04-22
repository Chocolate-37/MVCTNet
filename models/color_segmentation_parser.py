"""
Color Segmentation Parser for Pre-segmented Images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np
from PIL import Image
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ColorSegmentationParser:
    """"""
    
    def __init__(self):
        #  BGRRGB
        # BGR: 'crown': (0, 255, 0), 'trunk': (0, 165, 255), 'interference': (128, 128, 128), 'background': (255, 255, 255)
        self.color_map = {
            (0, 255, 0): 0,        #  →  (crown) - BGR(0,255,0) = RGB(0,255,0)
            (255, 165, 0): 1,      #  →  (trunk) - BGR(0,165,255) = RGB(255,165,0)
            (128, 128, 128): 2,    #  →  (interference) - BGR(128,128,128) = RGB(128,128,128)
            (255, 255, 255): 3     #  →  (background) - BGR(255,255,255) = RGB(255,255,255)
        }
        
        #  JPEG
        self.extended_color_map = {
            (0, 255, 0): 0, (0, 254, 0): 0, (1, 255, 0): 0, (0, 255, 1): 0,
            (1, 254, 0): 0, (0, 254, 1): 0, (1, 255, 1): 0, (1, 254, 1): 0,
            
            (255, 165, 0): 1, (254, 165, 0): 1, (255, 164, 0): 1, (255, 165, 1): 1,
            (254, 164, 0): 1, (254, 165, 1): 1, (255, 164, 1): 1, (254, 164, 1): 1,
            
            (128, 128, 128): 2, (127, 127, 127): 2, (129, 129, 129): 2,
            (127, 128, 128): 2, (128, 127, 128): 2, (128, 128, 127): 2,
            
            (255, 255, 255): 3, (254, 254, 254): 3, (254, 255, 255): 3,
            (255, 254, 255): 3, (255, 255, 254): 3, (254, 254, 255): 3,
            (254, 255, 254): 3, (255, 254, 254): 3,
        }
        
        self.class_names = ['crown', 'trunk', 'interference', 'background']
        
    def rgb_to_label(self, rgb_image):
        """
        RGB -  
        Args:
            rgb_image: [H, W, 3] numpy array, 0-255
        Returns:
            label_map: [H, W] numpy array, class label integer
        """
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()
        
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
        H, W = rgb_image.shape[:2]
        label_map = np.full((H, W), 3, dtype=np.int64)  # 
        
        #  1
        pixels_classified = 0
        for (r, g, b), label in self.extended_color_map.items():
            exact_match = ((rgb_image[:, :, 0] == r) & 
                          (rgb_image[:, :, 1] == g) & 
                          (rgb_image[:, :, 2] == b))
            if np.any(exact_match):
                label_map[exact_match] = label
                pixels_classified += np.sum(exact_match)
        
        print(f"     {pixels_classified} ")
        
                 #  2
        unclassified = (label_map == 3)  # 
        unclassified_count = np.sum(unclassified)
        
        if unclassified_count > 0:
            print(f"     {unclassified_count} ...")
            
            unclassified_coords = np.where(unclassified)
            pixels = rgb_image[unclassified_coords]
            
            target_colors = np.array(list(self.color_map.keys()))
            target_labels = np.array(list(self.color_map.values()))
            
            distances = np.sqrt(np.sum((pixels[:, None, :] - target_colors[None, :, :]) ** 2, axis=2))
            
            min_distances = np.min(distances, axis=1)
            closest_color_idx = np.argmin(distances, axis=1)
            
            distance_threshold = 100  # 
            valid_matches = min_distances < distance_threshold
            
            if np.any(valid_matches):
                valid_rows = unclassified_coords[0][valid_matches]
                valid_cols = unclassified_coords[1][valid_matches]
                valid_labels = target_labels[closest_color_idx[valid_matches]]
                
                label_map[valid_rows, valid_cols] = valid_labels
                
                distance_classified = np.sum(valid_matches)
                print(f"     {distance_classified} ")
        
        return label_map
    
    def analyze_segmentation(self, rgb_image):
        """
        statistics
        Args:
            rgb_image: [H, W, 3] 
        Returns:
            stats: contains
        """
        label_map = self.rgb_to_label(rgb_image)
        stats = {}
        
        total_pixels = label_map.size
        for label, name in enumerate(self.class_names):
            count = np.sum(label_map == label)
            percentage = count / total_pixels * 100
            stats[name] = {
                'count': count,
                'percentage': percentage
            }
        
        return stats
    
    def debug_image_colors(self, rgb_image, sample_points=500):
        """
         -  
        Args:
            rgb_image: [H, W, 3] numpy array
            sample_points: 
        Returns:
            color_samples: sample
        """
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()
        
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
        H, W = rgb_image.shape[:2]
        
        print("     (BGR):")
        print("      crown: RGB(0, 255, 0) - ")
        print("      trunk: RGB(255, 165, 0) - ") 
        print("      interference: RGB(128, 128, 128) - ")
        print("      background: RGB(255, 255, 255) - ")
        print()
        
        sample_indices = np.random.choice(H*W, min(sample_points, H*W), replace=False)
        flat_image = rgb_image.reshape(-1, 3)
        color_samples = flat_image[sample_indices]
        
        unique_colors, counts = np.unique(flat_image, axis=0, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]  # 
        
        print(f"     (15{len(unique_colors)}):")
        for i in range(min(15, len(unique_colors))):
            idx = sorted_indices[i]
            color = unique_colors[idx]
            count = counts[idx]
            total_pixels = H * W
            percentage = count / total_pixels * 100
            
            closest_match = ""
            min_dist = float('inf')
            for target_color, label in self.color_map.items():
                dist = np.sqrt(np.sum((color - np.array(target_color)) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    if dist < 5:
                        closest_match = f" {self.class_names[label]}"
                    elif dist < 20:
                        closest_match = f" ~{self.class_names[label]}({dist:.1f})"
            
            print(f"     RGB{tuple(color)} : {count:6d}  ({percentage:.2f}%){closest_match}")
        
        print("\n    :")
        for target_color, label in self.color_map.items():
            exact_count = np.sum(np.all(flat_image == np.array(target_color), axis=1))
            if exact_count > 0:
                percentage = exact_count / (H * W) * 100
                print(f"      {self.class_names[label]} RGB{target_color}: {exact_count}  ({percentage:.2f}%)")
            else:
                print(f"      {self.class_names[label]} RGB{target_color}: 0 ")
        
        return color_samples
    
    def improved_color_matching(self, rgb_image):
        """
        
        Args:
            rgb_image: [H, W, 3] numpy array
        Returns:
            label_map: [H, W] numpy array
        """
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()
        
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
        H, W = rgb_image.shape[:2]
        label_map = np.zeros((H, W), dtype=np.int64)
        
        pixels = rgb_image.reshape(-1, 3)
        
        target_colors = np.array(list(self.color_map.keys()))
        target_labels = np.array(list(self.color_map.values()))
        
        distances = np.sqrt(np.sum((pixels[:, None, :] - target_colors[None, :, :]) ** 2, axis=2))
        
        closest_color_idx = np.argmin(distances, axis=1)
        pixel_labels = target_labels[closest_color_idx]
        
        label_map = pixel_labels.reshape(H, W)
        
        return label_map


class MultiViewFeatureExtractor(nn.Module):
    """"""
    
    def __init__(self, feature_dim=256):
        super(MultiViewFeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        
        # ResNet50
        backbone = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.view_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim * 2, kernel_size=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, multi_view_images):
        """
        
        Args:
            multi_view_images: [B, 4, 3, H, W] 
        Returns:
            features: [B, feature_dim, H', W'] 
            global_features: [B, feature_dim] 
        """
        B, V, C, H, W = multi_view_images.shape
        
        images = multi_view_images.view(B * V, C, H, W)
        
        features = self.feature_extractor(images)  # [B*V, 2048, H', W']
        features = self.feature_adapter(features)  # [B*V, feature_dim, H', W']
        
        _, feat_dim, feat_h, feat_w = features.shape
        features = features.view(B, V, feat_dim, feat_h, feat_w)
        
        # 4
        fused_features = features.view(B, V * feat_dim, feat_h, feat_w)
        fused_features = self.view_fusion(fused_features)  # [B, feature_dim, H', W']
        
        global_features = self.global_pool(fused_features)  # [B, feature_dim, 1, 1]
        global_features = global_features.view(B, feat_dim)  # [B, feature_dim]
        
        return fused_features, global_features


class HybridMultiViewFeatureExtractor(nn.Module):
    """
      - CNNTransformer
    1. CNN ()
    2. Transformer ()
    3.  (front-back, left-right)
    4.  (MVCTNet)
    """
    
    def __init__(self, feature_dim=256, use_transformer=True):
        super(HybridMultiViewFeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        self.use_transformer = use_transformer
        
        #  1CNN ()
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )
        
        #  2Vision Transformer ()
        if self.use_transformer:
            # patch
            self.patch_embed = nn.Conv2d(512, feature_dim, kernel_size=4, stride=4)  # 16x16 -> 4x4
            self.pos_embed = nn.Parameter(torch.randn(1, 16, feature_dim))  # 4x4=16patches
            
            # Multi-head self-attention
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=feature_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
            
            self.cross_view_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
        
        if self.use_transformer:
            # Transformer
            self.view_fusion = nn.Sequential(
                nn.Linear(feature_dim * 4, feature_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(feature_dim * 2, feature_dim)
            )
        else:
            # CNN
            self.view_fusion = nn.Sequential(
                nn.Conv2d(512 * 4, feature_dim * 2, kernel_size=1),
                nn.BatchNorm2d(feature_dim * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1)
            )
        
        #   (MVCTNet)
        self.output_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),  # ResNet50
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """ResNet"""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, multi_view_images):
        """
        Args:
            multi_view_images: [B, 4, 3, H, W] 
        Returns:
            features: [B, feature_dim, 16, 16]  (ResNet50)
            global_features: [B, feature_dim] 
        """
        B, V, C, H, W = multi_view_images.shape
        
        # Step 1: CNN
        images = multi_view_images.view(B * V, C, H, W)
        cnn_features = self.cnn_backbone(images)  # [B*V, 512, 16, 16]
        
        if self.use_transformer:
            # Step 2: Transformer
            # Patch embedding
            patch_features = self.patch_embed(cnn_features)  # [B*V, feature_dim, 4, 4]
            patch_features = patch_features.flatten(2).transpose(1, 2)  # [B*V, 16, feature_dim]
            
            patch_features = patch_features + self.pos_embed
            
            # Self-attention
            transformer_features = self.transformer(patch_features)  # [B*V, 16, feature_dim]
            
            #  -  reshapeview
            transformer_features = transformer_features.transpose(1, 2).reshape(B*V, self.feature_dim, 4, 4)
            
            # 16x16
            transformer_features = F.interpolate(transformer_features, size=(16, 16), mode='bilinear', align_corners=False)
            
            # Step 3: 
            transformer_features = transformer_features.reshape(B, V, self.feature_dim, 16, 16)
            
            view_global_features = self.global_pool(transformer_features.view(B*V, self.feature_dim, 16, 16))
            view_global_features = view_global_features.reshape(B, V, self.feature_dim)
            
            #  (front-back, left-right)
            attended_features, _ = self.cross_view_attention(
                view_global_features, view_global_features, view_global_features
            )  # [B, V, feature_dim]
            
            # Step 4:  -  reshapeview
            fused_global = self.view_fusion(attended_features.reshape(B, V * self.feature_dim))  # [B, feature_dim]
            
            local_features = transformer_features.mean(dim=1)  # [B, feature_dim, 16, 16]
            
        else:
            # CNN
            _, feat_dim, feat_h, feat_w = cnn_features.shape
            cnn_features = cnn_features.view(B, V, feat_dim, feat_h, feat_w)
            
            fused_features = cnn_features.view(B, V * feat_dim, feat_h, feat_w)
            local_features = self.view_fusion(fused_features)
            
            fused_global = self.global_pool(local_features).reshape(B, self.feature_dim)
        
        # Step 5: 
        output_features = self.output_adapter(local_features)
        
        return output_features, fused_global


class TransformerMultiViewFeatureExtractor(nn.Module):
    
    def __init__(self, feature_dim=256, patch_size=16):
        super(TransformerMultiViewFeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.num_patches = (512 // patch_size) ** 2  # 32x32 = 1024 patches
        
        #   ()
        self.color_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        #  Patch embedding
        self.patch_embed = nn.Conv2d(feature_dim, feature_dim, kernel_size=patch_size, stride=patch_size)
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, feature_dim))
        self.dropout = nn.Dropout(0.1)
        
        #  Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.view_fusion_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, multi_view_images):
        """
        Args:
            multi_view_images: [B, 4, 3, H, W]
        Returns:
            features: [B, feature_dim, 16, 16] 
            global_features: [B, feature_dim] 
        """
        B, V, C, H, W = multi_view_images.shape
        
        # Step 1: 
        images = multi_view_images.view(B * V, C, H, W)
        color_features = self.color_encoder(images)  # [B*V, feature_dim, H, W]
        
        # Step 2: Patch embedding
        patch_features = self.patch_embed(color_features)  # [B*V, feature_dim, 32, 32]
        patch_features = patch_features.flatten(2).transpose(1, 2)  # [B*V, 1024, feature_dim]
        
        # Step 3: 
        patch_features = patch_features + self.pos_embed
        patch_features = self.dropout(patch_features)
        
        # Step 4: Transformer encoding
        encoded_features = self.transformer(patch_features)  # [B*V, 1024, feature_dim]
        
        # Step 5: 
        encoded_features = encoded_features.view(B, V, self.num_patches, self.feature_dim)
        
        view_features = encoded_features.mean(dim=2)  # [B, V, feature_dim]
        
        fused_views, attention_weights = self.view_fusion_attention(
            view_features, view_features, view_features
        )  # [B, V, feature_dim]
        
        global_features = fused_views.mean(dim=1)  # [B, feature_dim]
        global_features = self.output_proj(global_features)
        
        # Step 6:  (ResNet)
        weighted_patches = (encoded_features * fused_views.unsqueeze(2)).sum(dim=1)  # [B, 1024, feature_dim]
        
        reconstructed_features = weighted_patches.transpose(1, 2).view(B, self.feature_dim, 32, 32)
        
        # 16x16
        output_features = F.adaptive_avg_pool2d(reconstructed_features, (16, 16))
        
        return output_features, global_features


class SegmentedRubberTreeDataset(Dataset):
    """"""
    
    def __init__(self, data_root, transform=None, return_labels=True):
        self.data_root = data_root
        self.transform = transform
        self.return_labels = return_labels
        self.parser = ColorSegmentationParser()
        
        self.data_list = self._get_data_list()
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                # NOTE: no normalisation — raw RGB preserved for colour parsing
            ])
    
    def _get_data_list(self):
        pattern = os.path.join(self.data_root, "*_multi-view_images")
        folders = glob.glob(pattern)
        return sorted(folders)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        folder_path = self.data_list[idx]
        folder_name = os.path.basename(folder_path)
        base_name = folder_name.replace('_multi-view_images', '')
        
        views = ['front', 'back', 'left', 'right']
        images = []
        labels = []
        
        for view in views:
            img_path = os.path.join(folder_path, f"{base_name}_{view}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                
                if self.return_labels:
                    # numpy
                    img_np = np.array(img)
                    label_map = self.parser.rgb_to_label(img_np)
                    labels.append(torch.from_numpy(label_map).long())
                
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        
        if len(images) == 4:
            images = torch.stack(images)  # [4, 3, H, W]
            if self.return_labels:
                labels = torch.stack(labels)  # [4, H, W]
        else:
            while len(images) < 4:
                images.append(images[0])
                if self.return_labels:
                    labels.append(labels[0] if labels else torch.zeros(512, 512, dtype=torch.long))
            images = torch.stack(images)
            if self.return_labels:
                labels = torch.stack(labels)
        
        result = {
            'image': images,
            'folder_name': folder_name
        }
        
        if self.return_labels:
            result['label'] = labels
            
        return result


def test_color_parser():
    """"""
    print(" colour segmentation parser...")
    
    data_root = "./data/multi_view_predict_outputs_images"
    
    if os.path.exists(data_root):
        dataset = SegmentedRubberTreeDataset(data_root, return_labels=True)
        print(f"   : {len(dataset)}")
        
        if len(dataset) > 0:
            # sample
            sample = dataset[0]
            images = sample['image']  # [4, 3, 512, 512]
            labels = sample['label']  # [4, 512, 512] 
            
            print(f"   : {images.shape}")
            print(f"   : {labels.shape}")
            
            first_view_img = images[0].permute(1, 2, 0).numpy() * 255  # 0-255
            parser = ColorSegmentationParser()
            
            print("    :")
            parser.debug_image_colors(first_view_img.astype(np.uint8))
            
            stats = parser.analyze_segmentation(first_view_img.astype(np.uint8))
            print("\n    :")
            for class_name, stat in stats.items():
                print(f"     {class_name}: {stat['count']}  ({stat['percentage']:.1f}%)")
            
            first_label = labels[0]
            unique_labels = torch.unique(first_label)
            print(f"    : {unique_labels.tolist()}")
            
            if len(unique_labels) == 1 and unique_labels[0] == 3:
                print("\n    :")
                improved_labels = parser.improved_color_matching(first_view_img.astype(np.uint8))
                improved_unique = np.unique(improved_labels)
                print(f"    : {improved_unique.tolist()}")
            
            #  sample
            print(f"\n    sample...")
            sample_stats = []
            max_samples = min(10, len(dataset))  # 10sample
            
            for i in range(max_samples):
                sample = dataset[i]
                images = sample['image']
                labels = sample['label']
                
                first_view_img = images[0].permute(1, 2, 0).numpy() * 255
                stats = parser.analyze_segmentation(first_view_img.astype(np.uint8))
                
                sample_stats.append({
                    'sample_id': i,
                    'crown_pct': stats['crown']['percentage'],
                    'trunk_pct': stats['trunk']['percentage'], 
                    'interference_pct': stats['interference']['percentage'],
                    'background_pct': stats['background']['percentage']
                })
            
            avg_crown = np.mean([s['crown_pct'] for s in sample_stats])
            avg_trunk = np.mean([s['trunk_pct'] for s in sample_stats])
            avg_interference = np.mean([s['interference_pct'] for s in sample_stats])
            avg_background = np.mean([s['background_pct'] for s in sample_stats])
            
            print(f"\n    sample ({max_samples}sample):")
            print(f"      crown: {avg_crown:.2f}% (: {min([s['crown_pct'] for s in sample_stats]):.2f}% - {max([s['crown_pct'] for s in sample_stats]):.2f}%)")
            print(f"      trunk: {avg_trunk:.2f}% (: {min([s['trunk_pct'] for s in sample_stats]):.2f}% - {max([s['trunk_pct'] for s in sample_stats]):.2f}%)")
            print(f"      interference: {avg_interference:.2f}% (: {min([s['interference_pct'] for s in sample_stats]):.2f}% - {max([s['interference_pct'] for s in sample_stats]):.2f}%)")
            print(f"      background: {avg_background:.2f}% (: {min([s['background_pct'] for s in sample_stats]):.2f}% - {max([s['background_pct'] for s in sample_stats]):.2f}%)")
            
            # sample
            outliers = []
            for stat in sample_stats:
                if stat['trunk_pct'] == 0 or stat['crown_pct'] == 0:
                    outliers.append(stat['sample_id'])
            
            if outliers:
                print(f"\n    sample (): {outliers}")
            else:
                print(f"\n    samplecontains")


def test_feature_extractor():
    """ - """
    print("\n ...")
    
    x = torch.randn(2, 4, 3, 512, 512)  # batch size
    
    #   (CNN + Transformer)
    print("\n  ():")
    extractor_hybrid = HybridMultiViewFeatureExtractor(feature_dim=256, use_transformer=True)
    extractor_hybrid.eval()
    
    with torch.no_grad():
        features, global_features = extractor_hybrid(x)
        print(f"    : {x.shape}")
        print(f"    feature dim: {features.shape} (ResNet50)")
        print(f"    feature dim: {global_features.shape}")
        
        assert features.shape == (2, 256, 16, 16), f": {features.shape}"
        assert global_features.shape == (2, 256), f": {global_features.shape}"
        print("    !")
    
    #  ResNet50
    print("\n ResNet50:")
    extractor_resnet = MultiViewFeatureExtractor(feature_dim=256) 
    extractor_resnet.eval()
    
    with torch.no_grad():
        features_resnet, global_features_resnet = extractor_resnet(x)
        print(f"   ResNet50 - : {features_resnet.shape}")
        print(f"   ResNet50 - : {global_features_resnet.shape}")
        print(f"    - : {features.shape}")
        print(f"    - : {global_features.shape}")
        print("    !")
    
    #  params
    print("\n params:")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params_resnet = count_parameters(extractor_resnet)
    params_hybrid = count_parameters(extractor_hybrid)
    
    print(f"   ResNet50: {params_resnet:,} params")
    print(f"   : {params_hybrid:,} params")
    print(f"   params: {((params_hybrid - params_resnet) / params_resnet * 100):+.1f}%")
    
    #  CNN
    print("\n CNN ():")
    extractor_cnn = HybridMultiViewFeatureExtractor(feature_dim=256, use_transformer=False)
    extractor_cnn.eval()
    
    with torch.no_grad():
        features_cnn, global_features_cnn = extractor_cnn(x)
        print(f"   CNN - : {features_cnn.shape}")
        print(f"   CNN - : {global_features_cnn.shape}")
        
        params_cnn = count_parameters(extractor_cnn)
        print(f"   CNNparams: {params_cnn:,} params")
    
    print(f"\n : HybridMultiViewFeatureExtractor(use_transformer=True)")
    print(f"   : params")


def test_hybrid_extractor_detailed():
    """"""
    print("\n ...")
    
    extractor = HybridMultiViewFeatureExtractor(feature_dim=256, use_transformer=True)
    extractor.eval()
    
    x = torch.randn(1, 4, 3, 512, 512)
    
    print(f" : {x.shape}")
    print(" ...")
    
    with torch.no_grad():
        B, V, C, H, W = x.shape
        
        # Step 1: CNN
        images = x.view(B * V, C, H, W)
        print(f"   Step 1 - : {images.shape}")
        
        # CNN backbone
        features, global_features = extractor(x)
        
        print(f"    :")
        print(f"     : {features.shape}")
        print(f"     : {global_features.shape}")
        
        # statistics
        print(f"    :")
        print(f"     : {features.mean().item():.4f}")
        print(f"     : {features.std().item():.4f}")
        print(f"     : {global_features.mean().item():.4f}")
        print(f"     : {global_features.std().item():.4f}")
        
        #  (, NaN)
        assert not torch.isnan(features).any(), "local features contain NaN"
        assert not torch.isnan(global_features).any(), "global features contain NaN"
        assert features.abs().sum() > 0, "local features are all zeros"
        assert global_features.abs().sum() > 0, "global features are all zeros"
        
        print("    !")


def create_hybrid_extractor_example():
    """"""
    print("\n :")
    
    code_example = '''
from models.color_segmentation_parser import HybridMultiViewFeatureExtractor

# 1. 
feature_extractor = HybridMultiViewFeatureExtractor(
    feature_dim=256,           # feature dim
    use_transformer=True       # Transformer
)

# 2. 
def forward_pass(multi_view_images):
    # multi_view_images: [B, 4, 3, 512, 512] 
    # 4: front, back, left, right
    
    # 2D
    local_features, global_features = feature_extractor(multi_view_images)
    # local_features: [B, 256, 16, 16] - 3D
    # global_features: [B, 256] - 3D
    
    return local_features, global_features

# 3. MVCTNet (mvctnet_part_seg.py)
# 3D - 2D
# feat = torch.cat([feat, local_features_reshaped], dim=1)
'''
    
    print(code_example)
    
    return code_example


if __name__ == "__main__":
    print(" ...")
    print("=" * 50)
    
    test_color_parser()
    test_feature_extractor()
    
    print("\n ")
    print("=" * 50) 
