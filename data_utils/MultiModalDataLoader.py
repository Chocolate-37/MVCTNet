"""
Multimodal Rubber Tree Data Loader
Combines 3D point clouds with 2D multi-view images
"""

import os
import json
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from .ShapeNetDataLoader import PartNormalDataset

import sys
import os

# add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # go up to project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# attempt to import required modules
try:
    from models.color_segmentation_parser import HybridMultiViewFeatureExtractor
    _HAS_IMAGE_EXTRACTOR = True
except ImportError as e:
    print(f"WARNING: image feature extractor import failed: {e}")
    _HAS_IMAGE_EXTRACTOR = False

# path configuration
try:
    from utils.path_config import get_data_paths as _get_data_paths
    _HAS_PATH_CONFIG = True
except ImportError:
    _HAS_PATH_CONFIG = False

warnings.filterwarnings('ignore')


class MultiModalPartNormalDataset(PartNormalDataset):
    """
     Multimodal point cloud data loader
    
    supports simultaneous loading of 3D point clouds and 2D multi-view images
    fully backward compatible with PartNormalDataset
    """
    
    def __init__(self, 
                 root=None,  # 
                 npoints=2500,
                 split='train', 
                 class_choice=None,
                 normal_channel=False,
                 # multimodal arguments
                 enable_multimodal=False,
                 image_root=None,
                 image_feature_dim=256,
                 cache_image_features=True,
                 image_size=512):
        """
        multimodal data loader
        
        Args:
            params...
            enable_multimodal: enable multimodal mode (default False — preserves original 3D behaviour)
            image_root: 
            image_feature_dim: image feature dimension
            cache_image_features: cache image features in memory
            image_size: 
        """
        
        #  automatic path configuration
        if root is None:
            # path configuration
            if _HAS_PATH_CONFIG:
                try:
                    root, default_image_root = _get_data_paths()
                except:
                    # path configuration
                    root, default_image_root = self._fallback_path_detection()
            else:
                # use built-in path detection
                root, default_image_root = self._fallback_path_detection()
            
            # use default image_root if not explicitly provided
            if enable_multimodal and image_root is None:
                image_root = default_image_root
        
        #  call parent __init__ (preserves original 3D-only behaviour)
        super().__init__(root, npoints, split, class_choice, normal_channel)
        
        #  multimodal configuration
        self.enable_multimodal = enable_multimodal
        self.image_root = image_root
        self.image_feature_dim = image_feature_dim
        self.cache_image_features = cache_image_features
        self.image_size = image_size
        
        #  image processing components (initialised only when multimodal enabled)
        if self.enable_multimodal:
            self._initialize_multimodal_components()
        
        #  cache
        self.image_feature_cache = {} if cache_image_features else None
        
        #  statistics
        self.multimodal_stats = {
            'total_samples': len(self.datapath),
            'successful_loads': 0,
            'failed_loads': 0,
            'cached_features': 0
        }
    
    def _fallback_path_detection(self):
        """built-in path detection (fallback)"""
        
        cloud_pointcloud = './data/RubberTree'
        cloud_image = './data/multi_view_predict_outputs_images'
        
        local_pointcloud = './data/RubberTree'
        local_image = './data/multi_view_predict_outputs_images'
        
        # detect environment
        if os.path.exists(cloud_pointcloud):
            return cloud_pointcloud, cloud_image
        elif os.path.exists(local_pointcloud):
            return local_pointcloud, local_image
        else:
            raise ValueError("Cannot find dataset. Please specify the root parameter.")
    
    def _initialize_multimodal_components(self):
        """initialise multimodal components"""
        try:
            #  check image feature extractor availability
            if not _HAS_IMAGE_EXTRACTOR:
                raise ImportError("Image feature extractor unavailable. Please check the models module.")
            
            #  2D feature extractor (Hybrid CNN + Transformer)
            self.image_extractor = HybridMultiViewFeatureExtractor(
                feature_dim=self.image_feature_dim,
                use_transformer=True
            )
            
            #  image preprocessing
            self.image_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                # NOTE: no normalisation — raw RGB preserved for colour parsing
            ])
            
            #  validate image root directory
            if self.image_root is None:
                raise ValueError("image_root must be specified when multimodal mode is enabled.")
            
            if not os.path.exists(self.image_root):
                raise ValueError(f"Image root directory does not exist: {self.image_root}")
            
            # 2Dparams
            image_params = sum(p.numel() for p in self.image_extractor.parameters())
            image_trainable = sum(p.numel() for p in self.image_extractor.parameters() if p.requires_grad)
            print(f"   2D branch : {image_params:,} ({image_params/1e6:.2f}M)")
            
            print(f"Multimodal components initialised successfully")
            print(f"   image root: {self.image_root}")
            print(f"   feature dim: {self.image_feature_dim}")
            
        except Exception as e:
            print(f"ERROR: multimodal component initialisation failed: {e}")
            self.enable_multimodal = False
            raise
    
    def __getitem__(self, index):
        """
        retrieve a data sample by index
        
        Returns:
            enable_multimodal=False: (point_set, cls, seg) - 
            enable_multimodal=True: (point_set, cls, seg, image_features) - 
        """
        
        #  fetch 3D point cloud first (original logic unchanged)
        point_set, cls, seg = super().__getitem__(index)
        
        #  multimodal disabled — return 3D data only
        if not self.enable_multimodal:
            return point_set, cls, seg
        
        #  multimodal mode: load image features
        try:
            image_features = self._load_image_features(index)
            self.multimodal_stats['successful_loads'] += 1
            return point_set, cls, seg, image_features
            
        except Exception as e:
            #  log failure but do not interrupt training
            self.multimodal_stats['failed_loads'] += 1
            print(f"WARNING: sample {index} image loading failed: {e}")
            
            #  return zero-valued features to keep training running
            default_features = self._get_default_image_features()
            return point_set, cls, seg, default_features
    
    def _load_image_features(self, index):
        """load image features for this sample"""
        
        #  derive image folder path from point cloud filename
        pointcloud_path = self.datapath[index][1]  # : /path/to/part1_1.txt
        base_name = os.path.basename(pointcloud_path)[:-4]  # part1_1
        
        #  cache
        if self.cache_image_features and base_name in self.image_feature_cache:
            self.multimodal_stats['cached_features'] += 1
            return self.image_feature_cache[base_name]
        
        #  load multi-view images
        image_folder = os.path.join(self.image_root, f"{base_name}_multi-view_images")
        multi_view_images = self._load_multi_view_images(image_folder, base_name)
        
        #  extract 2D features
        with torch.no_grad():
            local_features, global_features = self.image_extractor(multi_view_images.unsqueeze(0))
            # local_features: [1, 256, 16, 16]
            # global_features: [1, 256]
        
        #  pack feature dict
        image_features = {
            'local': local_features.squeeze(0),   # [256, 16, 16]  
            'global': global_features.squeeze(0), # [256]
            'sample_id': base_name
        }
        
        #  cache
        if self.cache_image_features:
            self.image_feature_cache[base_name] = image_features
        
        return image_features
    
    def _load_multi_view_images(self, image_folder, base_name):
        """load 4-view images (front/back/left/right)"""
        
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        views = ['front', 'back', 'left', 'right']
        images = []
        
        for view in views:
            #  search for image file (try multiple naming conventions)
            possible_names = [
                f"{base_name}_{view}.jpg", 
                f"{base_name}_{view}.png",
                f"{view}.jpg",
                f"{view}.png"
            ]
            
            img_path = None
            for name in possible_names:
                candidate_path = os.path.join(image_folder, name)
                if os.path.exists(candidate_path):
                    img_path = candidate_path
                    break
            
            if img_path is None:
                raise FileNotFoundError(f"No image found for view '{view}' in: {image_folder}")
            
            #  load and preprocess image
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.image_transform(img)
                images.append(img_tensor)
            except Exception as e:
                raise ValueError(f"image loading failed {img_path}: {e}")
        
        #  stack into 4-view tensor [4, 3, H, W]
        multi_view_tensor = torch.stack(images)  # [4, 3, 512, 512]
        return multi_view_tensor
    
    def _get_default_image_features(self):
        """return zero-valued fallback features"""
        return {
            'local': torch.zeros(self.image_feature_dim, 16, 16),
            'global': torch.zeros(self.image_feature_dim),
            'sample_id': 'default'
        }
    
    def get_multimodal_stats(self):
        """statistics"""
        return self.multimodal_stats.copy()
    
    def clear_image_cache(self):
        """cache"""
        if self.image_feature_cache:
            self.image_feature_cache.clear()
            print("Image feature cache cleared")


def test_multimodal_dataloader():
    """multimodal data loader"""
    
    print("Testing multimodal data loader")
    print("=" * 50)
    
    #  configure data paths
    pointcloud_root = "./data/RubberTree_complex"
    image_root = "./data/multi_view_predict_outputs_images"
    
    #  test configurations
    test_configs = [
        {
            'name': ' (off)',
            'enable_multimodal': False,
            'image_root': None
        },
        {
            'name': ' (on)',
            'enable_multimodal': True,
            'image_root': image_root
        }
    ]
    
    for config in test_configs:
        print(f"\nTest: {config['name']}")
        print("-" * 30)
        
        try:
            #  create dataset
            dataset = MultiModalPartNormalDataset(
                root=pointcloud_root,
                npoints=2048,
                split='test',  # 
                normal_channel=True,
                **config
            )
            
            print(f"Dataset created successfully: {len(dataset)} samples")
            
            #  sample
            for i in range(min(3, len(dataset))):
                print(f"   Test sample {i}:")
                
                data = dataset[i]
                if config['enable_multimodal']:
                    point_set, cls, seg, image_features = data
                    print(f"      point cloud: {point_set.shape}")
                    print(f"      class label: {cls}")
                    print(f"      segmentation: {seg.shape}")
                    print(f"      image features - local: {image_features['local'].shape}")
                    print(f"      image features - global: {image_features['global'].shape}")
                    print(f"      sample ID: {image_features['sample_id']}")
                else:
                    point_set, cls, seg = data
                    print(f"      : {point_set.shape}")
                    print(f"      : {cls}")
                    print(f"      : {seg.shape}")
            
            #  multimodal statistics
            if config['enable_multimodal']:
                stats = dataset.get_multimodal_stats()
                print(f"   Multimodal statistics: {stats}")
            
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_multimodal_dataloader() 