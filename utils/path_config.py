"""
Path configuration utility for the MVCTNet multi-modal project.
"""

import os
from pathlib import Path

class PathConfig:
    """Path configuration manager."""

    def __init__(self):
        # Cloud server data paths
        self.cloud_config = {
            'pointcloud_root': './data/RubberTree',
            'image_root':      './data/multi_view_predict_outputs_images',
            'project_root':    './project',
            'env_name':        'cloud server'
        }
        self.current_config = self._detect_environment()

    def _detect_environment(self):
        """Auto-detect the current runtime environment."""
        if os.path.exists(self.cloud_config['pointcloud_root']):
            print('Environment detected: {}'.format(self.cloud_config['env_name']))
            return self.cloud_config
        else:
            print('WARNING: cloud data path not found. '
                  'Please verify the data mount or set paths manually.')
            return None

    def get_pointcloud_root(self):
        """Return the root directory of the point cloud dataset."""
        if self.current_config:
            return self.current_config['pointcloud_root']
        raise FileNotFoundError(
            'Pointcloud data path is not configured. '
            'Check whether the autodl-fs volume is mounted.')

    def get_image_root(self):
        """Return the root directory of the multi-view image dataset."""
        if self.current_config:
            return self.current_config['image_root']
        raise FileNotFoundError(
            'Image data path is not configured. '
            'Check whether the autodl-fs volume is mounted.')

    def get_project_root(self):
        """Return the project root directory."""
        if self.current_config:
            return self.current_config['project_root']
        return '.'

    def get_all_paths(self):
        """Return (pointcloud_root, image_root, project_root) as a tuple."""
        if self.current_config:
            return (
                self.current_config['pointcloud_root'],
                self.current_config['image_root'],
                self.current_config['project_root']
            )
        raise FileNotFoundError('Paths are not configured.')

    def verify_paths(self):
        """
        Check whether all configured paths actually exist on disk.
        Returns:
            all_ok  (bool): True if every path exists.
            results (dict): maps path name -> bool.
        """
        if not self.current_config:
            return False, {'error': 'no environment detected'}

        results = {
            'pointcloud': os.path.exists(self.current_config['pointcloud_root']),
            'image':      os.path.exists(self.current_config['image_root']),
            'project':    os.path.exists(self.current_config['project_root']),
        }
        all_ok = all(results.values())
        return all_ok, results

    def print_config(self):
        """Print the current path configuration and verification status."""
        print('Path Configuration')
        print('=' * 40)
        if self.current_config:
            print('Environment : {}'.format(self.current_config['env_name']))
            print('Pointcloud  : {}'.format(self.current_config['pointcloud_root']))
            print('Images      : {}'.format(self.current_config['image_root']))
            print('Project     : {}'.format(self.current_config['project_root']))
            all_ok, results = self.verify_paths()
            print('Path check:')
            for name, exists in results.items():
                status = 'OK     ' if exists else 'MISSING'
                print('  [{}] {}'.format(status, name))
            if all_ok:
                print('All paths verified successfully.')
            else:
                print('WARNING: one or more paths are missing.')
        else:
            print('ERROR: no valid environment configuration found.')

    def set_custom_paths(self, pointcloud_root=None, image_root=None, project_root=None):
        """
        Override paths with custom values (useful for non-standard setups).
        Args:
            pointcloud_root: custom point cloud data directory.
            image_root:      custom multi-view image directory.
            project_root:    custom project root directory.
        """
        self.current_config = {
            'pointcloud_root': pointcloud_root or self.cloud_config['pointcloud_root'],
            'image_root':      image_root      or self.cloud_config['image_root'],
            'project_root':    project_root    or self.cloud_config['project_root'],
            'env_name':        'custom'
        }
        print('Custom path configuration applied.')


# ---------------------------------------------------------------------------
# Global singleton — imported by other modules via utils/path_config.py
# ---------------------------------------------------------------------------
path_config = PathConfig()


# ---------------------------------------------------------------------------
# Convenience functions (preserving the original calling interface)
# ---------------------------------------------------------------------------

def get_data_paths():
    """Return (pointcloud_root, image_root). Compatible with existing import calls."""
    return path_config.get_pointcloud_root(), path_config.get_image_root()


def get_pointcloud_root():
    """Return the point cloud dataset root directory."""
    return path_config.get_pointcloud_root()


def get_image_root():
    """Return the multi-view image dataset root directory."""
    return path_config.get_image_root()


def verify_data_paths():
    """Verify that all data paths exist. Returns (all_ok, results_dict)."""
    return path_config.verify_paths()


def print_path_config():
    """Print the current path configuration to stdout."""
    path_config.print_config()
