"""
Batch Point Cloud Multi-View Image Generator (512x512)
Processes all point cloud files in a folder, generating a separate multi-view image folder for each file.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import os
import glob
from typing import List, Optional
from PIL import Image

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

class BatchAxisFreeGenerator:
    def __init__(self, input_folder: str, base_output_dir: str = "multi_6_view_outputs"):
        """Batch point cloud multi-view image generator

        Args:
            input_folder: Input folder path containing TXT point cloud files
            base_output_dir: The base output directory; each file will have its own subfolder created within this directory.
        """
        self.input_folder = Path(input_folder)
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)

        if not self.input_folder.exists():
            raise ValueError(f"folder does not exist: {input_folder}")

    def find_txt_files(self) -> List[Path]:

        txt_files = list(self.input_folder.glob("*.txt"))
        return sorted(txt_files)

    def get_output_folder_for_file(self, txt_file_path: Path) -> Path:
        """Generate the corresponding output folder path based on the txt file name.

        part1_1.txt -> part1_1_multi-view_images
        """
        base_name = txt_file_path.stem  #
        output_folder_name = f"{base_name}_multi-view_images"
        return self.base_output_dir / output_folder_name

    def load_point_cloud(self, txt_file_path: Path) -> Optional[np.ndarray]:

        try:
            data = np.loadtxt(txt_file_path)

            points = data[:, :3]
            return points
        except Exception as e:
            print(f"File loading failed {txt_file_path.name}: {e}")
            return None

    def get_view_angles(self):

        views = [
            {"name": "front", "elev": 0, "azim": 0},
            {"name": "back", "elev": 0, "azim": 180},
            {"name": "left", "elev": 0, "azim": 270},
            {"name": "right", "elev": 0, "azim": 90}

        ]
        return views

    def create_axis_free_view(self, points: np.ndarray, elev: float, azim: float, point_size: float = 1.0):

        plt.ioff()

        fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        ax.set_position([0, 0, 1, 1])

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c='green', s=point_size, alpha=0.8, edgecolors='none')

        ax.view_init(elev=elev, azim=azim)

        self._set_equal_aspect_3d(ax, points)

        self._remove_all_axes(ax)

        return fig

    def _set_equal_aspect_3d(self, ax, points: np.ndarray):

        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        z_range = points[:, 2].max() - points[:, 2].min()

        max_range = max(x_range, y_range, z_range) / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def _remove_all_axes(self, ax):

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.zaxis.set_visible(False)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)

        ax.grid(False)

        ax.w_xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))

        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        for spine in ax.spines.values():
            spine.set_visible(False)

    def ensure_512x512(self, image_path: Path) -> bool:

        try:
            with Image.open(image_path) as img:
                if img.size != (512, 512):

                    img_resized = img.resize((512, 512), Image.LANCZOS)
                    img_resized.save(image_path, 'JPEG', quality=95)
                    return True
                return True
        except Exception as e:
            print(f"Image resizing failed: {e}")
            return False

    def process_single_file(self, txt_file_path: Path, point_size: float = 1.0) -> bool:
        """Process a single point cloud file to generate multi-view images

        Args:
            txt_file_path: Point cloud file path
            point_size: Size of the point

        Returns:
            bool:
        """
        file_name = txt_file_path.stem
        output_folder = self.get_output_folder_for_file(txt_file_path)

        print(f"\n Processing files: {txt_file_path.name}")
        print(f"   Output directory: {output_folder.name}")

        output_folder.mkdir(exist_ok=True)

        points = self.load_point_cloud(txt_file_path)
        if points is None:
            return False

        if len(points) == 0:
            print(f"Point cloud is empty: {txt_file_path.name}")
            return False

        print(f"Point cloud information: {len(points)} Points")

        views = self.get_view_angles()
        generated_count = 0

        for view in views:
            try:

                fig = self.create_axis_free_view(points, view['elev'], view['azim'], point_size)

                # Generate filename: part1_1_back.jpg
                filename = f"{file_name}_{view['name']}.jpg"
                filepath = output_folder / filename

                plt.savefig(str(filepath),
                           format='jpg',
                           bbox_inches=None,
                           pad_inches=0,
                           facecolor='white',
                           edgecolor='none',
                           transparent=False,
                           dpi=80)  # 6.4 * 80 = 512

                plt.close(fig)


                if filepath.exists():
                    self.ensure_512x512(filepath)
                    file_size = filepath.stat().st_size / 1024


                    with Image.open(filepath) as img:
                        actual_size = img.size

                    print(f" {view['name']}: {filename} ({actual_size[0]}×{actual_size[1]}, {file_size:.1f} KB)")
                    generated_count += 1
                else:
                    print(f" No {view['name']}: File not created")

            except Exception as e:
                print(f"    No {view['name']}: Generation failed - {str(e)}")
                plt.close('all')
                continue

        success = generated_count == len(views)
        if success:
            print(f"  Yes Successfully generated {generated_count}/{len(views)} Images from different perspectives")
        else:
            print(f"   Partial success {generated_count}/{len(views)} Images from different perspectives")

        return success

    def process_all_files(self, point_size: float = 1.0) -> dict:

        txt_files = self.find_txt_files()

        if not txt_files:
            print(f"No In the folder {self.input_folder} No txt file found")
            return {"total": 0, "success": 0, "failed": 0}

        print(f" Batch point cloud multi-view image generator")
        print("=" * 80)
        print(f"Enter folder: {self.input_folder.absolute()}")
        print(f"Output base directory: {self.base_output_dir.absolute()}")
        print(f"turn up {len(txt_files)} A txt file")
        print(f"point_size: {point_size}")
        print(f"Image size: 512×512")
        print(f"Generate viewpoints: front view, rear view, left view, right view (4 views)")
        print("=" * 80)

        results = {"total": len(txt_files), "success": 0, "failed": 0, "files": []}

        for i, txt_file in enumerate(txt_files, 1):
            print(f"\n[{i}/{len(txt_files)}] Start processing...")

            try:
                success = self.process_single_file(txt_file, point_size)
                if success:
                    results["success"] += 1
                    results["files"].append({"file": txt_file.name, "status": "success"})
                else:
                    results["failed"] += 1
                    results["files"].append({"file": txt_file.name, "status": "failed"})

            except Exception as e:
                print(f"An error occurred while processing the file.: {str(e)}")
                results["failed"] += 1
                results["files"].append({"file": txt_file.name, "status": "error", "error": str(e)})


        print("\n" + "=" * 80)
        print("Batch processing completed!")
        print(f"total: {results['total']} A txt file")
        print(f"success: {results['success']} A txt file")
        print(f"failed: {results['failed']} A txt file")
        print(f"output_dir: {self.base_output_dir.absolute()}")
        print(f"Image size: 512×512")
        print(f"Each file generates four perspective images.")

        if results["failed"] > 0:
            print(f"\n Failed files:")
            for file_info in results["files"]:
                if file_info["status"] != "success":
                    error_msg = file_info.get("error", "Unknown error")
                    print(f"  • {file_info['file']}: {error_msg}")

        print("=" * 80)
        return results

def main():

    # Configuration parameters
    input_folder = r".\input_folde "  # Enter folder path
    base_output_dir = ".\multi_view_outputs"  # Basic output directory
    point_size = 1.0  # Size of the point

    # Check the input folder
    if not os.path.exists(input_folder):
        print(f"Enter folder does not exist: {input_folder}")
        print("Please modify the input_folder variable to the correct path.")
        return

    try:

        generator = BatchAxisFreeGenerator(input_folder, base_output_dir)


        results = generator.process_all_files(point_size=point_size)

        if results["success"] > 0:
            print(f"\n The generated images are ready for use in deep learning.!")
            print("✓ Precise 512×512 pixel size")
            print("✓ Only 4 horizontal perspectives")
            print("✓ No coordinate axis interference")
            print("✓ Pure point cloud features")

    except Exception as e:
        print(f"An unexpected error occurred.: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()