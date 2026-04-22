import cv2
import numpy as np
from skimage import morphology, measure, filters
from scipy import ndimage
import os
import glob
from pathlib import Path


class EnhancedTrunkSegmentation:
    def __init__(self):
        # Class colours in BGR format (OpenCV convention)
        self.colors = {
            'crown':         (0, 255, 0),       # green  - crown
            'trunk':         (0, 165, 255),      # orange - trunk
            'interference':  (128, 128, 128),    # grey   - interference / noise
            'background':    (255, 255, 255)     # white  - background
        }
        self.debug = True

    # ------------------------------------------------------------------
    # Plant region extraction
    # ------------------------------------------------------------------

    def extract_plant_mask(self, image):
        """Extract the foreground plant region using HSV colour thresholding."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Green hue range
            lower_green = np.array([25, 30, 30])
            upper_green = np.array([95, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)

            # Exclude white background
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            white_mask = gray > 220
            plant_mask = green_mask & (~white_mask)

            # Light morphological clean-up
            kernel = np.ones((3, 3), np.uint8)
            plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
            plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)

            return plant_mask.astype(bool)
        except Exception as e:
            if self.debug:
                print(f"Error in extract_plant_mask: {e}")
            return np.zeros((image.shape[0], image.shape[1]), dtype=bool)

    # ------------------------------------------------------------------
    # Main tree identification
    # ------------------------------------------------------------------

    def identify_main_tree_enhanced(self, plant_mask):
        """
        Enhanced main-tree identification with stricter handling for
        complex multi-plant scenes.
        """
        try:
            # Connected-component analysis
            labeled = measure.label(plant_mask)
            regions = measure.regionprops(labeled)

            if not regions:
                return plant_mask

            # Assess scene complexity by counting large connected regions
            large_regions = [r for r in regions if r.area > 500]
            scene_complexity = len(large_regions)
            is_complex_scene = scene_complexity > 1

            if self.debug:
                print(f"Scene complexity: {scene_complexity} large regions, "
                      f"complex scene: {is_complex_scene}")

            # Filter out obviously small regions
            valid_regions = [r for r in regions if r.area > 800]
            if not valid_regions:
                largest_region = max(regions, key=lambda x: x.area)
                valid_regions = [largest_region]

            # Compute per-region scoring features
            candidates = []
            h, w = plant_mask.shape

            for region in valid_regions:
                bbox = region.bbox
                height = bbox[2] - bbox[0]
                width  = bbox[3] - bbox[1]

                compactness  = region.area / (height * width)
                aspect_ratio = height / max(width, 1)

                # Position score: regions closer to image centre score higher
                center_y = (bbox[0] + bbox[2]) / 2
                center_x = (bbox[1] + bbox[3]) / 2
                distance_to_center = np.sqrt(
                    (center_y - h / 2) ** 2 + (center_x - w / 2) ** 2)
                position_score = 1.0 - min(
                    distance_to_center / (max(h, w) / 2), 1.0)

                # Vertical continuity: fraction of image height covered
                vertical_continuity = height / h

                # Shape regularity score
                shape_regularity = min(compactness * 1.5, 1.0)

                if is_complex_scene:
                    # Complex scene: weight compactness and position more heavily
                    total_score = (region.area / 10000 * 0.3 +
                                   shape_regularity * 0.3 +
                                   min(aspect_ratio / 1.5, 1.0) * 0.15 +
                                   position_score * 0.15 +
                                   vertical_continuity * 0.1)
                else:
                    # Simple scene: original weighting
                    total_score = (region.area / 10000 * 0.4 +
                                   min(compactness * 2, 1.0) * 0.2 +
                                   min(aspect_ratio / 1.5, 1.0) * 0.2 +
                                   position_score * 0.2)

                candidates.append({
                    'region':           region,
                    'score':            total_score,
                    'compactness':      compactness,
                    'aspect_ratio':     aspect_ratio,
                    'shape_regularity': shape_regularity
                })

            # Select the highest-scoring candidate
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best_candidate = candidates[0]

            # Validation thresholds (stricter for complex scenes)
            if is_complex_scene:
                selection_threshold = {
                    'compactness':   0.35,
                    'aspect_ratio':  1.0,
                    'area':          1500
                }
            else:
                selection_threshold = {
                    'compactness':   0.25,
                    'aspect_ratio':  0.8,
                    'area':          1000
                }

            if (best_candidate['compactness']      > selection_threshold['compactness'] and
                    best_candidate['aspect_ratio'] > selection_threshold['aspect_ratio'] and
                    best_candidate['region'].area  > selection_threshold['area']):

                main_region = best_candidate['region']
                if self.debug:
                    print(f"Selected main tree: area={main_region.area}, "
                          f"compactness={best_candidate['compactness']:.3f}, "
                          f"aspect_ratio={best_candidate['aspect_ratio']:.3f}")
            else:
                # Fallback: pick the best-scoring region above the minimum area
                suitable_candidates = [
                    c for c in candidates
                    if c['region'].area > selection_threshold['area']
                ]
                if suitable_candidates:
                    main_region = suitable_candidates[0]['region']
                    if self.debug:
                        print(f"Complex scene fallback selection: "
                              f"area={main_region.area}")
                else:
                    largest_region = max(
                        [c['region'] for c in candidates], key=lambda x: x.area)
                    main_region = largest_region
                    if self.debug:
                        print(f"Using largest region as main tree: "
                              f"area={main_region.area}")

            # Build main-tree mask
            main_tree_mask = np.zeros_like(plant_mask)
            for coord in main_region.coords:
                main_tree_mask[coord[0], coord[1]] = True

            # Further refinement in complex scenes
            if is_complex_scene:
                main_tree_mask = self.refine_main_tree_complex(
                    main_tree_mask, plant_mask, main_region)

            return main_tree_mask

        except Exception as e:
            if self.debug:
                print(f"Error in identify_main_tree_enhanced: {e}")
            return plant_mask

    def refine_main_tree_complex(self, main_tree_mask, plant_mask, main_region):
        """
        Refine the main-tree mask in complex scenes by removing pixels
        that deviate too far from the vertical centre axis.
        """
        try:
            coords = np.column_stack(np.where(main_tree_mask))
            if len(coords) == 0:
                return main_tree_mask

            min_row, max_row = coords[:, 0].min(), coords[:, 0].max()
            min_col, max_col = coords[:, 1].min(), coords[:, 1].max()

            center_col = (min_col + max_col) // 2
            tree_width  = max_col - min_col

            refined_mask = main_tree_mask.copy()

            for coord in coords:
                row, col = coord[0], coord[1]
                distance_to_center = abs(col - center_col)

                # Allow wider deviations near the top, stricter near the bottom
                relative_height = (row - min_row) / max(max_row - min_row, 1)

                if relative_height < 0.3:       # top 30 %
                    max_allowed_distance = tree_width * 0.6
                elif relative_height < 0.7:     # middle 40 %
                    max_allowed_distance = tree_width * 0.5
                else:                            # bottom 30 %
                    max_allowed_distance = tree_width * 0.4

                if distance_to_center > max_allowed_distance:
                    refined_mask[row, col] = False

            if self.debug:
                removed_pixels = np.sum(main_tree_mask) - np.sum(refined_mask)
                print(f"Complex scene refinement: removed {removed_pixels} "
                      f"interference pixels")

            return refined_mask

        except Exception as e:
            if self.debug:
                print(f"Error in refine_main_tree_complex: {e}")
            return main_tree_mask

    # ------------------------------------------------------------------
    # Colour and shape feature extraction
    # ------------------------------------------------------------------

    def analyze_color_features(self, image, plant_mask):
        """
        Compute per-pixel colour features within the plant mask:
          - green_strength:      green-hue saturation signal
          - saturation_feature:  raw HSV saturation (high = likely crown)
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            green_strength      = np.zeros(plant_mask.shape, dtype=np.float32)
            saturation_feature  = np.zeros_like(green_strength)

            if np.any(plant_mask):
                # Green strength: saturation of pixels whose hue is in the
                # green band [35, 85]; zero otherwise
                green_strength[plant_mask] = np.where(
                    (hsv[plant_mask, 0] >= 35) & (hsv[plant_mask, 0] <= 85),
                    hsv[plant_mask, 1] / 255.0,
                    0
                )

                # Saturation feature: high saturation -> more likely crown
                saturation_feature[plant_mask] = hsv[plant_mask, 1] / 255.0

            return green_strength, saturation_feature

        except Exception as e:
            if self.debug:
                print(f"Error in analyze_color_features: {e}")
            return (np.zeros_like(plant_mask, dtype=np.float32),
                    np.zeros_like(plant_mask, dtype=np.float32))

    def analyze_shape_features(self, plant_mask):
        """
        Compute per-pixel shape features via connected-component analysis:
          - aspect_ratio:  bounding-box height / width
          - solidity:      region area / bounding-box area (compactness)
          - circularity:   4*pi*area / perimeter^2
        """
        try:
            labeled = measure.label(plant_mask)

            shape_features = {
                'aspect_ratio': np.zeros_like(plant_mask, dtype=np.float32),
                'solidity':     np.zeros_like(plant_mask, dtype=np.float32),
                'circularity':  np.zeros_like(plant_mask, dtype=np.float32)
            }

            for region in measure.regionprops(labeled):
                if region.area < 50:
                    continue

                try:
                    bbox_height = region.bbox[2] - region.bbox[0]
                    bbox_width  = region.bbox[3] - region.bbox[1]
                    aspect_ratio = bbox_height / max(bbox_width, 1)

                    bbox_area   = bbox_height * bbox_width
                    compactness = region.area / max(bbox_area, 1)

                    if region.perimeter > 0:
                        circularity = (4 * np.pi * region.area /
                                       (region.perimeter * region.perimeter))
                        if np.isnan(circularity) or np.isinf(circularity):
                            circularity = 0.0
                    else:
                        circularity = 0.0

                    # Clamp to valid ranges
                    aspect_ratio = max(0.1, min(aspect_ratio, 10.0))
                    compactness  = max(0.1, min(compactness,  1.0))
                    circularity  = max(0.0, min(circularity,  1.0))

                    for coord in region.coords:
                        shape_features['aspect_ratio'][coord[0], coord[1]] = aspect_ratio
                        shape_features['solidity']    [coord[0], coord[1]] = compactness
                        shape_features['circularity'] [coord[0], coord[1]] = circularity

                except Exception:
                    # Assign safe default values when feature computation fails
                    for coord in region.coords:
                        shape_features['aspect_ratio'][coord[0], coord[1]] = 1.0
                        shape_features['solidity']    [coord[0], coord[1]] = 0.8
                        shape_features['circularity'] [coord[0], coord[1]] = 0.5

            return shape_features

        except Exception as e:
            if self.debug:
                print(f"Error in analyze_shape_features: {e}")
            return {
                'aspect_ratio': np.ones_like(plant_mask, dtype=np.float32),
                'solidity':     np.ones_like(plant_mask, dtype=np.float32) * 0.8,
                'circularity':  np.ones_like(plant_mask, dtype=np.float32) * 0.5
            }

    # ------------------------------------------------------------------
    # Crown / trunk boundary detection
    # ------------------------------------------------------------------

    def find_crown_trunk_boundary(self, plant_mask, green_strength,
                                   saturation_feature, min_row, max_row,
                                   tree_height):
        """
        Locate the vertical boundary between the crown and trunk regions.

        Strategy:
          1. Detect the ground / base region below 85 % of tree height.
          2. Set a reasonable maximum trunk height (40 % of tree height,
             capped at 150 px).
          3. Refine using colour features: rows with >= 80 % strong-crown
             pixels are likely crown, so the trunk starts below them.
          4. Enforce a minimum trunk height of 40 px.

        Returns:
            trunk_allowed_top    (float): upper boundary row for the trunk
            trunk_allowed_bottom (float): lower boundary row for the trunk
        """
        try:
            h, w = plant_mask.shape

            # 1. Potential ground area: bottom 15 % of tree extent
            ground_threshold = min_row + tree_height * 0.85
            potential_ground_mask = np.zeros_like(plant_mask, dtype=bool)

            for row in range(int(ground_threshold), h):
                for col in range(w):
                    if (row < plant_mask.shape[0] and
                            col < plant_mask.shape[1] and
                            plant_mask[row, col]):
                        potential_ground_mask[row, col] = True

            # 2. Find the topmost row of the ground area
            ground_coords = np.column_stack(np.where(potential_ground_mask))
            if len(ground_coords) > 0:
                ground_top = ground_coords[:, 0].min()
            else:
                ground_top = int(ground_threshold)

            # 3. Reasonable trunk height ceiling
            reasonable_trunk_height = min(tree_height * 0.4, 150)
            trunk_allowed_top = ground_top - reasonable_trunk_height

            # 4. Colour-feature refinement: detect rows that are strongly crown
            crown_start_candidates = []
            search_start = max(int(trunk_allowed_top), min_row)
            search_end   = int(ground_top - 20)  # leave at least 20 px for trunk

            for row in range(search_start, min(search_end, h)):
                crown_pixels_in_row = 0
                total_pixels_in_row = 0

                for col in range(w):
                    if plant_mask[row, col]:
                        total_pixels_in_row += 1
                        if (green_strength[row, col]     > 0.65 and
                                saturation_feature[row, col] > 0.55):
                            crown_pixels_in_row += 1

                if total_pixels_in_row > 0:
                    crown_ratio = crown_pixels_in_row / total_pixels_in_row
                    if crown_ratio > 0.8:   # row is >= 80 % strong-crown pixels
                        crown_start_candidates.append(row)

            if crown_start_candidates:
                detected_crown_start = min(crown_start_candidates)
                trunk_allowed_top = max(
                    trunk_allowed_top, detected_crown_start - 30)

            # 5. Enforce minimum trunk height (40 px)
            min_trunk_height = 40
            if ground_top - trunk_allowed_top < min_trunk_height:
                trunk_allowed_top = ground_top - min_trunk_height

            # 6. Boundary sanity check
            trunk_allowed_top    = max(trunk_allowed_top,
                                       min_row + tree_height * 0.2)
            trunk_allowed_bottom = ground_top

            if self.debug:
                actual_trunk_height = trunk_allowed_bottom - trunk_allowed_top
                print(f"Boundary detection: ground_top={ground_top}")
                print(f"Trunk region: rows {trunk_allowed_top:.0f} to "
                      f"{trunk_allowed_bottom:.0f} "
                      f"(height={actual_trunk_height:.0f} px)")

            return trunk_allowed_top, trunk_allowed_bottom

        except Exception as e:
            if self.debug:
                print(f"Error in find_crown_trunk_boundary: {e}")
            # Safe defaults
            return min_row + tree_height * 0.4, min_row + tree_height * 0.9

    # ------------------------------------------------------------------
    # Trunk skeleton and region construction
    # ------------------------------------------------------------------

    def find_boundary_limited_trunk(self, plant_mask, green_strength,
                                     trunk_allowed_top, trunk_allowed_bottom,
                                     min_row, max_row, h, w):
        """
        Compute the trunk centreline path and expand it into a trunk region,
        constrained to [trunk_allowed_top, trunk_allowed_bottom].
        """
        try:
            trunk_path = []

            start_row = max(int(trunk_allowed_top), min_row)
            end_row   = min(int(trunk_allowed_bottom), max_row)

            if start_row >= end_row:
                return (np.zeros_like(plant_mask),
                        [],
                        np.zeros_like(plant_mask))

            # Build centreline by averaging plant-pixel columns per row
            for row in range(start_row, min(end_row + 1, h)):
                row_mask = plant_mask[row, :]
                if np.any(row_mask):
                    plant_cols = np.where(row_mask)[0]
                    if len(plant_cols) > 0:
                        center_col = int(np.mean(plant_cols))
                        trunk_path.append([row, center_col])

            # Smooth the centreline with a sliding window
            if len(trunk_path) > 3:
                smoothed_path = []
                window_size   = min(3, len(trunk_path) // 2)

                for i in range(len(trunk_path)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx   = min(len(trunk_path), i + window_size // 2 + 1)
                    avg_col   = np.mean([p[1] for p in trunk_path[start_idx:end_idx]])
                    smoothed_path.append([trunk_path[i][0], int(avg_col)])

                trunk_path = smoothed_path

            # Rasterise centreline
            trunk_centerline = np.zeros_like(plant_mask)
            for point in trunk_path:
                if 0 <= point[0] < h and 0 <= point[1] < w:
                    trunk_centerline[point[0], point[1]] = True

            # Expand centreline into a trunk region
            trunk_region = self.create_boundary_limited_trunk_region(
                plant_mask, trunk_path, start_row, end_row, h, w)

            if self.debug:
                print(f"Bounded trunk path: {len(trunk_path)} points, "
                      f"rows {start_row}-{end_row}")

            return trunk_centerline, trunk_path, trunk_region

        except Exception as e:
            if self.debug:
                print(f"Error in find_boundary_limited_trunk: {e}")
            return (np.zeros_like(plant_mask),
                    [],
                    np.zeros_like(plant_mask))

    def create_boundary_limited_trunk_region(self, plant_mask, trunk_path,
                                              start_row, end_row, h, w):
        """
        Expand the trunk centreline path into a filled trunk region using
        an adaptive elliptical brush. In wide / complex scenes a larger
        brush is used to improve detection coverage.
        """
        trunk_region = np.zeros_like(plant_mask)

        if len(trunk_path) == 0:
            return trunk_region

        tree_section_height = max(end_row - start_row, 1)

        # Detect whether the scene is wide (complex)
        coords = np.column_stack(np.where(plant_mask))
        if len(coords) > 0:
            plant_width  = coords[:, 1].max() - coords[:, 1].min()
            plant_height = coords[:, 0].max() - coords[:, 0].min()
            aspect_ratio = plant_height / max(plant_width, 1)
            is_wide_scene = aspect_ratio < 1.5
        else:
            is_wide_scene = False

        for point in trunk_path:
            row, col = point[0], point[1]

            # Adaptive width: trunk gets wider towards the base
            relative_height = (row - start_row) / tree_section_height

            if is_wide_scene:
                # Larger brush for complex / wide scenes
                if relative_height < 0.3:
                    width = 8
                elif relative_height < 0.7:
                    width = 12
                else:
                    width = 15
            else:
                # Standard brush for simple scenes
                if relative_height < 0.3:
                    width = 6
                elif relative_height < 0.7:
                    width = 8
                else:
                    width = 10

            # Elliptical brush (narrower horizontally than vertically)
            for dr in range(-width, width + 1):
                for dc in range(-width // 2, width // 2 + 1):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < h and 0 <= nc < w and plant_mask[nr, nc]:
                        ellipse_dist = ((dr / width) ** 2 +
                                        (dc / (width // 2)) ** 2)
                        if ellipse_dist <= 1.0:
                            trunk_region[nr, nc] = True

        # Ensure vertical continuity along the centreline
        trunk_region = self.ensure_trunk_vertical_continuity(
            trunk_region, trunk_path, plant_mask)

        if self.debug and is_wide_scene:
            print("Wide / complex scene: using enlarged trunk brush")

        return trunk_region

    def ensure_trunk_vertical_continuity(self, trunk_region, trunk_path,
                                          plant_mask):
        """
        Fill gaps between consecutive centreline points by linear
        interpolation, guaranteeing a vertically continuous trunk mask.
        """
        try:
            if len(trunk_path) < 2:
                return trunk_region

            h, w = trunk_region.shape

            for i in range(len(trunk_path) - 1):
                start_row, start_col = trunk_path[i]
                end_row,   end_col   = trunk_path[i + 1]

                if abs(end_row - start_row) > 1:
                    steps = abs(end_row - start_row)
                    for j in range(steps):
                        t = j / max(steps, 1)
                        interp_row = int(start_row + t * (end_row - start_row))
                        interp_col = int(start_col + t * (end_col - start_col))

                        # Small neighbourhood around the interpolated point
                        for dr in range(-3, 4):
                            for dc in range(-3, 4):
                                nr, nc = interp_row + dr, interp_col + dc
                                if (0 <= nr < h and 0 <= nc < w and
                                        plant_mask[nr, nc]):
                                    if abs(dr) + abs(dc) <= 3:
                                        trunk_region[nr, nc] = True

            return trunk_region

        except Exception as e:
            if self.debug:
                print(f"Error in ensure_trunk_vertical_continuity: {e}")
            return trunk_region

    # ------------------------------------------------------------------
    # Multi-feature segmentation
    # ------------------------------------------------------------------

    def multi_feature_segmentation_enhanced(self, plant_mask, trunk_region,
                                             green_strength, saturation_feature,
                                             shape_features, min_row, max_row,
                                             tree_height, tree_width,
                                             trunk_allowed_top,
                                             trunk_allowed_bottom):
        """
        Per-pixel classification using a combination of:
          - Spatial constraints (forced crown above trunk zone, forced
            interference below ground threshold)
          - Distance to trunk centreline
          - Colour features (green strength, saturation)
          - Shape features (aspect ratio)
          - Vertical position within the trunk zone

        Label map: 1 = crown, 2 = trunk, 3 = interference
        """
        try:
            result_mask = np.zeros_like(plant_mask, dtype=np.uint8)

            # Detect whether the scene is complex (plant wider than main tree)
            coords = np.column_stack(np.where(plant_mask))
            plant_width    = coords[:, 1].max() - coords[:, 1].min()
            is_complex_scene = (plant_width / max(tree_width, 1)) > 1.3

            ground_threshold = min_row + tree_height * 0.85

            # Build a distance map to the trunk region
            trunk_distance_map = np.full(
                plant_mask.shape, np.inf, dtype=np.float32)
            trunk_coords = np.column_stack(np.where(trunk_region))

            if len(trunk_coords) > 0:
                for coord in coords:
                    row, col = coord[0], coord[1]
                    distances = np.sqrt(
                        (trunk_coords[:, 0] - row) ** 2 +
                        (trunk_coords[:, 1] - col) ** 2)
                    trunk_distance_map[row, col] = np.min(distances)

            # Per-pixel classification
            for coord in coords:
                row, col = coord[0], coord[1]

                # Forced interference: below the ground threshold
                if row > ground_threshold:
                    result_mask[row, col] = 3
                    continue

                # Forced crown: above the permitted trunk zone
                if row < trunk_allowed_top:
                    result_mask[row, col] = 1
                    continue

                # Forced interference: below the permitted trunk zone
                if row > trunk_allowed_bottom:
                    result_mask[row, col] = 3
                    continue

                # Feature-based scoring within the trunk zone
                green_val       = green_strength[row, col]
                saturation_val  = saturation_feature[row, col]
                aspect_ratio    = shape_features['aspect_ratio'][row, col]
                trunk_distance  = trunk_distance_map[row, col]

                trunk_score = 0
                crown_score = 0

                # 1. Trunk region membership
                if trunk_region[row, col]:
                    trunk_score += 3

                # 2. Distance to trunk centreline
                if   trunk_distance <= 5:
                    trunk_score += 3
                elif trunk_distance <= 10:
                    trunk_score += 2
                elif trunk_distance <= 15:
                    trunk_score += 1

                # 3. Colour features (relaxed thresholds in complex scenes)
                if is_complex_scene:
                    trunk_score += 1 if green_val     < 0.7 else 0
                    crown_score += 0 if green_val     < 0.7 else 1
                    trunk_score += 1 if saturation_val < 0.7 else 0
                    crown_score += 0 if saturation_val < 0.7 else 1
                else:
                    trunk_score += 1 if green_val     < 0.6 else 0
                    crown_score += 0 if green_val     < 0.6 else 1
                    trunk_score += 1 if saturation_val < 0.6 else 0
                    crown_score += 0 if saturation_val < 0.6 else 1

                # 4. Shape: tall, narrow shapes favour trunk
                if aspect_ratio > 1.5:
                    trunk_score += 1

                # 5. Vertical position: core of the trunk zone
                relative_height = ((row - trunk_allowed_top) /
                                   max(trunk_allowed_bottom - trunk_allowed_top, 1))
                if 0.2 < relative_height < 0.9:
                    trunk_score += 1

                # Decision
                decision_threshold = 3 if is_complex_scene else 2
                if trunk_score >= decision_threshold and trunk_score > crown_score:
                    result_mask[row, col] = 2   # trunk
                else:
                    result_mask[row, col] = 1   # crown

            # Post-processing
            result_mask = self.enhanced_post_process_v2(
                result_mask, plant_mask, trunk_region,
                green_strength, is_complex_scene)

            return result_mask

        except Exception as e:
            if self.debug:
                print(f"Error in multi_feature_segmentation_enhanced: {e}")
            # Fallback: label everything as crown
            result_mask = np.zeros_like(plant_mask, dtype=np.uint8)
            result_mask[plant_mask] = 1
            return result_mask

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def enhanced_post_process_v2(self, result_mask, plant_mask, trunk_region,
                                  green_strength, is_complex_scene):
        """
        Post-processing pipeline (V2):
          1. Convert small crown islands inside the trunk region to trunk.
          2. Region-grow the trunk outward within the trunk zone.
          3. Morphological closing to fill small holes in the trunk mask.
          4. Enforce a minimum trunk area by seeding from the trunk centroid.
        """
        try:
            trunk_mask = (result_mask == 2)
            crown_mask = (result_mask == 1)

            # 1. Small crown patches inside the trunk region -> trunk
            crown_in_trunk_region = crown_mask & trunk_region
            if np.any(crown_in_trunk_region):
                labeled_crown = measure.label(crown_in_trunk_region)
                for region in measure.regionprops(labeled_crown):
                    threshold = 30 if is_complex_scene else 20
                    if region.area < threshold:
                        for coord in region.coords:
                            if (0 <= coord[0] < result_mask.shape[0] and
                                    0 <= coord[1] < result_mask.shape[1]):
                                result_mask[coord[0], coord[1]] = 2

            # 2. Region-grow from existing trunk pixels
            trunk_mask = (result_mask == 2)
            if np.any(trunk_mask):
                grown_trunk = self.region_grow_trunk(
                    result_mask, trunk_region, green_strength, is_complex_scene)
                result_mask[grown_trunk & plant_mask] = 2

            # 3. Morphological closing: fill small holes
            trunk_mask = (result_mask == 2)
            if np.any(trunk_mask):
                kernel_size  = 5 if is_complex_scene else 3
                kernel       = np.ones((kernel_size, kernel_size), np.uint8)
                trunk_filled = cv2.morphologyEx(
                    trunk_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                new_trunk_pixels = (
                    (trunk_filled == 1) & (result_mask == 1) & plant_mask)
                result_mask[new_trunk_pixels] = 2

            # 4. Enforce minimum trunk area
            trunk_count    = np.sum(result_mask == 2)
            min_trunk_area = 150 if is_complex_scene else 100

            if trunk_count < min_trunk_area:
                trunk_coords = np.column_stack(np.where(trunk_region))
                if len(trunk_coords) > 0:
                    center_y = int(np.mean(trunk_coords[:, 0]))
                    center_x = int(np.mean(trunk_coords[:, 1]))

                    h, w   = result_mask.shape
                    radius = 15 if is_complex_scene else 12

                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius // 2, radius // 2 + 1):
                            ny, nx = center_y + dy, center_x + dx
                            if 0 <= ny < h and 0 <= nx < w and plant_mask[ny, nx]:
                                ellipse_dist = ((dy / radius) ** 2 +
                                                (dx / (radius // 2)) ** 2)
                                if ellipse_dist <= 1.0:
                                    result_mask[ny, nx] = 2

            if self.debug and is_complex_scene:
                final_trunk_count = np.sum(result_mask == 2)
                print(f"Complex scene post-processing: trunk pixels "
                      f"{trunk_count} -> {final_trunk_count}")

            return result_mask

        except Exception as e:
            if self.debug:
                print(f"Error in enhanced_post_process_v2: {e}")
            return result_mask

    def region_grow_trunk(self, result_mask, trunk_region, green_strength,
                           is_complex_scene):
        """
        Grow the trunk mask by one dilation step, accepting neighbouring
        pixels that lie inside the trunk region and have low green strength.
        """
        try:
            trunk_mask   = (result_mask == 2)
            grown_region = trunk_mask.copy()

            if not np.any(trunk_mask):
                return grown_region

            # Dilate the current trunk mask to find candidate border pixels
            kernel        = np.ones((3, 3), np.uint8)
            trunk_dilated = cv2.dilate(
                trunk_mask.astype(np.uint8), kernel, iterations=1)
            seed_points = (
                (trunk_dilated == 1) & (trunk_mask == 0) & trunk_region)

            seed_coords      = np.column_stack(np.where(seed_points))
            green_threshold  = 0.75 if is_complex_scene else 0.65

            for coord in seed_coords:
                row, col = coord[0], coord[1]
                # Accept: low green strength and currently labelled as crown
                if (green_strength[row, col] < green_threshold and
                        result_mask[row, col] == 1):
                    grown_region[row, col] = True

            return grown_region

        except Exception as e:
            if self.debug:
                print(f"Error in region_grow_trunk: {e}")
            return (result_mask == 2)

    # ------------------------------------------------------------------
    # Main segmentation entry point
    # ------------------------------------------------------------------

    def segment_single_tree(self, image):
        """
        Full segmentation pipeline for a single tree image:
          1.  Extract plant foreground mask
          2.  Identify the main tree (largest / best-scoring region)
          3.  Compute bounding geometry
          4.  Analyse colour features
          5.  Detect crown / trunk boundary
          6.  Analyse shape features
          7.  Build the bounded trunk region
          8.  Multi-feature pixel classification
          9.  Render the colour-coded result image
        """
        try:
            if self.debug:
                print("=== Starting enhanced segmentation ===")

            # Step 1: Extract plant region
            plant_mask = self.extract_plant_mask(image)
            if not np.any(plant_mask):
                return image.copy()

            # Step 2: Identify main tree
            main_tree_mask = self.identify_main_tree_enhanced(plant_mask)

            # Step 3: Bounding geometry
            coords = np.column_stack(np.where(main_tree_mask))
            min_row, max_row = coords[:, 0].min(), coords[:, 0].max()
            min_col, max_col = coords[:, 1].min(), coords[:, 1].max()
            tree_height = max_row - min_row
            tree_width  = max_col - min_col

            if self.debug:
                print(f"Tree dimensions: height={tree_height}, "
                      f"width={tree_width}")

            # Step 4: Colour features
            green_strength, saturation_feature = self.analyze_color_features(
                image, main_tree_mask)

            # Step 5: Crown / trunk boundary
            trunk_allowed_top, trunk_allowed_bottom = \
                self.find_crown_trunk_boundary(
                    main_tree_mask, green_strength, saturation_feature,
                    min_row, max_row, tree_height)

            # Step 6: Shape features
            shape_features = self.analyze_shape_features(main_tree_mask)

            # Step 7: Bounded trunk region
            trunk_centerline, trunk_path, trunk_region = \
                self.find_boundary_limited_trunk(
                    main_tree_mask, green_strength,
                    trunk_allowed_top, trunk_allowed_bottom,
                    min_row, max_row,
                    image.shape[0], image.shape[1])

            # Step 8: Multi-feature segmentation
            result_mask = self.multi_feature_segmentation_enhanced(
                main_tree_mask, trunk_region,
                green_strength, saturation_feature,
                shape_features,
                min_row, max_row, tree_height, tree_width,
                trunk_allowed_top, trunk_allowed_bottom)

            # Step 9: Render result
            result_image = image.copy()

            # Pixels in the plant mask but outside the main tree -> interference
            non_main_tree = plant_mask & (~main_tree_mask)
            result_image[non_main_tree] = self.colors['interference']

            crown_mask        = (result_mask == 1)
            trunk_mask        = (result_mask == 2)
            interference_mask = (result_mask == 3)

            result_image[crown_mask]        = self.colors['crown']
            result_image[trunk_mask]        = self.colors['trunk']
            result_image[interference_mask] = self.colors['interference']

            if self.debug:
                crown_count        = np.sum(crown_mask)
                trunk_count        = np.sum(trunk_mask)
                interference_count = (np.sum(interference_mask) +
                                      np.sum(non_main_tree))
                print(f"Segmentation result: crown={crown_count}, "
                      f"trunk={trunk_count}, "
                      f"interference={interference_count}")

            return result_image

        except Exception as e:
            if self.debug:
                import traceback
                print(f"Error in segment_single_tree: {e}")
                traceback.print_exc()
            return image.copy()

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    def read_image_chinese_path(self, image_path):
        """
        Read an image from a path that may contain non-ASCII characters.
        Uses binary file reading + cv2.imdecode to avoid OpenCV path issues.
        """
        try:
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                return None

            with open(image_path, 'rb') as f:
                image_data = f.read()

            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                print(f"Image decode failed: {image_path}")
                return None

            print(f"Loaded: {os.path.basename(image_path)}, "
                  f"size: {image.shape}")
            return image

        except Exception as e:
            print(f"Error reading image {image_path}: {str(e)}")
            return None

    def segment_image(self, image_path):
        """Load and segment a single image file."""
        image = self.read_image_chinese_path(image_path)
        if image is None:
            return None

        print(f"Segmenting: {os.path.basename(image_path)}")
        result_image = self.segment_single_tree(image)
        return result_image

    def save_image_chinese_path(self, image, output_path):
        """
        Save an image to a path that may contain non-ASCII characters.
        Uses cv2.imencode + binary write to avoid OpenCV path issues.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            _, ext = os.path.splitext(output_path)
            success, encoded_img = cv2.imencode(ext, image)

            if success:
                with open(output_path, 'wb') as f:
                    f.write(encoded_img.tobytes())
                return True
            else:
                print(f"Image encode failed: {output_path}")
                return False

        except Exception as e:
            print(f"Error saving image {output_path}: {str(e)}")
            return False

    # ------------------------------------------------------------------
    # Folder / batch processing
    # ------------------------------------------------------------------

    def process_folder(self, input_folder, output_folder):
        """Process all images in a single folder."""
        if not os.path.exists(input_folder):
            print(f"Error: input folder not found: {input_folder}")
            return

        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder: {output_folder}")

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(
                glob.glob(os.path.join(input_folder, ext.upper())))
        image_files = list(set(image_files))

        if not image_files:
            print(f"No images found in {input_folder}")
            return

        print(f"Found {len(image_files)} images. Processing...")
        success_count = 0

        for i, image_path in enumerate(image_files):
            print(f"\n{'=' * 60}")
            print(f"Processing {i + 1}/{len(image_files)}: "
                  f"{os.path.basename(image_path)}")

            result_image = self.segment_image(image_path)

            if result_image is not None:
                output_filename = os.path.basename(image_path)
                output_path     = os.path.join(output_folder, output_filename)

                if self.save_image_chinese_path(result_image, output_path):
                    print(f"  Saved: {output_filename}")
                    success_count += 1
                else:
                    print(f"  Save failed: {output_filename}")
            else:
                print(f"  Processing failed: "
                      f"{os.path.basename(image_path)}")

        print(f"\nFolder done. "
              f"Succeeded: {success_count}/{len(image_files)}")
        return success_count

    def process_multiple_folders(self, root_input_folder, root_output_folder):
        """Recursively process all sub-folders under root_input_folder."""
        if not os.path.exists(root_input_folder):
            print(f"Error: root input folder not found: "
                  f"{root_input_folder}")
            return

        os.makedirs(root_output_folder, exist_ok=True)
        print(f"Output root: {root_output_folder}")

        # Collect sub-folders
        subfolders = [
            item for item in os.listdir(root_input_folder)
            if os.path.isdir(os.path.join(root_input_folder, item))
        ]

        if not subfolders:
            print(f"No sub-folders found in {root_input_folder}")
            return

        subfolders.sort()
        print(f"Found {len(subfolders)} sub-folders: {subfolders}")

        total_processed = 0
        total_success   = 0

        for i, subfolder in enumerate(subfolders):
            print(f"\n{'=' * 80}")
            print(f"Processing folder {i + 1}/{len(subfolders)}: "
                  f"{subfolder}")
            print(f"{'=' * 80}")

            input_subfolder_path  = os.path.join(
                root_input_folder,  subfolder)
            output_subfolder_path = os.path.join(
                root_output_folder, subfolder)

            success_count = self.process_folder(
                input_subfolder_path, output_subfolder_path)

            if success_count is not None:
                total_success += success_count

                # Count images in this sub-folder for the summary
                image_extensions = [
                    '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(
                        glob.glob(os.path.join(input_subfolder_path, ext)))
                    image_files.extend(
                        glob.glob(os.path.join(
                            input_subfolder_path, ext.upper())))

                folder_total     = len(list(set(image_files)))
                total_processed += folder_total

                print(f"Folder {subfolder}: "
                      f"{success_count}/{folder_total} succeeded")
            else:
                print(f"Folder {subfolder}: processing failed")

        print(f"\n{'=' * 80}")
        print(f"All done.")
        print(f"Total: {total_success}/{total_processed} images succeeded")
        if total_processed > 0:
            print(f"Success rate: "
                  f"{(total_success / total_processed * 100):.1f}%")
        else:
            print("Success rate: 0%")
        print(f"Output directory: {root_output_folder}")
        print(f"{'=' * 80}")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def main():
    """Main function: segment all images across multiple sub-folders."""
    # Configure input / output root paths
    root_input_folder  = r".\input_folde"
    root_output_folder = r".\output_folder"

    segmenter = EnhancedTrunkSegmentation()
    segmenter.process_multiple_folders(root_input_folder, root_output_folder)

if __name__ == "__main__":
    main()