import os
import re
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import tifffile
import logging
import cv2
from skimage.measure import regionprops
from scipy.spatial import cKDTree
import pandas as pd
import image_processing
from shapely.geometry import Polygon
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global constants
OVERLAP = 0

# -------------------------------------------------------------------
# Regex patterns for input files
#
FILE_PATTERN = re.compile(
    r"^(?P<base>[A-Za-z0-9]+)_pos(?P<pos>\d+)_merged_(?P<version>\d{2})_(?P<type>cellProb|flows)\.tif$",
    re.IGNORECASE
)
MASK_FILE_PATTERN = re.compile(
    r"^(?P<base>[A-Za-z0-9]+)_pos(?P<pos>\d+)_merged_(?P<version>\d{2})_cp_masks\.tif$",
    re.IGNORECASE
)

# -------------------------------------------------------------------
# (Image I/O functions remain unchanged)

def group_masks(files):
    groups = defaultdict(list)
    for file in files:
        match = MASK_FILE_PATTERN.match(file.name)
        if match:
            group_key = f"{match.group('base')}_merged_{match.group('version')}_cp_masks"
            groups[group_key].append(file)
    return groups

# -------------------------------------------------------------------
# NEW HELPER: assign global colony ids to current colonies.
def assign_global_ids_to_current_colonies(current_colonies, prev_assignments, max_colony_movement=500, max_distance_from_colony=500, global_id_offset=0):
    """
    current_colonies: dict mapping current colony id (e.g. cluster id + 1) to a dict with keys:
        - "cells": list of cell centroids (each as [row, col])
        - "centroid": the colony centroid (mean of cell centroids)
    prev_assignments: dict mapping previous global_colony_id to a list of cell centroids (each as [row, col])
      (these will be aggregated to compute previous colony centroids)
    global_id_offset: starting global colony id (ensures ids do not reset per well)
    Returns a dict mapping current colony id to an assigned global_colony_id.
    """
    # Compute previous colony centroids by grouping cell centroids by global id.
    prev_colony_centroids = {}  # key: global id, value: mean centroid
    prev_cell_points = []       # list of previous cell centroids for KDTree search
    prev_cell_ids = []          # corresponding global id for each cell
    for gid, pts in prev_assignments.items():
        pts = np.vstack(pts)  # shape (n,2)
        prev_colony_centroids[gid] = np.mean(pts, axis=0)
        for pt in pts:
            prev_cell_points.append(pt)
            prev_cell_ids.append(gid)
    
    # Build KDTree for previous colony centroids (for colony-to-colony matching)
    if prev_colony_centroids:
        prev_centroids_array = np.vstack(list(prev_colony_centroids.values()))
        prev_gids = list(prev_colony_centroids.keys())
        tree_prev_colonies = cKDTree(prev_centroids_array)
    else:
        tree_prev_colonies = None

    # Build KDTree for previous cell centroids
    if len(prev_cell_points) > 0:
        prev_cell_points_array = np.vstack(prev_cell_points)
        tree_prev_cells = cKDTree(prev_cell_points_array)
    else:
        tree_prev_cells = None

    current_global_ids = {}
    # Determine starting new global id (largest previous id if available, else global_id_offset)
    if prev_assignments:
        max_gid = max(prev_assignments.keys())
    else:
        max_gid = global_id_offset

    # Step 1: For each current colony, try to match by colony centroid
    for cid, data in current_colonies.items():
        cur_centroid = data["centroid"]
        assigned = False
        if tree_prev_colonies is not None:
            dist, idx = tree_prev_colonies.query(cur_centroid, distance_upper_bound=max_colony_movement)
            if dist != np.inf:
                candidate_gid = prev_gids[idx]
                current_global_ids[cid] = candidate_gid
                assigned = True
        if not assigned:
            current_global_ids[cid] = None  # mark as not assigned yet

    # --- NEW STEP 2: assign based on nearest current cell with a global id ---
    # Build a list of current cell centroids that belong to colonies with an assigned global id.
    assigned_cell_points = []
    assigned_cell_ids = []
    for cid, data in current_colonies.items():
        if current_global_ids.get(cid) is not None:
            for pt in data["cells"]:
                assigned_cell_points.append(pt)
                assigned_cell_ids.append(current_global_ids[cid])
    if assigned_cell_points:
        assigned_cell_points_array = np.vstack(assigned_cell_points)
        tree_assigned_cells = cKDTree(assigned_cell_points_array)
    else:
        tree_assigned_cells = None

    # For each colony still unassigned, query the KDTree of current cells.
    for cid, data in current_colonies.items():
        if current_global_ids[cid] is None:
            cur_centroid = data["centroid"]
            if tree_assigned_cells is not None:
                dist, idx = tree_assigned_cells.query(cur_centroid, distance_upper_bound=max_distance_from_colony)
                if dist != np.inf:
                    candidate_gid = assigned_cell_ids[idx]
                    current_global_ids[cid] = candidate_gid
                else:
                    max_gid += 1
                    current_global_ids[cid] = max_gid
            else:
                max_gid += 1
                current_global_ids[cid] = max_gid

    return current_global_ids

from types import SimpleNamespace          # ← add near other imports
import pandas as pd                        # ← if not already imported


# -------------------------------------------------------------------
# Updated colony processing function.
def count_and_classify_colonies(mask_files, brightfield_files, colony_output_folder, outlines_output_folder, dead_mask_files=None, eps=50, metrics_list=None, prev_global_assignments=None, max_colony_movement=500, max_distance_from_colony=500):
    """
    Processes a single well. First, cells are clustered into colonies.
    Then, each colony (with at least two cells) is assigned a global_colony_id.
    For days > 0 (when prev_global_assignments is provided) the assignment is based on the closest previous-day colony centroid
    (if within max_colony_movement), and if not, based on the nearest previous cell (if within max_distance_from_colony).
    Colonies not matched are given a new global_colony_id.
    """
    well_coord = mask_files[0].name.split('_')[0]

    colony_path = colony_output_folder / f"colony_{well_coord}.tif"
    outlined_path = outlines_output_folder / f"colony_{well_coord}_outlines.tif"

    if not outlined_path.exists() or PROCESS_EXISTING:
        # if "B10" in well_coord or "A5" in well_coord or "B11" in well_coord:
        #     return
        # if "B11" not in well_coord and "G11" not in well_coord and "E10" not in well_coord:
        #     return
        # if "F9" not in well_coord:
        #     return
        # if "A10" in well_coord:
        #     return
        # if "B10" not in well_coord:
        #     return

        merged_mask = image_processing.merge_well_masks(mask_files)
        height, width = merged_mask.shape



        if not regionprops(merged_mask):
            # logging.info(f"No cell regions found in well {well_coord}. Using brightfield image for outlines.")
            # colony_color_image = np.zeros((height, width, 3), dtype=np.uint8)
            # border_thickness = 8
            # colony_color_image[:border_thickness, :] = 255
            # colony_color_image[-border_thickness:, :] = 255
            # colony_color_image[:, :border_thickness] = 255
            # colony_color_image[:, -border_thickness:] = 255

            # label_text = "0"
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # font_scale = 8
            # thickness = 12
            # text_size, baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            # text_width, text_height = text_size
            # text_x = (width - text_width) // 2
            # text_y = (height + text_height) // 2
            # cv2.putText(colony_color_image, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # border_color = (128, 128, 128)
            # cv2.rectangle(colony_color_image, (0, 0), (width-1, height-1), border_color, thickness=border_thickness)

            # if not colony_path.exists() or PROCESS_EXISTING:
            #     tifffile.imwrite(str(colony_path), colony_color_image)
            #     logging.info(f"Saved blank colony image for well {well_coord} to {colony_path}")
            # else:
            #     logging.info(f"Colony image for well {well_coord} already exists. Skipping writing.")
            return

        from sklearn.cluster import OPTICS

        # Assuming merged_mask, colony_output_folder, well_coord are defined elsewhere.
        regions = regionprops(merged_mask)
        regions = sorted(regions, key=lambda r: r.label)  # Ensure consistent ordering
        cell_labels = np.array([r.label for r in regions])
        centroids = np.array([r.centroid for r in regions])

        if len(centroids) < 3:
            logging.info(f"1 cell region found in well {well_coord}. Using brightfield image for outlines.")
            colony_color_image = np.zeros((height, width, 3), dtype=np.uint8)
            border_thickness = 8
            colony_color_image[:border_thickness, :] = 255
            colony_color_image[-border_thickness:, :] = 255
            colony_color_image[:, :border_thickness] = 255
            colony_color_image[:, -border_thickness:] = 255

            label_text = "0"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 8
            thickness = 12
            text_size, baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            text_width, text_height = text_size
            text_x = (width - text_width) // 2
            text_y = (height + text_height) // 2
            cv2.putText(colony_color_image, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            border_color = (128, 128, 128)
            cv2.rectangle(colony_color_image, (0, 0), (width-1, height-1), border_color, thickness=border_thickness)

            if not colony_path.exists() or PROCESS_EXISTING:
                tifffile.imwrite(str(colony_path), colony_color_image)
                logging.info(f"Saved blank colony image for well {well_coord} to {colony_path}")
            else:
                logging.info(f"Colony image for well {well_coord} already exists. Skipping writing.")
            return
        else:
            # ------------------------------------------------------------------
            # Just after you compute `centroids` …
            # ------------------------------------------------------------------
            csv_path = colony_output_folder / f"optics_{well_coord}.csv"
            use_cached_optics = csv_path.exists()

            candidate_clust_size = min(max(3, len(centroids) // 50), 10)
            min_clust_size       = min(candidate_clust_size, len(centroids))
            
            if use_cached_optics:
                logging.info(f"Loading cached OPTICS results for well {well_coord} "
                            f"from {csv_path}")
                df = pd.read_csv(csv_path)

                # ── restore the arrays we normally create with OPTICS ──────────
                ordered_indices      = df["ordering"].to_numpy()
                core_dist     = df["core_dist"].to_numpy()
                reachability  = df["reachability"].to_numpy()
                predecessor   = df["predecessor"].to_numpy()
                labels        = df["labels"].to_numpy()
                reach_iqr_labels = df["reach_iqr_labels"].to_numpy()

                # build a lightweight stand‑in so later code still does
                #       clust.ordering_ / clust.labels_
                clust = SimpleNamespace(ordering_=ordered_indices, labels_=labels)

                # we still need xi_vals for later logic
                xi_vals = np.divide(reachability[:-1], reachability[1:],
                                    out=np.full_like(reachability[:-1], np.nan),
                                    where=(reachability[1:] != 0))
                xi_vals[0] = np.nan
                xi_vals = np.append(xi_vals, np.nan)

                space = np.arange(len(centroids))

            else:
                # ── original OPTICS clustering block ───────────────────────────
                # never request more points than we actually have

                candidate_samples = min(max(2, len(centroids) // 30), 4)
                min_samples       = min(candidate_samples, len(centroids))

                clust = OPTICS(xi=0.45, leaf_size=30, algorithm='kd_tree',
                            max_eps=300, min_samples=min_samples,
                            min_cluster_size=min_clust_size).fit(centroids)
                space         = np.arange(len(centroids))
                reachability  = clust.reachability_[clust.ordering_]
                core_dist     = clust.core_distances_[clust.ordering_]
                predecessor   = clust.predecessor_[clust.ordering_]

                xi_vals = np.divide(reachability[:-1], reachability[1:],
                                    out=np.full_like(reachability[:-1], np.nan),
                                    where=(reachability[1:] != 0))
                xi_vals[0] = np.nan
                xi_vals = np.append(xi_vals, np.nan)

                ordered_indices = clust.ordering_
                labels          = clust.labels_[ordered_indices].copy()

            # Write the modified ordered labels back into the original array.
            clust.labels_[ordered_indices] = labels

            # Compute normalized transformations, handling infinite values by converting them to NaN.
            reachability_no_inf = np.where(np.isinf(reachability), np.nan, reachability)

            window_size = 300
            half_window = window_size // 2
            noise_threshold = np.zeros_like(reachability, dtype=float)
            n = len(reachability)

            for i in range(n):
                if i < half_window:
                    # Near the left boundary: use the first 100 values
                    start = 0
                    end = min(n, window_size)
                elif i > n - half_window:
                    # Near the right boundary: use the last 100 values
                    start = max(0, n - window_size)
                    end = n
                else:
                    # In the middle: use a centered window
                    start = i - half_window
                    end = i + half_window
                window_vals = reachability_no_inf[start:end]
                window_mean = np.nanmean(window_vals)
                q25, q75         = np.nanpercentile(window_vals, (25, 75))  
                noise_threshold[i] = window_mean + 6 * (q75 - q25)
            
            # ---------------------------
            # New: Compute Reachability IQR-based Labels using window-based noise threshold.
            # ---------------------------
            reach_iqr_labels = np.zeros_like(reachability, dtype=int)
            # Mark as noise (-1) any point with reachability > the window-based noise_threshold.
            for i in range(len(reachability)):
                if reachability[i] > noise_threshold[i]:
                    reach_iqr_labels[i] = -1

            # For each contiguous segment of points labeled 0 (non-noise) that is at least min_clust_size,
            # assign a new cluster label (starting from 1).
            current_label = 1
            start_index = None
            for i in range(len(reach_iqr_labels)):
                if reach_iqr_labels[i] == 0:
                    if start_index is None:
                        start_index = i
                else:
                    if start_index is not None:
                        if i - start_index >= (min_clust_size - 1):
                            reach_iqr_labels[start_index:i] = current_label
                            if np.linalg.norm(centroids[clust.ordering_][start_index-1] - centroids[clust.ordering_][start_index]) < 300:
                                reach_iqr_labels[start_index-1] = current_label
                            current_label += 1
                        start_index = None

            # Check if the last segment qualifies.
            if start_index is not None and (len(reach_iqr_labels) - start_index) >= (min_clust_size - 1):
                reach_iqr_labels[start_index - 1:] = current_label

            # Replace each unassigned non-noise point (0) with the first label to its left that is > 0
            for i in range(len(reach_iqr_labels)):
                if reach_iqr_labels[i] == 0:
                    if i < len(reach_iqr_labels)-2:
                        if reach_iqr_labels[i-1] == -1 and reach_iqr_labels[i+1] == -1:
                                reach_iqr_labels[i] = -1
                        else:
                            j = i - 1
                            # Move left until a valid label is found or the beginning is reached.
                            while j >= 0 and (reach_iqr_labels[j] == -1):
                                if reachability[j] > 300:
                                    j = -1
                                else:
                                    j -= 1
                            if j >= 0:
                                reach_iqr_labels[i] = reach_iqr_labels[j]
                            else:
                                reach_iqr_labels[i] = -1
                    else:
                        reach_iqr_labels[i] = -1      

            reach_iqr_labels[reach_iqr_labels == 0] = -1

            from scipy.spatial import cKDTree

            # Use the current clustering labels (here stored in clust.labels_)
            all_labels = reach_iqr_labels

            valid_noise_indices_list = []
            for i in range(n):
                if reachability[i] > noise_threshold[i] and core_dist[i] < 300:
                    # If the next point exists and its reachability is below (or equal to) its noise threshold,
                    # then skip current index — it is immediately followed by a point that would be considered in-cluster.
                    if (i < n - 1) and (reachability[i + 1] <= noise_threshold[i + 1]):
                        continue
                    valid_noise_indices_list.append(i)

            valid_noise_indices = np.array(valid_noise_indices_list)

            # ---------------------------
            # Updated: Assign valid noise indices per contiguous segment
            # with candidate pivot and new pivot condition.
            # ---------------------------
            # Assume valid_noise_indices is computed beforehand.
            # Ensure valid_noise_indices is sorted in ascending order.
            valid_noise_indices_sorted = np.sort(valid_noise_indices)

            # Group the valid noise indices into contiguous segments.
            contiguous_segments = []
            if valid_noise_indices_sorted.size > 0:
                current_segment = [valid_noise_indices_sorted[0]]
                for idx in valid_noise_indices_sorted[1:]:
                    # If the current index is consecutive, add it to the current segment.
                    if idx == current_segment[-1] + 1:
                        current_segment.append(idx)
                    else:
                        contiguous_segments.append(current_segment)
                        current_segment = [idx]
                contiguous_segments.append(current_segment)

            # Process each contiguous segment.
            for seg in contiguous_segments:
                seg = np.array(seg)  # Work with a NumPy array for convenience.
                
                # Find the candidate pivot: the index in seg with highest reachability.
                pivot_idx = seg[np.argmax(reachability[seg])]
                
                # Determine neighboring indices.
                left_neighbor_idx = seg[0] - 1
                right_neighbor_idx = seg[-1] + 1
                
                # Get the neighbor labels (if available and not noise).
                left_label = None
                if left_neighbor_idx >= 0 and all_labels[left_neighbor_idx] != -1 and np.linalg.norm(centroids[clust.ordering_][left_neighbor_idx] - centroids[clust.ordering_][seg[0]]) < 300:
                    left_label = all_labels[left_neighbor_idx]
                
                right_label = None
                if right_neighbor_idx < len(all_labels) and all_labels[right_neighbor_idx] != -1 and np.linalg.norm(centroids[clust.ordering_][right_neighbor_idx] - centroids[clust.ordering_][seg[-1]]) < 300:
                    right_label = all_labels[right_neighbor_idx]
                
                # If only one neighbor label is available, use it for the entire segment.
                if left_label is None and right_label is not None:
                    for idx in seg:
                        all_labels[idx] = right_label
                    continue
                if right_label is None and left_label is not None:
                    for idx in seg:
                        all_labels[idx] = left_label
                    continue
                if left_label is None and right_label is None:
                    # No valid neighbor labels; leave the segment as noise (-1).
                    continue

                # Now compare the candidate pivot's reachability with that of the right neighbor.
                # If the right neighbor's reachability is higher (or infinite) then the rule states:
                # "that label is the new pivot" and therefore the entire segment is assigned the left label.
                if (reachability[right_neighbor_idx] > reachability[pivot_idx]) or np.isinf(reachability[right_neighbor_idx]):
                    for idx in seg:
                        all_labels[idx] = left_label
                else:
                    # Otherwise, use the candidate pivot to split the segment:
                    # - For indices before the pivot, assign left neighbor's label.
                    # - For the pivot and indices after, assign right neighbor's label.
                    for idx in seg:
                        if idx < pivot_idx:
                            all_labels[idx] = left_label
                        else:
                            all_labels[idx] = right_label
                
                
            # If desired, update the clustering object's labels.
            reach_iqr_labels = all_labels
            for idx in reach_iqr_labels:
                if core_dist[idx] > 300:
                    reach_iqr_labels[idx] = -1

            # ---------------------------
            # Part 1: Compute colony hulls for each cluster
            # ---------------------------
            # Get the unique labels excluding noise (-1)
            unique_labels = np.unique(reach_iqr_labels)
            cluster_hulls = {}    # Stores convex hull for each colony label

            for label in unique_labels:
                if label == -1:
                    continue  # skip noise points
                # Get all points in the current colony
                colony_pts = centroids[clust.ordering_][reach_iqr_labels == label]
                # Only compute a hull if there are enough points
                if colony_pts.shape[0] >= 3:
                    # Note: switching x and y here so that x=column and y=row as done in your snippet
                    pts_xy = np.column_stack((colony_pts[:, 1], colony_pts[:, 0])).astype(np.float32)
                    hull = cv2.convexHull(pts_xy)
                    cluster_hulls[label] = hull

            # ---------------------------
            # Part 2: Merge colonies based on significant intersection of their hulls
            # ---------------------------
            # Define the fraction of the smaller hull area that must be intersected to trigger a merge.
            # Adjust the threshold as needed (e.g., 0.5 means 50% of the smaller area).
            intersection_threshold = 0.3

            # First convert hulls to shapely Polygofns for easy area and intersection computations.
            label_polygons = {}
            for label, hull in cluster_hulls.items():
                # cv2.convexHull returns an array of shape (N,1,2); squeeze removes the extra dimension.
                hull_points = hull.squeeze()
                # Create a shapely Polygon; hull_points are in (x, y) order.
                if hull_points.ndim == 1:
                    # In case the hull has only one point (should not happen if >= 3 points), skip.
                    continue
                poly = Polygon(hull_points)
                label_polygons[label] = poly

            # Use union–find (or disjoint set) to allow chained merging of colonies.
            # Initialize each label as its own parent.
            parent = {label: label for label in label_polygons.keys()}

            def find(x):
                # Find the ultimate parent of label x.
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                # Merge the groups of labels x and y.
                root_x = find(x)
                root_y = find(y)
                if root_x != root_y:
                    parent[root_y] = root_x

            labels_list = list(label_polygons.keys())
            num_labels = len(labels_list)

            # Check every pair for a significant intersection.
            for i in range(num_labels):
                for j in range(i+1, num_labels):
                    label1 = labels_list[i]
                    label2 = labels_list[j]
                    poly1 = label_polygons[label1]
                    poly2 = label_polygons[label2]
                    if poly1.intersects(poly2):
                        inter_area = poly1.intersection(poly2).area
                        # Compare the intersection area with the smaller hull's area.
                        if inter_area > intersection_threshold * min(poly1.area, poly2.area):
                            union(label1, label2)

            # Build a mapping from old label to new merged label (using the union-find parents).
            label_mapping = {}
            for label in label_polygons.keys():
                new_label = find(label)
                label_mapping[label] = new_label

            # Update reach_iqr_labels so that all points in merged colonies use the same label.
            for old_label, new_label in label_mapping.items():
                if old_label != new_label:
                    reach_iqr_labels[reach_iqr_labels == old_label] = new_label

            # Optionally, you may want to refresh the hulls for the merged colonies.
            merged_hulls = {}
            merged_polygons = {}
            for label in np.unique(reach_iqr_labels):
                if label == -1:
                    continue
                colony_pts = centroids[clust.ordering_][reach_iqr_labels == label]
                if colony_pts.shape[0] >= 3:
                    pts_xy = np.column_stack((colony_pts[:, 1], colony_pts[:, 0])).astype(np.float32)
                    hull = cv2.convexHull(pts_xy)
                    merged_hulls[label] = hull
                    
                    hull_points = hull.squeeze()
                    merged_polygons[label] = Polygon(hull_points)
            
            # Create a DataFrame with the relevant columns
            df = pd.DataFrame({
                "x" : centroids[clust.ordering_][:, 1],
                "y": centroids[clust.ordering_][:, 0],
                "ordering": clust.ordering_,
                "core_dist" : core_dist,
                "reachability": reachability,
                "predecessor" : predecessor,
                "labels": labels,
                "reach_iqr_labels" : reach_iqr_labels,
            })

            # Define the output CSV file path
            csv_file = os.path.join(colony_output_folder, f"optics_{well_coord}.csv")

            # Write the DataFrame to a CSV file
            df.to_csv(csv_file, index=False)

            # new_labels = clust.labels_ + 1
            
            reach_iqr_labels[reach_iqr_labels == -1] = 0
            new_labels = np.empty_like(reach_iqr_labels)
            new_labels[ordered_indices] = reach_iqr_labels

            # Build a mapping from cell label (from regionprops) to the new colony id.
            cell_to_colony = {}
            for i, cl in enumerate(cell_labels):
                cell_to_colony[cl] = new_labels[i]

            # Group cells into current colonies using the new cluster labels.
            all_colonies = {}
            for region, cl in zip(regions, cell_labels):
                colony_id = cell_to_colony[cl]
                if colony_id != 0:
                    all_colonies.setdefault(colony_id, {"cells": [], "centroid": None})
                    all_colonies[colony_id]["cells"].append(region.centroid)

            current_colonies = {cid: info for cid, info in all_colonies.items()}
            
            # Compute colony centroids for each colony.
            for cid, data in current_colonies.items():
                data["centroid"] = np.mean(data["cells"], axis=0)
            
            # Assign global colony ids.
            if prev_global_assignments is not None:
                # Aggregate previous assignments (prev_global_assignments is a dict mapping global id to cell centroid arrays)
                prev_groups = {}
                for gid, arr in prev_global_assignments.items():
                    # Ensure we have a list of centroids for each global id.
                    if gid not in prev_groups:
                        prev_groups[gid] = []
                    # arr may be a single centroid or an array of centroids.
                    arr = np.atleast_2d(arr)
                    for pt in arr:
                        prev_groups[gid].append(pt)
                current_global_ids = assign_global_ids_to_current_colonies(current_colonies, prev_groups, max_colony_movement, max_distance_from_colony)
            else:
                # For day 0, assign new global ids sequentially.
                current_global_ids = {}
                next_id = 1
                for cid in current_colonies.keys():
                    current_global_ids[cid] = next_id
                    next_id += 1

            # Now, build an array of global colony ids for each cell.
            global_ids_array = np.zeros(len(regions), dtype=np.int32)
            for i, r in enumerate(regions):
                cl = cell_labels[i]
                colony_id = cell_to_colony[cl]
                if colony_id == 0:
                    global_ids_array[i] = 0
                else:
                    global_ids_array[i] = current_global_ids.get(colony_id, 0)

            # Create per-cell mapping (mask label -> global id).
            mapping = np.zeros(merged_mask.max() + 1, dtype=np.int32)
            for cell_label, gid in zip(cell_labels, global_ids_array):
                mapping[cell_label] = gid
            colony_map = mapping[merged_mask]

            # Group cell centroids by global colony id for metrics.
            colony_centroid_groups = {}
            for region, cell_label in zip(regions, cell_labels):
                gid = mapping[cell_label]
                if gid != 0:
                    colony_centroid_groups.setdefault(gid, []).append(region.centroid)

            num_colonies = len(np.unique(colony_map[colony_map != 0]))

            # Create colony color image using the original global colony IDs.
            # Build a color lookup table that has an entry for every label up to the maximum global ID.
            max_label = colony_map.max()
            np.random.seed(0)
            color_lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
            color_lut[0] = [0, 0, 0]
            for label in range(1, max_label + 1):
                color_lut[label] = np.random.randint(0, 256, size=3, dtype=np.uint8)
            colony_color_image = color_lut[colony_map]

            # Display the number of unique global colonies on the image.
            label_text = str(num_colonies)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 8
            thickness = 12
            text_size, baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            text_width, text_height = text_size
            text_x = (width - text_width) // 2
            text_y = (height + text_height) // 2
            cv2.putText(colony_color_image, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            logging.info(f"Well {well_coord}: Found {len(regions)} cells grouped into {num_colonies} global colonies.")

            border_thickness = 8
            if num_colonies == 0:
                border_color = (128, 128, 128)
            elif num_colonies == 1:
                border_color = (0, 255, 0)
            else:
                border_color = (255, 0, 0)
            cv2.rectangle(colony_color_image, (0, 0), (width-1, height-1), border_color, thickness=border_thickness)
            
            if not colony_path.exists() or PROCESS_EXISTING:
                tifffile.imwrite(str(colony_path), colony_color_image)
                logging.info(f"Saved colony image for well {well_coord} to {colony_path}")
            else:
                logging.info(f"Colony image for well {well_coord} already exists. Skipping writing.")

            try:
                bf_mosaic = image_processing.merge_well_brightfield(brightfield_files)
            except Exception as e:
                logging.error(f"Error merging brightfield tiles for well {well_coord}: {e}")
                sample_tile = image_processing.read_brightfield_tile(brightfield_files[0])
                tile_height, tile_width = sample_tile.shape
                full_height = tile_height + 3 * (tile_height - 0)
                full_width = tile_width + 3 * (tile_width - 0)
                bf_mosaic = image_processing.create_placeholder_image(full_height, full_width, 3, border_thickness=8, dtype=sample_tile.dtype)
            
            blend_alpha = 0.8
            blend_beta = 0.15
            blend_gamma = 0
            shaded_img = bf_mosaic.copy()
            cell_pixels = colony_map > 0
            blended_cells = (bf_mosaic[cell_pixels].astype(np.float32) * blend_alpha +
                            colony_color_image[cell_pixels].astype(np.float32) * blend_beta +
                            blend_gamma).astype(np.uint8)
            shaded_img[cell_pixels] = blended_cells

            # If dead mask files are provided, merge them and compute their region properties.
            if dead_mask_files is not None:
                merged_dead_mask = image_processing.merge_well_masks(dead_mask_files)
                dead_regions = regionprops(merged_dead_mask)

            # ---------------- colony metrics loop ----------------
            for global_colony_id, pts in colony_centroid_groups.items():
                pts = np.array(pts)
                num_cells = pts.shape[0]
                colony_centroid = np.mean(pts, axis=0)

                # ------------------------------------------------------------------
                # >>> NEW: reachability statistics for the current colony <<<
                colony_idx   = np.where(global_ids_array == global_colony_id)[0]   # indices of its cells
                colony_reach = reachability[colony_idx]
                colony_reach = colony_reach[np.isfinite(colony_reach)]  # drop NaN/Inf first
                if colony_reach.size:
                    reach_90   = float(np.nanpercentile(colony_reach, 90))
                    reach_99   = float(np.nanpercentile(colony_reach, 99))
                    reach_mean = float(np.nanmean(colony_reach))
                else:
                    reach_90 = reach_99 = reach_mean = np.nan
                # ------------------------------------------------------------------

                hull = None
                hull_area = 0
                density = 0

                # cell‑count based death metrics
                dead_in_hull = 0           # dead cells whose centroid is inside hull
                dead_in_proximity = 0      # inside hull OR within PROXIMITY_PX outside

                # NEW ───────────────────────────────────────────────
                # area‑based death metrics (px²)
                dead_area_in_hull = 0          # summed area of dead regions whose centroid is inside hull
                dead_area_in_proximity = 0     # inside hull OR within halo
                # ───────────────────────────────────────────────────

                # ---------------------------------------------------
                # build convex hull around live‑cell centroids
                if pts.shape[0] >= 3:
                    pts_xy = np.column_stack((pts[:, 1], pts[:, 0])).astype(np.float32)
                    hull = cv2.convexHull(pts_xy)
                    hull_area = cv2.contourArea(hull)
                    density = num_cells / hull_area if hull_area > 0 else 0

                if hull is not None:
                    hull_points = hull.reshape(-1, 2)
                    # colony_color = tuple(int(c) for c in color_lut[global_colony_id])
                    # cv2.polylines(
                    #     shaded_img,
                    #     [np.int32(hull_points)],
                    #     isClosed=True,
                    #     color=colony_color,
                    #     thickness=3,
                    # )

                    # ---------------------------------------------------
                    # if dead masks are available, gather death metrics
                    if dead_mask_files is not None:
                        hull_diameter = math.sqrt(hull_area)          # px
                        PROXIMITY_PX = round(0.5 * hull_diameter)    # 50 % of diameter

                        # mask = np.zeros(shaded_img.shape[:2], np.uint8)
                        # cv2.fillConvexPoly(mask, np.int32(hull_points), 255)
                        # k = 2 * PROXIMITY_PX + 1
                        # struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                        # dilated = cv2.dilate(mask, struct)
                        # ring = cv2.subtract(dilated, mask)   # halo mask

                        # # optional display of halo region on overlay
                        # overlay = shaded_img.copy()
                        # overlay[ring == 255] = (0, 255, 255)
                        # cv2.addWeighted(overlay, 0.25, shaded_img, 0.75, 0, shaded_img)

                        # ---------------------------------------------------
                        # iterate over individual dead regions
                        for d in dead_regions:
                            # regionprops: centroid = (row, col)
                            dead_centroid = (d.centroid[1], d.centroid[0])

                            # Signed distance to hull (cv2.pointPolygonTest):
                            #   dist >  0  → inside hull
                            #   dist == 0  → on hull edge
                            #   dist <  0  → outside hull (|dist| = px to hull)
                            dist = cv2.pointPolygonTest(
                                np.int32(hull_points), dead_centroid, measureDist=True
                            )

                            # ---------- INSIDE HULL ----------
                            if dist >= 0:
                                dead_in_hull += 1
                                dead_in_proximity += 1      # inside counts for proximity too

                                # NEW: accumulate region area
                                dead_area_in_hull += d.area
                                dead_area_in_proximity += d.area

                            # ---------- HALO (outside but within proximity) ----------
                            elif -PROXIMITY_PX <= dist < 0:
                                dead_in_proximity += 1
                                # NEW: accumulate region area
                                dead_area_in_proximity += d.area

                # ---------------------------------------------------
                # stash metrics for the current colony
                if metrics_list is not None:
                    metrics_list.append(
                        {
                            "well": well_coord,
                            "global_colony_id": global_colony_id,
                            "num_cells": num_cells,
                            "centroid_x": float(colony_centroid[1]),
                            "centroid_y": float(colony_centroid[0]),
                            "hull_area": hull_area,
                            "density": density,

                            # ---------- reachability (NEW) ----------
                            "reach_p90": reach_90,
                            "reach_p99": reach_99,
                            "reach_mean": reach_mean,

                            # ---------------- count‑based ----------------
                            "num_dead_cells": dead_in_hull,
                            "num_dead_proximity": dead_in_proximity,
                            "dead_live_ratio": dead_in_hull / num_cells if num_cells > 0 else 0,
                            "dead_live_ratio_proximity": dead_in_proximity / num_cells if num_cells > 0 else 0,

                            # ---------------- area‑based (NEW) ----------------
                            "dead_area_in_hull": dead_area_in_hull,
                            "dead_area_in_proximity": dead_area_in_proximity,
                            "dead_live_ratio_area": dead_area_in_hull / hull_area if hull_area > 0 else 0,
                            "dead_live_ratio_area_proximity": dead_area_in_proximity / hull_area if hull_area > 0 else 0,
                        }
                    )
                    if dead_mask_files is not None:
                        metrics_list.append(
                            {
                                # ---------------- well‑level ----------------
                                "num_dead_well_total": len(dead_regions),
                                "num_live_well_total": len(regions),
                                "dead_live_ratio_well": len(dead_regions) / len(regions) if len(regions) > 0 else 0,
                            }
                        )

            # If dead mask files are provided, outline each dead cell in red on the shaded image.
            if dead_mask_files is not None:
                # Convert merged dead mask to uint8 if necessary.
                dead_mask_uint8 = (merged_dead_mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(dead_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cv2.drawContours(shaded_img, [cnt], -1, (255, 0, 0), thickness=1)

            cv2.rectangle(shaded_img, (0, 0), (width-1, height-1), border_color, thickness=border_thickness)

            if not outlined_path.exists() or PROCESS_EXISTING:
                tifffile.imwrite(str(outlined_path), shaded_img)
                logging.info(f"Saved outlined image for well {well_coord} to {outlined_path}")
            else:
                logging.info(f"Outlined image for well {well_coord} already exists. Skipping writing.")

def merge_plate_and_masks(
        masksDir, dead_masks_dir=None, prev_global_assignments_per_well=None):
    output_base = Path(masksDir + "_full_optics3")
    os.makedirs(output_base, exist_ok=True)

    # final plate‑wide file will go here
    csv_path = output_base / "colony_metrics.csv"

    cp_mask_files = [
        f for f in Path(masksDir).iterdir()
        if f.is_file() and f.name.lower().endswith("cp_masks.tif")
    ]
    if not cp_mask_files:
        logging.error(f"No cp_masks.tif files found in {masksDir}")
        return

    cp_mask_groups = group_masks(cp_mask_files)
    logging.info(f"Found {len(cp_mask_groups)} cp_mask groups.")

    colony_output_folder   = output_base / "colonies"
    outlines_output_folder = output_base / "colonies_outlines"
    colony_output_folder.mkdir(exist_ok=True)
    outlines_output_folder.mkdir(exist_ok=True)

    brightfield_dir = Path(masksDir).parent.parent
    per_well_csv_paths = []                                   # NEW

    # ─────────────────────────   main loop over wells   ──────────────────────
    for group_key, mask_files in cp_mask_groups.items():
        if len(mask_files) != 16:
            logging.warning(
                f"Skipping group {group_key}: expected 16 mask tiles, "
                f"found {len(mask_files)}")
            continue

        well_coord = mask_files[0].name.split('_')[0]
        brightfield_files = list(
            brightfield_dir.glob(f"{well_coord}_pos*_merged_??.tif"))
        if len(brightfield_files) != 16:
            logging.warning(
                f"Expected 16 brightfield tiles for well {well_coord}, "
                f"found {len(brightfield_files)}.  Skipping.")
            continue

        # match dead‑mask tiles (if provided)
        dead_mask_files = None
        if dead_masks_dir is not None:
            dead_mask_files = [
                Path(dead_masks_dir) / m.name for m in mask_files
                if (Path(dead_masks_dir) / m.name).exists()
            ]
            if len(dead_mask_files) != 16:
                logging.warning(
                    f"Expected 16 dead‑mask tiles for well {well_coord}, "
                    f"found {len(dead_mask_files)}.  Proceeding without them.")
                dead_mask_files = None

        prev_assignments = (
            prev_global_assignments_per_well.get(well_coord)
            if prev_global_assignments_per_well else None
        )

        # ── run the heavy lifting and collect metrics just for this well ──
        well_metrics: list[dict] = []

        # import cProfile, pstats, io, pathlib
        # pr = cProfile.Profile()
        # pr.enable()

        count_and_classify_colonies(
            mask_files,
            brightfield_files,
            colony_output_folder,
            outlines_output_folder,
            dead_mask_files,
            eps=50,
            metrics_list=well_metrics,                # << local list!
            prev_global_assignments=prev_assignments,
            max_colony_movement=500,
            max_distance_from_colony=500,
        )

        # pr.disable()
        # stats = pstats.Stats(pr).sort_stats("cumtime")   # or "tottime"
        # stats.print_stats(30)        # top 30 “hottest” functions
        

        # ── dump PER‑WELL metrics csv ──────────────────────────────────────
        if well_metrics:
            per_well_csv = colony_output_folder / f"colony_metrics_{well_coord}.csv"
            # collect a union of keys so every column is kept
            fieldnames = sorted({k for row in well_metrics for k in row})
            with open(per_well_csv, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=fieldnames)
                wr.writeheader()
                wr.writerows(well_metrics)

            logging.info(f"Saved colony metrics for well {well_coord} → {per_well_csv}")
            per_well_csv_paths.append(per_well_csv)

    # Uncomment the following lines if you wish to merge plate colonies/outlines.
    # plate_colonies_output = output_base / "plate_colonies.tif"
    # merge_plate_colonies(colony_output_folder, plate_colonies_output)
    #
    # plate_colony_outlines_output = output_base / "plate_colony_outlines.tif"
    # merge_plate_colony_outlines(outlines_output_folder, plate_colony_outlines_output)
    
    # ─────────────────────────── aggregate plate file ───────────────────────
    if per_well_csv_paths:
        df_all = pd.concat(
            (pd.read_csv(p) for p in per_well_csv_paths),
            ignore_index=True, sort=False
        )
        # optional: consistent ordering of useful columns
        df_all.to_csv(csv_path, index=False)
        logging.info(f"Saved aggregated colony metrics CSV → {csv_path}")
    else:
        logging.warning("No per‑well metrics were created; nothing to aggregate.")
        
# -------------------------------------------------------------------
# NEW HELPER: load previous assignments from colony_metrics CSV.
def load_prev_assignments_from_csv(csv_path):
    """
    Load a CSV of colony metrics and return a dict mapping well coordinate to a dictionary
    mapping global_colony_id to a list of cell centroids (each centroid is [row, col]).
    """
    prev_assignments = {}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Error reading CSV {csv_path}: {e}")
        return prev_assignments

    for _, row in df.iterrows():
        well = row['well']
        gid = int(row['global_colony_id'])
        centroid = np.array([row['centroid_y'], row['centroid_x']])
        if well not in prev_assignments:
            prev_assignments[well] = {}
        prev_assignments[well].setdefault(gid, []).append(centroid)
    return prev_assignments

# -------------------------------------------------------------------
# NEW HELPER: process a single plate (multiple timepoints) sequentially.
def process_plate_datasets(datasets, model):
    """
    Given a sorted list of dataset directories (each corresponding to one timepoint for a plate),
    process them sequentially so that for timepoints > 0 the previous day’s cell centroids (per well)
    are loaded and passed to merge_plate_and_masks.
    """
    prev_assignments_per_well = None
    for dataset in datasets:    
        # Each dataset is expected to contain a subfolder "model_outputs/{model}"
        masksDir = str(Path(dataset) / "model_outputs" / model)
        logging.info(f"Processing dataset {dataset} with masksDir {masksDir}")
        if "day1" not in str(dataset):
            merge_plate_and_masks(masksDir, prev_global_assignments_per_well=prev_assignments_per_well)
        # After processing, load previous assignments from the output colony metrics CSV.
        csv_path = Path(masksDir + "_full") / "colony_metrics.csv"
        if csv_path.exists():
            prev_assignments_per_well = load_prev_assignments_from_csv(csv_path)
            logging.info(f"Loaded previous assignments from {csv_path}")
        else:
            logging.warning(f"Colony metrics CSV not found at {csv_path}; previous assignments not updated.")

# -------------------------------------------------------------------
# NEW MAIN: Automatically find datasets and group by plate.
def main():
    # datasets = [d for d in Path(r"E:\MERGED_sspsygene_growthassay_colonies").iterdir() if d.is_dir() and (d / "model_outputs").exists() and ("day1" in d.name) and "blank" not in d.name]
    # for dataset in datasets:
    #     merge_plate_and_masks(str(dataset)+r"/model_outputs/2025_03_03_23_34_16.794792_epoch_899_0.4_2", str(dataset)+r"/model_outputs/2025_03_07_02_10_15.252341_epoch_3999_0.4_1")

    merge_plate_and_masks(r"E:\MERGED_sspsygene_growthassay_colonies\cluster_optimize_set\model_outputs\optimize")

    # ground_truth_csv = r"E:\MERGED_sspsygene_growthassay_colonies\ground_truth.csv"
    # # # Set the base input folder and the model name.

    # base_input = r"E:\MERGED_sspsygene_growthassay_colonies"
    # model = "2025_03_03_23_34_16.794792_epoch_899_0.4_2"

    # # Find subdirectories that contain a "model_outputs" folder.
    # datasets = [d for d in Path(base_input).iterdir() if d.is_dir() and (d / "model_outputs").exists() and ("day1" in d.name or "day4" in d.name)]
    # if not datasets:
    #     logging.error(f"No dataset subfolders found in {base_input} with a 'model_outputs' folder.")
    #     return

    # # Group datasets by plate id. Assume plate id is the first token (separated by underscore).
    # plates = {}
    # for d in datasets:
    #     plate_id = d.name.split('_')[0]
    #     plates.setdefault(plate_id, []).append(d)
    
    # # Process each plate sequentially.
    # for plate_id, plate_datasets in plates.items():
    #     logging.info(f"Processing plate {plate_id} with {len(plate_datasets)} timepoints.")
    #     # Sort datasets by time using extract_time_from_folder on the folder name.
    #     sorted_datasets = sorted(plate_datasets, key=lambda
    #     x: extract_time_from_folder(x.name))
    #     process_plate_datasets(sorted_datasets, model)

    # all_colonies_csv = plotting.make_all_colonies_csv(base_input, model)

    # plotting.optimize_density_threshold(ground_truth_csv, all_colonies_csv, days_list = [3])

    # plotting.plot_plate_colony_percentages(all_colonies_csv)

PROCESS_EXISTING = True

if __name__ == "__main__":
    main()
