import os
import json
import numpy as np
import pandas as pd
from tifffile import imread, imsave
import shutil
import matplotlib.pyplot as plt
import re
from concurrent.futures import ProcessPoolExecutor
import cv2
from natsort import natsorted
from scipy import ndimage
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib import colors
import matplotlib.colors as colors
from datetime import datetime
from itertools import product, combinations
import glob
from plotting import create_heatmaps
from config_utils import load_config, parse_filename, extract_time_from_folder as cfg_extract_time_from_folder, get_image_time
from pathlib import Path


def find_source_image(mask_filepath: str | Path) -> Path | None:
    """Return the original image used to create *mask_filepath* if known."""
    mask_path = Path(mask_filepath)
    mapping_file = mask_path.parent / "source_images.json"
    if mapping_file.exists():
        try:
            with open(mapping_file, "r") as f:
                mapping = json.load(f)
            key = mask_path.stem.replace("_cp_masks", "")
            paths = mapping.get(key)
            if paths:
                return Path(paths[0])
        except Exception:
            pass
    # Fallback to previous heuristic
    img_dir = mask_path.parents[2]
    img_name = mask_path.name.replace("_cp_masks.png", ".tif")
    return img_dir / img_name

def extract_metrics(mask_filepath, area, total_well_area, filter_min_size=None):
    cfg = load_config()
    if filter_min_size:
        #Filter masks by size (get rid of weird artifacts)
        unfiltered_dir = os.path.join(os.path.dirname(mask_filepath), "unfiltered_masks")
        os.makedirs(unfiltered_dir, exist_ok=True)

        unfiltered_mask_path = os.path.join(unfiltered_dir, os.path.basename(mask_filepath))

        if not os.path.exists(unfiltered_mask_path):
            shutil.move(mask_filepath, unfiltered_mask_path)
            if ".png" in unfiltered_mask_path:
                unfiltered_masks = cv2.imread(str(unfiltered_mask_path), cv2.IMREAD_UNCHANGED)
            else:
                unfiltered_masks = imread(unfiltered_mask_path)

            labels = np.unique(unfiltered_masks)
            labels = labels[labels != 0]
            areas = ndimage.sum(np.ones_like(unfiltered_masks), unfiltered_masks, index=labels[labels != 0])
            large_mask_indices = areas >= filter_min_size
            large_labels = labels[large_mask_indices]
            masks = np.zeros_like(unfiltered_masks)
            for label in large_labels:
                masks[unfiltered_masks == label] = label

            imsave(mask_filepath, masks)
    else:
        if ".png" in mask_filepath:
            masks = cv2.imread(str(mask_filepath), cv2.IMREAD_UNCHANGED)
        else:
            masks = imread(mask_filepath)

    well, position, channel, z, _ = parse_filename(os.path.basename(mask_filepath), cfg)
    if well:
        m = re.match(r"([A-Za-z]+)(\d+)", well)
        row = m.group(1) if m else None
        column = m.group(2) if m else None
    else:
        row = column = None
    pos = position

    # Count masks
    labels = np.unique(masks)
    mask_count = len(labels[labels != 0])

    results = {
        'file_path' : mask_filepath,
        'file': os.path.basename(mask_filepath),
        'p1': os.path.basename(os.path.dirname(mask_filepath)),
        'p2': os.path.basename(os.path.dirname(os.path.dirname(mask_filepath))),
        'p3': os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(mask_filepath)))),
        'predicted_count': mask_count,
        'cell_density': mask_count / area,
        'total_cells_in_well': mask_count / area * total_well_area,
        'predicted_confluency': 1 - (np.sum(masks == 0) / masks.size),
        'Well' : well,
        'Row' : row,
        "Column" : column,
        'Position' : pos
    }
    return results

def extract_well_info(file_path):
    cfg = load_config()
    well, position, _, _, _ = parse_filename(os.path.basename(file_path), cfg)
    if well and position:
        m = re.match(r"([A-Za-z]+)(\d+)", well)
        row = m.group(1) if m else None
        column = m.group(2) if m else None
        return well, row, column, position
    return None, None, None, None

def natural_sort_wells(well):
    match = re.match(r"([A-Za-z]+)([0-9]+)", well)
    if match:
        return (match.group(1), int(match.group(2)))
    return (well,)

def row_to_index(row_letter):
    return ord(row_letter.upper()) - ord('A')

def extract_column_number(column_label):
    return int(''.join(filter(str.isdigit, column_label)))

def extract_time_from_folder(folder_name):
    return cfg_extract_time_from_folder(folder_name)

def find_previous_timepoint_csv(current_csv_file):
    """
    Given the path to the current CSV, attempt to:
      1. Identify the parent folder that has the date/time in its name.
      2. Extract the 'prefix' before the first underscore.
      3. Find all sibling subfolders in the main experiment directory
         that have the same prefix and parse their times.
      4. Pick the folder with the largest date/time that is strictly
         earlier than the current folder’s date/time.
      5. Return the path to that folder's corresponding CSV (if found).
    
    Returns:
      (previous_csv_file, previous_folder_time, current_folder_time)
      or (None, None, None) if not found.
    """
    # 1) Identify the parent folder that includes the date/time in its name.
    #    e.g. ...\PTEN_Growth_Assay_..._20241227_185901\model_outputs\...
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_csv_file)))          # ...\model_outputs
    current_folder_name = os.path.basename(parent_dir)

    # 2) Extract the prefix before the first underscore (e.g. 'PTEN')
    split_folder = current_folder_name.split('_')
    if not split_folder:
        return None, None, None
    experiment_prefix = split_folder[0]  # e.g. PTEN

    # 3) The 'main experiment directory' is two levels above parent_parent_dir
    main_experiment_dir = os.path.dirname(parent_dir)
    current_time = extract_time_from_folder(current_folder_name)

    # Gather all subfolders that start with the same prefix:
    # e.g. PTEN_Growth_Assay_...
    candidate_folders = []
    for entry in os.scandir(main_experiment_dir):
        if entry.is_dir() and entry.name.startswith(experiment_prefix):
            try:
                folder_time = extract_time_from_folder(entry.name)
                candidate_folders.append((entry.path, folder_time))
            except Exception:
                # If parsing fails, skip
                pass

    # 4) Among candidate folders, pick the one with the largest date/time
    #    that is still strictly earlier than 'current_time'
    candidate_folders = sorted(candidate_folders, key=lambda x: x[1])  # sort by time
    previous_folder = None
    previous_time = None
    for folder_path, folder_time in candidate_folders:
        if folder_time < current_time:
            previous_folder = folder_path
            previous_time = folder_time
        else:
            # Once we hit a time >= current_time, stop
            break

    if previous_folder is None:
        return None, None, None  # no earlier timepoint found

    # 5) Attempt to locate the CSV within the 'model_outputs' subfolder
    #    (adjust wildcard as needed to find your actual results CSV)
    model_outputs_path = os.path.join(previous_folder, 'model_outputs')
    if not os.path.exists(model_outputs_path):
        return None, None, None

    # Example: searching recursively for "*_results.csv" in model_outputs
    # You may need to refine how you pick the correct CSV if multiple exist
    results_csvs = glob.glob(os.path.join(model_outputs_path, '**', '*_results.csv'), recursive=True)
    if not results_csvs:
        return None, None, None

    # In case there's more than one, pick any or pick the first natsorted
    results_csvs = natsorted(results_csvs)
    previous_csv_file = results_csvs[0]

    return previous_csv_file, previous_time, current_time

###############################################################################
# Custom normalization that reserves a slot for zero.
###############################################################################
class LogNormZeroReserved(colors.Normalize):
    def __init__(self, vmin, vmax, clip=False):
        """
        For cell_density:
          - 0 maps to 0 (drawn as black)
          - Positive values are mapped (in log10 space) from vmin to vmax
            into the range [1/N, 1] (with N=257).
        """
        new_vmin = vmin if vmin > 0 else 1e-6
        new_vmax = vmax if vmax > new_vmin else new_vmin * 10
        self.N = 257  # total number of discrete color slots
        super().__init__(new_vmin, new_vmax, clip)

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        value = np.asarray(value, dtype=np.float64)
        normed = np.empty_like(value)
        
        # For exactly zero, assign normalized 0.
        mask0 = (value == 0)
        normed[mask0] = 0.0
        
        # For positive values, apply log10 scaling.
        maskpos = (value > 0)
        if np.any(maskpos):
            try:
                log_vmin = np.log10(self.vmin)
                log_vmax = np.log10(self.vmax)

                denom = log_vmax - log_vmin
                if denom == 0:
                    denom = 1.0
                normed[maskpos] = (1/self.N) + (1 - 1/self.N) * (
                    (np.log10(value[maskpos]) - log_vmin) / denom
                )
            except Exception as e:
                print("[DEBUG] Exception in __call__ log scaling:", e)
                raise e
        return normed

    def inverse(self, normed):
        normed = np.asarray(normed, dtype=np.float64)
        inv = np.empty_like(normed)
        # For normalized values at or below 1/N, return 0.
        mask0 = (normed <= (1/self.N))
        inv[mask0] = 0.0
        
        # For positive normalized values, invert the mapping.
        maskpos = (normed > (1/self.N))
        try:
            log_vmin = np.log10(self.vmin)
            log_vmax = np.log10(self.vmax)
            denom = log_vmax - log_vmin
            if denom == 0:
                denom = 1.0
            inv[maskpos] = 10 ** (
                ((normed[maskpos] - 1/self.N) / (1 - 1/self.N)) * denom + log_vmin
            )
        except Exception as e:
            print("[DEBUG] Exception in inverse:", e)
            raise e
        return inv

def process_mask_files(masks_dir, cm_per_pixel, total_well_area, force_save=False, save_outlines=False, filter_min_size=None):
    csv_filepath = os.path.join(masks_dir, f"{os.path.basename(masks_dir)}_results.csv")
    if (not os.path.exists(csv_filepath)) or force_save:
        mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if '_cp_masks' in f]
        if mask_files:
            if ".png" in mask_files[0]:
                masks = cv2.imread(str(mask_files[0]), cv2.IMREAD_UNCHANGED)
            else:
                masks = imread(mask_files[0])
            area = masks.size * cm_per_pixel * cm_per_pixel

            # results = [extract_metrics(file, area, total_well_area) for file in mask_files]

            with ProcessPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(extract_metrics, mask_files, [area]*len(mask_files), [total_well_area]*len(mask_files), [filter_min_size]*len(mask_files)))

            if save_outlines:
                outline_cells_directory(os.path.dirname(os.path.dirname(masks_dir)), masks_dir, None)

            model_results = pd.DataFrame(results)
            if model_results['file'].any():
                model_results.to_csv(csv_filepath, index=False)
                print(f"Saved {csv_filepath}")

        create_heatmaps(csv_filepath, "cell_density")

    return csv_filepath

def outline_cells_directory(img_directory, live_directory, dead_directory=None, channel=1):
    if not live_directory:
        print("no live cells directory")
        return
    tif_files = [f for f in os.listdir(img_directory) if f.lower().endswith('.tif')]
    pairs = [(f, f.lower().replace('.tif', '_cp_masks.png')) for f in tif_files if '_cp_masks' not in f]
    for img_file, mask_file in pairs:
        img_path = os.path.join(img_directory, img_file)
        live_mask_path = os.path.join(live_directory, mask_file)
        
        if dead_directory is not None:
            dead_mask_path = os.path.join(dead_directory, mask_file)
        else:
            dead_mask_path = None

        color_and_outline_cells_per_channel(img_path, live_mask_path, dead_mask_path, channel=channel)

def color_and_outline_cells_per_channel(output_img_path, img, green_masks, red_masks=None, alpha=0.9, beta=0.1, gamma=0, thickness=3, quality=80, channel=2):
    try:
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

        # Define the channel indices you want to process
        channel_indices = [channel]  # Modify this list if you want to process more channels
        
        for channel_idx in channel_indices:
            # Check if the output file already exists
            if os.path.exists(output_img_path):
                continue  # Skip to the next channel if the output exists
            
            # Handle image dimensions
            if img.ndim == 2:
                img = np.stack((img, img, img), axis=0)
            if img.ndim == 3 and img.shape[0] == 5:
                img = img[[0, 1, 3], :, :]
            elif img.ndim == 3 and img.shape[0] != 3:
                raise ValueError("Image has 3 dimensions but does not have 3 channels")
    
            # Extract and normalize the specific channel
            channel_img = img[channel_idx, :, :]
            low = np.percentile(channel_img, 1)
            high = np.percentile(channel_img, 99.9)
            scale = 255
            low_clip = 0
            channel_img_normalized = ((channel_img - low) / (high - low) * scale).clip(low_clip, scale)
            
            # Convert to BGR format
            channel_img_colored = cv2.cvtColor(channel_img_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            # Create color overlays
            green_overlay = np.zeros_like(channel_img_colored)
            red_overlay = np.zeros_like(channel_img_colored)
            yellow_overlay = np.zeros_like(channel_img_colored)
            
            # Apply green mask
            green_overlay[green_masks > 0] = [0, 255, 0]
            
            # Apply red mask if provided
            if red_masks is not None:
                red_overlay[red_masks > 0] = [0, 0, 255]
            
            # Blend the overlays with the original image
            shaded_img = cv2.addWeighted(channel_img_colored, alpha, green_overlay, beta, 0)
            shaded_img = cv2.addWeighted(shaded_img, alpha, red_overlay, beta, 0)
            shaded_img = cv2.addWeighted(shaded_img, alpha, yellow_overlay, beta, gamma)
            
            # Define the structuring element for dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
            
            # Dilate green masks to get outlines
            dilated_green = cv2.dilate(green_masks, kernel)
            outlines_green = dilated_green - green_masks
            
            # Apply green outlines
            shaded_img[outlines_green > 0] = [0, 255, 0]
            
            # Apply red outlines if red_masks_path is provided
            if red_masks is not None:
                dilated_red = cv2.dilate(red_masks, kernel)
                outlines_red = dilated_red - red_masks
                shaded_img[outlines_red > 0] = [0, 0, 255]
            
            # Save the processed image
            cv2.imwrite(str(output_img_path), shaded_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    except Exception as e:
        print(f"Error processing {output_img_path}: {e}")

def outline_cells_per_channel(img_path, green_masks_path, red_masks_path, thickness=3, quality=60, model_name=""):
    try:
        img = imread(img_path).astype(np.float32)
            # Check the number of dimensions in the image
        if img.ndim == 2:
            # If the image is 2D (grayscale), add three channels
            img = np.stack((img, img, img), axis=0)
        if img.ndim == 3 and img.shape[0] == 5:
            img = img[[0, 1, 3], :, :]
        elif img.ndim == 3 and img.shape[0] != 3:
            # If the image is 3D but does not have 3 channels, we should handle it as needed
            # This condition is specific to images with non-standard shapes, modify if necessary
            raise ValueError("Image has 3 dimensions but does not have 3 channels")
        
        red_masks = imread(red_masks_path)
        green_masks = imread(green_masks_path)
        
        outlines_dir = os.path.join(os.path.dirname(green_masks_path), "outlines")
        os.makedirs(outlines_dir, exist_ok=True)

        for channel_idx in [1]:
        # for channel_idx in range(img.shape[0]):
            channel_img = img[channel_idx, :, :]
            low = np.percentile(channel_img, 1)
            high = np.percentile(channel_img, 99)
            scale = 255
            low_clip = 0
            channel_img_normalized = ((channel_img - low) / (high - low) * scale).clip(low_clip, scale)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))

            dilated_red = cv2.dilate(red_masks, kernel)
            outlines_red = dilated_red - red_masks

            dilated_green = cv2.dilate(green_masks, kernel)
            outlines_green = dilated_green - green_masks

            channel_img_outlined = cv2.cvtColor(channel_img_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Apply green outlines
            channel_img_outlined[outlines_green > 0] = [0, 255, 0]

            # Apply red outlines
            channel_img_outlined[outlines_red > 0] = [0, 0, 255]

            # Apply yellow outlines
            channel_img_outlined[np.logical_and(outlines_red > 0, outlines_green > 0)] = [0, 255, 255]

            outlined_img_path = os.path.join(outlines_dir, f"{os.path.basename(img_path).replace('.tif', '')}_channel_{channel_idx}.jpg")
            cv2.imwrite(str(outlined_img_path), channel_img_outlined, [cv2.IMWRITE_JPEG_QUALITY, quality])
    except Exception as e:
        print(e)


# Constants and configurations
PLATE_TYPE = "12W"
MAGNIFICATION = "20x"
CYTATION = True

PLATE_AREAS = {"6W": 9.6, "12W": 3.8, "24W": 2, "48W": 1.1, "96W": 0.32}
CM_PER_MICRON = 1 / 10000
if CYTATION:
    MICRONS_PER_PIXEL = 1389 / 1992 if MAGNIFICATION == "10x" else 694 / 1992
    IMAGE_AREA_CM = 1992 * 1992 * MICRONS_PER_PIXEL**2 * CM_PER_MICRON**2
else:
    if MAGNIFICATION=="10x":
        MICRONS_PER_PIXEL = 0.61922571983322461  # Taken from image metadata EVOS 10x
        IMAGE_AREA_CM = 0.0120619953 # Taken from image metadata EVOS 10x
    else:
        MICRONS_PER_PIXEL = 1.5188172690164046  # Taken from image metadata EVOS 4x
        IMAGE_AREA_CM = 0.0725658405  # Taken from image metadata EVOS 4x
CM_PER_PIXEL = CM_PER_MICRON*MICRONS_PER_PIXEL

# IMAGE_AREA_CM = 1992*1992 * MICRONS_PER_PIXEL**2 * CM_PER_MICRON**2

def process_subfolder(subfolder_name, base_dir, model_output_suffix):
    subfolder_path = os.path.join(base_dir, subfolder_name)
    # Check if the item is a directory
    if os.path.isdir(subfolder_path):
        param1 = subfolder_path
        param2 = os.path.join(subfolder_path, model_output_suffix)
        param3 = None
        # Call the function with the specified parameters
        outline_cells_directory(param1, param2, param3)

def extract_intensity_metrics(cellprob_filepath, area, total_well_area):
    """
    Reads the cell probability image and computes total intensity metrics.
    The total intensity is forced to be at least 0.
    """
    try:
        # Read the cell probability image (assumed to be numeric)
        cellprob_img = imread(cellprob_filepath)
    except Exception as e:
        print(f"Error reading {cellprob_filepath}: {e}")
        return None

    total_intensity = max(np.sum(cellprob_img), 0)
    max_intensity = max(np.max(cellprob_img), 0)

    # Extract well info (assuming the filename contains the well info in the same format)
    well, row, column, pos = extract_well_info(cellprob_filepath)

    results = {
        'file_path': cellprob_filepath,
        'file': os.path.basename(cellprob_filepath),
        'p1': os.path.basename(os.path.dirname(cellprob_filepath)),
        'p2': os.path.basename(os.path.dirname(os.path.dirname(cellprob_filepath))),
        'p3': os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(cellprob_filepath)))),
        'total_intensity': total_intensity,
        'max_intensity': max_intensity,
        'Well': well,
        'Row': row,
        'Column': column,
        'Position': pos
    }
    return results

def process_cellprob_files(masks_dir, cm_per_pixel, total_well_area, force_save=False):
    """
    Processes cell probability files by:
      - Replacing _cp_masks.tif with _cellProb.tif to obtain the intensity images.
      - Computing intensity metrics (total intensity, intensity density, etc.).
      - Scaling the dataset so that the minimum total_intensity is 0.
      - Merging (appending) these new intensity columns with the existing CSV (if available).
      - Creating a heatmap (using intensity_density) and a cumulative intensity plot.
    """
    # CSV to store intensity metrics.
    intensity_csv_filepath = os.path.join(masks_dir, f"{os.path.basename(masks_dir)}_results_intensity.csv")
    
    # Find the _cp_masks.tif files and derive the corresponding _cellProb.tif file paths.
    mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if '_cp_masks.tif' in f]
    if not mask_files:
        print("No mask files found.")
        return None

    cellprob_files = [f.replace('_cp_masks.tif', '_cellProb.tif') for f in mask_files]

    # Determine the image area (in cm²) from the first cellProb image.
    sample_img = imread(cellprob_files[0])
    area = sample_img.size * (cm_per_pixel ** 2)

    # Process all cellProb files in parallel.
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            extract_intensity_metrics,
            cellprob_files,
            [area] * len(cellprob_files),
            [total_well_area] * len(cellprob_files)
        ))
    results = [r for r in results if r is not None]
    intensity_df = pd.DataFrame(results)

    # Scale the entire intensity dataset so that the minimum total_intensity is 0.
    min_intensity = intensity_df['total_intensity'].min()
    intensity_df['total_intensity'] = intensity_df['total_intensity'] - min_intensity
    intensity_df['intensity_density'] = intensity_df['total_intensity'] / area
    intensity_df['total_intensity_in_well'] = intensity_df['intensity_density'] * total_well_area

    # Save the intensity metrics to CSV.
    intensity_df.to_csv(intensity_csv_filepath, index=False)
    print(f"Saved intensity results CSV to {intensity_csv_filepath}")

    # Check if the "old" CSV exists (from cell density metrics).
    old_csv_filepath = os.path.join(masks_dir, f"{os.path.basename(masks_dir)}_results.csv")
    if os.path.exists(old_csv_filepath):
        try:
            old_df = pd.read_csv(old_csv_filepath)
            # Create a common key 'base' by stripping the specific suffixes from the file names.
            old_df['base'] = old_df['file'].str.replace('_cp_masks.tif', '', regex=False)
            intensity_df['base'] = intensity_df['file'].str.replace('_cellProb.tif', '', regex=False)
            # Merge on the 'base' column.
            merged_df = pd.merge(
                old_df,
                intensity_df[['base', 'total_intensity', 'intensity_density', 'total_intensity_in_well']],
                on='base',
                how='left'
            )
            # Optionally drop the helper "base" column.
            merged_df = merged_df.drop(columns=['base'])
            # Save the merged CSV (you can choose to overwrite or create a new file).
            merged_csv_filepath = os.path.join(masks_dir, f"{os.path.basename(masks_dir)}_results_merged.csv")
            merged_df.to_csv(merged_csv_filepath, index=False)
            print(f"Saved merged CSV with appended intensity columns to {merged_csv_filepath}")
        except Exception as e:
            print(f"Error merging intensity data with old CSV: {e}")
    else:
        print("Old CSV not found; intensity CSV remains separate.")

    # Create a heatmap based on intensity_density using the intensity CSV.
    try:
        create_heatmaps(intensity_csv_filepath, value_col="intensity_density")
    except Exception as e:
        print(f"Failed to create intensity heatmap: {e}")

    # Create an additional plot: cumulative total intensity vs. number of images.
    try:
        plot_cumulative_intensity(intensity_csv_filepath)
    except Exception as e:
        print(f"Failed to create cumulative intensity plot: {e}")
        
    return intensity_csv_filepath


def plot_cumulative_intensity(csv_file):
    """
    Reads the CSV file with intensity metrics, sorts the images by total intensity (highest first),
    computes the cumulative sum of total intensity, and plots cumulative total intensity versus
    the number of images.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return

    # Sort by total_intensity in descending order
    df_sorted = df.sort_values('total_intensity', ascending=False).reset_index(drop=True)
    df_sorted['cumulative_total_intensity'] = df_sorted['total_intensity'].cumsum()

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.arange(1, len(df_sorted) + 1),
        df_sorted['cumulative_total_intensity'],
        marker='o',
        linestyle='-'
    )
    plt.xlabel("Number of Images (sorted by highest total intensity)")
    plt.ylabel("Cumulative Total Intensity")
    plt.title("Cumulative Total Intensity vs. Number of Images")
    plt.grid(True)

    output_plot = os.path.join(os.path.dirname(csv_file), "cumulative_total_intensity_plot.png")
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"Saved cumulative intensity plot to {output_plot}")

def make_all_data_csv(input_folder, model_name=None):
    if not model_name:
        return None

    cfg = load_config()
    all_data = []

    for subdir in glob.glob(os.path.join(input_folder, '*/')):
        try:
            folder_time = extract_time_from_folder(os.path.basename(subdir.rstrip('/')), cfg)

            for root, _, files in os.walk(subdir):
                for file in files:
                    if file.endswith('.csv') and (model_name in file):
                        data = pd.read_csv(os.path.join(root, file))
                        times = []
                        for mask_fp in data['file_path']:
                            src_img = find_source_image(mask_fp)
                            if cfg.get('time_source', 'folder') == 'date_created':
                                t = get_image_time(src_img, cfg) if src_img else None
                                if t is None:
                                    t = folder_time
                            else:
                                t = folder_time
                                if t is None and src_img:
                                    t = get_image_time(src_img, cfg)
                            times.append(t)
                        data['Time'] = times
                        all_data.append(data)
        except Exception:
            print(f"skipping{subdir}")

    output_folder = os.path.join(input_folder, f"{model_name}")
    os.makedirs(output_folder, exist_ok=True)

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data['Time'] = pd.to_datetime(combined_data['Time'])
    combined_data.sort_values(['Well', 'Time'], inplace=True)
    combined_data['Relative Time (hrs)'] = (
        combined_data.groupby('Well')['Time']
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    combined_data['Relative Time (hrs)'] = combined_data['Relative Time (hrs)'].fillna(0).astype(float)
    combined_data.drop(columns='Time', inplace=True)

    # Extract 'Plate', 'Row', 'Column', and 'PlateWell' information
    combined_data['Plate'] = combined_data['p3'].str.split('_').str[0]
    combined_data[['Row', 'Column']] = combined_data['Well'].str.extract(r'([A-Za-z]+)(\d+)')
    combined_data["Plate"] = combined_data["Plate"].str.lower()
    combined_data["PlateWell"] = (
            combined_data["Plate"]     
            + combined_data["Well"].astype(str)
        )
    combined_data['Column'] = combined_data['Column'].astype(int)

    def get_neighbor_wells(row, column):
        rows = [chr(ord(row) + i) for i in range(-1, 2) if 'A' <= chr(ord(row) + i) <= 'H']
        columns = [str(column + i) for i in range(-1, 2) if 1 <= column + i <= 12]
        return [r + c for r, c in product(rows, columns) if not (r == row and c == str(column))]

    def calculate_neighbors(df):
        unique_wells = set(df['PlateWell'])
        df['Neighbors'] = df.apply(lambda x: sum(1 for neighbor in get_neighbor_wells(x['Row'], x['Column']) if (x['Plate'] + neighbor) in unique_wells), axis=1)
        return df

    combined_data = calculate_neighbors(combined_data)

    group_map_path = os.path.join(input_folder, f'group_map.csv')
    if os.path.exists(group_map_path):
        group_map_df = pd.read_csv(group_map_path).astype(str)
        group_map_df = group_map_df.replace({"NA": np.nan, "nan": np.nan}).dropna()
        group_map_df.columns = (list(group_map_df.columns[:2]) + ['Group-' + col for col in group_map_df.columns[2:]])
        group_map_df["Plate"] = group_map_df["Plate"].str.lower()
        group_map_df["PlateWell"] = (
            group_map_df["Plate"]        
            + group_map_df["Well"].astype(str)        
        )
        combined_data = combined_data.merge(group_map_df, on='PlateWell', how='left', suffixes=('', '_drop'))   

    # Drop the duplicate columns that were appended with '_drop'
    combined_data = combined_data.loc[:, ~combined_data.columns.str.endswith('_drop')]
    combined_data_path = os.path.join(output_folder, f'{model_name}_all_data.csv')
    combined_data.to_csv(combined_data_path, index=False)
    return combined_data_path

if __name__ == "__main__":
    outline_cells_directory(r"E:\test\test_day5_20250519_153429", r"E:\test\test_day5_20250519_153429\model_outputs\2024_03_29_02_03_10.875324_epoch_960_0.4_0", channel=1)
    # outline_cells_directory(image_dir, masks_dir, channel=2)
