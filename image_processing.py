import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import tifffile
import logging
import cv2
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global constants
OVERLAP = 0

PROCESS_EXISTING = True

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

def parse_well_coordinate(well_str):
    match = re.match(r"^([A-Za-z]+)(\d+)$", well_str)
    if not match:
        raise ValueError(f"Invalid well coordinate format: {well_str}")
    row_letters = match.group(1).upper()
    col_number = int(match.group(2))
    row_index = 0
    for char in row_letters:
        row_index = row_index * 26 + (ord(char) - ord('A') + 1)
    return row_index - 1, col_number - 1

def create_placeholder_image(well_height, well_width, channels, border_thickness=8, dtype=np.uint8, intensity=255):
    if channels:
        placeholder = np.zeros((well_height, well_width, channels), dtype=dtype)
        placeholder[:border_thickness, :] = intensity
        placeholder[-border_thickness:, :] = intensity
        placeholder[:, :border_thickness] = intensity
        placeholder[:, -border_thickness:] = intensity
    else:
        placeholder = np.zeros((well_height, well_width), dtype=dtype)
        placeholder[:border_thickness, :] = intensity
        placeholder[-border_thickness:, :] = intensity
        placeholder[:, :border_thickness] = intensity
        placeholder[:, -border_thickness:] = intensity
    return placeholder

def read_tile_downscaled(file, scale_factor=0.5):
    try:
        tile = tifffile.imread(str(file))
        new_width = int(tile.shape[-1] * scale_factor)
        new_height = int(tile.shape[-2] * scale_factor)
        if tile.ndim == 2:
            tile = cv2.resize(tile, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        elif tile.ndim == 3:
            if tile.shape[0] in {1, 3} and tile.shape[1] > tile.shape[0]:
                tile = np.transpose(tile, (1, 2, 0))
            tile = cv2.resize(tile, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            raise ValueError("Unsupported image dimensions.")
        return tile
    except Exception as e:
        logging.error(f"Error reading and downscaling {file}: {e}")
        raise e

def read_masks_downscaled(file, scale_factor=0.5):
    try:
        mask = tifffile.imread(str(file))
        new_width = int(mask.shape[-1] * scale_factor)
        new_height = int(mask.shape[-2] * scale_factor)
        if mask.ndim == 2:
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        elif mask.ndim == 3:
            if mask.shape[0] in {1, 3} and mask.shape[1] > mask.shape[0]:
                mask = np.transpose(mask, (1, 2, 0))
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError("Unsupported mask image dimensions.")
        return mask
    except Exception as e:
        logging.error(f"Error reading and downscaling mask {file}: {e}")
        raise e

def read_brightfield_tile(file, scale_factor=0.5, channel_idx=1):
    try:
        img = tifffile.imread(str(file))
        channel_img = img[channel_idx, :, :]
        new_width = int(channel_img.shape[1] * scale_factor)
        new_height = int(channel_img.shape[0] * scale_factor)
        if scale_factor != 1:
            tile = cv2.resize(channel_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            tile = channel_img
        return (tile/255).astype(np.uint8)
    except Exception as e:
        logging.error(f"Error reading and downscaling brightfield tile {file}: {e}")
        raise e
# -------------------------------------------------------------------
# Merge functions for tiles and masks (unchanged)
def merge_well_brightfield(tile_files, overlap=OVERLAP):
    if len(tile_files) != 16:
        raise ValueError(f"Expected 16 brightfield tile images, but got {len(tile_files)}")
    
    brightfield_pattern = re.compile(
        r"^(?P<base>[A-Za-z0-9]+)_pos(?P<pos>\d+)_merged_(?P<version>\d{2})\.tif$", 
        re.IGNORECASE
    )
    def get_pos(file):
        match = brightfield_pattern.match(file.name)
        if match:
            return int(match.group('pos'))
        return 0

    sorted_files = sorted(tile_files, key=get_pos)
    tiles = [read_brightfield_tile(file) for file in sorted_files]
    
    first_shape = tiles[0].shape
    tile_height, tile_width = first_shape
    full_height = tile_height + 3 * (tile_height - overlap)
    full_width = tile_width + 3 * (tile_width - overlap)
    full_image = np.zeros((full_height, full_width), dtype=tiles[0].dtype)
    
    for file, tile in zip(sorted_files, tiles):
        pos = get_pos(file)
        if pos < 1 or pos > 16:
            raise ValueError(f"Tile position {pos} is out of the expected range (1-16).")
        row = (pos - 1) // 4
        col = (pos - 1) % 4
        start_row = row * (tile_height - overlap)
        start_col = col * (tile_width - overlap)
        full_image[start_row:start_row + tile_height, start_col:start_col + tile_width] = tile

    full_color = cv2.cvtColor(full_image, cv2.COLOR_GRAY2BGR)
    border_thickness = 8
    full_color[:border_thickness, :, :] = 255
    full_color[-border_thickness:, :, :] = 255
    full_color[:, :border_thickness, :] = 255
    full_color[:, -border_thickness:, :] = 255

    return full_color

def merge_well_tiles(tile_files, overlap=OVERLAP, scale_factor=0.5):
    if len(tile_files) != 16:
        raise ValueError(f"Expected 16 tile images, but got {len(tile_files)}")

    def get_pos(file):
        match = FILE_PATTERN.match(file.name)
        if match:
            return int(match.group('pos'))
        return 0

    sorted_files = sorted(tile_files, key=get_pos)
    tiles = [read_tile_downscaled(file, scale_factor) for file in sorted_files]

    first_shape = tiles[0].shape
    for tile in tiles:
        if tile.shape != first_shape:
            raise ValueError("All tile images must have the same dimensions and number of channels.")

    if len(first_shape) == 2:
        tile_height, tile_width = first_shape
        full_height = tile_height + 3 * (tile_height - overlap)
        full_width = tile_width + 3 * (tile_width - overlap)
        full_image = np.zeros((full_height, full_width), dtype=tiles[0].dtype)
    elif len(first_shape) == 3:
        tile_height, tile_width, channels = first_shape
        full_height = tile_height + 3 * (tile_height - overlap)
        full_width = tile_width + 3 * (tile_width - overlap)
        full_image = np.zeros((full_height, full_width, channels), dtype=tiles[0].dtype)
    else:
        raise ValueError("Unsupported tile image shape; expected 2D or 3D images.")

    for file, tile in zip(sorted_files, tiles):
        pos = get_pos(file)
        if pos < 1 or pos > 16:
            raise ValueError(f"Tile position {pos} is out of the expected range (1-16).")
        row = (pos - 1) // 4
        col = (pos - 1) % 4
        start_row = row * (tile_height - overlap)
        start_col = col * (tile_width - overlap)
        full_image[start_row:start_row + tile_height, start_col:start_col + tile_width] = tile

    if full_image.ndim == 2:
        full_image[:8, :] = 255
        full_image[-8:, :] = 255
        full_image[:, :8] = 255
        full_image[:, -8:] = 255
    else:
        full_image[:8, :, :] = 255
        full_image[-8:, :, :] = 255
        full_image[:, :8, :] = 255
        full_image[:, -8:, :] = 255

    return full_image

def merge_well_masks(mask_files, overlap=OVERLAP):
    if len(mask_files) != 16:
        raise ValueError(f"Expected 16 mask images, but got {len(mask_files)}")

    def get_pos(file):
        match = MASK_FILE_PATTERN.match(file.name)
        if match:
            return int(match.group('pos'))
        return 0

    sorted_files = sorted(mask_files, key=get_pos)
    tiles = [read_masks_downscaled(file) for file in sorted_files]

    first_shape = tiles[0].shape
    for tile in tiles:
        if tile.shape != first_shape:
            raise ValueError("All mask images must have the same dimensions.")

    tile_height, tile_width = first_shape
    full_height = tile_height + 3 * (tile_height - overlap)
    full_width = tile_width + 3 * (tile_width - overlap)
    
    full_mask = np.zeros((full_height, full_width), dtype=np.int32)
    current_offset = 0
    for file, tile in zip(sorted_files, tiles):
        tile = tile.astype(np.int32)
        tile_unique = tile.copy()
        nonzero = tile_unique > 0
        if np.any(nonzero):
            tile_unique[nonzero] += current_offset
            current_offset += tile_unique.max()
            
        pos = get_pos(file)
        if pos < 1 or pos > 16:
            raise ValueError(f"Tile position {pos} is out of the expected range (1-16).")
        row = (pos - 1) // 4
        col = (pos - 1) % 4
        start_row = row * (tile_height - overlap)
        start_col = col * (tile_width - overlap)
        full_mask[start_row:start_row + tile_height, start_col:start_col + tile_width] = tile_unique

    return full_mask

def group_tiles(files):
    groups = defaultdict(list)
    for file in files:
        match = FILE_PATTERN.match(file.name)
        if match:
            group_key = f"{match.group('base')}_merged_{match.group('version')}_{match.group('type')}"
            groups[group_key].append(file)
    return groups

# -------------------------------------------------------------------
# Merge functions for plate mosaics (unchanged)
def merge_plate_cellprob(input_folder, output_file, type_filter='cellProb'):
    logging.info(f"Starting full plate merge for type '{type_filter}' from folder: {input_folder}")
    
    files = [
        f for f in Path(input_folder).iterdir()
        if f.is_file() and f.name.lower().endswith(f"_{type_filter.lower()}.tif")
    ]
    if not files:
        logging.error(f"No files ending with _{type_filter}.tif found in {input_folder}")
        return

    groups = group_tiles(files)
    logging.info(f"Found {len(groups)} well groups in {input_folder}")

    well_images = {}
    for group_key, tile_files in groups.items():
        well_coord = group_key.split('_')[0]
        if len(tile_files) != 16:
            logging.warning(f"Group {group_key} for well {well_coord} does not have 16 tiles. Creating placeholder.")
            try:
                sample_tile = read_tile_downscaled(tile_files[0])
                tile_shape = sample_tile.shape
            except Exception as e:
                logging.error(f"Cannot create placeholder for well {well_coord}: {e}")
                continue
            if len(tile_shape) == 2:
                tile_height, tile_width = tile_shape
            else:
                tile_height, tile_width, _ = tile_shape
            full_height = tile_height + 3 * (tile_height - OVERLAP)
            full_width = tile_width + 3 * (tile_width - OVERLAP)
            if len(tile_shape) == 2:
                placeholder = create_placeholder_image(full_height, full_width, None, border_thickness=8, dtype=sample_tile.dtype)
            else:
                channels = tile_shape[2]
                placeholder = create_placeholder_image(full_height, full_width, channels, border_thickness=8, dtype=sample_tile.dtype)
            well_images[well_coord] = placeholder
            continue

        try:
            merged_well = merge_well_tiles(tile_files)
            well_images[well_coord] = merged_well
            logging.info(f"Merged well {well_coord} (group {group_key})")
        except Exception as e:
            logging.error(f"Failed to merge well {group_key}: {e}")

    if not well_images:
        logging.error("No well images were merged; aborting plate merge.")
        return

    layout = {}
    rows_set = set()
    cols_set = set()
    for well, img in well_images.items():
        try:
            row_idx, col_idx = parse_well_coordinate(well)
            layout[well] = (row_idx, col_idx)
            rows_set.add(row_idx)
            cols_set.add(col_idx)
        except Exception as e:
            logging.error(f"Skipping well {well} due to coordinate parsing error: {e}")

    if not layout:
        logging.error("No valid well coordinates were found; aborting plate merge.")
        return

    num_rows = max(rows_set) + 1
    num_cols = max(cols_set) + 1

    sample_img = next(iter(well_images.values()))
    well_height, well_width = sample_img.shape[:2]
    channels = sample_img.shape[2] if sample_img.ndim == 3 else None

    if channels:
        plate_img = np.zeros((num_rows * well_height, num_cols * well_width, channels), dtype=sample_img.dtype)
    else:
        plate_img = np.zeros((num_rows * well_height, num_cols * well_width), dtype=sample_img.dtype)

    for well, img in well_images.items():
        if well not in layout:
            continue
        r, c = layout[well]
        start_row = r * well_height
        start_col = c * well_width
        if channels:
            plate_img[start_row:start_row + well_height, start_col:start_col + well_width, :] = img
        else:
            plate_img[start_row:start_row + well_height, start_col:start_col + well_width] = img

    occupied_coords = { coord for coord in layout.values() }
    placeholder = create_placeholder_image(well_height, well_width, channels, border_thickness=8, dtype=sample_img.dtype)
    for r in range(num_rows):
        for c in range(num_cols):
            if (r, c) not in occupied_coords:
                start_row = r * well_height
                start_col = c * well_width
                if channels:
                    plate_img[start_row:start_row + well_height, start_col:start_col + well_width, :] = placeholder
                else:
                    plate_img[start_row:start_row + well_height, start_col:start_col + well_width] = placeholder

    if not output_file.exists() or PROCESS_EXISTING:
        try:
            tifffile.imwrite(str(output_file), plate_img)
            logging.info(f"Saved full plate mosaic to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save full plate mosaic {output_file}: {e}")
    else:
        logging.info(f"Full plate mosaic {output_file} already exists. Skipping writing.")

def merge_plate_colonies(colony_folder, output_file):
    colony_files = list(Path(colony_folder).glob("colony_*.tif"))
    if not colony_files:
        logging.error(f"No colony images found in {colony_folder}")
        return

    well_images = {}
    layout = {}
    rows_set = set()
    cols_set = set()
    for file in colony_files:
        parts = file.stem.split("_")
        if len(parts) < 2:
            logging.error(f"Unexpected file name format for colony image: {file.name}")
            continue
        well_coord = parts[1]
        try:
            row_idx, col_idx = parse_well_coordinate(well_coord)
        except Exception as e:
            logging.error(f"Error parsing well coordinate from {file.name}: {e}")
            continue
        layout[well_coord] = (row_idx, col_idx)
        rows_set.add(row_idx)
        cols_set.add(col_idx)
        try:
            img = tifffile.imread(str(file))
        except Exception as e:
            logging.error(f"Error reading colony image {file}: {e}")
            continue
        well_images[well_coord] = img

    if not well_images:
        logging.error("No valid colony images to merge.")
        return

    num_rows = max(rows_set) + 1
    num_cols = max(cols_set) + 1
    sample_img = next(iter(well_images.values()))
    well_height, well_width = sample_img.shape[:2]
    channels = sample_img.shape[2] if sample_img.ndim == 3 else None

    if channels:
        plate_img = np.zeros((num_rows * well_height, num_cols * well_width, channels), dtype=sample_img.dtype)
    else:
        plate_img = np.zeros((num_rows * well_height, num_cols * well_width), dtype=sample_img.dtype)

    for well, img in well_images.items():
        if well not in layout:
            continue
        r, c = layout[well]
        start_row = r * well_height
        start_col = c * well_width
        if channels:
            plate_img[start_row:start_row + well_height, start_col:start_col + well_width, :] = img
        else:
            plate_img[start_row:start_row + well_height, start_col:start_col + well_width] = img

    occupied_coords = { coord for coord in layout.values() }
    placeholder = create_placeholder_image(well_height, well_width, channels, border_thickness=8, dtype=sample_img.dtype)
    for r in range(num_rows):
        for c in range(num_cols):
            if (r, c) not in occupied_coords:
                start_row = r * well_height
                start_col = c * well_width
                if channels:
                    plate_img[start_row:start_row + well_height, start_col:start_col + well_width, :] = placeholder
                else:
                    plate_img[start_row:start_row + well_height, start_col:start_col + well_width] = placeholder

    if not output_file.exists() or PROCESS_EXISTING:
        try:
            tifffile.imwrite(str(output_file), plate_img)
            logging.info(f"Saved plate colony mosaic to {output_file}")
        except Exception as e:
            logging.error(f"Error saving plate colony mosaic to {output_file}: {e}")
    else:
        logging.info(f"Plate colony mosaic {output_file} already exists. Skipping writing.")

def merge_plate_colony_outlines(outlines_folder, output_file):
    outlined_files = list(Path(outlines_folder).glob("colony_*_outlines.tif"))
    if not outlined_files:
        logging.error(f"No outlined colony images found in {outlines_folder}")
        return

    well_images = {}
    layout = {}
    rows_set = set()
    cols_set = set()
    for file in outlined_files:
        parts = file.stem.split("_")
        if len(parts) < 3:
            logging.error(f"Unexpected file name format for outlined image: {file.name}")
            continue
        well_coord = parts[1]
        try:
            row_idx, col_idx = parse_well_coordinate(well_coord)
        except Exception as e:
            logging.error(f"Error parsing well coordinate from {file.name}: {e}")
            continue
        layout[well_coord] = (row_idx, col_idx)
        rows_set.add(row_idx)
        cols_set.add(col_idx)
        try:
            img = tifffile.imread(str(file))
        except Exception as e:
            logging.error(f"Error reading outlined image {file}: {e}")
            continue
        well_images[well_coord] = img

    if not well_images:
        logging.error("No valid outlined images to merge.")
        return

    num_rows = max(rows_set) + 1
    num_cols = max(cols_set) + 1
    sample_img = next(iter(well_images.values()))
    well_height, well_width = sample_img.shape[:2]
    channels = sample_img.shape[2] if sample_img.ndim == 3 else None

    if channels:
        plate_img = np.zeros((num_rows * well_height, num_cols * well_width, channels), dtype=sample_img.dtype)
    else:
        plate_img = np.zeros((num_rows * well_height, num_cols * well_width), dtype=sample_img.dtype)

    for well, img in well_images.items():
        if well not in layout:
            continue
        r, c = layout[well]
        start_row = r * well_height
        start_col = c * well_width
        if channels:
            plate_img[start_row:start_row + well_height, start_col:start_col + well_width, :] = img
        else:
            plate_img[start_row:start_row + well_height, start_col:start_col + well_width] = img

    occupied_coords = { coord for coord in layout.values() }
    placeholder = create_placeholder_image(well_height, well_width, channels, border_thickness=8, dtype=sample_img.dtype)
    for r in range(num_rows):
        for c in range(num_cols):
            if (r, c) not in occupied_coords:
                start_row = r * well_height
                start_col = c * well_width
                if channels:
                    plate_img[start_row:start_row + well_height, start_col:start_col + well_width, :] = placeholder
                else:
                    plate_img[start_row:start_row + well_height, start_col:start_col + well_width] = placeholder

    if not output_file.exists() or PROCESS_EXISTING:
        try:
            tifffile.imwrite(str(output_file), plate_img)
            logging.info(f"Saved plate colony outlines mosaic to {output_file}")
        except Exception as e:
            logging.error(f"Error saving plate colony outlines mosaic to {output_file}: {e}")
    else:
        logging.info(f"Plate colony outlines mosaic {output_file} already exists. Skipping writing.")

import logging
from pathlib import Path
import numpy as np
import tifffile
import pandas as pd

import logging
from pathlib import Path
import numpy as np
import tifffile
import pandas as pd

# Merge functions for plate mosaics (unchanged except for color conversion and border drawing)
def merge_plate_cellprob_with_outlines(input_folder, output_file, type_filter='cellProb', scale_factor=0.03125):
    logging.info(f"Starting full plate merge for type '{type_filter}' from folder: {input_folder}")

    csv_path = r"E:\MERGED_sspsygene_growthassay_colonies\ground_truth_with_simulated_prediction.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df[df["Plate"] == 54]
        df = df[df["Day"] == 8]
        # Assuming the CSV has columns 'well' and 'simulated_prediction'
        simulated_predictions = dict(zip(df['Well'], df['simulated_prediction']))
        logging.info("Loaded simulated predictions from CSV.")
    except Exception as e:
        logging.error(f"Failed to load CSV for simulated predictions: {e}")
        simulated_predictions = {}
    
    files = [
        f for f in Path(input_folder).iterdir()
        if f.is_file() and f.name.lower().endswith(f"_{type_filter.lower()}.tif")
    ]
    if not files:
        logging.error(f"No files ending with _{type_filter}.tif found in {input_folder}")
        return

    groups = group_tiles(files)
    logging.info(f"Found {len(groups)} well groups in {input_folder}")

    well_images = {}
    for group_key, tile_files in groups.items():
        well_coord = group_key.split('_')[0]
        if len(tile_files) != 16:
            logging.warning(f"Group {group_key} for well {well_coord} does not have 16 tiles. Creating placeholder.")
            try:
                sample_tile = read_tile_downscaled(tile_files[0], scale_factor=scale_factor)
                tile_shape = sample_tile.shape
            except Exception as e:
                logging.error(f"Cannot create placeholder for well {well_coord}: {e}")
                continue
            
            # Ensure placeholder is created as a color image.
            if len(tile_shape) == 2:
                tile_height, tile_width = tile_shape
                channels = 3  # Force color
            else:
                tile_height, tile_width, channels = tile_shape
            
            full_height = tile_height + 3 * (tile_height - OVERLAP)
            full_width = tile_width + 3 * (tile_width - OVERLAP)
            placeholder = create_placeholder_image(full_height, full_width, channels, border_thickness=8, dtype=sample_tile.dtype)
            well_images[well_coord] = placeholder
            continue

        try:
            merged_well = merge_well_tiles(tile_files, scale_factor=scale_factor)
            well_images[well_coord] = merged_well
            logging.info(f"Merged well {well_coord} (group {group_key})")
        except Exception as e:
            logging.error(f"Failed to merge well {group_key}: {e}")

    if not well_images:
        logging.error("No well images were merged; aborting plate merge.")
        return

    # Convert any grayscale well images to color by stacking channels.
    for well, img in well_images.items():
        if img.ndim == 2:
            img_color = np.stack([img] * 3, axis=-1)
            well_images[well] = img_color

    # Mapping from simulated_prediction to border color (RGB)
    prediction_to_color = {
        1: (0, 255, 0),      # Green
        2: (255, 0, 0),      # Red
        3: (255, 165, 0),    # Orange
        -1: (128, 0, 128)    # Purple
    }

    def add_border_to_image(img, color, thickness=8):
        # Draw a border on the image by replacing the edge pixels with the given color.
        # This function assumes the image is color (3D).
        # Top and bottom borders
        img[0:thickness, :, :] = color
        img[-thickness:, :, :] = color
        # Left and right borders
        img[:, 0:thickness, :] = color
        img[:, -thickness:, :] = color
        return img

    # Iterate over each well and add the border if the simulated_prediction warrants one.
    for well, img in well_images.items():
        pred_val = simulated_predictions.get(well, 0)
        if pred_val != 0 and pred_val in prediction_to_color:
            color = prediction_to_color[pred_val]
            well_images[well] = add_border_to_image(img, color, thickness=8)
    # --- End New Section ---

    layout = {}
    rows_set = set()
    cols_set = set()
    for well, img in well_images.items():
        try:
            row_idx, col_idx = parse_well_coordinate(well)
            layout[well] = (row_idx, col_idx)
            rows_set.add(row_idx)
            cols_set.add(col_idx)
        except Exception as e:
            logging.error(f"Skipping well {well} due to coordinate parsing error: {e}")

    if not layout:
        logging.error("No valid well coordinates were found; aborting plate merge.")
        return

    num_rows = max(rows_set) + 1
    num_cols = max(cols_set) + 1

    sample_img = next(iter(well_images.values()))
    well_height, well_width = sample_img.shape[:2]
    channels = sample_img.shape[2]  # Now all images are color (ndim==3)

    plate_img = np.zeros((num_rows * well_height, num_cols * well_width, channels), dtype=sample_img.dtype)

    for well, img in well_images.items():
        if well not in layout:
            continue
        r, c = layout[well]
        start_row = r * well_height
        start_col = c * well_width
        plate_img[start_row:start_row + well_height, start_col:start_col + well_width, :] = img

    occupied_coords = {coord for coord in layout.values()}
    placeholder = create_placeholder_image(well_height, well_width, channels, border_thickness=8, dtype=sample_img.dtype)
    for r in range(num_rows):
        for c in range(num_cols):
            if (r, c) not in occupied_coords:
                start_row = r * well_height
                start_col = c * well_width
                plate_img[start_row:start_row + well_height, start_col:start_col + well_width, :] = placeholder

    if not output_file.exists() or PROCESS_EXISTING:
        try:
            tifffile.imwrite(str(output_file), plate_img)
            logging.info(f"Saved full plate mosaic to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save full plate mosaic {output_file}: {e}")
    else:
        logging.info(f"Full plate mosaic {output_file} already exists. Skipping writing.")

def merge_plate_colonies_with_outlines(colony_folder, plate_calls_dir, plate, day):
    # Read the CSV and filter for Plate 54 and Day 8.
    csv_path = r"E:\MERGED_sspsygene_growthassay_colonies\ground_truth_with_simulated_prediction.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df[df["Plate"] == plate]
        df = df[df["Day"] == (day-1)]
        # Assuming the CSV has columns 'Well' and 'simulated_prediction'
        simulated_predictions = dict(zip(df['Well'], df['simulated_prediction']))
        logging.info("Loaded simulated predictions from CSV.")
    except Exception as e:
        logging.error(f"Failed to load CSV for simulated predictions: {e}")
        simulated_predictions = {}

    # Mapping from simulated_prediction to border color (RGB)
    prediction_to_color = {
        1: (0, 255, 0),      # Green
        2: (255, 0, 0),      # Red
        3: (255, 165, 0),    # Orange
        -1: (128, 0, 128),   # Purple
        0: (50, 50, 50)   # Gray
    }

    def add_border_to_image(img, color, thickness=4):
        """
        Draws a colored border around a color image.
        Assumes img is a 3D (color) array.
        """
        # Top border
        img[0:thickness, :, :] = color
        # Bottom border
        img[-thickness:, :, :] = color
        # Left border
        img[:, 0:thickness, :] = color
        # Right border
        img[:, -thickness:, :] = color
        return img

    # Gather colony images from the folder.
    colony_files = list(Path(colony_folder).glob("colony_*.tif"))
    if not colony_files:
        logging.error(f"No colony images found in {colony_folder}")
        return

    well_images = {}
    layout = {}
    rows_set = set()
    cols_set = set()

    for file in colony_files:
        parts = file.stem.split("_")
        if len(parts) < 2:
            logging.error(f"Unexpected file name format for colony image: {file.name}")
            continue
        well_coord = parts[1]
        try:
            row_idx, col_idx = parse_well_coordinate(well_coord)
        except Exception as e:
            logging.error(f"Error parsing well coordinate from {file.name}: {e}")
            continue
        layout[well_coord] = (row_idx, col_idx)
        rows_set.add(row_idx)
        cols_set.add(col_idx)
        try:
            img = tifffile.imread(str(file))
            img = cv2.resize(img, (img.shape[0]//16, img.shape[1]//16), interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            logging.error(f"Error reading colony image {file}: {e}")
            continue
        # The colony images are assumed to be in color.
        well_images[well_coord] = img

    if not well_images:
        logging.error("No valid colony images to merge.")
        return

    # Add outlines to each colony image based on simulated prediction.
    for well, img in well_images.items():
        pred_val = simulated_predictions.get(well, 0)
        if pred_val in prediction_to_color:
            color = prediction_to_color[pred_val]
            well_images[well] = add_border_to_image(img, color, thickness=6)

    # Determine mosaic layout dimensions.
    num_rows = max(rows_set) + 1
    num_cols = max(cols_set) + 1
    sample_img = next(iter(well_images.values()))
    well_height, well_width = sample_img.shape[:2]
    channels = sample_img.shape[2] if sample_img.ndim == 3 else None

    # Create an empty plate image.
    if channels:
        plate_img = np.zeros((num_rows * well_height, num_cols * well_width, channels), dtype=sample_img.dtype)
    else:
        plate_img = np.zeros((num_rows * well_height, num_cols * well_width), dtype=sample_img.dtype)

    # Place colony images into the plate mosaic.
    for well, img in well_images.items():
        if well not in layout:
            continue
        r, c = layout[well]
        start_row = r * well_height
        start_col = c * well_width
        if channels:
            plate_img[start_row:start_row + well_height, start_col:start_col + well_width, :] = img
        else:
            plate_img[start_row:start_row + well_height, start_col:start_col + well_width] = img

    # For empty well positions, create placeholders.
    occupied_coords = { coord for coord in layout.values() }
    placeholder = create_placeholder_image(well_height, well_width, channels, border_thickness=6, dtype=sample_img.dtype, intensity=50)
    for r in range(num_rows):
        for c in range(num_cols):
            if (r, c) not in occupied_coords:
                start_row = r * well_height
                start_col = c * well_width
                if channels:
                    plate_img[start_row:start_row + well_height, start_col:start_col + well_width, :] = placeholder
                else:
                    plate_img[start_row:start_row + well_height, start_col:start_col + well_width] = placeholder

    output_file = f"{plate_calls_dir}/plate_call_{plate}_day{day}.tif"
    # Save the mosaic image if appropriate.
    if not Path(output_file).exists() or PROCESS_EXISTING:
        try:
            tifffile.imwrite(output_file, plate_img)
            logging.info(f"Saved plate colony mosaic to {output_file}")
        except Exception as e:
            logging.error(f"Error saving plate colony mosaic to {output_file}: {e}")
    else:
        logging.info(f"Plate colony mosaic {output_file} already exists. Skipping writing.")

# brightfield_dir = r"E:\MERGED_sspsygene_growthassay_colonies\cluster_optimize_set"

# brightfield_files = list(Path(brightfield_dir).glob("C2_*"))

# merged = merge_well_brightfield(brightfield_files)
# tifffile.imwrite(str(brightfield_dir+"/C2_BF_no_outlines.tif"), merged)

# print("ok")
# # # merge_plate_colonies_with_outlines(r"E:\MERGED_sspsygene_growthassay_colonies\54_day9_20250218_172845\model_outputs\2025_03_03_23_34_16.794792_epoch_899_0.4_2_full\colonies", Path(r"E:\MERGED_sspsygene_growthassay_colonies\54_day9_20250218_172845\model_outputs\2025_03_03_23_34_16.794792_epoch_899_0.4_2_full\cellProb_with_outlines.tif"))

# import re
# from pathlib import Path

# # Assuming merge_plate_colonies_with_outlines is already defined/imported
# # def merge_plate_colonies_with_outlines(colonies_path, plate_calls_path, plate, day):
# #     # Your implementation here
# #     pass

# # Base directories
# base_dir = Path(r"E:\MERGED_sspsygene_growthassay_colonies")
# plate_calls_dir = Path(r"E:\MERGED_sspsygene_growthassay_colonies\plate_calls")

# # Regex pattern to match 'day1', 'day4', or 'day9'
# day_pattern = re.compile(r'day(1|4|9)', re.IGNORECASE)

# # Iterate over each subdirectory in the base folder
# for subdir in base_dir.iterdir():
#     if subdir.is_dir() and day_pattern.search(subdir.name):
#         # Attempt to extract the plate number (assumes the folder name starts with the plate number)
#         parts = subdir.name.split('_')
#         try:
#             plate = int(parts[0])
#         except ValueError:
#             print(f"Skipping {subdir.name}: unable to parse plate number.")
#             continue

#         # Extract day number from the folder name
#         day_match = re.search(r'day(\d+)', subdir.name, re.IGNORECASE)
#         if day_match:
#             day = int(day_match.group(1))
#         else:
#             print(f"Skipping {subdir.name}: 'day' not found in the folder name.")
#             continue

#         # Construct the expected model_outputs directory path
#         model_outputs_dir = subdir / "model_outputs"
#         if not model_outputs_dir.exists():
#             print(f"Skipping {subdir.name}: 'model_outputs' directory not found.")
#             continue

#         # Look for a subfolder within model_outputs that contains a 'colonies' folder
#         found = False
#         for output_subdir in model_outputs_dir.iterdir():
#             colonies_dir = output_subdir / "colonies"
#             if colonies_dir.exists() and colonies_dir.is_dir():
#                 # Call your function with the colonies path, plate_calls directory, plate, and day
#                 merge_plate_colonies_with_outlines(str(colonies_dir), plate_calls_dir, plate=plate, day=day)
#                 found = True
#                 break  # Remove this break if you expect multiple colonies folders per plate folder

#         if not found:
#             print(f"Skipping {subdir.name}: no 'colonies' folder found in any subfolder of 'model_outputs'.")
