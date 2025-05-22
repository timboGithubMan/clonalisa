from __future__ import annotations
from itertools import repeat
import concurrent.futures as cf
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from skimage import io

from importlib.resources import files
# from omnipose import utils # utils is not used
from cellpose_omni import models, core
import cellpose_omni.io

core.use_gpu()

def _load_model(model_info: tuple[str, float, float]):
    """
    model_info = (filename, flow_threshold, mask_threshold)

    filename must exist inside clonalisa/omnipose_models/
    (they're shipped with the wheel).
    """
    weights_path = (
        files("clonalisa")
        .joinpath("omnipose_models")
        .joinpath(model_info[0])
    )

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    try:
        nchan = int(model_info[0].split("nchan_")[1][0])
    except Exception as e:  # pragma: no cover
        raise ValueError(
            "Cannot parse nchan from model filename "
            f"‘{model_info[0]}’"
        ) from e

    print(f"→ loading model {weights_path.name}  (nchan={nchan})")
    return models.CellposeModel(
         pretrained_model=str(weights_path),
         nclasses=2,
         nchan=nchan,
         gpu=True if models.torch.cuda.is_available() else False # Simpler: core.use_gpu() is already called
                                                                # and model checks for available CUDA
    )

def _run_on_subset(
    image_paths: list[Path],
    model_info: tuple[str, float, float],
    out_dir: Path,
    save_cellProb = True,
    save_flows = False, save_outlines = False
):
    """
    Worker entry-point.
    """
    
    model_filename, flow_thresh, mask_thresh = model_info
    model = _load_model(model_info)

    for img_idx, img_path in enumerate(image_paths):
        current_image_name = img_path.name # For logging
        try:
            img_data = [io.imread(str(img_path))] # model.eval expects a list of images
            masks_list, flows_list, _ = model.eval(
                img_data, # List containing a single image
                omni=True,
                normalize=True,
                channels=None,
                resample=False,
                tile=True,
                # bsize=448,
                flow_threshold=flow_thresh,
                mask_threshold=mask_thresh,
                affinity_seg=True,
                suppress=False,
            )
            masks = masks_list[0]
            flows = flows_list[0]

            print(f"Worker ({current_image_name}): Saving outputs...")
            cellpose_omni.io.imwrite(out_dir / (img_path.stem + "_cp_masks.png"), masks)
            if save_cellProb and flows is not None and len(flows) > 2 and flows[2] is not None:
                cell_prob_map = flows[2]
                scaled_prob_map = np.clip(cell_prob_map * 21.25, 0, 255).astype(np.uint8)
                cellpose_omni.io.imwrite(out_dir / (img_path.stem + "_cellProb.png"), scaled_prob_map)
            else:
                print(f"Warning: Could not extract cell_prob_map for {img_path.name}")
            if save_flows and flows is not None and len(flows) > 2 and flows[0] is not None:
                cellpose_omni.io.imwrite(out_dir / (img_path.stem + "_flows.png"), flows[0].astype(np.uint8))
            else:
                print(f"Warning: Could not extract flows_rgb_map for {img_path.name}")

            print(f"✓ {img_path.name}")

        except Exception as e:
            print(f"✗ {img_path.name}  (Error: {e})")
            import traceback
            traceback.print_exc() # Print full traceback for errors in worker

# -----------------------------------------------------------------------------

def run_omnipose(
    directory: str | Path,
    model_info: tuple[str, float, float],
    *,
    save_cellProb: bool = True,
    save_flows: bool = False,
    save_outlines: bool = False,
    num_threads: int = 4,
) -> Path:
    """
    Segment all .tif[f] in *directory* using the given model.

    Parameters
    ----------
    directory        : folder with input images
    model_info       : (weights_filename, flow_thresh, mask_thresh)
    save_cellprob    : save cellprob image
    save_flows       : save flows image
    save_outlines    : save outline image
    num_threads      : how many worker processes to spawn

    Returns
    -------
    Path to the folder holding Omnipose outputs.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    out_dir = directory / "model_outputs" / (
        "_".join(model_info[0].split("_")[-8:])
        + f"_{model_info[1]}_{model_info[2]}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    masks_done = {p.stem.replace("_cp_masks", "") for p in out_dir.glob("*_cp_masks.png")}
    imgs_iterable: Iterable[Path] = (
        p
        for p in directory.glob("*.tif*") # Handles .tif and .tiff
        if all(s not in p.name for s in ("masks", "dP.tif", "outlines.tif", "flows.tif", "Wells", "cellProb"))
        and p.stem not in masks_done
    )
    all_imgs = list(imgs_iterable) # Make it a list to shuffle and chunk
    
    if not all_imgs:
        print("No new images to process.")
        return out_dir
        
    random.shuffle(all_imgs) # Shuffle before chunking

    num_workers = min(num_threads, len(all_imgs))
    if num_workers == 0: # Should not happen if all_imgs is not empty, but good practice
        print("No workers to spawn (num_threads or image count is 0).")
        return out_dir
        
    # Split into roughly equal chunks for each worker
    chunks: list[list[Path]] = [[] for _ in range(num_workers)]
    for i, img_path in enumerate(all_imgs):
        chunks[i % num_workers].append(img_path)
    
    # Filter out empty chunks if len(all_imgs) < num_threads
    chunks = [chunk for chunk in chunks if chunk]
    actual_num_workers = len(chunks) # This will be the actual number of processes

    if actual_num_workers == 0:
        print("No images assigned to any worker.") # Should be caught by 'if not all_imgs'
        return out_dir

    print(f"Processing {len(all_imgs)} images using {actual_num_workers} worker(s).")
    
    with cf.ProcessPoolExecutor(max_workers=actual_num_workers) as pool:
        results = list(pool.map(
            _run_on_subset,
            chunks,                        # iterable ➊ – images for each worker
            repeat(model_info),            # broadcast constants ⤵
            repeat(out_dir),
            repeat(save_cellProb),
            repeat(save_flows),
            repeat(save_outlines)
        ))


    return out_dir

if __name__ == "__main__":
    start = time.time() 
    try:
        _model = ("10x_NPC_nclasses_2_nchan_3_dim_2_2024_03_29_02_03_10.875324_epoch_960", 0.4, 0.0)
        image_dir = r"E:\test\test_day5_20250519_153429" 
        print(f"Looking for images in: {Path(image_dir).resolve()}")
    
        output_path = run_omnipose(
            image_dir,
            _model,
            num_threads=8,       # Number of processes
        )
        print(f"Omnipose processing complete. Outputs in: {output_path}")

    except ImportError as e:
        print(f"ImportError: {e}. Make sure 'clonalisa' is installed and accessible.")
        print("You might need to run `pip install .` if clonalisa is a local package.")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}. Check model paths and image directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        elapsed = time.time() - start           # <── stop timer
        print(f"\nTotal elapsed time: {elapsed:.2f} s")