from __future__ import annotations
import time
import os
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from skimage import io

from importlib.resources import files
# from omnipose import utils # utils is not used
from cellpose_omni import models, core
import cellpose_omni.io
from process_masks import color_and_outline_cells_per_channel
from config_utils import parse_filename, load_config
import concurrent.futures as cf
import multiprocessing as mp

core.use_gpu()


def _extract_well_info(filename: str):
    well, position, _, _, _ = parse_filename(filename)
    if well and position:
        return well, f"pos{position}"
    return None, None


def _collect_image_groups(
    directory: Path,
    keyword: str = "bright",
    z_indices: Sequence[int] | None = None,
) -> dict[str, list[Path]]:
    """Return a mapping of output name -> list of image paths.

    Parameters
    ----------
    directory : Path
        Folder containing images.
    keyword : str
        Only files containing this substring (case insensitive) are used.
    z_indices : Sequence[int] | None
        If given, only the specified Z indices are kept for each group.
    """

    cfg = load_config()
    tif_files = [
        f
        for f in os.listdir(directory)
        if f.lower().endswith(".tif") and keyword.lower() in f.lower()
    ]

    groups: dict[str, list[tuple[int | None, Path]]] = {}
    for file in tif_files:
        well_name, position, _, z_index, step = parse_filename(file, cfg)
        if well_name and position:
            step_str = step or "0"
            output_name = f"{well_name}_pos{position}_merged_{step_str}"
            groups.setdefault(output_name, []).append((z_index, directory / file))

    filtered: dict[str, list[Path]] = {}
    for name, items in groups.items():
        items.sort(key=lambda t: (t[0] if t[0] is not None else 0))
        if z_indices is not None:
            filtered[name] = [p for z, p in items if z in z_indices]
        else:
            filtered[name] = [p for _, p in items]

    return filtered


def _load_and_merge(paths: list[Path], order: Sequence[int] | None, crop_size: int = 1992) -> np.ndarray:
    imgs = []
    for p in paths:
        img = io.imread(str(p))
        if img.shape[0] > crop_size or img.shape[1] > crop_size:
            center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
            start_x = center_x - (crop_size // 2)
            start_y = center_y - (crop_size // 2)
            img = img[start_y:start_y + crop_size, start_x:start_x + crop_size]
        imgs.append(img)

    if order:
        imgs = [imgs[i] for i in order if i < len(imgs)]

    if len(imgs) > 1:
        return np.stack(imgs, axis=0)
    return imgs[0]

def _load_model(model_info: tuple[str, float, float]):
    """
    model_info = (file path, flow_threshold, mask_threshold)
    """
    model_filepath, flow_thresh, mask_thresh = model_info
    try:
        nchan = int(os.path.basename(model_filepath).split("nchan_")[1][0])
    except Exception as e:
        raise ValueError(
            "Cannot parse nchan from model filename "
            f"‘{model_info[0]}’"
        ) from e

    print(f"→ loading model {os.path.basename(model_filepath)}  (nchan={nchan})")
    return models.CellposeModel(
         pretrained_model=model_info[0],
         nclasses=2,
         nchan=nchan,
         gpu=True if models.torch.cuda.is_available() else False # Simpler: core.use_gpu() is already called
                                                                # and model checks for available CUDA
    )

def _run_on_subset(
    groups: list[tuple[str, list[Path]]],
    model_info: tuple[str, float, float],
    out_dir: Path,
    z_indices: Sequence[int] | None,
    save_cellProb=False,
    save_flows=False,
    save_outlines=False,
    progress_queue: mp.Queue | None = None,
):
    """
    Worker entry-point.
    """
    
    model_filepath, flow_thresh, mask_thresh = model_info
    model = _load_model(model_info)

    for img_idx, (name, paths) in enumerate(groups):
        current_image_name = name  # For logging
        try:
            img = _load_and_merge(paths, z_indices)
            img_data = [img]  # model.eval expects a list of images
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
            cellpose_omni.io.imwrite(out_dir / f"{name}_cp_masks.png", masks)
            if save_cellProb and flows is not None and len(flows) > 2 and flows[2] is not None:
                cell_prob_map = flows[2]
                scaled_prob_map = np.clip(cell_prob_map * 21.25, 0, 255).astype(np.uint8)
                cellpose_omni.io.imwrite(out_dir / f"{name}_cellProb.png", scaled_prob_map)
            if save_flows and flows is not None and len(flows) > 2 and flows[0] is not None:
                cellpose_omni.io.imwrite(out_dir / f"{name}_flows.png", flows[0].astype(np.uint8))
            if save_outlines and masks is not None:
                color_and_outline_cells_per_channel(out_dir / "outlines" / f"{name}_outlines.jpg", img, masks)

            print(f"✓ {name}")

        except Exception as e:
            print(f"✗ {name}  (Error: {e})")
            import traceback
            traceback.print_exc() # Print full traceback for errors in worker
        finally:
            if progress_queue is not None:
                progress_queue.put(1)

# -----------------------------------------------------------------------------

def run_omnipose(
    directory: str | Path,
    model_info: tuple[str, float, float],
    *,
    filter_keyword: str = "bright",
    z_indices: Sequence[int] | None = None,
    save_cellProb: bool = False,
    save_flows: bool = False,
    save_outlines: bool = True,
    num_threads: None,
    progress_callback=None,
) -> Path:
    """
    Segment all .tif[f] in *directory* using the given model.

    Parameters
    ----------
    directory        : folder with input images
    model_info       : (weights_filename, flow_thresh, mask_thresh)
    filter_keyword   : filter images containing this substring
    z_indices        : which Z indices to use (None = all)
    save_cellprob    : save cellprob image
    save_flows       : save flows image
    save_outlines    : save outline image
    num_threads      : how many worker processes to spawn
    progress_callback: optional callback(done, total) for progress reporting

    Returns
    -------
    Path to the folder holding Omnipose outputs.
    """

    num_threads = os.cpu_count() // 2 if num_threads <= 0 else num_threads

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    out_dir = directory / "model_outputs" /(os.path.basename(model_info[0]) + f"_{model_info[1]}_{model_info[2]}")
    out_dir.mkdir(parents=True, exist_ok=True)

    masks_done = {p.stem.replace("_cp_masks", "") for p in out_dir.glob("*_cp_masks.png")}
    groups_dict = _collect_image_groups(directory, keyword=filter_keyword, z_indices=z_indices)
    with open(out_dir / "source_images.json", "w") as f:
        json.dump({k: [str(p) for p in v] for k, v in groups_dict.items()}, f, indent=2)
    all_groups = [(name, paths) for name, paths in groups_dict.items() if name not in masks_done]

    if not all_groups:
        print("No new images to process.")
        return out_dir

    total = len(all_groups)
    num_workers = min(num_threads, len(all_groups))
    if num_workers == 0:
        print("No workers to spawn (num_threads or image count is 0).")
        return out_dir

    chunks: list[list[tuple[str, list[Path]]]] = [[] for _ in range(num_workers)]
    for i, grp in enumerate(all_groups):
        chunks[i % num_workers].append(grp)

    chunks = [chunk for chunk in chunks if chunk]
    actual_num_workers = len(chunks)

    if progress_callback:
        progress_callback(0, total)
        mgr = mp.Manager()
        q: mp.Queue = mgr.Queue()
    else:
        q = None

    with cf.ProcessPoolExecutor(max_workers=actual_num_workers) as pool:
        futures = [
            pool.submit(
                _run_on_subset,
                chunk,
                model_info,
                out_dir,
                z_indices,
                save_cellProb,
                save_flows,
                save_outlines,
                q,
            )
            for chunk in chunks
        ]

        done = 0
        if progress_callback:
            while done < total:
                q.get()
                done += 1
                progress_callback(done, total)

        for fut in futures:
            fut.result()

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