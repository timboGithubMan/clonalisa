import tkinter as tk
from tkinter import filedialog, Listbox, Label, Button, Scrollbar, Frame, messagebox
from threading import Thread
import os
import subprocess
from skimage.io import imread, imsave
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from omnipose_threaded import run_omnipose
from cellpose_omni import models

MAX_CP_WORKERS = os.cpu_count() or 8  # Default to 8 if os.cpu_count() returns None
MAX_OMNIPOSE_WORKERS = os.cpu_count() or 8  # Default to 8 if os.cpu_count() returns None

def count_images(dir_path):
    dapi_images = 0
    mask_images = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith((".tif", ".tiff", ".png")):
                if "_DAPI" in filename and "masks" not in filename and "Wells.tif" not in filename:
                    dapi_images += 1
                elif "masks" in filename:
                    mask_images += 1
    return dir_path, dapi_images, mask_images


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Cell Analysis Tool")
        self.root.geometry('1280x720')  # Set the window size
        self.selected_dirs = {}
        self.status_label = Label(self.root, text="Ready to process.")
        self.status_label.pack(pady=(10, 0), fill=tk.X)

        self.setup_ui()

        self.pipeline_paths = {
            "CY5": r"Z:\Wellslab\ToolsAndScripts\CellProfiler\pipelines\CY5_masks_from_cellpose.cppipe",
            "GFP": r"Z:\Wellslab\ToolsAndScripts\CellProfiler\pipelines\GFP_masks_from_cellpose.cppipe",
            "RFP": r"Z:\Wellslab\ToolsAndScripts\CellProfiler\pipelines\RFP_masks_from_cellpose.cppipe",
        }

    def setup_ui(self):
        frame_dirs = Frame(self.root)
        frame_dirs.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        Label(frame_dirs, text="Directories").pack()

        scrollbar = Scrollbar(frame_dirs)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox_dirs = Listbox(frame_dirs, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set, width=80)
        self.listbox_dirs.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox_dirs.yview)

        Button(frame_dirs, text="Add Directories", command=self.add_directories).pack(pady=(10, 0))
        Button(frame_dirs, text="Remove Directory", command=self.remove_directory).pack(pady=(0, 10))
        Button(frame_dirs, text="Process with Cellpose", command=lambda: self.process_with_cellpose(False)).pack(
            pady=(0, 10))
        Button(frame_dirs, text="Process with Cellpose + CellProfiler", command=self.process_with_omnipose).pack(
            pady=(0, 10))

    def add_directories(self):
        dir_path = filedialog.askdirectory(title="Select Directories", mustexist=True)
        if dir_path and dir_path not in self.selected_dirs:
            Thread(target=self.add_directory, args=(dir_path,)).start()

    def add_directory(self, dir):
        result = count_images(dir)
        self.selected_dirs[dir] = result[1:]
        self.listbox_dirs.insert(tk.END, f"{dir} (DAPI: {result[1]}, Masks: {result[2]})")

    def populate_directories(self):
        base_path = r"Z:\Wellslab\Cytation_overflow_040723"
        if os.path.exists(base_path):
            for dir_name in os.listdir(base_path):
                full_path = os.path.join(base_path, dir_name)
                if os.path.isdir(full_path):
                    self.add_directory(full_path)
        else:
            print(f"The path {base_path} does not exist.")

    def remove_directory(self):
        selected_indices = list(self.listbox_dirs.curselection())
        for index in selected_indices[::-1]:
            dir_path = self.listbox_dirs.get(index).split(" (DAPI")[0]
            if dir_path in self.selected_dirs:
                del self.selected_dirs[dir_path]
            self.listbox_dirs.delete(index)

    def process_csv(self, directory, channel):
        """
        Combine every results_extended_Image.csv produced for <channel>,
        aggregate counts by Metadata_Well, and save the roll-up.
        """
        base_dir = os.path.join(directory, "cellProfiler_results", channel)
        if not os.path.isdir(base_dir):
            print(f"[WARN] No CellProfiler results for {channel} in {directory}")
            return

        # 1) gather every CSV under base_dir (well sub-folders, etc.)
        csv_paths = []
        for root, _, files in os.walk(base_dir):
            csv_paths += [os.path.join(root, f)
                          for f in files
                          if "results_" in f and not "extended" in f and not "by_well" in f]

        if not csv_paths:
            print(f"[WARN] No CSV files found for {channel} in {base_dir}")
            return

        # 2) concatenate
        df = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)

        # 3) aggregate
        agg_cols = [c for c in df.columns if c.startswith("Count_")]
        agg_funcs = {c: "sum" for c in agg_cols}
        agg_funcs["Metadata_Row"] = "first"
        agg_funcs["Metadata_Column"] = "first"

        df_agg = df.groupby("Metadata_Well").agg(agg_funcs).reset_index()

        # 4) add “bin or greater” percentages
        bin_cols = [c for c in df_agg.columns
                    if "_bin" in c and "Count_nuclei_accepted" not in c]
        for col in bin_cols:
            base = col.rsplit("_", 1)[0]
            bin_num = int(col.split("_bin")[-1])
            greater = [f"{base}_bin{i}" for i in range(bin_num, 7)]
            df_agg[f"{col}_or_greater_percent"] = (
                df_agg[greater].sum(axis=1) / df_agg["Count_nuclei_accepted"] * 100
            )

        # 5) write the roll-up next to the per-well folders
        out_path = os.path.join(base_dir, f"results_by_well_{channel}.csv")
        df_agg.to_csv(out_path, index=False)
        print(f"[OK] Aggregated CSV written → {out_path}")

    def get_well_channel_groups(self, directory):
        """
        Return a dict keyed by (well, channel) → list[(core, filename)].

        core = "well_pos_id_step"  (e.g. A01_pos1_001_02)
        channel is one of self.pipeline_paths (GFP / CY5 / RFP).
        DAPI is ignored because it is only the reference channel.
        """
        groups = {}
        for fname in os.listdir(directory):
            if not fname.lower().endswith((".tif", ".tiff", ".png")):
                continue

            parts = fname.split('_')
            if len(parts) < 6:
                continue                    # malformed name

            well, pos, channel = parts[0], parts[1], parts[2]
            if channel not in self.pipeline_paths or channel == "DAPI":
                continue                    # skip non-pipeline or DAPI files

            id_part = parts[4]
            step = parts[5].split('.')[0]   # strip extension
            core = f"{well}_{pos}_{id_part}_{step}"

            groups.setdefault((well, channel), []).append((core, fname))
        return groups

    def write_file_list(self, file_names, directory, well, channel):
        parent_directory = os.path.basename(os.path.dirname(directory))
        output_directory = os.path.join(directory, "cellProfiler_results", channel)
        #output_directory = f"Z:\\Wellslab\\cellProfiler_runs\\{parent_directory}\\{os.path.basename(directory)}\\{channel}"
        os.makedirs(output_directory, exist_ok=True)
        file_list_path = os.path.join(output_directory, f"{well}_{channel}_file_list.txt")
        # file_list_path = os.path.join(output_directory, f"{channel}_file_list.txt")
        with open(file_list_path, 'w') as file_list:
            for file_name in file_names:
                file_list.write(f"{os.path.join(directory, file_name)}\n")
        return file_list_path

    def call_cellprofiler_pipeline(self, directory):
        """
        Launch up to MAX_CP_WORKERS parallel CellProfiler jobs.
        Each job processes one (well, channel) group.
        """
        directory = directory.replace(r"//hg-fs01/research", "Z:") \
                             .replace(r"//hg-fs01/Research", "Z:")

        dir_files = os.listdir(directory)
        groups = self.get_well_channel_groups(directory)

        def results_csv_exists(directory, well, channel):
            """
            Return True if a results_<well>.csv already exists for <channel>.
            We look in:
                <dir>/cellProfiler_results/<channel>/
                <dir>/cellProfiler_results/<channel>/<well>/
            """
            base = os.path.join(directory, "cellProfiler_results", channel)
            fname = f"results_{well}.csv"

            return (
                os.path.isfile(os.path.join(base, fname)) or
                os.path.isfile(os.path.join(base, well, fname))
            )
    
        def run_single_pipeline(well, channel, files):
            if results_csv_exists(directory, well, channel):
                print(f"[SKIP] {well}/{channel}: results CSV already present")
                return
            """Internal worker that actually calls CellProfiler."""
            valid_files = []
            for core, ch_file in files:
                well_, pos, id_part, step = core.split('_')
                dapi_base = f"{well_}_{pos}_DAPI_1_{id_part}_{step}"

                dapi_file = next(
                    (f for f in dir_files if f.startswith(dapi_base)
                     and f.endswith(".tif") and "Wells.tif" not in f),
                    None)
                mask_file = next(
                    (f for f in dir_files if f.startswith(dapi_base)
                     and f.endswith("_masks.png")),
                    None)

                if dapi_file and mask_file:
                    valid_files += [ch_file, dapi_file, mask_file]

            if not valid_files:
                print(f"[SKIP] {well}/{channel}: no matching DAPI+mask")
                return

            file_list = self.write_file_list(valid_files, directory, well, channel)
            cp_exe = r"C:\Program Files\CellProfiler\CellProfiler.exe"
            cmd = [
                cp_exe,
                "-c", "-r",
                "-p", self.pipeline_paths[channel],
                "--file-list", file_list,
            ]
            try:
                subprocess.run(cmd, cwd=os.path.dirname(cp_exe), check=True)
                self.process_csv(directory, channel)      # optional post-proc
                print(f"[DONE] {well}/{channel}")
            except subprocess.CalledProcessError as e:
                print(f"[ERR]  {well}/{channel}: {e}")

        # ---- PARALLEL EXECUTION ----
        with ThreadPoolExecutor(max_workers=MAX_CP_WORKERS) as pool:
            futures = [
                pool.submit(run_single_pipeline, well, channel, files)
                for (well, channel), files in groups.items()
            ]
            for f in as_completed(futures):
                # Will re-raise any exception from the worker
                f.result()
        
        for ch in self.pipeline_paths:
            self.process_csv(directory, ch)

    def process_file(self, full_filepath):
        try:
            mask_filename = full_filepath.split(".ti")[0] + "_masks.png"
            # Skip if a mask file already exists
            if os.path.exists(mask_filename):
                print(f"Mask file already exists for: {full_filepath}")
                return

            print(f"Processing file: {full_filepath}")
            img = imread(full_filepath)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            masks, flows, styles = self.model.eval(img, batch_size=8, diameter=None, channels=[0, 0],
                                                   flow_threshold=0.4, cellprob_threshold=4)
            Image.fromarray(masks).save(mask_filename)
            print(f"Finished processing file: {full_filepath}")
        except Exception as e:
            print(f"Failed to process file {full_filepath}. Error: {e}")

        return full_filepath

    def process_with_omnipose(self, process_with_cellProfiler=True):
        if not self.selected_dirs:
            messagebox.showwarning("No Directories Selected",
                                   "Please add at least one directory to process.")
            return

        dirs = list(self.selected_dirs.keys())
        Thread(target=self._process_with_omnipose,
               args=(dirs, process_with_cellProfiler),
               daemon=True).start()

    def _process_with_omnipose(self, dirs, process_with_cellProfiler=True):
        total_dirs = len(dirs)

        for idx, dir in enumerate(dirs, start=1):
            self.status_label.config(text=f"Omnipose: {dir}  ({idx}/{total_dirs})")
            self.root.update_idletasks()

            try:
                out_dir = run_omnipose(
                    directory           = dir,
                    model_info          = self.model_info,
                    num_threads         = MAX_OMNIPOSE_WORKERS,          # spawn up to 8 worker processes
                    output_in_same_dir  = True,       # write masks next to images
                    save_cellProb       = True,
                    save_flows          = False,
                    save_outlines       = False,
                )
                print(f"[✓] Omnipose finished → {out_dir}")

            except Exception as e:
                print(f"[✗] Omnipose failed in {dir}: {e}")
                continue

            # optional post-processing in CellProfiler -----------------
            if process_with_cellProfiler:
                self.call_cellprofiler_pipeline(dir)

        self.status_label.config(text="All Omnipose jobs complete.")
        messagebox.showinfo("Omnipose", "Processing complete.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
