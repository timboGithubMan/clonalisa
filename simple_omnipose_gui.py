import os
import tkinter as tk
from tkinter import filedialog, messagebox

import omnipose_threaded
import process_masks

import subprocess, sys, pathlib

# Constants from omnipose_pipeline
MAX_WORKERS = os.cpu_count() // 2
PLATE_TYPE = "96W"
MAGNIFICATION = "10x"
CYTATION = True
CHANNEL_ORDER = [1, 2, 0]

PLATE_AREAS = {"6W": 9.6, "12W": 3.8, "24W": 2, "48W": 1.1, "96W": 0.32}
CM_PER_MICRON = 1 / 10000
if CYTATION:
    MICRONS_PER_PIXEL = 1389 / 1992 if MAGNIFICATION == "10x" else 694 / 1992
    IMAGE_AREA_CM = 1992 * 1992 * MICRONS_PER_PIXEL**2 * CM_PER_MICRON**2
else:
    if MAGNIFICATION == "10x":
        MICRONS_PER_PIXEL = 0.61922571983322461
        IMAGE_AREA_CM = 0.0120619953
    else:
        MICRONS_PER_PIXEL = 1.5188172690164046
        IMAGE_AREA_CM = 0.0725658405

CM_PER_PIXEL = CM_PER_MICRON * MICRONS_PER_PIXEL


def run_pipeline(input_dir: str, model_file: str, *, save_flows: bool,
                 save_cellprob: bool, save_outlines: bool) -> None:
    """Run omnipose on all subdirectories of *input_dir*."""
    model_info = (model_file, 0.4, 0)
    subdirs = [d for d in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, d)) and "epoch" not in d]
    for sub in subdirs:
        dir_path = os.path.join(input_dir, sub)
        out_dir = omnipose_threaded.run_omnipose(
            dir_path,
            model_info,
            num_threads=MAX_WORKERS,
            channel_order=CHANNEL_ORDER,
            save_flows=save_flows,
            save_cellProb=save_cellprob,
            save_outlines=save_outlines,
        )
        process_masks.process_mask_files(
            out_dir,
            CM_PER_PIXEL,
            PLATE_AREAS.get(PLATE_TYPE),
            force_save=False,
            filter_min_size=None,
        )
    all_data_csv = process_masks.make_all_data_csv(input_dir, os.path.basename(model_info[0]))

def launch_gui() -> None:
    root = tk.Tk()
    root.title("Omnipose Pipeline")

    input_var = tk.StringVar()
    model_var = tk.StringVar()
    save_flows_var = tk.BooleanVar(value=False)
    save_cellprob_var = tk.BooleanVar(value=False)
    save_outlines_var = tk.BooleanVar(value=True)

    def browse_input() -> None:
        path = filedialog.askdirectory()
        if path:
            input_var.set(path)

    def browse_model() -> None:
        path = filedialog.askopenfilename(initialdir="omnipose_models")
        if path:
            model_var.set(path)

    def run_clicked() -> None:
        input_dir = input_var.get()
        model_file = model_var.get()
        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Select a valid input directory")
            return
        if not os.path.isfile(model_file):
            messagebox.showerror("Error", "Select a model file")
            return
        run_pipeline(
            input_dir,
            model_file,
            save_flows=save_flows_var.get(),
            save_cellprob=save_cellprob_var.get(),
            save_outlines=save_outlines_var.get(),
        )
        messagebox.showinfo("Finished", "Processing complete")

    # Layout
    tk.Label(root, text="Input directory:").grid(row=0, column=0, sticky="e")
    tk.Entry(root, textvariable=input_var, width=40).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=browse_input).grid(row=0, column=2)

    tk.Label(root, text="Model file:").grid(row=1, column=0, sticky="e")
    tk.Entry(root, textvariable=model_var, width=40).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=browse_model).grid(row=1, column=2)

    tk.Checkbutton(root, text="Save flows", variable=save_flows_var).grid(row=2, column=0, sticky="w")
    tk.Checkbutton(root, text="Save cell probability", variable=save_cellprob_var).grid(row=2, column=1, sticky="w")
    tk.Checkbutton(root, text="Save outlines", variable=save_outlines_var).grid(row=2, column=2, sticky="w")

    tk.Button(root, text="Run", command=run_clicked).grid(row=3, column=1, pady=10)

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
