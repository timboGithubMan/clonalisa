import os
import re
import numpy as np
import tifffile as tiff
from concurrent.futures import ThreadPoolExecutor

# New imports for GUI preview and image handling
import tkinter as tk
from PIL import Image, ImageTk

def create_merged_directory(input_dir, base_dir):
    if base_dir in input_dir:
        output_dir = input_dir
    else:
        merged_dir_name = base_dir + os.path.basename(input_dir)
        output_dir = os.path.join(base_dir, merged_dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    return output_dir

def process_images(pair, merged_dir, output_name):
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir, exist_ok=True)

    output_path = os.path.join(merged_dir, output_name)

    imgs = []
    crop_size = 1992

    for img_path in pair:
        try:
            img = tiff.imread(img_path)
            if img.shape[0] > crop_size or img.shape[1] > crop_size:
                center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
                start_x = center_x - (crop_size // 2)
                start_y = center_y - (crop_size // 2)
                img = img[start_y:start_y + crop_size, start_x:start_x + crop_size]
            imgs.append(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return

    if len(imgs) > 1:
        new_img = np.stack(imgs, axis=0)
    else:
        new_img = imgs[0]  # Single channel image

    tiff.imwrite(output_path, new_img, imagej=True)

def extract_well_info(filename):
    match = re.match(r'([A-Z]\d+)_pos(\d+)(Z\d+)?', filename)
    if match:
        well_name = match.group(1)
        position = f"pos{match.group(2)}"
        return well_name, position
    return None, None

def make_multi_channels(input_dir, subdir, output_dir, num_workers=4):
    if input_dir != output_dir:
        print(f"make multi channels {subdir}")
        subdir_path = os.path.join(input_dir, subdir)
        merged_subdir = os.path.join(output_dir, subdir)
    
        os.makedirs(merged_subdir, exist_ok=True)
        existing_merged_files = [img for img in os.listdir(merged_subdir)
                if img.endswith('.tif') and not any(suffix in img for suffix in ['masks.tif', 'dP.tif', 'outlines.tif', 'flows.tif', 'Wells'])]

        tif_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.tif') and not f.startswith('.') and "Bright" in f]

        pairs = {}
        for file in tif_files:
            well_name, position = extract_well_info(file)
            if well_name and position:
                parts = file.split('_')
                step = parts[-1].split('.')[0]
                output_name = f"{well_name}_{position}_merged_{step}.tif"
                if output_name not in pairs:
                    pairs[output_name] = []
                pairs[output_name].append(os.path.join(subdir_path, file))
     
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for output_name, file_paths in pairs.items():
                if output_name not in existing_merged_files:
                    futures.append(executor.submit(process_images, file_paths, merged_subdir, output_name))
            for future in futures:
                future.result()
        
    return subdir

def merge_z_stack_groups_multi_pos(input_dir, output_dir, crop_size=1992):
    """
    Processes Z-stack images in input_dir for multiple positions.
    For each position, groups images by Z index and merges groups [i, i+2, i+4],
    where i ranges from 0 to 15 (i.e. merging Z0,Z2,Z4; Z1,Z3,Z5; ...; Z15,Z17,Z19).
    
    Merged images are saved in output_dir with filenames indicating the position and Z indexes used.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    tif_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith('.tif') and "Bright Field" in f and not f.startswith('.')]
    
    pos_z_dict = {}
    for f in tif_files:
        match = re.search(r'_pos(\d+)Z(\d+)_', f)
        if match:
            pos = f"pos{match.group(1)}"
            z_index = int(match.group(2))
            if pos not in pos_z_dict:
                pos_z_dict[pos] = {}
            pos_z_dict[pos][z_index] = f
        else:
            print(f"Filename did not match expected pattern: {f}")

    for pos, z_dict in pos_z_dict.items():
        print(f"Processing {pos} with {len(z_dict)} Z images.")
        for i in range(16):
            if i in z_dict and (i + 2) in z_dict and (i + 4) in z_dict:
                group_files = [z_dict[i], z_dict[i + 2], z_dict[i + 4]]
                img_paths = [os.path.join(input_dir, file) for file in group_files]
                imgs = []
                for path in img_paths:
                    try:
                        img = tiff.imread(path)
                        if img.shape[0] > crop_size or img.shape[1] > crop_size:
                            center_x = img.shape[1] // 2
                            center_y = img.shape[0] // 2
                            start_x = center_x - (crop_size // 2)
                            start_y = center_y - (crop_size // 2)
                            img = img[start_y:start_y + crop_size, start_x:start_x + crop_size]
                        imgs.append(img)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
                        continue
                if len(imgs) == 3:
                    merged_img = np.stack(imgs, axis=0)
                    output_name = f"Merged_{pos}_Z{i}_{i+2}_{i+4}.tif"
                    output_path = os.path.join(output_dir, output_name)
                    tiff.imwrite(output_path, merged_img, imagej=True)
                    print(f"Saved merged image: {output_path}")
            else:
                print(f"For {pos}: Group Z{i}, Z{i+2}, Z{i+4} is missing one or more images.")

def shift_image(img, x_offset, y_offset):
    """
    Shifts a 2D image by x_offset and y_offset without wrapping around.
    The empty regions are filled with zeros.
    """
    shifted = np.zeros_like(img)
    h, w = img.shape
    if x_offset >= 0:
        src_x = 0
        dest_x = x_offset
        width = w - x_offset
    else:
        src_x = -x_offset
        dest_x = 0
        width = w + x_offset
    if y_offset >= 0:
        src_y = 0
        dest_y = y_offset
        height = h - y_offset
    else:
        src_y = -y_offset
        dest_y = 0
        height = h + y_offset

    if width > 0 and height > 0:
        shifted[dest_y:dest_y+height, dest_x:dest_x+width] = img[src_y:src_y+height, src_x:src_x+width]
    return shifted

def adjust_image_range_16bit(img, low, high):
    """
    Adjusts a 16-bit image by linearly mapping intensities so that:
      - values <= low become 0,
      - values >= high become 65535,
      - values in between are scaled linearly.
    """
    img_f = img.astype(np.float32)
    if high <= low:
        return np.zeros_like(img, dtype=np.uint16)
    normalized = (img_f - low) / (high - low)
    normalized = np.clip(normalized, 0, 1)
    return (normalized * 65535).astype(np.uint16)

def convert_16bit_to_8bit(img):
    """
    Converts a 16-bit image to 8-bit for display.
    Assumes the image uses the full 0-65535 range.
    """
    return (img / 256).astype(np.uint8)

def alignment_gui(img1, img2):
    """
    Presents a Tkinter GUI with sliders to adjust:
      - x and y offsets (for img1),
      - low and high intensity values for each channel,
      - plus checkboxes to toggle the display of each channel.
      
    For channel 1, img1 is shifted by x/y offsets and then its intensities are remapped
    from the chosen low/high values. For channel 2, the remapping is applied directly.
    
    For display, the 16-bit results are converted to 8-bit.
    
    When Save is clicked, the GUI returns:
      (x_offset, y_offset, ch1_low, ch1_high, ch2_low, ch2_high, channel1_visible, channel2_visible)
    """
    root = tk.Tk()
    root.title("Custom Alignment Merge")
    
    # Offset variables
    x_offset = tk.IntVar(value=0)
    y_offset = tk.IntVar(value=0)
    
    # Visibility checkboxes for channels
    channel1_visible = tk.BooleanVar(value=True)
    channel2_visible = tk.BooleanVar(value=True)
    
    # Low/High variables for each channel (default: full range)
    ch1_low = tk.IntVar(value=0)
    ch1_high = tk.IntVar(value=65535)
    ch2_low = tk.IntVar(value=0)
    ch2_high = tk.IntVar(value=65535)
    
    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.TOP, fill=tk.X)
    preview_frame = tk.Frame(root)
    preview_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    
    # Offsets sliders
    x_slider = tk.Scale(control_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                        label="X Offset", variable=x_offset)
    x_slider.grid(row=0, column=0, padx=5, pady=5)
    y_slider = tk.Scale(control_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                        label="Y Offset", variable=y_offset)
    y_slider.grid(row=0, column=1, padx=5, pady=5)
    
    # Checkboxes for channel display
    channel1_cb = tk.Checkbutton(control_frame, text="Show Channel 1", variable=channel1_visible)
    channel1_cb.grid(row=0, column=2, padx=5, pady=5)
    channel2_cb = tk.Checkbutton(control_frame, text="Show Channel 2", variable=channel2_visible)
    channel2_cb.grid(row=0, column=3, padx=5, pady=5)
    
    # Frame for low/high adjustments per channel
    adjust_frame = tk.Frame(control_frame)
    adjust_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5)
    
    # Channel 1 low/high sliders
    ch1_low_slider = tk.Scale(adjust_frame, from_=0, to=65535, orient=tk.HORIZONTAL,
                              label="Ch1 Low", variable=ch1_low, resolution=1, length=200)
    ch1_low_slider.grid(row=0, column=0, padx=5, pady=5)
    ch1_high_slider = tk.Scale(adjust_frame, from_=0, to=65535, orient=tk.HORIZONTAL,
                               label="Ch1 High", variable=ch1_high, resolution=1, length=200)
    ch1_high_slider.set(65535)
    ch1_high_slider.grid(row=0, column=1, padx=5, pady=5)
    
    # Channel 2 low/high sliders
    ch2_low_slider = tk.Scale(adjust_frame, from_=0, to=65535, orient=tk.HORIZONTAL,
                              label="Ch2 Low", variable=ch2_low, resolution=1, length=200)
    ch2_low_slider.grid(row=0, column=2, padx=5, pady=5)
    ch2_high_slider = tk.Scale(adjust_frame, from_=0, to=65535, orient=tk.HORIZONTAL,
                               label="Ch2 High", variable=ch2_high, resolution=1, length=200)
    ch2_high_slider.set(65535)
    ch2_high_slider.grid(row=0, column=3, padx=5, pady=5)
    
    preview_label = tk.Label(preview_frame)
    preview_label.pack(padx=5, pady=5)
    
    def update_preview(*args):
        x = x_offset.get()
        y = y_offset.get()
        # For channel 1, shift then apply range adjustment if visible.
        if channel1_visible.get():
            shifted_img1 = shift_image(img1, x, y)
            adjusted_ch1 = adjust_image_range_16bit(shifted_img1, ch1_low.get(), ch1_high.get())
        else:
            adjusted_ch1 = np.zeros_like(img1)
        # For channel 2, apply range adjustment if visible.
        if channel2_visible.get():
            adjusted_ch2 = adjust_image_range_16bit(img2, ch2_low.get(), ch2_high.get())
        else:
            adjusted_ch2 = np.zeros_like(img2)
        
        # Convert to 8-bit for display.
        disp_ch1 = convert_16bit_to_8bit(adjusted_ch1)
        disp_ch2 = convert_16bit_to_8bit(adjusted_ch2)
        
        composite = np.stack([disp_ch1, disp_ch2, np.zeros_like(disp_ch1)], axis=2)
        composite_img = Image.fromarray(np.uint8(composite))
        imgtk = ImageTk.PhotoImage(image=composite_img)
        preview_label.config(image=imgtk)
        preview_label.image = imgtk
    
    # Bind changes to update preview.
    x_slider.config(command=lambda val: update_preview())
    y_slider.config(command=lambda val: update_preview())
    ch1_low_slider.config(command=lambda val: update_preview())
    ch1_high_slider.config(command=lambda val: update_preview())
    ch2_low_slider.config(command=lambda val: update_preview())
    ch2_high_slider.config(command=lambda val: update_preview())
    channel1_visible.trace_add("write", lambda *args: update_preview())
    channel2_visible.trace_add("write", lambda *args: update_preview())
    
    update_preview()
    
    def save_and_close():
        root.quit()
        root.destroy()
    
    save_button = tk.Button(control_frame, text="Save", command=save_and_close)
    save_button.grid(row=0, column=4, padx=5, pady=5)
    
    root.mainloop()
    return (x_offset.get(), y_offset.get(),
            ch1_low.get(), ch1_high.get(),
            ch2_low.get(), ch2_high.get(),
            channel1_visible.get(), channel2_visible.get())

def custom_align_merge(input_dir, subdir, output_dir):
    """
    Similar to make_multi_channels, but for each merged image (with at least two channels),
    a GUI is presented to fine tune the alignment between the first and second channels.
    The GUI allows you to set:
      - x/y offsets for the first channel,
      - a low/high intensity range for each channel,
      - and toggling channel visibility.
      
    The final merged multi-channel image (in 16-bit) is then saved.
    """
    subdir_path = os.path.join(input_dir, subdir)
    merged_subdir = os.path.join(output_dir, subdir)
    os.makedirs(merged_subdir, exist_ok=True)

    tif_files = [f for f in os.listdir(subdir_path)
                 if f.lower().endswith('03.tif') and not f.startswith('.') and "Bright" in f]

    pairs = {}
    for file in tif_files:
        well_name, position = extract_well_info(file)
        if well_name and position:
            parts = file.split('_')
            step = parts[-1].split('.')[0]
            output_name = f"{well_name}_{position}_merged_{step}.tif"
            if output_name not in pairs:
                pairs[output_name] = []
            pairs[output_name].append(os.path.join(subdir_path, file))

    for output_name, file_paths in pairs.items():
        if len(file_paths) >= 2:
            imgs = []
            crop_size = 1992
            for img_path in file_paths[:2]:
                try:
                    img = tiff.imread(img_path)
                    if img.shape[0] > crop_size or img.shape[1] > crop_size:
                        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
                        start_x = center_x - (crop_size // 2)
                        start_y = center_y - (crop_size // 2)
                        img = img[start_y:start_y + crop_size, start_x:start_x + crop_size]
                    imgs.append(img)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            if len(imgs) >= 2:
                print(f"Custom aligning {output_name} ...")
                params = alignment_gui(imgs[0], imgs[1])
                (x_off, y_off, ch1_low_val, ch1_high_val, ch2_low_val, ch2_high_val, ch1_vis, ch2_vis) = params
                shifted_img1 = shift_image(imgs[0], x_off, y_off)
                if ch1_vis:
                    merged_ch1 = adjust_image_range_16bit(shifted_img1, ch1_low_val, ch1_high_val)
                else:
                    merged_ch1 = np.zeros_like(imgs[0])
                if ch2_vis:
                    merged_ch2 = adjust_image_range_16bit(imgs[1], ch2_low_val, ch2_high_val)
                else:
                    merged_ch2 = np.zeros_like(imgs[1])
                
                # If additional channels exist, append them unmodified.
                if len(imgs) > 2:
                    merged_img = np.stack([merged_ch1, merged_ch2] + imgs[2:], axis=0)
                else:
                    merged_img = np.stack([merged_ch1, merged_ch2], axis=0)
                output_path = os.path.join(merged_subdir, output_name)
                tiff.imwrite(output_path, merged_img, imagej=True)
                print(f"Saved custom aligned merged image: {output_path}")
        else:
            print(f"Skipping {output_name} as it does not have at least two channels.")
    return subdir

def make_multi_channels_reorder(input_dir, subdir, output_dir, num_workers=4):
    if input_dir != output_dir:
        print(f"make multi channels {subdir}")
        subdir_path = os.path.join(input_dir, subdir)
        merged_subdir = os.path.join(output_dir, subdir)
    
        os.makedirs(merged_subdir, exist_ok=True)
        existing_merged_files = [img for img in os.listdir(merged_subdir)
                if img.endswith('.tif') and not any(suffix in img for suffix in ['masks.tif', 'dP.tif', 'outlines.tif', 'flows.tif', 'Wells'])]

        tif_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.tif') and not f.startswith('.') and "Bright" in f]

        pairs = {}
        for file in tif_files:
            well_name, position = extract_well_info(file)
            if well_name and position:
                parts = file.split('_')
                step = parts[-1].split('.')[0]
                output_name = f"{well_name}_{position}_merged_{step}.tif"
                if output_name not in pairs:
                    pairs[output_name] = []
                pairs[output_name].append(os.path.join(subdir_path, file))
     
        pairs = {key: [value[1], value[2], value[0]] for key, value in pairs.items()}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for output_name, file_paths in pairs.items():
                if output_name not in existing_merged_files:
                    futures.append(executor.submit(process_images, file_paths, merged_subdir, output_name))
            for future in futures:
                future.result()

    return subdir

def make_multi_channels_reorder_custom(input_dir, subdir, output_dir, num_workers=4):
    if input_dir != output_dir:
        print(f"make multi channels {subdir}")
        subdir_path = os.path.join(input_dir, subdir)
        merged_subdir = os.path.join(output_dir, subdir)
    
        os.makedirs(merged_subdir, exist_ok=True)
        existing_merged_files = [img for img in os.listdir(merged_subdir)
                if img.endswith('.tif') and not any(suffix in img for suffix in ['masks.tif', 'dP.tif', 'outlines.tif', 'flows.tif', 'Wells'])]

        tif_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.tif') and not f.startswith('.') and "Bright" in f and ("Z1" in f or "Z2" in f or "Z3" in f)]

        pairs = {}
        for file in tif_files:
            well_name, position = extract_well_info(file)
            if well_name and position:
                parts = file.split('_')
                step = parts[-1].split('.')[0]
                output_name = f"{well_name}_{position}_merged_{step}.tif"
                if output_name not in pairs:
                    pairs[output_name] = []
                pairs[output_name].append(os.path.join(subdir_path, file))
     
        pairs = {key: [value[1], value[2], value[0]] for key, value in pairs.items()}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for output_name, file_paths in pairs.items():
                if output_name not in existing_merged_files:
                    futures.append(executor.submit(process_images, file_paths, merged_subdir, output_name))
            for future in futures:
                future.result()

    return subdir

# if __name__ == "__main__":
    # For custom alignment merge with GUI fine-tuning (using low/high mapping for 16-bit images):
    # custom_align_merge(input_directory, "", output_directory)
