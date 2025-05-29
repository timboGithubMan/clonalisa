import os
import omnipose_threaded
import process_masks
from importlib.resources import files

PLATE_TYPE = "96W"
MAGNIFICATION = "10x"
CYTATION = True
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

PRETRAINED_MODEL_INFOS = [
    [str(files("clonalisa").joinpath("omnipose_models") /
         "10x_NPC_nclasses_2_nchan_3_dim_2_2024_03_29_02_03_10.875324_epoch_960"), 0.4, 0],
]

DEAD_MODEL_INFO = [
    [str(files("clonalisa").joinpath("omnipose_models") /
         "dead_nclasses_2_nchan_3_dim_2_ded_2025_03_07_02_10_15.252341_epoch_3999"), 0.4, 0],
]

CHANNEL_ORDER = [1, 2, 0]


def process_directory(input_dir: str) -> None:
    subdirs = [d for d in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, d)) and "epoch" not in d]
    for subdir in subdirs:
        dir_path = os.path.join(input_dir, subdir)
        for model_info in PRETRAINED_MODEL_INFOS:
            live_dir = omnipose_threaded.run_omnipose(
                dir_path,
                model_info,
                num_threads=8,
                channel_order=CHANNEL_ORDER,
                save_flows=True,
            )
            process_masks.process_mask_files(
                live_dir,
                CM_PER_PIXEL,
                PLATE_AREAS.get(PLATE_TYPE),
                force_save=False,
                filter_min_size=None,
            )
        if DEAD_MODEL_INFO:
            dead_dir = omnipose_threaded.run_omnipose(
                dir_path,
                DEAD_MODEL_INFO[0],
                num_threads=8,
                channel_order=CHANNEL_ORDER,
            )
            process_masks.process_mask_files(
                dead_dir,
                CM_PER_PIXEL,
                PLATE_AREAS.get(PLATE_TYPE),
                force_save=False,
                filter_min_size=None,
            )


def main() -> None:
    input_dirs = [r"E:\\test"]
    for input_dir in input_dirs:
        try:
            print(f"starting {input_dir}")
            process_directory(input_dir)
        except Exception:
            print(f"failed {input_dir}")


if __name__ == "__main__":
    main()
