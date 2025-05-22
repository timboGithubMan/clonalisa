import os
import make_multi_channels
import omnipose_threaded
import process_masks
import time
import threading
import queue
from importlib.resources import files
import measure_growth_rates

def producer(inputDir, subdir, outputDir, queue):
    subdir = make_multi_channels.make_multi_channels_reorder(inputDir, subdir, outputDir, 4)
    queue.put(subdir)

def producer_worker(inputDir, outputDir, dir_queue):
    subdirs = [d for d in os.listdir(inputDir) if os.path.isdir(os.path.join(inputDir, d)) and not "epoch" in d]
    
    for subdir in subdirs:
        while dir_queue.full():
            time.sleep(0.25)

        producer(inputDir, subdir, outputDir, dir_queue)
    
    dir_queue.put(None)

def consumer(outputDir, lock, queue):
    while True:
        subdir = queue.get()
        if subdir is None:
            break

        merged_img_dir = os.path.join(outputDir, subdir)
        dead_dir = None
        for model_info_array in PRETRAINED_MODEL_INFOS:
            try:
                live_dir = omnipose_threaded.run_omnipose(merged_img_dir, model_info_array, num_threads=8)
                results_csv = process_masks.process_mask_files(live_dir, CM_PER_PIXEL, PLATE_AREAS.get(PLATE_TYPE), force_save=False, filter_min_size=None)
            except Exception as e:
                print(f"Error processing directory {merged_img_dir} with {model_info_array}: {e}")
        if DEAD_MODEL_INFO:
            try:
                dead_dir = omnipose_threaded.run_omnipose(merged_img_dir, DEAD_MODEL_INFO[0], num_threads=8)
                results_csv = process_masks.process_mask_files(dead_dir, CM_PER_PIXEL, PLATE_AREAS.get(PLATE_TYPE), force_save=False, filter_min_size=None)
            except Exception as e:
                print(f"Error processing directory {merged_img_dir} with dead model: {e}")

        queue.task_done()

PLATE_TYPE = "96W"
MAGNIFICATION = "10x"
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


PRETRAINED_MODEL_INFOS = [
    [str(files("clonalisa").joinpath("omnipose_models") / r"10x_NPC_nclasses_2_nchan_3_dim_2_2024_03_29_02_03_10.875324_epoch_960"), 0.4, 0],
]

DEAD_MODEL_INFO = [
    [str(files("clonalisa").joinpath("omnipose_models") / r"dead_nclasses_2_nchan_3_dim_2_ded_2025_03_07_02_10_15.252341_epoch_3999"), 0.4, 0],
]
    
def main():
    inputDirs = [r"E:\MERGED_20250515_RGF_ISO_SAG_GROWTH_ROUND2"]

    for inputDir in inputDirs:
        try:
            print(f"starting {inputDir}")
            base_dir = r"E:/"
            output_dir = make_multi_channels.create_merged_directory(inputDir, base_dir)

            # max_queue_size=999
            # dir_queue = queue.Queue(maxsize=max_queue_size)
            # lock = threading.Lock()

            # consumer_threads = []  # List to keep track of threads
            # for _ in range(1):  # Create and start consumer threads
            #     consumer_thread = threading.Thread(target=consumer, args=(output_dir, lock, dir_queue))
            #     consumer_thread.start()
            #     consumer_threads.append(consumer_thread)

            # producer_thread = threading.Thread(target=producer_worker, args=(inputDir, output_dir, dir_queue))
            # producer_thread.start()
            # producer_thread.join()

            # for _ in range(len(consumer_threads)):
            #     dir_queue.put(None)

            # for consumer_thread in consumer_threads:
            #     consumer_thread.join()

            for model_info_array in PRETRAINED_MODEL_INFOS:
                model_name = "_".join(model_info_array[0].split("_")[-8:])

                dead_model_name = "_".join(DEAD_MODEL_INFO[0][0].split("_")[-8:]) if DEAD_MODEL_INFO else None
                model_output_dir = os.path.join(os.path.join(output_dir,f'{model_name}'))

                all_data_csv = process_masks.make_all_data_csv(output_dir, model_name)
                per_well_data_csv_path = measure_growth_rates.calculate_growth_rates_per_well(all_data_csv, time_lower=80, time_upper=100)
                group_columns =  [col for col in pd.read_csv(per_well_data_csv_path).columns if ("Group" in col and not "_Group" in col and not "merged" in col)]

                import pandas as pd
                per_well_data_csv_path = os.path.join(model_output_dir, f'{model_name}_per_well_data.csv')
                group_columns =  [col for col in pd.read_csv(os.path.join(model_output_dir, f'{model_name}_per_well_data.csv')).columns if ("Group" in col and not "_Group" in col and not "merged" in col)]

                for group_col in group_columns:
                    measure_growth_rates.plot_over_time(per_well_data_csv_path, group_to_plot=group_col, y_values="exp_rate_cell_density", p_values_col="exp_rate_cell_density", log=True)
                    measure_growth_rates.plot_over_time(per_well_data_csv_path, group_to_plot=group_col, y_values="logistic_k_corr_fit_cell_density", p_values_col="logistic_k_corr_fit_cell_density", log=True)
                    measure_growth_rates.plot_over_time(per_well_data_csv_path, group_to_plot=group_col, y_values="logistic_k_CV_fit_cell_density", p_values_col="logistic_k_CV_fit_cell_density", log=True)
                    measure_growth_rates.plot_over_time(per_well_data_csv_path, group_to_plot=group_col, y_values="cell_density", p_values_col="cell_density", log=True)

        except:
            print(f'failed {inputDir}')


if __name__ == "__main__":
    main()