from __future__ import annotations
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
import math
import logging
from pathlib import Path
import image_processing
import os
import glob
from datetime import datetime
import shutil
from typing import Dict, List, Tuple
from scipy.stats import sem
from math import ceil, sqrt
from matplotlib.colors import LogNorm
from natsort import natsorted
from matplotlib.colors import ListedColormap


def plot_all_plate_visualizations(input_csv, tp1, min_cells_tp1, tp2, min_cells_tp2):
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Filter rows for each timepoint based on the min cells threshold.
    df_tp1 = df[(df["Relative Time (days)"] == tp1) & (df["num_cells"] >= min_cells_tp1)]
    df_tp2 = df[(df["Relative Time (days)"] == tp2) & (df["num_cells"] >= min_cells_tp2)]
    
    # Get unique plates from the data
    plates = df["Plate"].unique()
    n_plates = len(plates)
    
    # Define the layout for a 96-well plate.
    plate_rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
    plate_cols = list(range(1, 13))
    
    # We want 3 plates per column, so set n_rows to 3 and compute n_cols.
    n_rows = 3
    n_cols = math.ceil(n_plates / n_rows)
    
    # Create a figure with a grid of subplots.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    
    # Ensure axes is a 2D array.
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = np.atleast_2d(axes)
    
    # Flatten the axes array for easy indexing.
    axes_flat = axes.flatten()
    
    # Loop over each plate and plot it in the corresponding subplot.
    for i, plate in enumerate(plates):
        ax = axes_flat[i]
        # Loop over each well position on the plate.
        for j, row_letter in enumerate(plate_rows):
            # Compute y so that row A is at the top.
            y = (len(plate_rows) - 1) - j
            for col in plate_cols:
                x = col - 1  # 0-indexed x coordinate
                well = f"{row_letter}{col}"
                
                # Count colonies in this well for each timepoint.
                count_tp1 = len(df_tp1[(df_tp1["Plate"] == plate) & (df_tp1["well"] == well)])
                count_tp2 = len(df_tp2[(df_tp2["Plate"] == plate) & (df_tp2["well"] == well)])
                
                # Determine well color based on criteria:
                # 1. If there is more than 1 colony at the first timepoint -> blue.
                if count_tp1 > 1:
                    well_color = '#FF7F6A'
                # 2. If there is exactly 1 colony at tp1:
                elif count_tp1 == 1:
                    if count_tp2 > 1:
                        well_color = 'orange'   # 1 colony at tp1 but more than 1 at tp2.
                    elif count_tp2 == 1:
                        well_color = '#98DFAF'
                    elif count_tp2 == 0:
                        well_color = 'purple'
                    else:
                        well_color = 'white'
                # 3. Default: no colony at tp1 -> white.
                else:
                    well_color = 'white'
                
                # Draw the well rectangle.
                rect = patches.Rectangle((x, y), 1, 1, facecolor=well_color, edgecolor='black')
                ax.add_patch(rect)
                # Optionally, display colony counts.
                ax.text(x + 0.5, y + 0.5, f"{count_tp1}/{count_tp2}",
                        ha='center', va='center', fontsize=8)
        
        # Configure subplot axes to mimic the 96-well grid.
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.set_xticks(np.arange(12) + 0.5)
        ax.set_yticks(np.arange(8) + 0.5)
        ax.set_xticklabels(plate_cols)
        ax.set_yticklabels(list(reversed(plate_rows)))
        ax.set_aspect('equal')
        ax.set_title(f"Plate {plate}")
    
    # Turn off any unused subplots if there are fewer plates than grid slots.
    for j in range(i + 1, n_rows * n_cols):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    # Save the combined figure to one file.
    output_filename = input_csv.replace(".csv", "_all_plate_results.png")
    plt.savefig(output_filename)
    plt.close()

def plot_plate_colony_percentages(csv_path, output_folder=None):
    """
    For each unique plate in the CSV file, create a scatter plot (with connecting lines)
    of "Percentage of Total Colony Count" vs "Relative Time (days)". The three series are:
      - Single Undifferentiated (green)
      - Differentiated (orange)
      - Dead (purple)
    
    Baseline: For every plate, only consider wells that had exactly one colony at day 0,
              where that colony had > 5 cells.
              At day 0, these wells are set to 100% Single Undifferentiated, 0% Differentiated,
              and 0% Dead.
    
    For subsequent days, for each valid well:
      - If no colony is detected (i.e. count==0), it is classified as "Dead".
      - If exactly one colony is detected:
            * Compare the current colony's num_cells to that from the previous day.
              If the current num_cells is ≤ 80% of the previous day's, mark as "Dead".
            * Else if the colony's density exceeds 0.044, mark as "Differentiated".
            * Otherwise, classify as "Single Undifferentiated".
      - If more than one colony is detected, classify the well as "Differentiated".
    
    The percentages are calculated relative to the total number of valid wells for that plate.
    
    Args:
        csv_path (str): Path to the CSV file containing colony metrics. 
                        The CSV must have columns "Plate", "well", "Relative Time (days)",
                        "num_cells", and "density".
        output_folder (str, optional): Folder where the plots will be saved.
                                       If None, plots will be saved in the same directory as csv_path.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Read the CSV and ensure the "Relative Time (days)" column is integer type.
    df = pd.read_csv(csv_path)
    df['Relative Time (days)'] = df['Relative Time (days)'].astype(int)

    # Determine valid wells: those with exactly one colony at day 0 that has > 5 cells.
    baseline_df = df[df['Relative Time (days)'] == 0]
    valid_baseline = baseline_df.groupby(['Plate', 'well']).filter(
        lambda group: (len(group) == 1) and (group.iloc[0]['num_cells'] > 3)
    )
    
    # Create a mapping from Plate to the set of valid wells.
    valid_wells_dict = {}
    for _, row in valid_baseline.iterrows():
        plate = row['Plate']
        well = row['well']
        valid_wells_dict.setdefault(plate, set()).add(well)

    # Get the complete set of days present in the CSV.
    days = sorted(df['Relative Time (days)'].unique())

    # For each plate, compute the state counts per day.
    for plate, wells in valid_wells_dict.items():
        # Initialize a dictionary: key = day, value = dict with counts for each category.
        day_category_counts = {day: {'Single Undifferentiated': 0,
                                     'Differentiated': 0,
                                     'Dead': 0} for day in days}
        
        # Process each valid well.
        for well in wells:
            for day in days:
                # For day 0, all valid wells are "Single Undifferentiated".
                if day == 0:
                    current = df[(df['Plate'] == plate) &
                        (df['well'] == well) &
                        (df['Relative Time (days)'] == day)]
                    if current['dead_live_ratio_proximity'].item() > 0.27:
                        category = 'Dead'
                    else:
                        category = 'Single Undifferentiated'
                else:
                    # Retrieve all colony entries for the given plate, well, and day.
                    current = df[(df['Plate'] == plate) &
                                 (df['well'] == well) &
                                 (df['Relative Time (days)'] == day)]
                    if current.empty:
                        category = 'Dead'
                    elif len(current) == 1:
                        row_current = current.iloc[0]
                        # Attempt to retrieve the previous day's measurement.
                        prev = df[(df['Plate'] == plate) &
                                  (df['well'] == well) &
                                  (df['Relative Time (days)'] == (day - 1))]
                        if not prev.empty and len(prev) == 1:
                            prev_num_cells = prev.iloc[0]['num_cells']
                        else:
                            prev_num_cells = None
                        # Check if the colony's cell count dropped by 10% or more from the previous day.
                        if prev_num_cells is not None and row_current['num_cells'] <= 0.9 * prev_num_cells:
                            category = 'Dead'
                        if row_current['dead_live_ratio_proximity'] > 0.27:
                            category = 'Dead'
                        # Check for differentiation based on density.
                        elif row_current['reach_p99'] > 24.08027:
                            category = 'Differentiated'
                        else:
                            category = 'Single Undifferentiated'
                    else:
                        # More than one colony detected implies differentiation.
                        category = 'Differentiated'
                day_category_counts[day][category] += 1

        # Calculate percentages (per day) relative to the total number of valid wells.
        total_wells = len(wells)
        days_list = []
        single_pct = []
        diff_pct = []
        dead_pct = []
        for day in days:
            days_list.append(day)
            single_pct.append(day_category_counts[day]['Single Undifferentiated'] / total_wells * 100)
            diff_pct.append(day_category_counts[day]['Differentiated'] / total_wells * 100)
            dead_pct.append(day_category_counts[day]['Dead'] / total_wells * 100)
        
        # Create a scatter plot with lines connecting the points.
        plt.figure(figsize=(8, 6))
        plt.scatter(days_list, single_pct, color='#98DFAF', label='Single Undifferentiated')
        plt.plot(days_list, single_pct, color='#98DFAF')
        plt.scatter(days_list, diff_pct, color='orange', label='Differentiated')
        plt.plot(days_list, diff_pct, color='orange')
        plt.scatter(days_list, dead_pct, color='purple', label='Dead')
        plt.plot(days_list, dead_pct, color='purple')
        plt.xlabel('Relative Time (days)')
        plt.ylabel('Percentage of Total Colony Count (%)')
        plt.title(f'Plate {plate}: Colony State Percentages Over Time')
        plt.legend()
        plt.ylim(0, 110)
        plt.grid(True)
        
        # Determine output folder.
        if output_folder is None:
            output_folder = os.path.dirname(csv_path)
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f'plate_{plate}_colony_percentages.png')
        plt.savefig(output_file)
        plt.close()
        print(f"Saved plot for Plate {plate} to {output_file}")

def plot_metric_with_sem(csv_path, output_folder=None):
    # Load CSV and ensure "Relative Time (days)" is an integer.
    df = pd.read_csv(csv_path)
    df['Relative Time (days)'] = df['Relative Time (days)'].astype(int)

    # Identify valid wells based on baseline (day 0): exactly one colony with > 5 cells.
    baseline_df = df[df['Relative Time (days)'] == 0]
    valid_baseline = baseline_df.groupby(['Plate', 'well']).filter(
        lambda group: (len(group) == 1) and (group.iloc[0]['num_cells'] > 5)
    )
    
    # Build a mapping from Plate to its set of valid wells.
    valid_wells_dict = {}
    for _, row in valid_baseline.iterrows():
        plate = row['Plate']
        well = row['well']
        valid_wells_dict.setdefault(plate, set()).add(well)

    # Group data by Plate, well, and day, summing num_cells so that each record represents 
    # the total cells for that well at that day.
    # grouped = df.groupby(['Plate', 'well', 'Relative Time (days)'])['num_cells'].sum().reset_index()
    grouped = (
        df
        .groupby(['Plate', 'well', 'Relative Time (days)'])
        .agg(
            num_cells=('num_cells', 'sum'),
            num_dead=('num_dead_proximity', 'sum')
        )
        .assign(dead_fraction=lambda g: g.num_dead / g.num_cells)  # or *100 for %
        .reset_index()
    )
    # For each plate, create a dictionary mapping day -> list of total cell counts (excluding 0 or NaN)
    plate_day_data = {}
    for plate, wells in valid_wells_dict.items():
        plate_day_data.setdefault(plate, {})
        for well in wells:
            well_data = grouped[(grouped['Plate'] == plate) & (grouped['well'] == well)]
            for _, row in well_data.iterrows():
                day = row['Relative Time (days)']
                total_cells = row['dead_fraction']
                # Exclude invalid values (0 or NaN)
                if pd.notnull(total_cells) and total_cells > 0:
                    plate_day_data[plate].setdefault(day, []).append(total_cells)

    # Prepare the plot.
    plt.figure(figsize=(10, 8))
    for plate, day_dict in plate_day_data.items():
        valid_days = []
        means = []
        sems = []
        for day in sorted(day_dict.keys()):
            values = np.array(day_dict[day])
            if len(values) == 0:
                continue  # Skip days with no valid measurements
            mean_val = values.mean()
            sem_val = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
            valid_days.append(day)
            means.append(mean_val)
            sems.append(sem_val)
        if valid_days:
            plt.errorbar(valid_days, means, yerr=sems, marker='o', capsize=5, label=f"Plate {plate}")

    plt.xlabel('Day')
    plt.ylabel('Cells Per Colony')
    plt.title('Cells Per Colony Over Time')
    # plt.yscale('log')  # Logarithmic y-axis
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Determine the output folder.
    if output_folder is None:
        output_folder = os.path.dirname(csv_path)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'plate_total_cells_mean_sem_log.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Saved total cells plot with mean +/- SEM to {output_file}")
    
def plot_all_histograms(csv_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Ensure faceting columns are treated as categorical
    df['Plate'] = df['Plate'].astype(str)
    df['Relative Time (days)'] = df['Relative Time (days)'].astype(str)
    
    # List of variables for which we want to create histograms
    variables = ['num_cells', 'density', 'mean_nnd', 'max_nnd', 'std_nnd']
    
    for var in variables:
        # Filter for positive values since log scale doesn't support zero or negatives.
        positive_values = df[var][df[var] > 0]
        # If there are no positive values, skip plotting this variable.
        if positive_values.empty:
            continue
        x_min = positive_values.min()
        x_max = positive_values.max()
        # Create logarithmically spaced bins (30 bins here; adjust as needed)
        bins = np.logspace(np.log10(x_min), np.log10(x_max), 50)
        
        # Create a FacetGrid with Plate as rows and Relative Time (days) as columns.
        g = sns.FacetGrid(df, row="Plate", col="Relative Time (days)",
                          margin_titles=True, sharex=True, sharey=False, aspect=2)
        
        # Map a histogram of the variable onto each facet using the common bins.
        g.map(plt.hist, var, bins=bins, color="steelblue", edgecolor="black")
        
        # Set both x and y axes to log scale for each subplot.
        for ax in g.axes.flat:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(bins[0], bins[-1])
        
        # Add axis labels and facet titles.
        g.set_axis_labels(f"{var} (log scale)", "Count (log scale)")
        g.set_titles(row_template='Plate: {row_name}', col_template='Relative Time (days): {col_name}')
        
        # Adjust layout to ensure labels and titles are clearly visible.
        plt.tight_layout()
        
        # Save the plot as an image file. The filename includes the variable name.
        output_filename = csv_path.replace(".csv", f"_{var}_histogram.png")
        plt.savefig(output_filename)
        plt.close()

def plot_plate_diagrams_by_day(input_csv, ground_truth_csv=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Load simulation CSV file.
    df = pd.read_csv(input_csv)
    
    # Identify unique days and plates from simulation data.
    days = sorted(df["Relative Time (days)"].unique())
    plates = sorted(df["Plate"].unique())
    # plates = sorted(df["Plate"].unique())[:4]
    first_day = days[0]
    
    # Precompute wells with colonies on the first day for simulation.
    df_first = df[df["Relative Time (days)"] == first_day]
    first_day_wells_sim = {
        plate: set(df_first[df_first["Plate"] == plate]["well"].unique())
        for plate in plates
    }
    
    # Load ground truth CSV and precompute first-day wells if provided.
    if ground_truth_csv:
        gt_df = pd.read_csv(ground_truth_csv)
        gt_df.loc[gt_df["Status"] == -1, "Status"] = 1
        gt_df["Plate"] = gt_df["Plate"].astype(str)
        gt_df["Day"] = gt_df["Day"].astype(str)
        gt_days = sorted(gt_df["Day"].unique())
        first_day_gt = gt_days[0]
        # Only include wells where Status != 0 for dead-colony inference.
        first_day_wells_gt = {
            plate: set(gt_df[(gt_df["Plate"] == plate) &
                             (gt_df["Day"] == first_day_gt) &
                             (gt_df["Status"] != 0)]["Well"].unique())
            for plate in plates
        }
        # --- NEW: Append simulation prediction to ground truth data ---
        def compute_sim_pred(row):
            plate = str(row["Plate"])
            day = row["Day"]
            # If day comes from ground truth as a string, it might be converted to a float/int as needed.
            try:
                day_val = float(day)
            except Exception:
                day_val = 0
            well = row["Well"]
            # Filter simulation data to the matching plate, day, and well.
            df_subset = df[(df["Plate"].astype(str) == plate) &
                           (df["Relative Time (days)"] == float(day)) &  # ensure numeric match
                           (df["well"] == well)]
            count = len(df_subset)
            if count > 1:
                return 2
            elif count == 1:
                density = df_subset.iloc[0]["reach_p99"] if "reach_p99" in df_subset.columns else 0
                # For day 0, always use 1; for later days use density to decide.
                return 3 if (density > 24.08027 and day_val > 0) else 1
            else:
                return 0
        
        gt_df["simulated_prediction"] = gt_df.apply(compute_sim_pred, axis=1)
        output_csv = ground_truth_csv.replace(".csv", "_with_simulated_prediction.csv")
        gt_df.to_csv(output_csv, index=False)
    else:
        gt_df = None

    # Define the 96-well layout.
    plate_rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
    plate_cols = list(range(1, 13))
    
    # Determine subplot grid dimensions.
    n_cols = len(plates)
    n_rows = 5 if gt_df is not None else 1
    
    # Loop over each day and generate the diagrams.
    for day in days:
        # Filter simulation data for the current day.
        df_day = df[df["Relative Time (days)"] == day]
        # (The filtering on num_cells remains the same.)
        if str(day) == "0":
            df_day = df_day[df_day["num_cells"] > 2]
        else:
            df_day = df_day[df_day["num_cells"] > 2]
        # Create subplots.
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])
        
        # --- Row 1: Simulation diagram ---
        for i, plate in enumerate(plates):
            ax_sim = axes[1, i]
            for j, row_letter in enumerate(plate_rows):
                y = len(plate_rows) - 1 - j  # so that row A is at the top
                for col in plate_cols:
                    x = col - 1  # 0-indexed
                    well = f"{row_letter}{col}"
                    df_well = df_day[(df_day["Plate"] == plate) & (df_day["well"] == well)]
                    count = len(df_well)
                    
                    if count > 1:
                        well_color = '#FF7F6A'
                    elif count == 1:
                        density = df_well.iloc[0]["reach_p99"] if "reach_p99" in df_well.columns else 0
                        if density > 24.08027 and day > 0:
                            well_color = 'orange'
                        else:
                            well_color = '#98DFAF'
                    else:
                        well_color = 'white'
                    
                    rect = patches.Rectangle((x, y), 1, 1, facecolor=well_color, edgecolor='black')
                    ax_sim.add_patch(rect)
            
            ax_sim.set_xlim(0, 12)
            ax_sim.set_ylim(0, 8)
            ax_sim.set_xticks(np.arange(12) + 0.5)
            ax_sim.set_yticks(np.arange(8) + 0.5)
            ax_sim.set_xticklabels(plate_cols)
            ax_sim.set_yticklabels(list(reversed(plate_rows)))
            ax_sim.set_aspect('equal')
            ax_sim.set_title(f"Plate {plate} - ClonaLiSA")
        
        if gt_df is not None:
            # --- Row 0: Ground Truth diagram ---
            for i, plate in enumerate(plates):
                ax_gt = axes[0, i]
                df_well_day = gt_df[(gt_df["Plate"] == str(plate)) & (gt_df["Day"] == str(day))]
                for j, row_letter in enumerate(plate_rows):
                    y = len(plate_rows) - 1 - j
                    for col in plate_cols:
                        x = col - 1
                        well = f"{row_letter}{col}"
                        df_well = df_well_day[df_well_day["Well"] == well]
                        if len(df_well) > 0:
                            status = int(df_well.iloc[0]["Status"])
                            if status == 0:
                                well_color = 'white'
                            elif status == 1:
                                well_color = '#98DFAF'
                            elif status == 2:
                                well_color = '#FF7F6A'
                            elif status == 3:
                                well_color = 'orange'
                            elif status == -1:
                                well_color = 'purple'
                        else:
                            well_color = 'white'
                        
                        rect = patches.Rectangle((x, y), 1, 1, facecolor=well_color, edgecolor='black')
                        ax_gt.add_patch(rect)
                
                ax_gt.set_xlim(0, 12)
                ax_gt.set_ylim(0, 8)
                ax_gt.set_xticks(np.arange(12) + 0.5)
                ax_gt.set_yticks(np.arange(8) + 0.5)
                ax_gt.set_xticklabels(plate_cols)
                ax_gt.set_yticklabels(list(reversed(plate_rows)))
                ax_gt.set_aspect('equal')
                ax_gt.set_title(f"Plate {plate} - Ground Truth")
            
            # --- Row 2: Error diagram ---
            for i, plate in enumerate(plates):
                ax_err = axes[2, i]
                # In this block, we now use the same simulation prediction logic as used in compute_sim_pred:
                for j, row_letter in enumerate(plate_rows):
                    y = len(plate_rows) - 1 - j
                    for col in plate_cols:
                        x = col - 1
                        well = f"{row_letter}{col}"
                        df_day["Plate"] = df_day["Plate"].astype(str)
                        df_well_sim = df_day[(df_day["Plate"] == str(plate)) & (df_day["well"] == well)]
                        count = len(df_well_sim)
                        if count > 1:
                            sim_status = 2
                        elif count == 1:
                            density = df_well_sim.iloc[0]["reach_p99"] if "reach_p99" in df_well_sim.columns else 0
                            sim_status = 3 if (density > 24.08027 and int(day) > 0) else 1
                        else:
                            sim_status = 0

                        # Determine ground truth status.
                        df_well_gt = gt_df[(gt_df["Plate"] == str(plate)) &
                                           (gt_df["Day"] == str(day)) &
                                           (gt_df["Well"] == well)]
                        if len(df_well_gt) > 0:
                            gt_status = int(df_well_gt.iloc[0]["Status"])
                            if gt_status == -1:
                                gt_color = 'purple'
                            elif gt_status == 1:
                                gt_color = '#98DFAF'
                            elif gt_status == 2:
                                gt_color = '#FF7F6A'
                            elif gt_status == 3:
                                gt_color = 'orange'
                            else:
                                gt_color = 'white'
                        else:
                            gt_status = 0
                            gt_color = 'white'
                        
                        # Color the well only if simulation and ground truth predictions do not match.
                        fill_color = gt_color if sim_status != gt_status else 'black'
                        rect = patches.Rectangle((x, y), 1, 1, facecolor=fill_color, edgecolor='white')
                        ax_err.add_patch(rect)
                
                ax_err.set_xlim(0, 12)
                ax_err.set_ylim(0, 8)
                ax_err.set_xticks(np.arange(12) + 0.5)
                ax_err.set_yticks(np.arange(8) + 0.5)
                ax_err.set_xticklabels(plate_cols)
                ax_err.set_yticklabels(list(reversed(plate_rows)))
                ax_err.set_aspect('equal')
                ax_err.set_title(f"Plate {plate} - Error")

            # # --- Row 2 (Alternative): Manual diagram ---
            # # (If you have a separate manual column and want to plot it, make sure to use a distinct axis, for example by adjusting n_rows.)
            # for i, plate in enumerate(plates):
            #     ax_manual = axes[2, i]  # Adjust the row index if needed.
            #     df_well_day = gt_df[(gt_df["Plate"] == str(plate)) & (gt_df["Day"] == str(day))]
            #     for j, row_letter in enumerate(plate_rows):
            #         y = len(plate_rows) - 1 - j
            #         for col in plate_cols:
            #             x = col - 1
            #             well = f"{row_letter}{col}"
            #             df_well = df_well_day[df_well_day["Well"] == well]
            #             if len(df_well) > 0 and not np.isnan(df_well.iloc[0]["manual"]):
            #                 status = int(df_well.iloc[0]["manual"])
            #                 if status == 0:
            #                     well_color = 'white'
            #                 elif status == 1:
            #                     well_color = '#98DFAF'
            #                 elif status == 2:
            #                     well_color = '#FF7F6A'
            #                 elif status == 3:
            #                     well_color = 'orange'
            #                 elif status == -1:
            #                     well_color = 'purple'
            #             else:
            #                 well_color = 'white'
                        
            #             rect = patches.Rectangle((x, y), 1, 1, facecolor=well_color, edgecolor='black')
            #             ax_manual.add_patch(rect)
                
            #     ax_manual.set_xlim(0, 12)
            #     ax_manual.set_ylim(0, 8)
            #     ax_manual.set_xticks(np.arange(12) + 0.5)
            #     ax_manual.set_yticks(np.arange(8) + 0.5)
            #     ax_manual.set_xticklabels(plate_cols)
            #     ax_manual.set_yticklabels(list(reversed(plate_rows)))
            #     ax_manual.set_aspect('equal')
            #     ax_manual.set_title(f"Plate {plate} - Manual")

            # --- Row 3: Confusion Matrix (ClonaLiSA) ---
            for i, plate in enumerate(plates):
                day_str = str(day)
                plate_str = str(plate)
                ax_cm = axes[3, i]
                if day_str == str(first_day):
                    num_classes = 3
                    class_labels = [0, 1, 2]
                    cm = np.zeros((num_classes, num_classes), dtype=int)
                else:
                    num_classes = 4
                    class_labels = [0, 1, 2, 3]
                    cm = np.zeros((num_classes, num_classes), dtype=int)
                                
                for row_letter in plate_rows:
                    for col in plate_cols:
                        well = f"{row_letter}{col}"
                        df_well_sim = df_day[(df_day["Plate"] == plate_str) & (df_day["well"] == well)]
                        count = len(df_well_sim)
                        if count > 1:
                            sim_status = 2
                        elif count == 1:
                            density = df_well_sim.iloc[0]["reach_p99"] if "reach_p99" in df_well_sim.columns else 0
                            sim_status = 3 if (density > 24.08027 and int(day) > 0) else 1
                        else:
                            sim_status = 0
                                    
                        df_well_gt = gt_df[(gt_df["Plate"] == plate_str) &
                                           (gt_df["Day"] == day_str) &
                                           (gt_df["Well"] == well)]
                        if len(df_well_gt) > 0:
                            gt_status = int(df_well_gt.iloc[0]["Status"])
                        else:
                            gt_status = 0
                                    
                        if day == first_day:
                            if gt_status == 3:
                                gt_status = 1
                            if sim_status == 3:
                                sim_status = 1
                                    
                        if gt_status in class_labels and sim_status in class_labels:
                            cm[gt_status, sim_status] += 1
                                    
                total = np.sum(cm)
                correct = np.trace(cm)
                overall_accuracy = correct / total if total > 0 else 1.0
                ax_cm.set_title(f"Overall Accuracy - {overall_accuracy:.0%}")
                
                # Normalize each row and plot the heatmap.
                cm_norm = np.zeros_like(cm, dtype=float)
                for r in range(num_classes):
                    row_sum = np.sum(cm[r, :])
                    if row_sum > 0:
                        cm_norm[r, :] = cm[r, :] / row_sum
                    else:
                        cm_norm[r, :] = 0
                im = ax_cm.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
                tick_marks = np.arange(num_classes)
                ax_cm.set_xticks(tick_marks)
                ax_cm.set_xticklabels(class_labels)
                ax_cm.set_yticks(tick_marks)
                ax_cm.set_yticklabels(class_labels)
                
                thresh = cm_norm.max() / 2.
                for m in range(cm_norm.shape[0]):
                    for n in range(cm_norm.shape[1]):
                        ax_cm.text(n, m, f"{cm_norm[m, n]:.0%}",
                                   ha="center", va="center",
                                   color="white" if cm_norm[m, n] > thresh else "black")
                ax_cm.set_ylabel('Ground Truth')
                ax_cm.set_xlabel('Predicted')

            # --- Row 4: Confusion Matrix (Manual) ---
            for i, plate in enumerate(plates):
                day_str = str(day)
                plate_str = str(plate)
                ax_cm = axes[4, i]
                if day_str == str(first_day):
                    num_classes = 3
                    class_labels = [0, 1, 2]
                    cm = np.zeros((num_classes, num_classes), dtype=int)
                else:
                    num_classes = 4
                    class_labels = [0, 1, 2, 3]
                    cm = np.zeros((num_classes, num_classes), dtype=int)
                                
                for row_letter in plate_rows:
                    for col in plate_cols:
                        well = f"{row_letter}{col}"
                        df_well_gt = gt_df[(gt_df["Plate"] == plate_str) &
                                           (gt_df["Day"] == day_str) &
                                           (gt_df["Well"] == well)]
                        if not df_well_gt.empty:
                            val = df_well_gt.iloc[0].get("manual", 0)
                            if pd.isnull(val):
                                manual_status = np.nan
                            else:
                                manual_status = int(val)
                        else:
                            manual_status = 0
                                    
                        if len(df_well_gt) > 0:
                            gt_status = int(df_well_gt.iloc[0]["Status"])
                        else:
                            gt_status = 0
                                    
                        if day == first_day:
                            if gt_status == 3:
                                gt_status = 1
                            if manual_status == 3:
                                manual_status = 1
                                    
                        if gt_status in class_labels and manual_status in class_labels:
                            cm[gt_status, manual_status] += 1
                                    
                total = np.sum(cm)
                correct = np.trace(cm)
                overall_accuracy = correct / total if total > 0 else 1.0
                ax_cm.set_title(f"Overall Accuracy - {overall_accuracy:.0%}")
                
                cm_norm = np.zeros_like(cm, dtype=float)
                for r in range(num_classes):
                    row_sum = np.sum(cm[r, :])
                    if row_sum > 0:
                        cm_norm[r, :] = cm[r, :] / row_sum
                    else:
                        cm_norm[r, :] = 0
                im = ax_cm.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
                tick_marks = np.arange(num_classes)
                ax_cm.set_xticks(tick_marks)
                ax_cm.set_xticklabels(class_labels)
                ax_cm.set_yticks(tick_marks)
                ax_cm.set_yticklabels(class_labels)
                
                thresh = cm_norm.max() / 2.
                for m in range(cm_norm.shape[0]):
                    for n in range(cm_norm.shape[1]):
                        ax_cm.text(n, m, f"{cm_norm[m, n]:.0%}",
                                   ha="center", va="center",
                                   color="white" if cm_norm[m, n] > thresh else "black")
                ax_cm.set_ylabel('Ground Truth')
                ax_cm.set_xlabel('Manual')
        
        fig.suptitle(f"Day: {day}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_filename = input_csv.replace(".csv", f"_day_{day}_plate_results.pdf")
        plt.savefig(output_filename)
        plt.close()

def plot_confusion_matrix_from_csv(csv_path, col1, col1_label, col2, col2_label):
    from sklearn.metrics import confusion_matrix
    """
    Reads a CSV file, computes and plots a normalized confusion matrix using two specified columns.

    Parameters:
        csv_path (str): Path to the CSV file.
        col1 (str): Name of the column to be used as ground truth.
        col1_label (str): Label for the ground truth axis.
        col2 (str): Name of the column to be used as predictions.
        col2_label (str): Label for the prediction axis.
    """
    # Read the CSV into a DataFrame.
    df = pd.read_csv(csv_path)
    # df = df[df["Day"] == 3]
    # df = df[df["Plate"] != 92]
    # df = df[df["Plate"] != 93]
    df.loc[df["semi"] == -1, "semi"] = np.nan
    df.loc[df["manual"] == -1, "manual"] = np.nan
    df.loc[df["semi"] > 2, "semi"] = 2
    df.loc[df["manual"] > 2, "manual"] = 2
    df = df.dropna(subset=[col1, col2])
    # df.loc[df["Status"] == 3, "Status"] = 1
    # df.loc[df["simulated_prediction"] == 3, "simulated_prediction"] = 1

    # Extract the two columns of interest.
    y_true = df[col1]
    y_pred = df[col2]

    # Determine all unique classes from the two columns and sort them.
    classes = sorted(set(y_true).union(set(y_pred)))

    # Compute the confusion matrix.
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Calculate overall accuracy.
    total = np.sum(cm)
    correct = np.trace(cm)
    overall_accuracy = correct / total if total > 0 else 1.0

    # Normalize the confusion matrix so that each row sums to 1.
    cm_norm = np.zeros_like(cm, dtype=float)
    for i in range(len(classes)):
        row_sum = np.sum(cm[i, :])
        if row_sum > 0:
            cm_norm[i, :] = cm[i, :] / row_sum
        else:
            cm_norm[i, :] = 0

    # Create the plot.
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"Overall Accuracy - {overall_accuracy:.0%}")

    # Set tick marks for each class.
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Annotate each cell with the normalized percentage.
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.0%}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")

    ax.set_ylabel(col1_label)
    ax.set_xlabel(col2_label)
    plt.tight_layout()
    output_filename = os.path.join(os.path.dirname(csv_path), f"confusion_matrix_{col1_label}_{col2_label}.pdf")
    plt.savefig(output_filename, dpi=150)
    plt.close()

def optimize_density_threshold(ground_truth_csv, colonies_csv, metric_col="density", days_list=[3],
                               threshold_range_flat=(0, 0.01), num_steps_flat=500,
                               std_range=(0, 5), num_steps_std=500,
                               mad_range=(0, 7), num_steps_mad=500):
    """
    Optimize the density threshold for classifying a single colony in a well as differentiated,
    for a list of specified days plus their combination.

    Three methods are considered:
      1. Flat thresholds: candidate density thresholds linearly spaced over a given range.
      2. Std method: thresholds defined as mean - k*std (k is varied over a candidate range).
      3. MAD method: thresholds defined as median - k*MAD (k is varied over a candidate range).

    For each candidate threshold, the predicted status is:
      - 3 (differentiated) if density < threshold,
      - 1 (undifferentiated) otherwise.

    Only wells with a single colony are considered and only rows where ground truth Status is 1 or 3.
    Performance metrics (precision, recall, and F1 score for the differentiated class) are computed
    separately for each specified day and for the combined set of days.

    A facet grid of 3 rows x 3 columns is plotted:
      - Row 1: Flat threshold analysis (x-axis: threshold value)
      - Row 2: Std method analysis (x-axis: k in standard deviations)
      - Row 3: MAD method analysis (x-axis: k in MAD units)
    
    Each subplot plots one line per condition (each day from days_list and one combined line).

    Parameters:
      ground_truth_csv (str): Path to the ground truth CSV file (columns: Plate, Well, Day, Status).
      colonies_csv (str): Path to the colonies metrics CSV file (columns including Plate, well,
                          Relative Time (days), density). 'well' and 'Relative Time (days)' are renamed
                          to 'Well' and 'Day'.
      days_list (list): List of days to analyze (e.g. [3, 8]).
      threshold_range_flat (tuple): Range of flat threshold candidates.
      num_steps_flat (int): Number of flat threshold candidates.
      std_range (tuple): Range for candidate k values (in standard deviations).
      num_steps_std (int): Number of candidate k values for std method.
      mad_range (tuple): Range for candidate k values (in MAD units).
      num_steps_mad (int): Number of candidate k values for MAD method.
    
    Returns:
      results (dict): Dictionary with best parameters and performance metrics for each method,
                      separately for each day in days_list and for the combined condition.
    """
    # Load ground truth CSV and filter rows with Day > 1.
    try:
        df_gt = pd.read_csv(ground_truth_csv)
    except Exception as e:
        logging.error(f"Error reading ground truth CSV {ground_truth_csv}: {e}")
        return None
    df_gt["Day"] = pd.to_numeric(df_gt["Day"], errors="coerce")
    df_gt = df_gt[df_gt["Day"] > 1]
    
    # Load colonies CSV and rename columns.
    try:
        df_col = pd.read_csv(colonies_csv)
    except Exception as e:
        logging.error(f"Error reading colonies CSV {colonies_csv}: {e}")
        return None
    df_col.rename(columns={"well": "Well", "Relative Time (days)": "Day"}, inplace=True)
    df_col["Day"] = pd.to_numeric(df_col["Day"], errors="coerce")
    df_col["Plate"] = pd.to_numeric(df_col["Plate"], errors="coerce")
    df_col = df_col[df_col["Day"] > 1]
    
    # Use only ground truth rows with Status 1 or 3.
    df_gt = df_gt[df_gt["Status"].isin([1, 3])]
    
    # Merge on Plate, Well, and Day.
    merged = pd.merge(df_gt, df_col, on=["Plate", "Well", "Day"], how="inner")
    if merged.empty:
        logging.error("Merged DataFrame is empty. Check the CSV files and merge keys.")
        return None

    # Keep only wells with a single colony.
    well_counts = merged.groupby(["Plate", "Well", "Day"]).size().reset_index(name="count")
    single_colony_keys = well_counts[well_counts["count"] == 1][["Plate", "Well", "Day"]]
    merged = pd.merge(merged, single_colony_keys, on=["Plate", "Well", "Day"], how="inner")
    if merged.empty:
        logging.error("No wells with a single colony found after filtering.")
        return None

    # Build data for each individual day in days_list and for the combined condition.
    def get_plate_data(df):
        groups = df.groupby("Plate")
        plate_data = {}
        for plate, group in groups:
            densities = group[metric_col].values
            true_status = group["Status"].values
            plate_data[plate] = {"densities": densities, "true_status": true_status}
        return plate_data

    data_by_day = {}
    for d in days_list:
        df_day = merged[merged["Day"] == d]
        if not df_day.empty:
            data_by_day[str(d)] = get_plate_data(df_day)
    
    # Combined condition: all rows corresponding to days in days_list.
    combined_data = get_plate_data(merged[merged["Day"].isin(days_list)])
    
    # Dictionary with all conditions: individual days plus "Combined".
    all_conditions = data_by_day.copy()
    all_conditions["Combined"] = combined_data

    # Helper function to compute metrics.
    def compute_metrics(densities, true_status, threshold):
        pred_status = np.where(densities > threshold, 3, 1)
        tp = np.sum((pred_status == 3) & (true_status == 3))
        fp = np.sum((pred_status == 3) & (true_status != 3))
        fn = np.sum((pred_status != 3) & (true_status == 3))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    #############################
    # Method 1: Flat thresholds.
    flat_thresholds = np.linspace(threshold_range_flat[0], threshold_range_flat[1], num_steps_flat)
    flat_metrics = {cond: {"precision": [], "recall": [], "f1": []} for cond in all_conditions}

    for t in flat_thresholds:
        for cond, plate_data in all_conditions.items():
            metrics = [compute_metrics(data["densities"], data["true_status"], t)
                       for data in plate_data.values()]
            if metrics:
                p, r, f = zip(*metrics)
                flat_metrics[cond]["precision"].append(np.mean(p))
                flat_metrics[cond]["recall"].append(np.mean(r))
                flat_metrics[cond]["f1"].append(np.mean(f))
            else:
                flat_metrics[cond]["precision"].append(np.nan)
                flat_metrics[cond]["recall"].append(np.nan)
                flat_metrics[cond]["f1"].append(np.nan)

    # Convert lists to numpy arrays.
    for cond in flat_metrics:
        flat_metrics[cond]["precision"] = np.array(flat_metrics[cond]["precision"])
        flat_metrics[cond]["recall"] = np.array(flat_metrics[cond]["recall"])
        flat_metrics[cond]["f1"] = np.array(flat_metrics[cond]["f1"])

    # Find best flat threshold (max F1) for each condition.
    best_idx_flat = {cond: np.nanargmax(flat_metrics[cond]["f1"]) for cond in all_conditions}
    best_threshold_flat = {cond: flat_thresholds[best_idx_flat[cond]] for cond in all_conditions}

    #############################
    # Method 2: Std method.
    k_std_candidates = np.linspace(std_range[0], std_range[1], num_steps_std)
    std_metrics = {cond: {"precision": [], "recall": [], "f1": []} for cond in all_conditions}

    for k in k_std_candidates:
        for cond, plate_data in all_conditions.items():
            metrics = []
            for data in plate_data.values():
                mean_val = np.mean(data["densities"])
                std_val = np.std(data["densities"])
                thresh = mean_val + k * std_val
                metrics.append(compute_metrics(data["densities"], data["true_status"], thresh))
            if metrics:
                p, r, f = zip(*metrics)
                std_metrics[cond]["precision"].append(np.mean(p))
                std_metrics[cond]["recall"].append(np.mean(r))
                std_metrics[cond]["f1"].append(np.mean(f))
            else:
                std_metrics[cond]["precision"].append(np.nan)
                std_metrics[cond]["recall"].append(np.nan)
                std_metrics[cond]["f1"].append(np.nan)

    for cond in std_metrics:
        std_metrics[cond]["precision"] = np.array(std_metrics[cond]["precision"])
        std_metrics[cond]["recall"] = np.array(std_metrics[cond]["recall"])
        std_metrics[cond]["f1"] = np.array(std_metrics[cond]["f1"])

    best_idx_std = {cond: np.nanargmax(std_metrics[cond]["f1"]) for cond in all_conditions}
    best_k_std = {cond: k_std_candidates[best_idx_std[cond]] for cond in all_conditions}
    best_threshold_std = {}
    for cond, plate_data in all_conditions.items():
        # Average threshold across plates for the best k.
        thresholds = [np.mean(data["densities"]) - best_k_std[cond] * np.std(data["densities"])
                      for data in plate_data.values()]
        best_threshold_std[cond] = np.mean(thresholds) if thresholds else np.nan

    #############################
    # Method 3: MAD method.
    k_mad_candidates = np.linspace(mad_range[0], mad_range[1], num_steps_mad)
    mad_metrics = {cond: {"precision": [], "recall": [], "f1": []} for cond in all_conditions}

    for k in k_mad_candidates:
        for cond, plate_data in all_conditions.items():
            metrics = []
            for data in plate_data.values():
                med = np.median(data["densities"])
                mad = np.median(np.abs(data["densities"] - med))
                thresh = med + k * mad
                metrics.append(compute_metrics(data["densities"], data["true_status"], thresh))
            if metrics:
                p, r, f = zip(*metrics)
                mad_metrics[cond]["precision"].append(np.mean(p))
                mad_metrics[cond]["recall"].append(np.mean(r))
                mad_metrics[cond]["f1"].append(np.mean(f))
            else:
                mad_metrics[cond]["precision"].append(np.nan)
                mad_metrics[cond]["recall"].append(np.nan)
                mad_metrics[cond]["f1"].append(np.nan)

    for cond in mad_metrics:
        mad_metrics[cond]["precision"] = np.array(mad_metrics[cond]["precision"])
        mad_metrics[cond]["recall"] = np.array(mad_metrics[cond]["recall"])
        mad_metrics[cond]["f1"] = np.array(mad_metrics[cond]["f1"])

    best_idx_mad = {cond: np.nanargmax(mad_metrics[cond]["f1"]) for cond in all_conditions}
    best_k_mad = {cond: k_mad_candidates[best_idx_mad[cond]] for cond in all_conditions}
    best_threshold_mad = {}
    for cond, plate_data in all_conditions.items():
        thresholds = [np.median(data["densities"]) - best_k_mad[cond] * np.median(np.abs(data["densities"] - np.median(data["densities"])))
                      for data in plate_data.values()]
        best_threshold_mad[cond] = np.mean(thresholds) if thresholds else np.nan

    #############################
    # Plotting the facet grid.
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Define a helper for plotting metrics per method.
    def plot_metric(ax, x_vals, metrics_dict, x_label, best_vals, metric_name, marker):
        for cond in metrics_dict:
            line, = ax.plot(x_vals, metrics_dict[cond][metric_name], marker=marker, label=cond)
            # Draw a vertical line at the best x value for this condition.
            ax.axvline(best_vals[cond], color=line.get_color(), linestyle="--",
                       label=f"Best {cond}: {best_vals[cond]:.5f}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()
    
    # Row 1: Flat thresholds.
    plot_metric(axes[0, 0], flat_thresholds, flat_metrics, "Density Threshold", 
                best_threshold_flat, "precision", marker="o")
    axes[0, 0].set_title("Flat - Precision")
    plot_metric(axes[0, 1], flat_thresholds, flat_metrics, "Density Threshold", 
                best_threshold_flat, "recall", marker="o")
    axes[0, 1].set_title("Flat - Recall")
    plot_metric(axes[0, 2], flat_thresholds, flat_metrics, "Density Threshold", 
                best_threshold_flat, "f1", marker="o")
    axes[0, 2].set_title("Flat - F1 Score")
    
    # Row 2: Std method.
    plot_metric(axes[1, 0], k_std_candidates, std_metrics, "k (Std Deviations)", 
                best_k_std, "precision", marker="s")
    axes[1, 0].set_title("Std Dev - Precision")
    plot_metric(axes[1, 1], k_std_candidates, std_metrics, "k (Std Deviations)", 
                best_k_std, "recall", marker="s")
    axes[1, 1].set_title("Std Dev - Recall")
    plot_metric(axes[1, 2], k_std_candidates, std_metrics, "k (Std Deviations)", 
                best_k_std, "f1", marker="s")
    axes[1, 2].set_title("Std Dev - F1 Score")
    
    # Row 3: MAD method.
    plot_metric(axes[2, 0], k_mad_candidates, mad_metrics, "k (MAD units)", 
                best_k_mad, "precision", marker="^")
    axes[2, 0].set_title("MAD - Precision")
    plot_metric(axes[2, 1], k_mad_candidates, mad_metrics, "k (MAD units)", 
                best_k_mad, "recall", marker="^")
    axes[2, 1].set_title("MAD - Recall")
    plot_metric(axes[2, 2], k_mad_candidates, mad_metrics, "k (MAD units)", 
                best_k_mad, "f1", marker="^")
    axes[2, 2].set_title("MAD - F1 Score")
    
    plt.tight_layout()
    output_filename = colonies_csv.replace(".csv", f"_optimal_density_threshold_{metric_col}.png")
    plt.savefig(output_filename, dpi=150)
    plt.close()

    # Logging the best parameters for each condition.
    for cond in all_conditions:
        logging.info(f"Flat method {cond}: Best threshold = {best_threshold_flat[cond]:.5f}, F1 = {flat_metrics[cond]['f1'][best_idx_flat[cond]]:.4f}")
        logging.info(f"Std method {cond}: Best k = {best_k_std[cond]:.2f}, average threshold = {best_threshold_std[cond]:.5f}, F1 = {std_metrics[cond]['f1'][best_idx_std[cond]]:.4f}")
        logging.info(f"MAD method {cond}: Best k = {best_k_mad[cond]:.2f}, average threshold = {best_threshold_mad[cond]:.5f}, F1 = {mad_metrics[cond]['f1'][best_idx_mad[cond]]:.4f}")

    results = {
        "flat": {cond: {"best_threshold": best_threshold_flat[cond],
                        "F1": flat_metrics[cond]["f1"][best_idx_flat[cond]]} for cond in all_conditions},
        "std": {cond: {"best_k": best_k_std[cond],
                       "average_threshold": best_threshold_std[cond],
                       "F1": std_metrics[cond]["f1"][best_idx_std[cond]]} for cond in all_conditions},
        "mad": {cond: {"best_k": best_k_mad[cond],
                       "average_threshold": best_threshold_mad[cond],
                       "F1": mad_metrics[cond]["f1"][best_idx_mad[cond]]} for cond in all_conditions}
    }
    return results

def plot_plate_colony_percentages_single_figure(csv_path, output_folder=None):
    """
    Produces a two‑panel stacked‑bar figure summarising colony calls per plate
    across time.  **Four mutually‑exclusive categories** are now used:

        • Single Undifferentiated – single colony, dead_live_ratio_proximity ≤ 0.27
        • Differentiated          – ≥ 2 colonies *or* reach_p99 > 24.08027
        • Dying                   – single colony with dead_live_ratio_proximity > 0.27
        • Dead                    – well has disappeared on that day (no row)

    TOP panel   → raw well counts
    BOTTOM panel→ percentages of valid wells

    The PNG is saved as “..._counts_and_percentages.png”.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # ------------------------- 1) read & filter  ------------------------- #
    df = pd.read_csv(csv_path)
    df["Relative Time (days)"] = df["Relative Time (days)"].astype(int)

    baseline_df = df.query("`Relative Time (days)` == 0")
    valid_wells_df = baseline_df.groupby(["Plate", "well"]).filter(
        lambda g: len(g) == 1 and g.iloc[0]["num_cells"] > 3
    )

    valid_wells_dict = {}
    for _, r in valid_wells_df.iterrows():
        valid_wells_dict.setdefault(r["Plate"], set()).add(r["well"])

    all_days = sorted(df["Relative Time (days)"].unique())
    all_plates = sorted(valid_wells_dict)

    # -------------------- 2) classification & aggregation ---------------- #
    day_plate_counts = {
        d: {
            p: {
                "Single Undifferentiated": 0,
                "Differentiated": 0,
                "Dying": 0,
                "Dead": 0,
            }
            for p in all_plates
        }
        for d in all_days
    }

    for plate in all_plates:
        valid_wells = valid_wells_dict[plate]
        for well in valid_wells:
            for day in all_days:
                cur = df[
                    (df["Plate"] == plate)
                    & (df["well"] == well)
                    & (df["Relative Time (days)"] == day)
                ]

                if day == 0:
                    # Day‑0 wells always exist; classify as Dying or Single
                    category = (
                        "Dying"
                        if cur.iloc[0]["dead_live_ratio_proximity"] > 0.27
                        else "Single Undifferentiated"
                    )
                else:
                    # Subsequent days
                    if cur.empty:
                        category = "Dead"  # well disappeared ↦ Dead
                    elif len(cur) == 1:
                        row = cur.iloc[0]
                        if row["dead_live_ratio_proximity"] > 0.27:
                            category = "Dying"
                        elif row["reach_p99"] > 24.08027:
                            category = "Differentiated"
                        else:
                            category = "Single Undifferentiated"
                    else:
                        category = "Differentiated"  # ≥2 colonies

                day_plate_counts[day][plate][category] += 1

    # convert to percentages ----------------------------------------------------------
    day_plate_pct = {d: {} for d in all_days}
    ordered_cats = (
        "Single Undifferentiated",
        "Differentiated",
        "Dying",
        "Dead",
    )
    for plate in all_plates:
        denom = len(valid_wells_dict[plate])
        for d in all_days:
            cts = day_plate_counts[d][plate]
            if denom:
                day_plate_pct[d][plate] = tuple(cts[k] / denom * 100 for k in ordered_cats)
            else:
                day_plate_pct[d][plate] = (0.0, 0.0, 0.0, 0.0)

    # ------------------------- 3) plotting  ----------------------------------------- #
    if output_folder is None:
        output_folder = os.path.dirname(csv_path)
    os.makedirs(output_folder, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 10),
        sharex=False,
        gridspec_kw={"height_ratios": [1.1, 1]},
    )
    ax_cnt, ax_pct = axes

    n_days = len(all_days)
    n_plates = len(all_plates)
    gap = 1  # horizontal gap between clusters

    colors = {
        "Single": "#98DFAF",  # green (unchanged)
        "Diff": "orange",
        "Dying": "purple",  # inherits original “Dead” colour
        "Dead": "red",  # new colour per user request
    }

    def legend_label(cat, d_idx, p_idx):
        return cat if (d_idx == 0 and p_idx == 0) else None

    for d_idx, d in enumerate(all_days):
        for p_idx, p in enumerate(all_plates):
            x = d_idx * (n_plates + gap) + p_idx
            c_single = day_plate_counts[d][p]["Single Undifferentiated"]
            c_diff = day_plate_counts[d][p]["Differentiated"]
            c_dying = day_plate_counts[d][p]["Dying"]
            c_dead = day_plate_counts[d][p]["Dead"]
            pct_single, pct_diff, pct_dying, pct_dead = day_plate_pct[d][p]

            # counts panel
            ax_cnt.bar(
                x,
                c_single,
                width=0.8,
                color=colors["Single"],
                edgecolor="black",
                label=legend_label("undifferentiated", d_idx, p_idx),
            )
            ax_cnt.bar(
                x,
                c_diff,
                width=0.8,
                bottom=c_single,
                color=colors["Diff"],
                edgecolor="black",
                label=legend_label("differentiated", d_idx, p_idx),
            )
            ax_cnt.bar(
                x,
                c_dying,
                width=0.8,
                bottom=c_single + c_diff,
                color=colors["Dying"],
                edgecolor="black",
                label=legend_label("dying", d_idx, p_idx),
            )
            ax_cnt.bar(
                x,
                c_dead,
                width=0.8,
                bottom=c_single + c_diff + c_dying,
                color=colors["Dead"],
                edgecolor="black",
                label=legend_label("dead", d_idx, p_idx),
            )

            # percentage panel
            ax_pct.bar(x, pct_single, width=0.8, color=colors["Single"], edgecolor="black")
            ax_pct.bar(
                x,
                pct_diff,
                width=0.8,
                bottom=pct_single,
                color=colors["Diff"],
                edgecolor="black",
            )
            ax_pct.bar(
                x,
                pct_dying,
                width=0.8,
                bottom=pct_single + pct_diff,
                color=colors["Dying"],
                edgecolor="black",
            )
            ax_pct.bar(
                x,
                pct_dead,
                width=0.8,
                bottom=pct_single + pct_diff + pct_dying,
                color=colors["Dead"],
                edgecolor="black",
            )

    # ------------- cosmetics & labels ---------------------------------------------- #
    plate_label_map = {
        0: "NTC",
        54: "CUL3",
        66: "MED23",
        70: "NIPBL",
        92: "BAP1",
        93: "BCKDK",
    }
    x_positions = [d_idx * (n_plates + gap) + p_idx for d_idx in range(n_days) for p_idx in range(n_plates)]
    x_labels = [plate_label_map.get(p, str(p)) for _ in all_days for p in all_plates]
    ax_pct.set_xticks(x_positions)
    ax_pct.set_xticklabels(x_labels, rotation=45, ha="right")
    ax_cnt.set_xticks(x_positions)
    ax_cnt.set_xticklabels(x_labels, rotation=45, ha="right")

    for d_idx, d in enumerate(all_days):
        center = d_idx * (n_plates + gap) + (n_plates - 1) / 2
        ax_cnt.text(center, ax_cnt.get_ylim()[1] * 1.02, f"Day {d}", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax_pct.text(center, 102, f"Day {d}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    max_wells = max(len(v) for v in valid_wells_dict.values())
    ax_cnt.set_ylim(0, max_wells * 1.15)
    ax_cnt.set_ylabel("Number of Colonies")
    ax_pct.set_ylim(0, 110)
    ax_pct.set_ylabel("Percentage of Colonies (%)")

    ax_cnt.set_title("Colony Calls by Plate over Time — Raw Counts")
    ax_pct.set_title("Colony Calls by Plate over Time — Percentages")
    # ── move the legend outside the axes ─────────────────────────────────────────────
    handles, labels = ax_cnt.get_legend_handles_labels()

    # put the legend just to the right of ax_cnt, centred vertically
    fig.legend(handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),   #   (x, y) in figure-coords
            borderaxespad=0)

    # make the figure a little narrower on the right so tight_layout
    # doesn’t squash the subplot to make room for the legend
    plt.tight_layout()   

    out_file = os.path.join(output_folder,
                            "all_plates_stacked_counts_and_percentages.png")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved counts/percentages figure to {out_file}")
    
def plot_plate_average_density_excluding_dead(csv_path, output_folder=None):
    """
    Reads a CSV of colony data, filters valid wells based on the criteria:
      - Exactly one colony at day 0
      - That colony has > 5 cells
    Then, for each valid well on each subsequent day, classifies the well as Dead or Alive:
      - Dead if:
          * No colony detected, or
          * Colony cell count <= 90% of previous day's count
      - Alive if:
          * Exactly one colony and >90% previous day's cell count
             -> further subdivided into Single Undifferentiated vs Differentiated 
                (but both are considered 'alive' for density averaging)
          * More than one colony => Differentiated => also 'alive'
    
    Finally, we compute (and plot) the average density of all 'alive' wells (excluding dead)
    for each plate on each day. The result is a single figure with multiple lines (one per plate).
    
    Args:
        csv_path (str): Path to the CSV file containing colony metrics.
                        Must have columns: "Plate", "well", "Relative Time (days)",
                        "num_cells", and "density".
        output_folder (str, optional): Folder where the plot will be saved.
                                       If None, uses the same directory as csv_path.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # --------------------
    # 1) Read and filter data
    # --------------------
    df = pd.read_csv(csv_path)
    # Make sure "Relative Time (days)" is integer
    df['Relative Time (days)'] = df['Relative Time (days)'].astype(int)

    # Determine valid wells: those with exactly one colony at day 0 that has > 5 cells
    baseline_df = df[df['Relative Time (days)'] == 0]
    valid_baseline = baseline_df.groupby(['Plate', 'well']).filter(
        lambda g: (len(g) == 1) and (g.iloc[0]['num_cells'] > 5)
    )

    # Dictionary {plate: set_of_valid_wells}
    valid_wells_dict = {}
    for _, row in valid_baseline.iterrows():
        plate = row['Plate']
        well = row['well']
        valid_wells_dict.setdefault(plate, set()).add(well)

    # Collect all unique days and the plates that have at least one valid well
    all_days = sorted(df['Relative Time (days)'].unique())
    all_plates = sorted(valid_wells_dict.keys())

    # --------------------
    # 2) Classify each well as dead or alive (per day)
    #    We'll store day_plate_well_category[day][plate][well] = 'Dead' / 'Alive'
    # --------------------
    day_plate_well_category = {
        day: {
            plate: {}
            for plate in all_plates
        }
        for day in all_days
    }

    for plate in all_plates:
        wells = valid_wells_dict[plate]
        for well in wells:
            for day in all_days:
                if day == 0:
                    # By definition, day 0 is "alive" (Single Undifferentiated)
                    # for valid wells
                    day_plate_well_category[day][plate][well] = 'Alive'
                else:
                    current = df[
                        (df['Plate'] == plate) &
                        (df['well'] == well) &
                        (df['Relative Time (days)'] == day)
                    ]
                    if current.empty:
                        # No colony => dead
                        day_plate_well_category[day][plate][well] = 'Dead'
                    elif len(current) == 1:
                        row_current = current.iloc[0]
                        # Attempt to retrieve previous day's measurement
                        prev = df[
                            (df['Plate'] == plate) &
                            (df['well'] == well) &
                            (df['Relative Time (days)'] == day - 1)
                        ]
                        if not prev.empty and len(prev) == 1:
                            prev_num_cells = prev.iloc[0]['num_cells']
                        else:
                            prev_num_cells = None
                        
                        # If cell count ≤ 90% of previous => dead
                        if (prev_num_cells is not None) and (row_current['num_cells'] <= 0.9 * prev_num_cells):
                            day_plate_well_category[day][plate][well] = 'Dead'
                        else:
                            # single colony but not <=90% => alive
                            # (differentiated vs single doesn't matter here;
                            #  either way it's 'Alive' for averaging density)
                            day_plate_well_category[day][plate][well] = 'Alive'
                    else:
                        # More than one colony => 'Differentiated' => 'Alive'
                        day_plate_well_category[day][plate][well] = 'Alive'

    # --------------------
    # 3) Compute the average density per plate, per day, excluding dead
    # --------------------
    # We'll store day_plate_avg_density[day][plate] = float or np.nan
    day_plate_avg_density = {
        day: {plate: np.nan for plate in all_plates}
        for day in all_days
    }

    for day in all_days:
        for plate in all_plates:
            wells = valid_wells_dict[plate]
            # Find all wells that are 'Alive' on this day
            alive_wells = [w for w in wells if day_plate_well_category[day][plate][w] == 'Alive']

            if len(alive_wells) == 0:
                # no alive wells => avg density = NaN
                day_plate_avg_density[day][plate] = np.nan
            else:
                # For each alive well, get that day's density from df
                # Note: some wells might have multiple rows if the CSV has duplicates,
                # but we assume there's only 1 row for that well+day if it's "Alive."
                densities = []
                for w in alive_wells:
                    row = df[
                        (df['Plate'] == plate) &
                        (df['well'] == w) &
                        (df['Relative Time (days)'] == day)
                    ]
                    if not row.empty:
                        # Take the first row's density
                        densities.append(row.iloc[0]['density'])
                if len(densities) == 0:
                    day_plate_avg_density[day][plate] = np.nan
                else:
                    day_plate_avg_density[day][plate] = np.mean(densities)

    # --------------------
    # 4) Plot a single figure with day on x-axis and average density on y-axis
    #    for each plate. We exclude dead wells from the average.
    # --------------------
    if output_folder is None:
        output_folder = os.path.dirname(csv_path)
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Make a line for each plate
    for plate in all_plates:
        # Collect the day values and corresponding avg densities
        x_vals = []
        y_vals = []
        for day in all_days:
            avg_dens = day_plate_avg_density[day][plate]
            x_vals.append(day)
            y_vals.append(avg_dens)

        plt.plot(x_vals, y_vals, marker='o', label=f"Plate {plate}")

    plt.xlabel('Relative Time (days)')
    plt.ylabel('Average Density (excluding dead wells)')
    plt.title('Average Density Over Days, by Plate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_file = os.path.join(output_folder, "plate_average_density_excluding_dead.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved average density plot (excluding dead) to {out_file}")

def plot_plate_growth_rates(csv_path, output_folder=None):
    """
    For each unique plate in the CSV file, fit an exponential growth curve for each valid well,
    extract the exponential growth rate (r), and then plot a bar plot of the mean exponential growth 
    rate with SEM error bars per plate.

    Valid well criteria:
      - At day 0, the well must have exactly one colony with > 5 cells.
      - On subsequent days, valid wells may have more than one colony.
    The total number of cells per well at a given day is calculated as the sum of num_cells for that well.
    
    For each valid well, an exponential growth model is assumed: N(t) = A * exp(r * t).
    The growth rate r is determined by performing a linear regression on the natural logarithm of 
    total cells versus time. Wells with less than 2 valid time points (i.e. total cells > 0) are skipped.

    Args:
        csv_path (str): Path to the CSV file containing colony metrics.
                        The CSV must include the columns "Plate", "well", "Relative Time (days)", and "num_cells".
        output_folder (str, optional): Folder where the plot will be saved.
                                       If None, the plot is saved in the same directory as csv_path.
    """

    # Load CSV and ensure "Relative Time (days)" is integer.
    df = pd.read_csv(csv_path)
    df['Relative Time (days)'] = df['Relative Time (days)'].astype(int)

    # Identify valid wells at baseline (day 0): exactly one colony with > 5 cells.
    baseline_df = df[df['Relative Time (days)'] == 0]
    valid_baseline = baseline_df.groupby(['Plate', 'well']).filter(
        lambda group: (len(group) == 1) and (group.iloc[0]['num_cells'] > 5)
    )
    
    # Build mapping from Plate to its set of valid wells.
    valid_wells_dict = {}
    for _, row in valid_baseline.iterrows():
        plate = row['Plate']
        well = row['well']
        valid_wells_dict.setdefault(plate, set()).add(well)

    # Group data by Plate, well, and day, summing num_cells so that each record represents 
    # the total cells for that well at that day.
    grouped = df.groupby(['Plate', 'well', 'Relative Time (days)'])['num_cells'].sum().reset_index()

    # For each valid well, fit an exponential curve and extract the growth rate.
    plate_growth_rates = {}  # {plate: list of growth rates (r)}
    for plate, wells in valid_wells_dict.items():
        plate_growth_rates.setdefault(plate, [])
        for well in wells:
            well_data = grouped[(grouped['Plate'] == plate) & (grouped['well'] == well)]
            # Sort by time.
            well_data = well_data.sort_values('Relative Time (days)')
            # Exclude time points with non-positive cell counts (log undefined).
            valid_data = well_data[well_data['num_cells'] > 0]
            if len(valid_data) < 2:
                continue  # Not enough data to fit an exponential curve.
            t = valid_data['Relative Time (days)'].values
            cells = valid_data['num_cells'].values
            # Fit a line to the log-transformed cell counts: log(N) = log(A) + r*t.
            log_cells = np.log(cells)
            slope, intercept = np.polyfit(t, log_cells, 1)
            # Exclude negative growth rates.
            # if slope < 0:
            #     continue
            # 'slope' is the exponential growth rate.
            plate_growth_rates[plate].append(slope)

    # Calculate the mean and SEM of growth rates for each plate.
    plates = []
    means = []
    sems = []
    for plate, rates in plate_growth_rates.items():
        if len(rates) == 0:
            continue
        rates = np.array(rates)
        mean_rate = rates.mean()
        sem_rate = rates.std(ddof=1) / np.sqrt(len(rates)) if len(rates) > 1 else 0
        plates.append(plate)
        means.append(mean_rate)
        sems.append(sem_rate)

    # Create a bar plot with error bars for the mean exponential growth rate per plate.
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(plates))
    plt.bar(x_pos, means, yerr=sems, capsize=5, align='center', alpha=0.7)
    plt.xticks(x_pos, [f"Plate {plate}" for plate in plates])
    plt.ylabel('Exponential Growth per Day')
    plt.title('Growth Rate per Colony')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Determine the output folder.
    if output_folder is None:
        output_folder = os.path.dirname(csv_path)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'plate_growth_rates.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Saved growth rates bar plot to {output_file}")

def make_all_colonies_csv(input_folder, model_name=None):
    """
    Combines colony metrics from multiple subfolders and adds a 'source_dir' column.

    For each subdirectory (each representing a plate/day run), the function:
      - Extracts the run time via extract_time_from_folder(subdir).
      - Loads the colony_metrics CSV from:
            <subdir>/model_outputs/<model_name>_full_optics2/colony_metrics.csv
      - Adds the run time (Time) and extracts the Plate info.
      - Adds a 'source_dir' column, which is set to:
            <subdir>/model_outputs/<model_name>_full_optics2/colonies
    Then the function concatenates all dataframes, computes relative time in hours and days,  
    and saves the combined CSV file.
    
    Parameters:
      input_folder (str): The root folder containing subdirectories for runs.
      model_name (str): The model name used to locate the proper subfolder.
    
    Returns:
      str: The path to the combined output CSV.
    """
    if not model_name:
        return None
    
    all_data = []
    min_time = None

    for subdir in glob.glob(os.path.join(input_folder, '*/')):
        try:
            # Assume extract_time_from_folder is defined elsewhere.
            time = extract_time_from_folder(subdir)
            if min_time is None or time < min_time:
                min_time = time

            # Construct the full path to the colony_metrics CSV.
            metrics_csv = os.path.join(subdir, "model_outputs", f"{model_name}_full_optics3", "colony_metrics.csv")
            data = pd.read_csv(metrics_csv)

            # Add the run time.
            data['Time'] = time

            # Extract Plate number (assumes the folder name structure encodes plate info).
            data['Plate'] = os.path.basename(os.path.dirname(subdir)).split('_')[0]

            # Add the source_dir column pointing to the colonies folder.
            data['source_dir'] = os.path.join(subdir, "model_outputs", f"{model_name}_full_optics3", "colonies")
            
            all_data.append(data)
        except Exception as e:
            print(f"skipping {subdir}: {e}")

    # Create an output folder for the model.
    output_folder = os.path.join(input_folder, f"{model_name}")
    os.makedirs(output_folder, exist_ok=True)

    # Combine all data frames and compute Relative Time (hrs and days)
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data['Relative Time (hrs)'] = ((combined_data['Time'] - min_time).dt.total_seconds() / 3600).astype(float)
    combined_data['Relative Time (days)'] = np.floor((combined_data['Relative Time (hrs)'] + 12) / 24).astype(int)

    combined_data_path = os.path.join(output_folder, f'{model_name}_all_colonies.csv')
    combined_data.to_csv(combined_data_path, index=False)
    return combined_data_path

def extract_time_from_folder(folder_name):
    folder_name = folder_name.rstrip('\\').rstrip('/')
    parts = folder_name.split('_')
    date_str = parts[-2]
    time_str = parts[-1]
    return datetime.strptime(f'{date_str} {time_str}', '%Y%m%d %H%M%S')

def organize_incorrect_predictions(simulation_csv, ground_truth_csv, dst_root):
    """
    Organizes incorrectly predicted wells into a folder structure for further analysis.
    
    The function reads both the simulation CSV (which contains the simulation results) 
    and the ground truth CSV. It assumes the ground truth CSV has the columns:
        "Plate", "Day", "Well", "Status"
    and that the simulation CSV uses the following logic for prediction:
      - If more than one simulation entry exists for a well, prediction = 2.
      - If exactly one simulation entry exists and its "density" value is available, then:
            if (density < 0.0036 and day > 0) → prediction = 3,
            otherwise → prediction = 1.
      - If no simulation entry is found, prediction = 0.
      
    With the new update, each well record in the ground truth CSV (or the joined CSV)
    should contain a column "source_dir" giving the directory where the colony images are located.
    
    For every ground truth record where the simulated prediction does NOT match the actual Status,
    the function copies two files:
      - Colony image: <source_dir>/colony_<Well>.tif
      - Optics image: <source_dir>/optics_<Well>.png
    into a subfolder of the destination defined as:
         <dst_root>/incorrect_predictions/day_<Day>/
    The files are renamed as:
         plate_<Plate>_colony_<Well>.tif and plate_<Plate>_colony_<Well>_optics.tif respectively.
    
    Parameters:
      simulation_csv (str): Path to the CSV with simulation results.
      ground_truth_csv (str): Path to the ground truth CSV.
      dst_root (str): Destination root folder where the "incorrect_predictions" folder will be created.
    """
    
    # Load the simulation and ground truth CSV files.
    sim_df = pd.read_csv(simulation_csv)
    gt_df = pd.read_csv(ground_truth_csv)
    gt_df.loc[gt_df["Status"] == -1, "Status"] = 1
    
    # Ensure that Plate and Day values are strings for consistent merging.
    gt_df["Plate"] = gt_df["Plate"].astype(str)
    gt_df["Day"] = gt_df["Day"].astype(str)
    sim_df["Plate"] = sim_df["Plate"].astype(str)
    
    # Prepare a subset of simulation data with the "source_dir" info.
    # The simulation CSV is assumed to have the well column named "well" and day info as "Relative Time (days)".
    # We rename them to match the ground truth keys ("Well" and "Day"). Convert Day to string for the join.
    sim_sub = sim_df[['Plate', 'well', 'Relative Time (days)', 'source_dir']].drop_duplicates()
    sim_sub = sim_sub.rename(columns={"well": "Well", "Relative Time (days)": "Day"})
    sim_sub["Day"] = sim_sub["Day"].astype(str)
    
    # Merge the ground truth with simulation to add the source_dir value.
    merged_df = pd.merge(gt_df, sim_sub, on=["Plate", "Well", "Day"], how="left")
    
    # Define a helper function to compute the simulated prediction for a given record.
    def compute_sim_pred(row):
        plate = str(row["Plate"])
        well = row["Well"]
        try:
            day_val = int(row["Day"])
        except Exception:
            day_val = 0
        
        # Filter the simulation CSV for entries that match the given well.
        df_subset = sim_df[
            (sim_df["Plate"].astype(str) == plate) &
            (sim_df["well"] == well) &
            (sim_df["Relative Time (days)"] == day_val)
        ]
        count = len(df_subset)
        if count > 1:
            return 2
        elif count == 1:
            density = df_subset.iloc[0]["density"] if "density" in df_subset.columns else 0
            return 3 if (density < 0.0036 and day_val > 0) else 1
        else:
            return 0

    # Identify records where the simulated prediction and ground truth Status differ.
    incorrect_records = []
    for index, row in merged_df.iterrows():
        sim_pred = compute_sim_pred(row)
        gt_status = int(row["Status"])
        if sim_pred != gt_status:
            incorrect_records.append({
                "Plate": row["Plate"],
                "Day": row["Day"],
                "Well": row["Well"],
                # This should now be available from the merged DataFrame.
                "source_dir": row.get("source_dir", ""),
                "sim_pred": sim_pred,
                "gt_status": gt_status
            })
    
    # Process each incorrect record by copying the respective image files.
    for record in incorrect_records:
        plate = record["Plate"]
        day = record["Day"]
        well = record["Well"]
        source_dir = record["source_dir"]
        
        if not source_dir or not os.path.exists(str(source_dir)):
            print(f"Warning: Source directory not found for Plate {plate}, Day {day}, Well {well}")
            continue
        
        # Construct the source file paths.
        colony_src = os.path.join(source_dir, f"colony_{well}.tif")
        optics_src = os.path.join(source_dir, f"optics_{well}.png")
        outlines_src = os.path.join(os.path.join(os.path.dirname(source_dir), "colonies_outlines"), f"colony_{well}_outlines.tif")
        
        # Construct the destination folder for the specific day.
        dst_day_folder = os.path.join(dst_root, "incorrect_predictions", f"day_{day}")
        os.makedirs(dst_day_folder, exist_ok=True)
        
        # Define destination file names.
        colony_dst = os.path.join(dst_day_folder, f"plate_{plate}_colony_{well}.tif")
        optics_dst = os.path.join(dst_day_folder, f"plate_{plate}_colony_{well}_optics.png")
        outlines_dst = os.path.join(dst_day_folder, f"plate_{plate}_colony_{well}_outlines.tif")
        
        # Copy colony image file.
        if os.path.exists(colony_src):
            shutil.copy(colony_src, colony_dst)
        else:
            print(f"Warning: Colony image not found for well {well} in {source_dir}")
        
        # Copy optics image file.
        if os.path.exists(optics_src):
            shutil.copy(optics_src, optics_dst)
        else:
            print(f"Warning: Optics image not found for well {well} in {source_dir}")
            
        # Copy outlines image file.
        if os.path.exists(outlines_src):
            shutil.copy(outlines_src, outlines_dst)
        else:
            print(f"Warning: Outlines image not found for well {well} in {outlines_src}")
        
    print("Incorrect predictions have been organized successfully.")

def plot_plate_death(csv_path, metric_col='dead_live_ratio', output_folder=None):
    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Ensure "Relative Time (days)" is treated as an integer for grouping purposes
    df['Relative Time (days)'] = df['Relative Time (days)'].astype(int)
    
    # Filter the DataFrame to retain only wells with exactly 1 colony per (Plate, Day) group
    filtered_df = df.groupby(['Plate', 'Relative Time (days)', 'well']).filter(lambda group: len(group) == 1)
    
    # Compute the global minimum and maximum for the metric to use a global color scale.
    # These statistics are computed only from the valid data (ignoring potential NaNs).
    global_min = filtered_df[metric_col].min()
    global_max = np.nanpercentile(filtered_df[metric_col],90)
    
    # Get the unique days and plates (sorted)
    days = sorted(filtered_df['Relative Time (days)'].unique())
    plates = sorted(filtered_df['Plate'].unique())
    
    # Set up the figure grid: rows correspond to days, and columns to plates.
    fig, axes = plt.subplots(nrows=len(days), ncols=len(plates), figsize=(len(plates)*4, len(days)*4))
    
    # Ensure axes is a 2D array even if there's only one row or one column.
    if len(days) == 1 and len(plates) == 1:
        axes = np.array([[axes]])
    elif len(days) == 1:
        axes = np.array([axes])
    elif len(plates) == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Helper function to convert a well identifier (e.g., "A1") to numerical row and column indices.
    def well_to_indices(well):
        m = re.match(r"([A-Za-z]+)(\d+)", well)
        if m:
            row_letters = m.group(1).upper()
            col = int(m.group(2)) - 1  # zero-based index for columns
            # Convert row letters to a number (e.g., "A"->0, "B"->1, etc.)
            row = 0
            for char in row_letters:
                row = row * 26 + (ord(char) - ord('A') + 1)
            row = row - 1  # convert to zero-based index
            return row, col
        else:
            return None, None
    
    # Process each combination of day and plate.
    for i, day in enumerate(days):
        for j, plate in enumerate(plates):
            ax = axes[i, j]
            sub_df = filtered_df[(filtered_df['Relative Time (days)'] == day) & (filtered_df['Plate'] == plate)]
            
            # If no data exists for this combination, create an empty heatmap using a default plate size (8x12)
            if sub_df.empty:
                n_rows, n_cols = 8, 12
                heatmap = np.full((n_rows, n_cols), np.nan)
            else:
                # Copy sub-dataframe to avoid SettingWithCopyWarning
                sub_df = sub_df.copy()
                # Convert the "Well" identifiers to numeric indices for rows and columns.
                sub_df['Row_index'], sub_df['Col_index'] = zip(*sub_df['well'].map(well_to_indices))
                
                # Determine grid size from the available well positions.
                max_row = sub_df['Row_index'].max()
                max_col = sub_df['Col_index'].max()
                n_rows = int(max_row + 1)
                n_cols = int(max_col + 1)
                # Initialize a grid filled with NaN values
                heatmap = np.full((n_rows, n_cols), np.nan)
                for idx, row_data in sub_df.iterrows():
                    r = int(row_data['Row_index'])
                    c = int(row_data['Col_index'])
                    heatmap[r, c] = row_data[metric_col]
            
            # Plot a heatmap for this plate/day with global scale (vmin/vmax) applied;
            # no color bar is added.
            im = ax.imshow(heatmap, interpolation='none', cmap='viridis', vmin=global_min, vmax=global_max)
            ax.set_title(f"Plate {plate}, Day {day}")
            # Set tick marks for each column and row.
            ax.set_xticks(range(n_cols))
            ax.set_yticks(range(n_rows))
            # Convert row indices to letters (e.g., 0 -> 'A') and column indices to one-based numbering.
            ax.set_yticklabels([chr(i + ord('A')) for i in range(n_rows)])
            ax.set_xticklabels([str(i+1) for i in range(n_cols)])
    
    plt.tight_layout()
    
    # Define the output folder for the plot; if none is provided, use the CSV’s directory.
    if output_folder is None:
        output_folder = os.path.dirname(csv_path)
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the figure to a file.
    out_file = os.path.join(output_folder, f"plate_dead_ratio_heatmap_{metric_col}.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved plate heatmap plot to {out_file}")

import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
#  NOTE:
#  Ground‑truth death (‑1) is now assigned when **exactly one** colony is observed in
#  a well at time *t* **and** that colony has disappeared at the *very next recorded
#  time‑point* for the same well (regardless of whether that is t + 1 day, t + 2 days,
#  etc.).  If no later time‑point exists the sample is treated as *alive* (+1) because
#  its fate is unknown.
#
#  The optional ``ground_truth_csv`` parameter is retained for API compatibility but
#  is ignored under the new logic.
# --------------------------------------------------------------------------------------

TARGET_PRECISION: float = 0.50  # the 50 % PPV requirement
PREC_TOL: float = 0.02          # acceptable ± tolerance around the target

__all__ = ["optimize_death_threshold"]


def optimize_death_threshold(
    ground_truth_csv: str,  # kept for backward compatibility – ignored
    colonies_csv: str,
    metric_col: str = "dead_live_ratio",
    *,
    days_list: List[int] = (0, 3),
    threshold_range_flat: Tuple[float, float] = (0.0, 0.5),
    num_steps_flat: int = 300,
    std_range: Tuple[float, float] = (0.0, 5.0),
    num_steps_std: int = 50,
    mad_range: Tuple[float, float] = (0.0, 7.0),
    num_steps_mad: int = 50,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Optimise density thresholds separating *death* (‑1) from *alive* (+1).

    Parameters
    ----------
    ground_truth_csv
        Ignored – maintained for backward compatibility.
    colonies_csv
        CSV with per‑colony measurements. Must contain ``Plate``, ``well``/``Well``,
        ``Relative Time (days)``/``Day`` and the chosen ``metric_col``.
    metric_col
        Column holding the scalar score (default ``dead_live_ratio``).
    days_list
        Days evaluated individually **and** together under a "Combined" condition.
    threshold_range_flat, num_steps_flat
        Grid for the flat threshold search.
    std_range, num_steps_std
        Grid for ``mean + k·σ`` per‑plate search.
    mad_range, num_steps_mad
        Grid for ``median + k·MAD`` per‑plate search.

    Returns
    -------
    dict
        ``{method → {condition → result‑dict}}`` where *result‑dict* contains the
        best hyper‑parameter and the achieved precision, recall & f1.
    """

    # ------------------------------------------------------------------
    # 0. Load colonies CSV ------------------------------------------------
    try:
        df_col = pd.read_csv(colonies_csv)
    except Exception as exc:
        logging.error("Error reading colonies CSV %s: %s", colonies_csv, exc)
        return {}

    df_col.rename(columns={"well": "Well", "Relative Time (days)": "Day"}, inplace=True)
    df_col["Day"] = pd.to_numeric(df_col["Day"], errors="coerce")
    df_col["Plate"] = pd.to_numeric(df_col["Plate"], errors="coerce")

    req_cols = {"Plate", "Well", "Day", metric_col}
    missing = req_cols.difference(df_col.columns)
    if missing:
        raise ValueError(f"Missing expected columns in colonies CSV: {sorted(missing)}")

    # ------------------------------------------------------------------
    # 1. Derive ground‑truth labels from *next* recorded time‑point -------
    counts = (
        df_col.groupby(["Plate", "Well", "Day"], observed=False).size().reset_index(name="colony_count")
    )
    counts.sort_values(["Plate", "Well", "Day"], inplace=True)
    counts["next_colony_count"] = counts.groupby(["Plate", "Well"], observed=False)["colony_count"].shift(-1)

    single = counts[counts["colony_count"] == 1].copy()
    single["Status"] = np.where(
        (single["next_colony_count"].isna()) | (single["next_colony_count"] == 0),
        -1,
        1,
    )

    merged = df_col.merge(
        single[["Plate", "Well", "Day", "Status"]],
        on=["Plate", "Well", "Day"],
        how="inner",
        validate="many_to_one",
    )

    if merged.empty:
        logging.error("No eligible single‑colony wells found with a following time‑point.")
        return {}

    # ------------------------------------------------------------------
    # 2. Build data dictionaries -----------------------------------------
    def _by_plate(df: pd.DataFrame):
        out = {}
        for plate, grp in df.groupby("Plate", observed=False):
            out[plate] = {
                "metrics": grp[metric_col].to_numpy(float),
                "labels": grp["Status"].to_numpy(int),
            }
        return out

    data_by_day: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for d in days_list:
        subset = merged[merged["Day"] == d]
        if not subset.empty:
            data_by_day[str(d)] = _by_plate(subset)

    combined_data: Dict[str, Dict[str, np.ndarray]] = {}
    for d in days_list:
        subset = merged[merged["Day"] == d]
        if subset.empty:
            continue
        for plate, grp in subset.groupby("Plate", observed=False):
            combined_data[f"{d}|{plate}"] = {
                "metrics": grp[metric_col].to_numpy(float),
                "labels": grp["Status"].to_numpy(int),
            }

    all_conditions = {**data_by_day, "Combined": combined_data}

    # ------------------------------------------------------------------
    # 3. Metric helper ---------------------------------------------------
    def _metrics(vals: np.ndarray, labs: np.ndarray, thr: float):
        pred = np.where(vals > thr, -1, 1)
        tp = np.sum((pred == -1) & (labs == -1))
        fp = np.sum((pred == -1) & (labs == 1))
        fn = np.sum((pred == 1) & (labs == -1))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    # ------------------------------------------------------------------
    # 4. Helper to choose best index given metric arrays -----------------
    def _choose_best_idx(prec_arr: np.ndarray, rec_arr: np.ndarray):
        """Return index that maximises recall under precision≈TARGET_PRECISION.
        If *no* candidate meets the tolerance, choose the index whose precision
        is closest to the target (then still maximise recall among ties)."""
        ok = np.abs(prec_arr - TARGET_PRECISION) <= PREC_TOL
        if ok.any():
            idx = np.argmax(np.where(ok, rec_arr, -np.inf))
        else:
            # distance of every point from the target precision
            dist = np.abs(prec_arr - TARGET_PRECISION)
            best_dist = np.nanmin(dist)
            # among those closest, take the one with maximal recall
            candidates = np.where(dist == best_dist)[0]
            idx = candidates[np.argmax(rec_arr[candidates])]
        return idx

    # ------------------------------------------------------------------
    # 5. Grid‑search helpers -------------------------------------------
    def _search_flat():
        grid = np.linspace(*threshold_range_flat, num_steps_flat)
        metrics = {c: {"precision": [], "recall": [], "f1": []} for c in all_conditions}

        for t in grid:
            for cond, by_plate in all_conditions.items():
                rows = [_metrics(d["metrics"], d["labels"], t) for d in by_plate.values()]
                if rows:
                    p, r, f = zip(*rows)
                    metrics[cond]["precision"].append(np.mean(p))
                    metrics[cond]["recall"].append(np.mean(r))
                    metrics[cond]["f1"].append(np.mean(f))
                else:
                    metrics[cond]["precision"].append(np.nan)
                    metrics[cond]["recall"].append(np.nan)
                    metrics[cond]["f1"].append(np.nan)

        for cond in metrics:
            metrics[cond]["precision"] = np.array(metrics[cond]["precision"])
            metrics[cond]["recall"] = np.array(metrics[cond]["recall"])
            metrics[cond]["f1"] = np.array(metrics[cond]["f1"])

        best_idx = {c: _choose_best_idx(metrics[c]["precision"], metrics[c]["recall"]) for c in all_conditions}
        best_thr = {c: grid[best_idx[c]] for c in all_conditions}
        return grid, metrics, best_idx, best_thr

    def _search_k(method: str, rng: Tuple[float, float], steps: int):
        ks = np.linspace(*rng, steps)
        metrics = {c: {"precision": [], "recall": [], "f1": []} for c in all_conditions}
        avg_thr: Dict[str, List[float]] = {c: [] for c in all_conditions}

        for k in ks:
            for cond, by_plate in all_conditions.items():
                prec_list, rec_list, f1_list, thr_list = [], [], [], []
                for d in by_plate.values():
                    if method == "std":
                        m, s = np.mean(d["metrics"]), np.std(d["metrics"])
                        thr = m + k * s
                    else:  # mad
                        med = np.median(d["metrics"])
                        mad = np.median(np.abs(d["metrics"] - med))
                        thr = med + k * mad
                    p, r, f = _metrics(d["metrics"], d["labels"], thr)
                    prec_list.append(p)
                    rec_list.append(r)
                    f1_list.append(f)
                    thr_list.append(thr)

                if prec_list:
                    metrics[cond]["precision"].append(np.mean(prec_list))
                    metrics[cond]["recall"].append(np.mean(rec_list))
                    metrics[cond]["f1"].append(np.mean(f1_list))
                    avg_thr[cond].append(np.mean(thr_list))
                else:
                    metrics[cond]["precision"].append(np.nan)
                    metrics[cond]["recall"].append(np.nan)
                    metrics[cond]["f1"].append(np.nan)
                    avg_thr[cond].append(np.nan)

        for cond in metrics:
            metrics[cond]["precision"] = np.array(metrics[cond]["precision"])
            metrics[cond]["recall"] = np.array(metrics[cond]["recall"])
            metrics[cond]["f1"] = np.array(metrics[cond]["f1"])
            avg_thr[cond] = np.array(avg_thr[cond])

        best_idx = {c: _choose_best_idx(metrics[c]["precision"], metrics[c]["recall"]) for c in all_conditions}
        best_k = {c: ks[best_idx[c]] for c in all_conditions}
        best_avg_thr = {c: float(avg_thr[c][best_idx[c]]) for c in all_conditions}
        return ks, metrics, best_idx, best_k, best_avg_thr

    thr_flat, flat_metrics, idx_flat, best_thr_flat = _search_flat()
    ks_std, std_metrics, idx_std, best_k_std, avg_thr_std = _search_k("std", std_range, num_steps_std)
    ks_mad, mad_metrics, idx_mad, best_k_mad, avg_thr_mad = _search_k("mad", mad_range, num_steps_mad)

    # ------------------------------------------------------------------
    # 6. Plot -----------------------------------------------------------
    def _plot(ax, xs, m_dict, xlabel, best_x, metric, marker):
        for cond, col in zip(m_dict, plt.cm.tab10.colors):
            ys = m_dict[cond][metric]
            ax.plot(xs, ys, marker, color=col, label=cond)
            ax.axvline(best_x[cond], color=col, linestyle="--")
            idx = np.abs(xs - best_x[cond]).argmin()
            ax.axhline(ys[idx], color=col, linestyle=":", alpha=0.7)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Flat
    _plot(axes[0, 0], thr_flat, flat_metrics, "Threshold", best_thr_flat, "precision", "o")
    axes[0, 0].set_title("Flat – Precision")
    _plot(axes[0, 1], thr_flat, flat_metrics, "Threshold", best_thr_flat, "recall", "o")
    axes[0, 1].set_title("Flat – Recall")
    _plot(axes[0, 2], thr_flat, flat_metrics, "Threshold", best_thr_flat, "f1", "o")
    axes[0, 2].set_title("Flat – F1")

    # Std
    _plot(axes[1, 0], ks_std, std_metrics, "k (σ)", best_k_std, "precision", "s")
    axes[1, 0].set_title("Std – Precision")
    _plot(axes[1, 1], ks_std, std_metrics, "k (σ)", best_k_std, "recall", "s")
    axes[1, 1].set_title("Std – Recall")
    _plot(axes[1, 2], ks_std, std_metrics, "k (σ)", best_k_std, "f1", "s")
    axes[1, 2].set_title("Std – F1")

    # MAD
    _plot(axes[2, 0], ks_mad, mad_metrics, "k (MAD)", best_k_mad, "precision", "^")
    axes[2, 0].set_title("MAD – Precision")
    _plot(axes[2, 1], ks_mad, mad_metrics, "k (MAD)", best_k_mad, "recall", "^")
    axes[2, 1].set_title("MAD – Recall")
    _plot(axes[2, 2], ks_mad, mad_metrics, "k (MAD)", best_k_mad, "f1", "^")
    axes[2, 2].set_title("MAD – F1")

    plt.tight_layout()
    out_png = colonies_csv.replace(".csv", f"_optimal_death_threshold_{metric_col}.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def optimize_death_threshold_back(ground_truth_csv, colonies_csv, metric_col="dead_live_ratio", days_list=[0],
                               threshold_range_flat=(0, 0.5), num_steps_flat=300,
                               std_range=(0, 5), num_steps_std=50,
                               mad_range=(0, 7), num_steps_mad=50):
    # Load ground truth CSV and filter rows with Day > 1.
    try:
        df_gt = pd.read_csv(ground_truth_csv)
    except Exception as e:
        logging.error(f"Error reading ground truth CSV {ground_truth_csv}: {e}")
        return None
    df_gt["Day"] = pd.to_numeric(df_gt["Day"], errors="coerce")
    
    # Load colonies CSV and rename columns.
    try:
        df_col = pd.read_csv(colonies_csv)
    except Exception as e:
        logging.error(f"Error reading colonies CSV {colonies_csv}: {e}")
        return None
    df_col.rename(columns={"well": "Well", "Relative Time (days)": "Day"}, inplace=True)
    df_col["Day"] = pd.to_numeric(df_col["Day"], errors="coerce")
    df_col["Plate"] = pd.to_numeric(df_col["Plate"], errors="coerce")
    
    # Use only ground truth rows with Status 1 or -1
    df_gt = df_gt[df_gt["Status"].isin([1, -1])]
    
    # Merge on Plate, Well, and Day.
    merged = pd.merge(df_gt, df_col, on=["Plate", "Well", "Day"], how="inner")
    if merged.empty:
        logging.error("Merged DataFrame is empty. Check the CSV files and merge keys.")
        return None

    # Keep only wells with a single colony.
    well_counts = merged.groupby(["Plate", "Well", "Day"]).size().reset_index(name="count")
    single_colony_keys = well_counts[well_counts["count"] == 1][["Plate", "Well", "Day"]]
    merged = pd.merge(merged, single_colony_keys, on=["Plate", "Well", "Day"], how="inner")
    if merged.empty:
        logging.error("No wells with a single colony found after filtering.")
        return None

    # Build data for each individual day in days_list and for the combined condition.
    def get_plate_data(df):
        groups = df.groupby("Plate")
        plate_data = {}
        for plate, group in groups:
            metrics = group[metric_col].values
            true_status = group["Status"].values
            plate_data[plate] = {"metrics": metrics, "true_status": true_status}
        return plate_data

    data_by_day = {}
    for d in days_list:
        df_day = merged[merged["Day"] == d]
        if not df_day.empty:
            data_by_day[str(d)] = get_plate_data(df_day)
    
    # Combined condition: all rows corresponding to days in days_list.
    combined_data = get_plate_data(merged[merged["Day"].isin(days_list)])
    
    # Dictionary with all conditions: individual days plus "Combined".
    all_conditions = data_by_day.copy()
    all_conditions["Combined"] = combined_data

    # Helper function to compute metrics.
    def compute_metrics(values, true_status, threshold):
        pred_status = np.where(values > threshold, -1, 1)
        tp = np.sum((pred_status == -1) & (true_status == -1))
        fp = np.sum((pred_status == -1) & (true_status != -1))
        fn = np.sum((pred_status != -1) & (true_status == -1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    #############################
    # Method 1: Flat thresholds.
    flat_thresholds = np.linspace(threshold_range_flat[0], threshold_range_flat[1], num_steps_flat)
    flat_metrics = {cond: {"precision": [], "recall": [], "f1": []} for cond in all_conditions}

    for t in flat_thresholds:
        for cond, plate_data in all_conditions.items():
            metrics = [compute_metrics(data["metrics"], data["true_status"], t)
                       for data in plate_data.values()]
            if metrics:
                p, r, f = zip(*metrics)
                flat_metrics[cond]["precision"].append(np.mean(p))
                flat_metrics[cond]["recall"].append(np.mean(r))
                flat_metrics[cond]["f1"].append(np.mean(f))
            else:
                flat_metrics[cond]["precision"].append(np.nan)
                flat_metrics[cond]["recall"].append(np.nan)
                flat_metrics[cond]["f1"].append(np.nan)

    # Convert lists to numpy arrays.
    for cond in flat_metrics:
        flat_metrics[cond]["precision"] = np.array(flat_metrics[cond]["precision"])
        flat_metrics[cond]["recall"] = np.array(flat_metrics[cond]["recall"])
        flat_metrics[cond]["f1"] = np.array(flat_metrics[cond]["f1"])

    # Find best flat threshold (max F1) for each condition.
    best_idx_flat = {cond: np.nanargmax(flat_metrics[cond]["f1"]) for cond in all_conditions}
    best_threshold_flat = {cond: flat_thresholds[best_idx_flat[cond]] for cond in all_conditions}

    #############################
    # Method 2: Std method.
    k_std_candidates = np.linspace(std_range[0], std_range[1], num_steps_std)
    std_metrics = {cond: {"precision": [], "recall": [], "f1": []} for cond in all_conditions}

    for k in k_std_candidates:
        for cond, plate_data in all_conditions.items():
            metrics = []
            for data in plate_data.values():
                mean_val = np.mean(data["metrics"])
                std_val = np.std(data["metrics"])
                thresh = mean_val + k * std_val
                metrics.append(compute_metrics(data["metrics"], data["true_status"], thresh))
            if metrics:
                p, r, f = zip(*metrics)
                std_metrics[cond]["precision"].append(np.mean(p))
                std_metrics[cond]["recall"].append(np.mean(r))
                std_metrics[cond]["f1"].append(np.mean(f))
            else:
                std_metrics[cond]["precision"].append(np.nan)
                std_metrics[cond]["recall"].append(np.nan)
                std_metrics[cond]["f1"].append(np.nan)

    for cond in std_metrics:
        std_metrics[cond]["precision"] = np.array(std_metrics[cond]["precision"])
        std_metrics[cond]["recall"] = np.array(std_metrics[cond]["recall"])
        std_metrics[cond]["f1"] = np.array(std_metrics[cond]["f1"])

    best_idx_std = {cond: np.nanargmax(std_metrics[cond]["f1"]) for cond in all_conditions}
    best_k_std = {cond: k_std_candidates[best_idx_std[cond]] for cond in all_conditions}
    best_threshold_std = {}
    for cond, plate_data in all_conditions.items():
        # Average threshold across plates for the best k.
        thresholds = [np.mean(data["metrics"]) - best_k_std[cond] * np.std(data["metrics"])
                      for data in plate_data.values()]
        best_threshold_std[cond] = np.mean(thresholds) if thresholds else np.nan

    #############################
    # Method 3: MAD method.
    k_mad_candidates = np.linspace(mad_range[0], mad_range[1], num_steps_mad)
    mad_metrics = {cond: {"precision": [], "recall": [], "f1": []} for cond in all_conditions}

    for k in k_mad_candidates:
        for cond, plate_data in all_conditions.items():
            metrics = []
            for data in plate_data.values():
                med = np.median(data["metrics"])
                mad = np.median(np.abs(data["metrics"] - med))
                thresh = med + k * mad
                metrics.append(compute_metrics(data["metrics"], data["true_status"], thresh))
            if metrics:
                p, r, f = zip(*metrics)
                mad_metrics[cond]["precision"].append(np.mean(p))
                mad_metrics[cond]["recall"].append(np.mean(r))
                mad_metrics[cond]["f1"].append(np.mean(f))
            else:
                mad_metrics[cond]["precision"].append(np.nan)
                mad_metrics[cond]["recall"].append(np.nan)
                mad_metrics[cond]["f1"].append(np.nan)

    for cond in mad_metrics:
        mad_metrics[cond]["precision"] = np.array(mad_metrics[cond]["precision"])
        mad_metrics[cond]["recall"] = np.array(mad_metrics[cond]["recall"])
        mad_metrics[cond]["f1"] = np.array(mad_metrics[cond]["f1"])

    best_idx_mad = {cond: np.nanargmax(mad_metrics[cond]["f1"]) for cond in all_conditions}
    best_k_mad = {cond: k_mad_candidates[best_idx_mad[cond]] for cond in all_conditions}
    best_threshold_mad = {}
    for cond, plate_data in all_conditions.items():
        thresholds = [np.median(data["metrics"]) - best_k_mad[cond] * np.median(np.abs(data["metrics"] - np.median(data["metrics"])))
                      for data in plate_data.values()]
        best_threshold_mad[cond] = np.mean(thresholds) if thresholds else np.nan

    #############################
    # Plotting the facet grid.
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Define a helper for plotting metrics per method.
    def plot_metric(ax, x_vals, metrics_dict, x_label, best_vals, metric_name, marker):
        for cond in metrics_dict:
            line, = ax.plot(x_vals, metrics_dict[cond][metric_name], marker=marker, label=cond)
            # Draw a vertical line at the best x value for this condition.
            ax.axvline(best_vals[cond], color=line.get_color(), linestyle="--",
                       label=f"Best {cond}: {best_vals[cond]:.5f}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()
        ax.set_ylim(0, 1)
    
    # Row 1: Flat thresholds.
    plot_metric(axes[0, 0], flat_thresholds, flat_metrics, "Threshold", 
                best_threshold_flat, "precision", marker="o")
    axes[0, 0].set_title("Flat - Precision")
    plot_metric(axes[0, 1], flat_thresholds, flat_metrics, "Threshold", 
                best_threshold_flat, "recall", marker="o")
    axes[0, 1].set_title("Flat - Recall")
    plot_metric(axes[0, 2], flat_thresholds, flat_metrics, "Threshold", 
                best_threshold_flat, "f1", marker="o")
    axes[0, 2].set_title("Flat - F1 Score")
    
    # Row 2: Std method.
    plot_metric(axes[1, 0], k_std_candidates, std_metrics, "k (Std Deviations)", 
                best_k_std, "precision", marker="s")
    axes[1, 0].set_title("Std Dev - Precision")
    plot_metric(axes[1, 1], k_std_candidates, std_metrics, "k (Std Deviations)", 
                best_k_std, "recall", marker="s")
    axes[1, 1].set_title("Std Dev - Recall")
    plot_metric(axes[1, 2], k_std_candidates, std_metrics, "k (Std Deviations)", 
                best_k_std, "f1", marker="s")
    axes[1, 2].set_title("Std Dev - F1 Score")
    
    # Row 3: MAD method.
    plot_metric(axes[2, 0], k_mad_candidates, mad_metrics, "k (MAD units)", 
                best_k_mad, "precision", marker="^")
    axes[2, 0].set_title("MAD - Precision")
    plot_metric(axes[2, 1], k_mad_candidates, mad_metrics, "k (MAD units)", 
                best_k_mad, "recall", marker="^")
    axes[2, 1].set_title("MAD - Recall")
    plot_metric(axes[2, 2], k_mad_candidates, mad_metrics, "k (MAD units)", 
                best_k_mad, "f1", marker="^")
    axes[2, 2].set_title("MAD - F1 Score")
    
    plt.tight_layout()
    output_filename = colonies_csv.replace(".csv", f"_optimal_density_threshold_{metric_col}.png")
    plt.savefig(output_filename, dpi=150)
    plt.close()

    # Logging the best parameters for each condition.
    for cond in all_conditions:
        logging.info(f"Flat method {cond}: Best threshold = {best_threshold_flat[cond]:.5f}, F1 = {flat_metrics[cond]['f1'][best_idx_flat[cond]]:.4f}")
        logging.info(f"Std method {cond}: Best k = {best_k_std[cond]:.2f}, average threshold = {best_threshold_std[cond]:.5f}, F1 = {std_metrics[cond]['f1'][best_idx_std[cond]]:.4f}")
        logging.info(f"MAD method {cond}: Best k = {best_k_mad[cond]:.2f}, average threshold = {best_threshold_mad[cond]:.5f}, F1 = {mad_metrics[cond]['f1'][best_idx_mad[cond]]:.4f}")

    results = {
        "flat": {cond: {"best_threshold": best_threshold_flat[cond],
                        "F1": flat_metrics[cond]["f1"][best_idx_flat[cond]]} for cond in all_conditions},
        "std": {cond: {"best_k": best_k_std[cond],
                       "average_threshold": best_threshold_std[cond],
                       "F1": std_metrics[cond]["f1"][best_idx_std[cond]]} for cond in all_conditions},
        "mad": {cond: {"best_k": best_k_mad[cond],
                       "average_threshold": best_threshold_mad[cond],
                       "F1": mad_metrics[cond]["f1"][best_idx_mad[cond]]} for cond in all_conditions}
    }
    return results

def plate_mean_sem_jitter(
    csv_path: str,
    column_to_plot: str,
    *,
    time_col: str = "Relative Time (days)",
    plate_col: str = "Plate",
    figsize: tuple = (10, 6),
    capsize: int = 3,
    base_marker: str = "o",
    jitter_frac: float | None = 0.05,
    line_alpha: float = 0.35,
    elinewidth: float = 2.0,
    logy: bool = True,
) -> None:
    """Plot plate‑level mean ± SEM over time.

    Enhancements compared with the original version
    ----------------------------------------------
    1. **Dynamic ranking** – at every time point the plates are ordered from
       highest to lowest mean, and their x‑jitter is assigned according to
       that rank so error bars never overlap.
    2. **Clear visual separation** between the connecting lines (faint,
       semi‑transparent) and the points + error bars (full‑opacity).

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    column_to_plot : str
        Metric to plot on the y‑axis.
    time_col : str, default "Relative Time (days)"
        X‑axis column (numeric).
    plate_col : str, default "Plate"
        Column identifying the plate.
    figsize : tuple, default (10, 6)
        Figure size.
    capsize : int, default 3
        Length of the error‑bar caps.
    base_marker : str, default "o"
        Marker for mean points.
    jitter_frac : float | None, default 0.05
        Fraction of the **median time‑step** used as maximum jitter.
        If ``None`` or 0, no jitter is applied.
    line_alpha : float, default 0.35
        Transparency for the connecting lines (0–1).
    elinewidth : float, default 2.0
        Thickness for the error‑bar arms (vertical lines).
    logy : bool, default True
        If *True*, use a log‑scaled y‑axis.
    """

    import matplotlib.colors as mcolors
    
    # ---------- load & basic sanity checks ----------------------------------
    df = pd.read_csv(csv_path)
    for col in (column_to_plot, time_col, plate_col):
        if col not in df.columns:
            raise ValueError(f"Column “{col}” not in CSV.")

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # ---------- determine jitter amplitude ----------------------------------
    unique_times = np.sort(df[time_col].unique())
    if len(unique_times) > 1:
        median_dt = np.median(np.diff(unique_times))
    else:
        median_dt = 1.0  # fallback when we have a single time point

    jitter_amp = (jitter_frac or 0.0) * median_dt

    # ---------- compute mean & SEM for every plate / time -------------------
    stats = (
        df.groupby([time_col, plate_col])[column_to_plot]
          .agg(mean="mean", sem=sem)
          .reset_index()
    )

    # ---------- normalise to Plate 0 ---------------------------------------
    baseline = (
        stats[stats[plate_col] == 0][[time_col, "mean"]]
        .rename(columns={"mean": "baseline_mean"})
    )
    stats = stats.merge(baseline, on=time_col, how="left")
    stats["mean"] = stats["mean"] / stats["baseline_mean"]
    stats["sem"] = stats["sem"] / stats["baseline_mean"]

    CUSTOM_PALETTE = [
        "#f26d6a",  # coral-red
        "#e86ab2",  # pink
        "#83aabd",  # light-blue
        "#31abaa",  # teal
        "#4daa41",  # green
        "#a6931e",  # mustard-yellow
    ]
    # pre‑allocate a colour per plate (stable across the whole figure)
    plate_names = sorted(df[plate_col].unique())
    plate_colours = {
        p: CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)] for i, p in enumerate(plate_names)
    }

    # ---------- build per‑time ranked offsets -------------------------------
    rank_offsets = {}
    for t in unique_times:
        # sort plates present at this t by descending mean value
        subset = stats[stats[time_col] == t]
        subset = subset.sort_values("mean", ascending=False)
        n_here = len(subset)
        for i, (_, row) in enumerate(subset.iterrows()):
            offset = (i - (n_here - 1) / 2) * jitter_amp
            rank_offsets.setdefault(t, {})[row[plate_col]] = offset
    
    # ---------- plotting -----------------------------------------------------
    plt.figure(figsize=figsize)

    # ------------------------- Style adjustments ---------------------------
    # Apply a dark theme (black background, white text)
    plt.style.use("dark_background")
    plt.rcParams["savefig.facecolor"] = "black"

    
    plate_label_map = {
        0: "NTC",
        54: "CUL3",
        66: "MED23",
        70: "NIPBL",
        92: "BAP1",
        93: "BCKDK",
    }

    for p in plate_names:
        p_stats = stats[stats[plate_col] == p].sort_values(time_col)
        if p_stats.empty:
            continue  # plate has no data

        # build vector of jittered x positions based on the dynamic ranking
        jittered_time = p_stats[time_col] + [rank_offsets[t][p] for t in p_stats[time_col]]

        colour = plate_colours[p]
        faint_colour = list(mcolors.to_rgba(colour))
        faint_colour[3] = line_alpha  # adjust alpha only

        # 1. faint line connecting the means
        plt.plot(
            jittered_time,
            p_stats["mean"],
            lw=1.5,
            ls="-",
            marker=None,
            color=faint_colour,
            zorder=1,
        )

        # 2. points + error bars (opaque, drawn on top)
        plt.errorbar(
            jittered_time,
            p_stats["mean"],
            yerr=p_stats["sem"],
            fmt=base_marker,
            ms=5,
            ls="none",  # avoid re‑drawing the line inside errorbar
            ecolor=colour,
            elinewidth=elinewidth,
            capsize=capsize,
            color=colour,
            label=f"{plate_label_map[p]}",
            zorder=2,
        )

    plt.xlabel(time_col)
    # plt.ylabel(column_to_plot)
    # plt.title(f"{column_to_plot} over time – mean ± SEM per Plate (rank‑jittered)")
    plt.title(f"Dying Cells Per Colony by Target over Time")
    plt.ylabel("Percent Dying Cells Per Colony (Normalized to NTC)")
    plt.xlabel("Timepoint (days)")
    plt.tight_layout()
    plt.legend(frameon=False, ncol=2)
    if logy:
        plt.yscale("log")
    output_filename = csv_path.replace(".csv", f"{column_to_plot}_per_colony.png")
    plt.savefig(output_filename)
    plt.close() 

def dying_ratio_by_plate_day(
    csv_path: str,
    ratio_col: str = "dead_live_ratio_proximity",
    *,
    time_col: str = "Relative Time (days)",
    plate_col: str = "Plate",
    threshold_k: float = 1.33,
    save_csv: bool = True,
    plot: bool = True,
    figsize: tuple = (10, 6),
    save_plot: bool = True,
) -> pd.DataFrame:
    # ---------------- load & sanity checks ----------------------------------
    df = pd.read_csv(csv_path)
    for col in (ratio_col, time_col, plate_col):
        if col not in df.columns:
            raise ValueError(f'Column "{col}" not found in {csv_path}')

    # ensure numeric day + drop NAs in critical columns
    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, ratio_col])

    # ---------------- per‑day global stats & thresholds --------------------
    day_stats = (
        df.groupby(time_col)[ratio_col]
          .agg(mean="mean", std="std")
          .assign(threshold=lambda g: g["mean"] + threshold_k * g["std"])
    )

    # merge threshold and flag dying colonies
    df = df.merge(day_stats["threshold"], left_on=time_col, right_index=True)
    df["dying"] = df[ratio_col] > 0.27

    # ---------------- summarise per plate per day --------------------------
    result = (
        df.groupby([time_col, plate_col])["dying"]
          .agg(total_colonies="size", dying_colonies="sum")
          .assign(dying_ratio=lambda g: g["dying_colonies"] / g["total_colonies"])
          .reset_index()
          .sort_values([time_col, plate_col])
    )

    # ---------------- optional CSV -----------------------------------------
    if save_csv:
        out_csv = csv_path.replace(".csv", "_dying_ratio_by_plate_day.csv")
        result.to_csv(out_csv, index=False)
        print(f'Saved per‑plate/day summary to "{out_csv}"')

    # ---------------- optional plotting ------------------------------------
    if plot:
        plt.figure(figsize=figsize)
        for plate, grp in result.groupby(plate_col):
            plt.plot(
                grp[time_col],
                grp["dying_ratio"],
                marker="o",
                ls="-",
                label=f"Plate {plate}",
            )
        plt.xlabel(time_col)
        plt.ylabel("Dying colonies / total colonies")
        plt.title("Fraction of dying colonies per plate over time")
        plt.ylim(0, 1)
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()

        if save_plot:
            out_fig = csv_path.replace(".csv", "_dying_ratio_plot.png")
            plt.savefig(out_fig, dpi=300)
            print(f'Saved figure to "{out_fig}"')
        else:
            plt.show()
        plt.close()

    return result

def create_custom_viridis():
    # Get 256 colors from viridis using the new API.
    base = plt.colormaps['viridis'](np.linspace(0, 1, 256))
    # Prepend black so that index 0 is black.
    newcolors = np.vstack((np.array([0, 0, 0, 1]), base))
    return ListedColormap(newcolors)

def create_heatmaps(
    csv_file: str,
    value_col: str = "cell_density",
    simple: bool = True,
    histogram_bins = 60
):
    """
    Creates a grid of heat-maps (one per well) **and** a histogram showing the distribution
    of mean cell-densities across wells.

    Parameters
    ----------
    csv_file : str
        Path to the input CSV.
    value_col : str
        Column whose values are plotted.
    simple : bool
        Kept for API compatibility (unused here but left intact).
    histogram_bins : int | str
        Bin spec passed to `plt.hist`; default `"auto"` chooses a good rule automatically.
    """

    plt.switch_backend("Agg")   # compatibility with worker threads
    plt.ioff()                  # turn off interactive mode

    # ---------- Read & clean data ----------
    df = pd.read_csv(csv_file)
    df["Row"]    = df["Row"].str.upper().str[0].map(lambda x: ord(x) - ord("A")).astype(int)
    df["Column"] = df["Well"].str.extract(r"(\d+)").astype(int)
    if "Position" not in df.columns:
        df["Position"] = 1
    df["Position"]  = pd.to_numeric(df["Position"],  errors="coerce").astype(int)
    df[value_col]   = pd.to_numeric(df[value_col],   errors="coerce")
    df              = df.dropna(subset=["Row", "Column", "Position", value_col])

    # ---------- Figure-out plate layout ----------
    unique_rows = natsorted(df["Row"].unique())
    unique_cols = natsorted(df["Column"].unique())
    row_idx  = {r: i for i, r in enumerate(unique_rows)}
    col_idx  = {c: i for i, c in enumerate(unique_cols)}
    nrows, ncols = len(unique_rows), len(unique_cols)

    fig_side = 5
    fig_total, axs_total = plt.subplots(nrows, ncols,
                                        figsize=(ncols * fig_side, nrows * fig_side),
                                        squeeze=False)

    # ---------- Colour normalisation ----------
    non_zero  = df[df[value_col] > 0][value_col]
    vmin      = non_zero.min() if not non_zero.empty else 1e-6
    vmax      = non_zero.max() if not non_zero.empty else vmin * 10
    norm      = LogNorm(vmin=vmin, vmax=vmax)         # use LogNormZeroReserved if you have it
    cmap      = create_custom_viridis()

    # ---------- Plot each well ----------
    grouped = df.groupby("Well", sort=False)
    for well, well_df in grouped:
        well_df = well_df.sort_values("Position")

        r, c    = int(well_df["Row"].iloc[0]), int(well_df["Column"].iloc[0])
        ax      = axs_total[row_idx[r], col_idx[c]]

        # fill a square grid with NaNs first
        grid_n      = int(ceil(sqrt(well_df["Position"].max())))
        heat        = np.full((grid_n, grid_n), np.nan)

        pos0        = well_df["Position"] - 1
        xs          = (pos0 % grid_n).astype(int)
        ys          = (grid_n - 1 - (pos0 // grid_n)).astype(int)

        heat[ys, xs] = well_df[value_col].to_numpy()

        # stats
        mean_val = np.nanmean(heat)
        med_val  = np.nanmedian(heat)
        std_val  = np.nanstd(heat)
        cov_val  = std_val / mean_val if mean_val else np.nan

        ax.imshow(
            heat,
            cmap=cmap,
            norm=norm,
            extent=[0, grid_n, 0, grid_n],
            interpolation="nearest",
            origin="lower",
        )

        for (yy, xx), val in np.ndenumerate(heat):
            if np.isfinite(val):
                ax.text(xx + .5, yy + .5, f"{val:.1f}",
                        ha="center", va="center", fontsize=8,
                        color="white", backgroundcolor="black")

        ax.set_title(f"Well {well}\nμ={mean_val:.2f}, ẋ={med_val:.2f}, COV={cov_val:.2f}")
        ax.set_xlim(0, grid_n); ax.set_ylim(0, grid_n)
        ax.axis("off")

    plt.tight_layout()

    # ---------- Colour-bar ----------
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig_total.colorbar(sm, ax=axs_total.ravel().tolist(), shrink=0.5, extend="min")

    # ---------- Save heat-map grid -----------
    basename   = os.path.splitext(os.path.basename(csv_file))[0]
    heat_path  = os.path.join(os.path.dirname(csv_file),
                              f"{basename}_heatmap_{value_col}.png")
    fig_total.savefig(heat_path, dpi=200)
    plt.close(fig_total)
    print(f"Saved heat-maps to {heat_path}")

    # ======================================================================
    #  NEW:  Histogram of mean density *per well*
    # ======================================================================
    per_well_means = grouped[value_col].mean().dropna()

    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
    ax_hist.hist(per_well_means, bins=histogram_bins)
    ax_hist.set_xlabel(f"Mean {value_col} per well")
    ax_hist.set_ylabel("Number of wells")
    ax_hist.set_title("Histogram of per-well mean cell densities")
    plt.tight_layout()

    hist_path = os.path.join(os.path.dirname(csv_file),
                             f"{basename}_histogram_{value_col}.png")
    fig_hist.savefig(hist_path, dpi=200)
    plt.close(fig_hist)
    print(f"Saved histogram   to {hist_path}")


def main():
    ground_truth_csv = r"E:\MERGED_sspsygene_growthassay_colonies\ground_truth.csv"


    base_input = r"E:\MERGED_sspsygene_growthassay_colonies"
    model = "2025_03_03_23_34_16.794792_epoch_899_0.4_2"
    # all_colonies_csv = make_all_colonies_csv(base_input, model)

    all_colonies_csv = r'E:\\MERGED_sspsygene_growthassay_colonies\\2025_03_03_23_34_16.794792_epoch_899_0.4_2\\2025_03_03_23_34_16.794792_epoch_899_0.4_2_all_colonies.csv'

    # # plot_plate_death(all_colonies_csv, metric_col="dead_live_ratio_proximity",)
    # # plot_plate_death(all_colonies_csv, metric_col="dead_live_ratio_area_proximity",)
    # # plot_plate_death(all_colonies_csv, metric_col="dead_live_ratio_well",)

    optimize_death_threshold(ground_truth_csv, all_colonies_csv, metric_col="dead_live_ratio_proximity", days_list = [0, 3], threshold_range_flat = (0.0, 1.5))
    # optimize_death_threshold(ground_truth_csv, all_colonies_csv, metric_col="dead_live_ratio_area_proximity", days_list = [0, 3], threshold_range_flat = (0.0, 2.5))
    optimize_death_threshold(ground_truth_csv, all_colonies_csv, metric_col="dead_live_ratio_well", days_list = [0, 3], threshold_range_flat = (0.0, 60))

    # optimize_death_threshold(colonies_csv=all_colonies_csv, ground_truth_csv=None, metric_col="dead_live_ratio_proximity")
    # optimize_death_threshold(colonies_csv=all_colonies_csv, ground_truth_csv=None, metric_col="dead_live_ratio_area_proximity")
    # optimize_death_threshold(colonies_csv=all_colonies_csv, ground_truth_csv=None, metric_col="dead_live_ratio_well")

    # dying_ratio_by_plate_day(all_colonies_csv)

    # organize_incorrect_predictions(all_colonies_csv, ground_truth_csv, os.path.dirname(all_colonies_csv))

    # plot_plate_diagrams_by_day(all_colonies_csv, ground_truth_csv)

    # plot_confusion_matrix_from_csv(r"E:\MERGED_sspsygene_growthassay_colonies\ground_truth_with_simulated_prediction.csv", "Status", "Ground Truth", "simulated_prediction", "ClonaLiSA")
    # plot_confusion_matrix_from_csv(r"E:\MERGED_sspsygene_growthassay_colonies\ground_truth_with_simulated_prediction.csv", "Status", "Ground Truth", "manual", "Manual")
    # plot_confusion_matrix_from_csv(r"D:\sspsygene\colony_counting\merged_216px\truth.csv", "semi", "Ground Truth", "manual", "Manual")

    # plot_plate_diagrams_by_day(all_colonies_csv, ground_truth_csv)

    # optimize_density_threshold(ground_truth_csv, all_colonies_csv, metric_col = "density", days_list = [0,3,8])
    # optimize_density_threshold(ground_truth_csv, all_colonies_csv, metric_col = "reach_p99", days_list = [0,3,8], threshold_range_flat=(0,100))

    # plot_plate_colony_percentages(all_colonies_csv)
    # plot_plate_diagrams_by_day_w_manual(all_colonies_csv, ground_truth_csBv)
    # plot_plate_colony_percentages_single_figure(all_colonies_csv)
    # plot_plate_growth_rates(all_colonies_csv)

    plate_mean_sem_jitter(all_colonies_csv, "num_cells", logy=True)


if __name__ == "__main__":
    main()