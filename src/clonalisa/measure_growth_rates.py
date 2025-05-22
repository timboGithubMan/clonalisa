import os
import pandas as pd
import numpy as np
import math
from datetime import datetime
from itertools import product, combinations
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
from natsort import natsorted
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def logistic_growth_model(t, K, r, P0):
    return K / (1 + ((K - P0) / P0) * np.exp(-r * (t/24)))   

def calculate_best_fit_logistic_params(group, K, value_col):
    times = group['Relative Time (hrs)'].values
    densities = group[value_col].values
    P0 = densities[0]
    
    # Use curve_fit to find the best r value for the logistic model
    try:
        params, _ = curve_fit(lambda t, r: logistic_growth_model(t, K, r, P0), times, densities, p0=0.01)
        return params[0], P0  # Return the growth rate (r) and initial population (P0)
    except RuntimeError:
        return np.nan, np.nan  # If the fitting fails, return 'na'

def plot_cv_and_correlation_vs_capacity(output_dir, data_df, group_to_plot, value_col):
    # Ensure necessary data conversions
    data_df[['Merged Time', 'Relative Time (hrs)', value_col]] = data_df[['Merged Time', 'Relative Time (hrs)', value_col]].apply(pd.to_numeric)

    # Group by necessary columns
    unique_groups = (data_df.groupby([group_to_plot, 'Plate', 'Well', 'Position'])
                     if 'Position' in data_df.columns
                     else data_df.groupby([group_to_plot, 'Plate', 'Well']))

    step_size = 100000
    max_capacity = 20000000
    max_Pt = data_df[value_col].max()
    max_time = data_df['Merged Time'].max()
    min_capacity = np.ceil(max_Pt / step_size) * step_size
    capacities = range(int(min_capacity), max_capacity, step_size)  # Adjust range and step as needed

    avg_cvs = []
    median_cvs = []
    min_avg_cv = float('inf')
    min_median_cv = float('inf')
    best_avg_cv_capacity = max_capacity
    best_median_cv_capacity = max_capacity

    slopes = []
    median_slopes = []
    min_slope = float('inf')
    min_median_slope = float('inf')
    best_slope_capacity = max_capacity
    best_median_slope_capacity = max_capacity

    for K in capacities:
        growth_rates = []
        # Calculate logistic fit parameters for each group at the current carrying capacity K
        for group_keys, group in unique_groups:
            group_sorted = group.sort_values('Merged Time')
            if len(group_sorted) > 1:
                r, P0 = calculate_best_fit_logistic_params(group_sorted, K, value_col)
                if not np.isnan(r):
                    for _, row in group_sorted.iterrows():
                        if isinstance(group_keys, tuple):  # When group_keys is a tuple
                            group_dict = {group_to_plot: group_keys[0],
                                          'Plate': row['Plate'],
                                          'Well': row['Well'],
                                          'K': K,
                                          'P0': P0,
                                          'logistic_growth': r}
                        else:
                            group_dict = {group_to_plot: group_keys,
                                          'Plate': row['Plate'],
                                          'Well': row['Well'],
                                          'K': K,
                                          'P0': P0,
                                          'logistic_growth': r}
                        growth_rates.append(group_dict)

        growth_rates_df = pd.DataFrame(growth_rates)
        if growth_rates_df.empty:
            continue

        # Compute coefficient of variation (CV) metrics
        cv_grouped = growth_rates_df.groupby([group_to_plot])['logistic_growth'].apply(
            lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan)
        avg_cv = cv_grouped.mean()
        avg_cvs.append(avg_cv)
        median_cv = cv_grouped.median()
        median_cvs.append(median_cv)
        if median_cv < min_median_cv:
            min_median_cv = median_cv
            best_median_cv_capacity = K
        if avg_cv < min_avg_cv:
            min_avg_cv = avg_cv
            best_avg_cv_capacity = K
        
        # Instead of computing correlations, compute the slope of the linear regression
        slopes_within_groups = []
        for _, group in growth_rates_df.groupby([group_to_plot]):
            group_data = group.dropna(subset=['P0', 'logistic_growth'])
            if group_data.shape[0] > 1:
                # Compute the slope (and intercept) of the best-fit line
                slope, intercept = np.polyfit(group_data['P0'], group_data['logistic_growth'], 1)
                slopes_within_groups.append(slope)
        if len(slopes_within_groups) == 0:
            continue
        mean_slope = np.mean(slopes_within_groups)
        slopes.append(mean_slope)
        if abs(mean_slope) < abs(min_slope):
            min_slope = mean_slope
            best_slope_capacity = K
        median_slope = np.median(slopes_within_groups)
        median_slopes.append(median_slope)
        if abs(median_slope) < abs(min_median_slope):
            min_median_slope = median_slope
            best_median_slope_capacity = K

    # Plot the CV and Slope vs. Carrying Capacity
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # CV Values plot on primary y-axis
    ax1.set_xlim(0, max_capacity)
    color = 'tab:purple'
    ax1.set_xlabel('Carrying Capacity')
    ax1.set_ylabel('CV Values', color=color)
    ax1.plot(capacities, median_cvs, marker='x', color='tab:purple', label='Median CV')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(x=best_median_cv_capacity, color='purple', linestyle='--',
                label=f'Best Median CV: {best_median_cv_capacity}, {min_median_cv}')
    lines, labels = ax1.get_legend_handles_labels()

    # Slope Values plot on secondary y-axis
    ax2 = ax1.twinx()
    ax2.spines["right"].set_position(("axes", 1.2))
    color = 'tab:red'
    ax2.set_ylabel('Slope of Fit', color=color)
    ax2.plot(capacities, median_slopes, marker='o', color='tab:red', label='Median Slope')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axvline(x=best_median_slope_capacity, color='red', linestyle='--',
                label=f'Best Median Slope: {best_median_slope_capacity}, {min_slope}')
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Combine legends from both axes
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Carrying Capacity Metrics')
    fig.tight_layout()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'stdev_and_slope_vs_capacity_{max_time}_{group_to_plot}_{value_col}.png'),
                bbox_inches='tight', dpi=400)
    plt.close()

    return best_median_cv_capacity, best_median_slope_capacity

def add_growth_rate_columns(df, group_col, value_cols, output_folder):
    df = merge_timepoints(df, time_threshold=12)
    df.sort_values(by=[group_col] + ['Merged Time'], inplace=True)
    
    unique_times = df['Merged Time'].unique()

    for i in range(1, len(unique_times)):
        current_time = unique_times[i]
        prev_time = unique_times[i - 1]
        sub_df = df[df['Merged Time'] <= current_time]

        output_dir = os.path.join(output_folder, f"{int(sub_df['Merged Time'].max())}")
        for value_col in value_cols:
            per_well_data = sub_df.groupby([group_col, 'Plate', 'Well']).filter(lambda group: group['Merged Time'].max() > prev_time)

            best_capacity_CV, best_capacity_corr = plot_cv_and_correlation_vs_capacity(output_dir, per_well_data, group_col, value_col)
            unique_groups = per_well_data.groupby([group_col, 'Plate', 'Well'])
            for _, group in unique_groups:
                if len(group) > 1:
                    for idx in group.index:     
                        if group.loc[idx, 'Merged Time'] == current_time:
                            # Calculate growth rates using the best capacities
                            exp_r, P0 = calculate_best_fit_logistic_params(group, 1e9, value_col)
                            df.at[idx, f'exp_rate_{value_col}'] = exp_r * 100
                            logistic_r_cv, _ = calculate_best_fit_logistic_params(group, best_capacity_CV, value_col)
                            df.at[idx, f'logistic_k_CV_fit_{value_col}'] = logistic_r_cv * 100
                            logistic_r_corr, _ = calculate_best_fit_logistic_params(group, best_capacity_corr, value_col)
                            df.at[idx, f'logistic_k_corr_fit_{value_col}'] = logistic_r_corr * 100
    return df

def extract_time_from_folder(folder_name):
    # Remove trailing slash if present
    folder_name = folder_name.rstrip('\\').rstrip('/')

    parts = folder_name.split('_')
    date_str = parts[-2]
    time_str = parts[-1]
    return datetime.strptime(f'{date_str} {time_str}', '%Y%m%d %H%M%S')

def custom_agg(x):
    if np.issubdtype(x.dtype, np.number):
        return x.mean()  # For numerical columns, calculate the mean
    else:
        # For categorical columns, return the first value
        # This assumes all values within each group are the same for categorical columns
        return x.iloc[0]

def merge_timepoints(df, time_threshold=8):
    df = df.rename(columns={'Relative Time (hrs)': 'Relative_Time_unshifted'})
    
    # 2) Shift each PlateWell’s times so its minimum is 0
    df['Relative Time (hrs)'] = (
        df.groupby('PlateWell')['Relative_Time_unshifted']
          .transform(lambda x: x - x.min())
    )
    
    # 3) Sort by the shifted time
    df = df.sort_values(by='Relative Time (hrs)')
    
    # Create a new column for merged timepoints
    df['Merged Time'] = df['Relative Time (hrs)']

    # Merge timepoints
    for i in range(len(df) - 1):
        if df.iloc[i + 1]['Relative Time (hrs)'] - df.iloc[i]['Relative Time (hrs)'] <= time_threshold:
            df.iloc[i + 1, df.columns.get_loc('Merged Time')] = df.iloc[i]['Merged Time']

    return df

def format_scientific(num):
    """Custom format for scientific notation to remove leading zero in exponent."""
    return "{:.0e}".format(num).replace('e-0', 'e-').replace('e+0', 'e+')

def calculate_growth_rates_per_well(main_model_all_data_csv, dead_model_all_data_csv=None, minimum_density=0, time_upper=np.inf, time_lower=-np.inf):
    main_model_data = pd.read_csv(main_model_all_data_csv)
    model_name = os.path.basename(main_model_all_data_csv).split("_all_data")[0]
    output_folder = os.path.dirname(main_model_all_data_csv)

    mask_inside = main_model_data['Relative Time (hrs)'].between(time_lower, time_upper, inclusive='both')
    main_model_data = main_model_data[~mask_inside]      # the tilde (~) negates the mask

    group_columns = [col for col in main_model_data.columns if "Group" in col and not "merged" in col]
    if not group_columns:
        group_columns = ['Plate']
    main_model_data['combined_group'] = main_model_data[group_columns].astype(str).agg('_'.join, axis=1)

    value_cols = ['cell_density']
    
    per_well_data = main_model_data.groupby(['PlateWell', 'Relative Time (hrs)']).agg(custom_agg).reset_index()
    per_well_data.drop('Position', axis=1, inplace=True)
    # per_well_data = per_well_data[per_well_data['Column'] != 2]
    # per_well_data = per_well_data[per_well_data['Column'] != 11]
    per_well_data = add_growth_rate_columns(per_well_data, 'combined_group', value_cols, output_folder)
    per_well_data['cell_density_p0'] = per_well_data.groupby(['Plate', 'Well'])['cell_density'].transform(lambda x: x.iloc[0])

    per_well_data_csv_path = os.path.join(output_folder, f'{model_name}_per_well_data.csv')
    per_well_data.to_csv(per_well_data_csv_path, index=False)
    for value_col in value_cols:
        plot_growth_rates_bars_and_scatter(os.path.join(output_folder, f'{model_name}_per_well_data.csv'), value_col)
    return per_well_data_csv_path

def is_nested(df, parent_cols, child_cols):
    """
    Check if child grouping is nested within parent grouping.

    Parameters:
    - df: DataFrame to check.
    - parent_cols: List of parent grouping columns.
    - child_cols: List of child grouping columns.

    Returns:
    - Boolean indicating if child is nested within parent.
    """
    parent_groups = df.groupby(list(parent_cols)).ngroups
    combined_groups = df.groupby(list(child_cols)).ngroups
    return combined_groups == parent_groups

def plot_bar_with_points(ax, agg_data, df, group_cols, method, bar_colors):
    """
    Plots a bar graph with individual data points overlaid, showing mean with 95% confidence interval.

    Parameters:
    - ax: Matplotlib Axes object to plot on.
    - agg_data: Aggregated DataFrame containing mean and standard error.
    - df: Original DataFrame containing individual data points.
    - group_cols: List of columns used for grouping.
    - method: The method/metric being plotted.
    - bar_colors: List of colors for each bar based on group.
    """
    # Create x-axis labels by joining group columns
    x_labels = agg_data.apply(lambda row: '_'.join([str(row[col]) for col in group_cols]), axis=1)
    x = np.arange(len(agg_data))
    
    # Calculate the 95% confidence interval multiplier
    # Using t-distribution for more accurate CI with small sample sizes
    ci_multiplier = agg_data['Count'].apply(lambda n: stats.t.ppf(0.975, df=n-1) if n > 1 else np.nan)
    
    # Calculate the error bars (upper and lower bounds)
    # yerr in bar plot expects the error magnitude (half the CI width)
    agg_data['ci_lower'] = agg_data['Mean_Growth_Rate'] - 1.96 * agg_data['Std_Err']
    agg_data['ci_upper'] = agg_data['Mean_Growth_Rate'] + 1.96 * agg_data['Std_Err']

    lower_errors = (agg_data['Mean_Growth_Rate'] - agg_data['ci_lower']).tolist()
    upper_errors = (agg_data['ci_upper'] - agg_data['Mean_Growth_Rate']).tolist()
    
    # Plot bars with 95% CI error bars
    bars = ax.bar(
        x, 
        agg_data['Mean_Growth_Rate'], 
        yerr=[lower_errors, upper_errors], 
        capsize=5, 
        color=bar_colors, 
        edgecolor='black', 
        linewidth=1.5,  # Thickness of bar edges
        error_kw={'elinewidth': 2, 'ecolor': 'black'},
        zorder=1,
        label='Mean with 95% CI'
    )
    
    # Scatter the individual data points colored black with black edges
    for idx, row in agg_data.iterrows():
        subset = df.copy()
        for col in group_cols:
            subset = subset[subset[col] == row[col]]
        y = subset[method].dropna()
        if len(y) == 0:
            continue  # Skip if no data points in the subset
        # Add jitter to x-axis for better visibility
        x_jittered = np.random.normal(x[idx], 0.05, size=len(y))
        ax.scatter(
            x_jittered, 
            y, 
            color='black',  # Black color for data points
            edgecolor='black',  # Black edge for points
            alpha=0.6, 
            s=30,  # Smaller size for data points
            zorder=2  # Ensure points are above error bars
        )
    
    # Set x-axis labels and positions
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Set axis labels and title (customize as needed)
    ax.set_ylabel('Mean Growth Rate')
    ax.set_xlabel('Groups')
    ax.set_title(f'Mean {method} with 95% Confidence Interval')
    
    # Optionally, add a legend
    ax.legend()
    
    # Improve layout
    plt.tight_layout()

def plot_scatter(ax, agg_data, df, group_cols, method, bar_colors, color_dict):
    """
    Plots a scatter plot of Mean P0 vs Mean Growth Rate.

    Parameters:
    - ax: Matplotlib Axes object to plot on.
    - agg_data: Aggregated DataFrame containing mean values.
    - df: Original DataFrame containing individual data points.
    - group_cols: List of columns used for grouping.
    - method: The method/metric being plotted.
    - bar_colors: List of colors for aggregated points based on group.
    - color_dict: Dictionary mapping group labels to colors.
    """
    # Plot aggregated mean points
    aggregated_scatter = ax.scatter(
        agg_data['Mean_P0'],
        agg_data['Mean_Growth_Rate'],
        c=bar_colors,  # Use the same bar colors
        marker='D',
        edgecolors='black',  # Black edge for aggregated points
        s=140,
        label='Aggregated Mean'
    )
    
    # Plot individual data points colored by group
    scatter = ax.scatter(
        df['P0'],
        df[method],
        c=df['group_label'].map(color_dict),  # Color based on group
        cmap='tab10' if len(color_dict) <= 10 else 'tab20',
        alpha=1,
        edgecolors='black',  # Black edge for individual points
        s=70,
        label='Individual Data Points'
    )
    
    ax.set_xlabel('P0')
    ax.set_ylabel('Growth Rate')
    
    return aggregated_scatter, scatter  # Return scatter objects for legend

def plot_growth_rates_bars_and_scatter(csv_file1, value_col):
    """
    Main function to plot growth rates as bar graphs with overlaid points and scatter plots.

    Parameters:
    - csv_file1: Path to the input CSV file.
    - value_col: Column representing the initial cell density P0.
    """
    # Read and preprocess the data
    og_df = pd.read_csv(csv_file1)
    og_df = og_df.groupby(['Plate', 'Well']).filter(lambda x: len(x['Merged Time'].unique()) > 1)

    # Identify group columns
    group_columns = [col for col in og_df.columns if ("Group" in col and not "_Group" in col and not "merged" in col)]

    # Generate all combinations of group_columns
    group_cols_combinations = []
    for r in range(1, len(group_columns)+1):
        combinations_r = list(combinations(group_columns, r))
        group_cols_combinations.extend(combinations_r)

    # Remove redundant groupings due to nesting
    non_redundant_group_cols_list = []
    for group_cols in group_cols_combinations:
        is_redundant = False
        for other_group_cols in group_cols_combinations:
            if set(group_cols) != set(other_group_cols) and set(group_cols).issubset(set(other_group_cols)):
                if is_nested(og_df, other_group_cols, group_cols):
                    is_redundant = True
                    break
        if not is_redundant:
            non_redundant_group_cols_list.append(group_cols)

    non_redundant_group_cols_list.append(('Row',))
    non_redundant_group_cols_list.append(('Column',))

    # Define growth rate methods
    growth_rate_methods = [
        f'exp_rate_{value_col}',
        f'logistic_k_CV_fit_{value_col}',
        f'logistic_k_corr_fit_{value_col}',
    ]

    fixed_height_per_plot = 8  # Height per subplot in inches

    for current_time in og_df['Merged Time'].unique()[1:]:
        for method in growth_rate_methods:
            # Check if method column exists and has data
            if method not in og_df.columns:
                print(f"Column {method} not found in DataFrame.")
                continue
            if not og_df[method].any():
                print(f"Column {method}.any() is False")
                continue

            for rate_type in ['Growth Rate', 'Doubling Time']:
                num_plots = len(non_redundant_group_cols_list)
                num_rows = math.ceil(num_plots / 2)
                fig_height = num_rows * fixed_height_per_plot
                num_cols = 4
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(32, fig_height), squeeze=False)
                # axs is always a 2D array

                for plot_idx, group_cols in enumerate(non_redundant_group_cols_list):
                    # Add these lines to calculate row and column
                    row = plot_idx // 2
                    col_base = (plot_idx % 2) * 2

                    if isinstance(group_cols, tuple):
                        group_cols = list(group_cols)

                    df_group_cols = list(group_cols) + ['Plate', 'Well'] + (['Position'] if 'Position' in og_df.columns else [])
                    df = og_df.sort_values(by=df_group_cols + ['Merged Time'])
                    df['P0'] = df.groupby(df_group_cols)[value_col].transform('first')
                    df = df[df['Merged Time'] == current_time]
                    df = df.dropna(subset=group_cols + [method])

                    if df.empty:
                        continue

                    df = df.dropna(subset=[method])

                    if rate_type == 'Doubling Time':
                        df[method] = np.log(2) / (df[method]/100)

                    agg_data = df.groupby(group_cols).agg(
                        Mean_P0=('P0', 'mean'),
                        Mean_Growth_Rate=(method, 'mean'),
                        Std_Dev=(method, 'std'),
                        Count=(method, 'count')  # Needed to calculate standard error
                    ).reset_index()

                    # Calculate Standard Error using std / sqrt(n)
                    agg_data['Std_Err'] = agg_data['Std_Dev'] / np.sqrt(agg_data['Count'])
                    agg_data['Std_Error_Percent'] = agg_data['Std_Err']  / agg_data['Mean_Growth_Rate']
                    

                    # Calculate Coefficient of Variation (CV)
                    agg_data['CV'] = agg_data['Std_Dev'] / agg_data['Mean_Growth_Rate']

                    # Calculate average and median CV
                    avg_stderr = agg_data['Std_Error_Percent'].mean()
                    median_stderr = agg_data['Std_Error_Percent'].median()

                    # Calculate correlation between 'P0' and the specified method within each group
                    corr_data = df.groupby(group_cols).apply(lambda x: x['P0'].corr(x[method]))
                    avg_corr = corr_data.mean()
                    median_corr = corr_data.median()

                    # Assign a unique color to each group
                    if len(group_cols) == 1:
                        # Single group column
                        unique_groups = agg_data[group_cols[0]].astype(str)
                    else:
                        # Multiple group columns
                        unique_groups = agg_data[group_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

                    unique_group_labels = unique_groups.unique()
                    num_unique_groups = len(unique_group_labels)

                    # Use a colormap with enough distinct colors
                    cmap = plt.get_cmap('tab10') if num_unique_groups <= 10 else plt.get_cmap('tab20')
                    colors = cmap(np.linspace(0, 1, num_unique_groups))
                    color_dict = dict(zip(unique_group_labels, colors))

                    # Prepare bar colors based on group
                    bar_colors = agg_data.apply(lambda row: color_dict['_'.join([str(row[col]) for col in group_cols])], axis=1).tolist()

                    # Assign group labels to individual data points for scatter plot coloring
                    if len(group_cols) == 1:
                        df['group_label'] = df[group_cols[0]].astype(str)
                    else:
                        df['group_label'] = df[group_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

                    # Plot bar with points
                    ax_bar = axs[row, col_base]
                    plot_bar_with_points(ax_bar, agg_data, df, group_cols, method, bar_colors)
                    group_cols_str = ', '.join([col.replace("Group-", "") for col in group_cols])
                    ax_bar.set_title(f'{method}\nAvg_StdErr (% of mean): {avg_stderr:.4f} Median_StdErr: {median_stderr:.4f}')

                    if rate_type == 'Doubling Time':
                        ax_bar.set_ylabel('Doubling Time (days)')
                    else:
                        ax_bar.set_ylabel('Growth Rate (% per day)')

                    # Plot scatter with all individual data points
                    ax_scatter = axs[row, col_base + 1]
                    aggregated_scatter, scatter = plot_scatter(ax_scatter, agg_data, df, group_cols, method, bar_colors, color_dict)

                    ax_scatter.set_title(
                        f'Growth Rate vs P0 - Groups: {group_cols_str}\nAvg_corr: {avg_corr:.4f} Median_corr: {median_corr:.4f}'
                    )
                    ax_scatter.set_xlabel('P0')

                    if rate_type == 'Doubling Time':
                        ax_scatter.set_ylabel('Doubling Time')
                    else:
                        ax_scatter.set_ylabel('Growth Rate (% per day)')

                # Adjust layout to prevent overlap, leaving space on the right for the centralized legend
                plt.tight_layout(rect=[0, 0, 0.95, 1])  # [left, bottom, right, top]

                # Save the figure for the method and current_time
                output_dir = os.path.join(os.path.dirname(csv_file1), f"{int(current_time)}", rate_type)
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(output_dir, f'{current_time}_{method}.png'),
                    bbox_inches='tight',
                    dpi=400
                )
                plt.close()

def perform_anova_followed_by_tukey(df, metric_column, ANOVA_group, csv_file=None):
    import pandas as pd
    from scipy import stats

    p_values = {}
    results_list = []  # Will hold all results to be saved if csv_file is provided

    # Get unique values for 'Merged Time'
    timepoints = df['Merged Time'].unique()

    for time in timepoints:
        p_values[time] = {}
        try:
            # Filter out rows with missing values for the metric or grouping column
            sub_df = df[(df['Merged Time'] == time) & 
                        df[metric_column].notna() & 
                        df[ANOVA_group].notna()]
            
            # Group data by the specified ANOVA group
            groups = [sub_df[sub_df[ANOVA_group] == group][metric_column] 
                      for group in sub_df[ANOVA_group].unique()]
            
            # Perform ANOVA
            f_stat, p_val = stats.f_oneway(*groups)
            
            # Save ANOVA results in the results list
            results_list.append({
                'Time': time,
                'Test': 'ANOVA',
                'f_stat': f_stat,
                'p_val': p_val,
                'group1': None,
                'group2': None,
                'meandiff': None,
                'lower': None,
                'upper': None,
                'reject': None,
                'full_precision_p_value': None
            })
            
            # If ANOVA shows significant differences, proceed with Tukey HSD for pairwise comparisons
            if p_val < 0.05:
                tukey_results = pairwise_tukeyhsd(sub_df[metric_column], sub_df[ANOVA_group])
                comparisons = tukey_results.summary().data[1:]  # Skip header row

                for i, comparison in enumerate(comparisons):
                    group1, group2, meandiff, _, lower, upper, reject = comparison
                    full_precision_p_value = tukey_results.pvalues[i]

                    # Save Tukey HSD results in the results list
                    results_list.append({
                        'Time': time,
                        'Test': 'Tukey',
                        'f_stat': None,
                        'p_val': None,
                        'group1': group1,
                        'group2': group2,
                        'meandiff': meandiff,
                        'lower': lower,
                        'upper': upper,
                        'reject': reject,
                        'full_precision_p_value': full_precision_p_value
                    })

                    # Save the significant pairwise comparison in p_values (if the test rejects the null)
                    if reject:
                        p_values[time][(group1, group2)] = full_precision_p_value

        except Exception as e:
            print(f'Skipping ANOVA for time {time} due to error: {e}')

    # If a csv_file is provided, save the entire results to a CSV file.
    if csv_file:
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(csv_file.replace(".csv", f"_ANOVA_{metric_column}.csv"), index=False)

    return p_values

def plot_over_time(csv_file, group_to_plot="", y_values="", p_values_col="", log=True):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=[group_to_plot])

    # Sort groups using natsort
    groups = natsorted(df[group_to_plot].unique())
    group_offsets = np.linspace(-2, 2, len(groups))

    # Create figure with two subfigures: one for ANOVA text, one for the plot
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 1, height_ratios=[1])
    # ax_text = fig.add_subplot(gs[0])
    ax_plot = fig.add_subplot(gs[0])

    cmap = plt.get_cmap('nipy_spectral')
    color_cycle = [cmap(i/len(groups)) for i in range(len(groups))]
    group_colors = {group: color for group, color in zip(groups, color_cycle)}

    group_means = {}
    for i, group in enumerate(groups):
        group_data = df[df[group_to_plot] == group]
        ax_plot.scatter(group_data['Relative Time (hrs)'] + group_offsets[i], group_data[y_values],
                    color=group_colors[group], label=group, alpha=0.7)
        mean_group_data = group_data.groupby('Merged Time')[y_values].mean().reset_index()
        ax_plot.plot(mean_group_data['Merged Time'], mean_group_data[y_values], 
                 color=group_colors[group], alpha=0.9, linewidth=2.0)
        group_means[group] = mean_group_data[y_values].mean()

    p_values = perform_anova_followed_by_tukey(df, f'{p_values_col}', group_to_plot, csv_file)
    
    if log:
        ax_plot.set_yscale('log')
        ymin, ymax = df[y_values][df[y_values] > 0].min(), df[y_values].max()
        dynamic_range = ymax / ymin
        ax_plot.set_ylim(ymin / np.power(dynamic_range, 1/16), ymax * np.power(dynamic_range, 1/16))
    else:
        ymin, ymax = df[y_values].min(), df[y_values].max()
        ax_plot.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))

    # # Add ANOVA text to the text axes
    # ax_text.axis('off')
    # for time, pairs in p_values.items():
    #     x_pos = ax_plot.transData.transform((time, 0))[0]
    #     x_pos = ax_text.transAxes.inverted().transform((x_pos, 0))[0]
    #     y_pos = 1.0
    #     for (group1, group2), p_val in sorted(pairs.items(), key=lambda x: (x[1], max(group_means.get(x[0][0], float('-inf')), group_means.get(x[0][1], float('-inf')))), reverse=True):
    #         if group_means[group2] > group_means[group1]:
    #             group1, group2 = group2, group1
    #         stars = '*' * sum([p_val < threshold for threshold in [0.05, 0.01, 0.001]])
    #         group1_color = group_colors[group1]
    #         group2_color = group_colors[group2]
            
    #         ax_text.text(x_pos - 0.015, y_pos, f'{group1}', ha='right', va='top', fontsize=8,
    #                 color=group1_color, fontweight='bold', transform=ax_text.transAxes)
    #         ax_text.text(x_pos + 0.015, y_pos, f'{group2}', ha='left', va='top', fontsize=8,
    #                 color=group2_color, fontweight='bold', transform=ax_text.transAxes)
    #         ax_text.text(x_pos, y_pos, f'{stars}', ha='center', va='top', fontsize=8,
    #                 color=group1_color if p_val < 0.05 else group2_color, fontweight='bold', transform=ax_text.transAxes)
            
    #         y_pos -= 0.1  # Adjust this value to change vertical spacing
    ax_plot.set_title(f'{p_values_col}', color='black')

    ax_plot.set_xlabel('Relative Time (hrs)')
    ax_plot.set_ylabel(y_values)
    ax_plot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    output_file = os.path.join(os.path.dirname(csv_file), f'{group_to_plot}_{y_values}_{p_values_col}.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()