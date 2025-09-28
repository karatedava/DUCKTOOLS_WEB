"""
MODULE PURPOSE
general functions independet on the model itself
    - plotting
    - analysis
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np
from scipy.interpolate import griddata

from pathlib import Path

from src.config import *

def load_gsreport(filename:str) -> pd.DataFrame:

    """
    Load and return corresponding grid search report
    """
    
    filepath = Path(filename)
    df = pd.read_csv(filepath, sep='\t').sort_values(by='HARVEST', ascending=False)

    return df

def plot_distribution(gs_report:pd.DataFrame) -> None:

    """
    Plot a simple distributions of cultivation trend 
    """

    sns.set()
    sns.pairplot(gs_report, x_vars=['HF', 'HR', 'IDENS'], y_vars='HARVEST', kind='kde')

def plot_curves(biomass_path:Path, harvest_path:Path, fname:str) -> None:

    biomass_df = pd.read_csv(biomass_path)
    harvest_df = pd.read_csv(harvest_path)

    plt.figure(figsize=(15, 6))
    sns.set()
    sns.lineplot(data=biomass_df,x='time',y='biomass',c='green', label='biomass left')
    sns.lineplot(data=harvest_df,x='time',y='biomass',c='red',marker='o', label='biomass harvested')
    plt.xlabel('time')
    plt.ylabel('biomass [g/m^2]')

    plt.legend()

    plt.savefig(DATA_PATH / f'{fname}_curves.png')
    plt.close()

def smooth_heatmap(gs_report:pd.DataFrame, x='HP', y='HR', z='HARVEST', cmap='plasma', scatter_indices:int=None, fname:str=None) -> None:

    """
    Plot interpolated (smoothed) heatmap from grid searh report
    """

    X = gs_report[x]
    Y = gs_report[y]
    Z = gs_report[z]

    # Create a grid for interpolation
    xi = np.linspace(min(X), max(X), 100)
    yi = np.linspace(min(Y), max(Y), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the data
    zi = griddata((X, Y), Z, (xi, yi), method='cubic')

    # Create the 2D contour plot
    fig = plt.figure(figsize=(12, 9), dpi=100)  # Larger figure with higher DPI
    ax = fig.add_subplot(111)
    
    # Plot filled contours
    contour = ax.contourf(xi, yi, zi, cmap=cmap, levels=20)  # Increased levels for smoother gradients

        # Scatter selected points and add annotations
    if scatter_indices is not None:
        # Ensure scatter_indices is a list
        scatter_indices = np.arange(0,scatter_indices,1)
        # Plot scatter points
        ax.scatter(X.iloc[scatter_indices], Y.iloc[scatter_indices], c='red', s=50, edgecolors='black')
        # Add annotations in (x,y) format
        for idx in scatter_indices:
            x_val, y_val, D0 = X.iloc[idx], Y.iloc[idx], gs_report['IDENS'].iloc[idx]
            ax.annotate(f'({x_val:.2f},{y_val:.2f} | {D0})', 
                        (x_val, y_val), 
                        xytext=(5, 5),  # Offset annotation
                        textcoords='offset points', 
                        fontsize=10, 
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', alpha=0.8))
    
    # Customize the plot
    ax.set_xlabel(x, fontsize=16, labelpad=10)
    ax.set_ylabel(y, fontsize=16, labelpad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    #plt.title(f'2D Contour Plot of {z} vs {x} and {y}', fontsize=14, pad=15)
    
    # Adjust colorbar
    cbar = fig.colorbar(contour, ax=ax, label=z, shrink=0.8, pad=0.1)
    cbar.set_ticks([])
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout to avoid overlap
    plt.tight_layout()
    
    if fname:
        plt.savefig(DATA_PATH / f'{fname}_HMapSmooth.png')
    else:
        # Show the plot
        plt.show()
    
    plt.close()

def heatmap(gs_report:pd.DataFrame, x='HP', y='HR', z='HARVEST', cmap='plasma', fname:str = None) -> None:

    """
    Construct precise heatmap (discrete)
    """

    ### select only rows with W0 leading to highest HARVEST ###
    selected_idens = gs_report[gs_report['HARVEST'] == gs_report['HARVEST'].max()]['IDENS'].values[0]
    df_selected = gs_report[gs_report['IDENS'] == selected_idens]

    df = df_selected.pivot(index=y, columns=x, values=z).sort_values(by=y,ascending=False)

    # Set up the plot
    plt.figure(figsize=(15, 12))
    sns.heatmap(df, annot=False, fmt='.1f', cmap=cmap, cbar_kws={'label': 'Yield','ticks': []})

    # Customize the plot
    #plt.title('Heatmap of Yield for Different Hr and Hf Values', fontsize=14)
    plt.xlabel('Harvest period [day]', fontsize=16)
    plt.ylabel('Harvest ratio [%]', fontsize=16)

    plt.tick_params(axis='x', labelsize=16)  # Font size for x-axis ticks
    plt.tick_params(axis='y', labelsize=16)  # Font size for y-axis ticks

    if fname:
        plt.savefig(DATA_PATH / f'{fname}_HMap.png')
    else:
        # Show the plot
        plt.show()
    
    plt.close()

def table(gs_report:pd.DataFrame, N:int=5, fname:str = None):
    
    Harvest_sorted = gs_report.sort_values(by='HARVEST', ascending=False)
    top_n_df = Harvest_sorted.head(N)

    _, ax = plt.subplots(figsize=(8, N * 0.5))  # Adjust height based on number of rows
    
    # Hide axes
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=top_n_df.values,
        colLabels=top_n_df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(top_n_df.columns),
        bbox=[0, 0, 1, 1]
    )
    
    # Adjust table properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    # Set title
    plt.title(f'Top {N} optimal predictions', pad=20)
    # Adjust layout
    plt.tight_layout()

    if fname:
        plt.savefig(DATA_PATH / f'{fname}_table.png')
    else:
        plt.show()
    plt.close()


def gen_report(report_path):
    """
    Generate PNG heatmap files, a table, and optionally biomass/harvest curves.
    Returns a list of paths to the generated PNG files.
    Args:
        report_path: Path or string to the report file.
    """
    # Convert report_path to Path object if it's a string
    report_path = Path(report_path) if isinstance(report_path, str) else report_path

    report_df = load_gsreport(report_path)
    filename = str(report_path).split('/')[1]  # Get filename without extension

    # Ensure static/images directory exists
    static_images_dir = Path('static/images')
    static_images_dir.mkdir(parents=True, exist_ok=True)

    # List to store generated file paths
    generated_files = []

    # Generate heatmap
    heatmap(gs_report=report_df, fname=filename)
    heatmap_path = DATA_PATH / f'{filename}_HMap.png'
    static_heatmap_path = static_images_dir / f'{filename}_HMap.png'
    heatmap_path.rename(static_heatmap_path)  # Move to static/images
    generated_files.append(f'images/{filename}_HMap.png')

    # Generate smooth heatmap
    smooth_heatmap(gs_report=report_df, fname=filename, scatter_indices=5)
    smooth_heatmap_path = DATA_PATH / f'{filename}_HMapSmooth.png'
    static_smooth_heatmap_path = static_images_dir / f'{filename}_HMapSmooth.png'
    smooth_heatmap_path.rename(static_smooth_heatmap_path)  # Move to static/images
    generated_files.append(f'images/{filename}_HMapSmooth.png')

    # Generate table
    table(gs_report=report_df, N=5, fname=filename)
    table_path = DATA_PATH / f'{filename}_table.png'
    static_table_path = static_images_dir / f'{filename}_table.png'
    table_path.rename(static_table_path)  # Move to static/images
    generated_files.append(f'images/{filename}_table.png')

    # Generate biomass/harvest curves if available
    bc_file = f"{report_path}.biomass_df.csv"
    hv_file = f"{report_path}.harvest_df.csv"

    if Path(bc_file).exists() and Path(hv_file).exists():
        plot_curves(
            biomass_path=bc_file,
            harvest_path=hv_file,
            fname=filename
        )
        curves_path = DATA_PATH / f'{filename}_curves.png'
        static_curves_path = static_images_dir / f'{filename}_curves.png'
        curves_path.rename(static_curves_path)  # Move to static/images
        generated_files.append(f'images/{filename}_curves.png')

    return generated_files
