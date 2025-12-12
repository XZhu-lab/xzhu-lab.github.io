# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:19:14 2025

@author: TAN Xiaoyue
"""

# -*- coding: utf-8 -*-


"""
NLFI Calculation Demo (Single City)
-----------------------------------
Demonstrates how to calculate the Night Light Fluctuation Index (NLFI) 
from "Mean" and "Standard Deviation" raster images.

==============================================================================
IMPORTANT NOTES & DATA PREPARATION
==============================================================================


1. Input Data Generation: 
   The "Mean" and "Standard Deviation" input images (MEAN_PATH, STD_PATH) 
   should be calculated from an annual daily NTL time series.

2. Outlier Removal (Recommended): 
   When generating these annual statistics, it is highly recommended to use a 
   3-sigma filter (exclude values deviating more than 3 times the standard 
   deviation from the mean) to remove extreme values. 

3. Recommended Data Source (HDNTL): 
   It is strongly suggested to use the HDNTL dataset for the underlying daily time series.
   
   - HDNTL is derived from NASA Black Marble (VNP46A2) but includes 
     crucial corrections for spatial mismatch and angular effects. Using 
     HDNTL minimizes fluctuations caused by sensor artifacts.
   - The dataset is publicly available.
   - Pei, Z., Zhu, X., Hu, Y., Chen, J., & Tan, X. (2025). A high-quality daily 
     nighttime light (HDNTL) dataset for global 600+ cities (2012–2024). 
     Earth System Science Data, 17(10), 5675–5691. 
     https://doi.org/10.5194/essd-17-5675-2025
   

REFERENCES:
- NLFI Index Methodology: 
  Tan, X., Chen, J., & Zhu, X. (2025). Beyond static brightness: daily nighttime 
  light fluctuations enrich nighttime vitality evaluation for urban zones. 
  Sustainable Cities and Society, 107043. 
  
  
==============================================================================

Workflow:
1. Read Data: Load Mean and Std rasters.
2. Fit Baseline: Find the "inherent background noise" level (y = kx) via binning statistics.
3. Calculate Index: Calculate the deviation of each pixel from the baseline.
4. Visualize Results: Plot scatter diagrams and the result map.
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ================= 1. Configuration =================

# Input file paths (Please replace with your actual TIF file paths)
# Note: Ensure these are calculated from annual HDNTL data with 3-sigma filtering.
MEAN_PATH = "path/to/your/city_mean_radiance.tif"  # Mean Radiance Image
STD_PATH  = "path/to/your/city_std_deviation.tif"   # Standard Deviation Image

# Algorithm Parameters
MEAN_THRESHOLD = 0.5     # Radiance Threshold: Exclude dark background areas (e.g., water, uninhabited areas)
BIN_WIDTH      = 10.0    # Bin Width: Group by every 10 radiance units when fitting the baseline
MIN_BIN_COUNT  = 70      # Min Bin Count: If pixels in a bin are too few, exclude from fitting
PICK_PER_BIN   = 5       # Points per Bin: Select the 5 points with the lowest fluctuation in each group to represent "inherent noise"

# ================= 2. Core Functions =================

def read_data(mean_file, std_file):
    """Read raster data and handle invalid values."""
    with rasterio.open(mean_file) as src_m, rasterio.open(std_file) as src_s:
        mean_arr = src_m.read(1).astype(np.float32)
        std_arr  = src_s.read(1).astype(np.float32)
        
        # Mark Nodata as NaN to avoid interference with calculations
        if src_m.nodata is not None:
            mean_arr[mean_arr == src_m.nodata] = np.nan
        if src_s.nodata is not None:
            std_arr[std_arr == src_s.nodata] = np.nan
            
        # Ensure Standard Deviation is non-negative (data anomaly handling)
        std_arr[std_arr < 0] = np.nan
        
        profile = src_m.profile.copy()
        
    return mean_arr, std_arr, profile

def fit_baseline(mean_arr, std_arr):
    """
    Fit Inherent Variation Baseline.
    Logic: In each radiance bin, find the points with the lowest Standard Deviation,
    assuming they represent the normal physical fluctuation for that brightness level.
    """
    # 1. Flatten arrays, keep only valid pixels above the threshold
    mask = np.isfinite(mean_arr) & np.isfinite(std_arr) & (mean_arr >= MEAN_THRESHOLD)
    m_flat = mean_arr[mask]
    s_flat = std_arr[mask]
    
    if m_flat.size == 0:
        print("Warning: Number of valid pixels is 0. Cannot fit.")
        return None, [], []

    # 2. Binning based on Mean Radiance
    m_min, m_max = np.min(m_flat), np.max(m_flat)
    edges = np.arange(np.floor(m_min), np.ceil(m_max) + BIN_WIDTH, BIN_WIDTH)
    indices = np.digitize(m_flat, edges)
    
    rep_m, rep_s = [], [] # To store selected representative points
    
    # 3. Iterate through each bin, select points with the lowest Std
    unique_bins = np.unique(indices)
    for b_idx in unique_bins:
        bin_mask = (indices == b_idx)
        # Only calculate if the bin has enough pixels to avoid accidental errors
        if np.count_nonzero(bin_mask) >= MIN_BIN_COUNT:
            bin_m = m_flat[bin_mask]
            bin_s = s_flat[bin_mask]
            
            # Find indices of the k points with the smallest Std
            k = min(PICK_PER_BIN, bin_m.size)
            smallest_idx = np.argpartition(bin_s, k-1)[:k]
            
            rep_m.append(bin_m[smallest_idx])
            rep_s.append(bin_s[smallest_idx])
            
    if not rep_m:
        return None, [], []

    # Concatenate all representative points
    rep_m = np.concatenate(rep_m)
    rep_s = np.concatenate(rep_s)
    
    # 4. Fit a line passing through the origin: y = kx
    # Formula: k = sum(x*y) / sum(x*x)
    k_slope = np.dot(rep_m, rep_s) / np.dot(rep_m, rep_m)
    
    return k_slope, rep_m, rep_s

def calculate_nlfi_map(mean_arr, std_arr, k):
    """
    Calculate the NLFI index for the whole map.
    NLFI is defined as the geometric vertical distance from point (Mean, Std) to the baseline y=kx.
    """
    nlfi_map = np.full_like(mean_arr, np.nan)
    
    # Only calculate for areas above the threshold
    mask = np.isfinite(mean_arr) & np.isfinite(std_arr) & (mean_arr >= MEAN_THRESHOLD)
    
    if k is not None:
        # Distance from point to line formula: |kx - y| / sqrt(k^2 + 1)
        # Here x = Mean, y = Std
        dist = np.abs(k * mean_arr[mask] - std_arr[mask]) / np.sqrt(k**2 + 1)
        nlfi_map[mask] = dist
        
    return nlfi_map

# ================= 3. Main Process =================

def main():
    # Check if file exists (For demonstration purposes only)
    if not os.path.exists(MEAN_PATH):
        print(f"Note: File not found: {MEAN_PATH}")
        print("Please update MEAN_PATH and STD_PATH at the top of the code with your actual TIF file paths.")
        return

    print("--- Start Processing ---")
    
    # 1. Read
    print("1. Reading data...")
    mean_img, std_img, profile = read_data(MEAN_PATH, STD_PATH)
    
    # 2. Fit
    print("2. Fitting baseline...")
    slope_k, rep_m, rep_s = fit_baseline(mean_img, std_img)
    
    if slope_k is None:
        print("Fitting failed. Cannot calculate.")
        return

    print(f"   Fitting Result: Slope k = {slope_k:.4f}")
    print(f"   (Baseline Equation: Std = {slope_k:.4f} * Mean)")

    # 3. Calculate
    print("3. Calculating NLFI index...")
    nlfi_result = calculate_nlfi_map(mean_img, std_img, slope_k)
    
    # 4. Save (Optional)
    output_path = "NLFI_Output.tif"
    profile.update(dtype='float32', count=1, compress='deflate')
    with rasterio.open(output_path, 'w', **profile) as dst:
        # Fill NaN back to Nodata value for GIS software recognition
        save_data = nlfi_result.copy()
        save_data[np.isnan(save_data)] = profile['nodata'] if profile['nodata'] else -9999
        dst.write(save_data, 1)
    print(f"   Result saved to: {output_path}")

    # ================= 4. Plotting Demo =================
    print("4. Plotting analysis charts...")
    
    plt.figure(figsize=(12, 5), dpi=120)

    # Left Plot: Scatter Fit Diagram
    ax1 = plt.subplot(1, 2, 1)
    
    # Randomly sample background points for faster plotting
    valid_mask = np.isfinite(mean_img) & (mean_img > MEAN_THRESHOLD)
    sample_m = mean_img[valid_mask]
    sample_s = std_img[valid_mask]
    if sample_m.size > 10000:
        idx = np.random.choice(sample_m.size, 10000, replace=False)
        sample_m = sample_m[idx]
        sample_s = sample_s[idx]
    
    ax1.scatter(sample_m, sample_s, c='gray', s=1, alpha=0.3, label='All Pixels (Sampled)')
    ax1.scatter(rep_m, rep_s, c='red', s=15, marker='x', label='Baseline Rep Points')
    
    # Draw Line
    x_line = np.linspace(0, np.max(rep_m), 100)
    y_line = slope_k * x_line
    ax1.plot(x_line, y_line, 'b-', lw=2, label=f'Baseline y={slope_k:.3f}x')
    
    ax1.set_xlabel("Mean Radiance")
    ax1.set_ylabel("Std Deviation")
    ax1.set_title("Baseline Fitting Illustration")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Right Plot: NLFI Result Preview
    ax2 = plt.subplot(1, 2, 2)
    # Simple stretch display
    valid_vals = nlfi_result[~np.isnan(nlfi_result)]
    if valid_vals.size > 0:
        vmin, vmax = np.percentile(valid_vals, [2, 98])
        im = ax2.imshow(nlfi_result, cmap='magma', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax2, label='NLFI Value')
    else:
        ax2.text(0.5, 0.5, "No Valid Data", ha='center')
        
    ax2.set_title("NLFI Spatial Distribution")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()