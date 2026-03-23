import asdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

def cross_match_sky(true_ra, true_dec, pred_ra, pred_dec, radius_arcsec=2.0):
    """
    Cross-match using RA/Dec. 
    radius_arcsec is matching radius in arcseconds.
    """
    def to_cartesian(ra, dec):
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        return np.column_stack((x, y, z))

    true_xyz = to_cartesian(true_ra, true_dec)
    pred_xyz = to_cartesian(pred_ra, pred_dec)
    
    radius_rad = np.deg2rad(radius_arcsec / 3600.0)
    chord_len = 2 * np.sin(radius_rad / 2.0)
    
    tree = cKDTree(true_xyz)
    dist, idx = tree.query(pred_xyz, distance_upper_bound=chord_len)
    matched = dist < chord_len
    
    dist_arcsec = np.rad2deg(2 * np.arcsin(np.clip(dist / 2.0, 0, 1))) * 3600.0
    
    return matched, dist_arcsec, idx

def main():
    prototype_path = "data/prototype/pollux_prototype_l2.asdf"
    catalog_path = "output_catalog.asdf"
    
    if not os.path.exists(catalog_path):
        print(f"Error: {catalog_path} not found.")
        return

    # Load Ground Truth
    with asdf.open(prototype_path) as af:
        gt = af['ground_truth']
        true_df = pd.DataFrame({
            'ra': gt['ra'],
            'dec': gt['dec'],
            'mag': gt['mag']
        })
        image_data = np.array(af['roman']['data'])

    # Load Detections
    with asdf.open(catalog_path) as af:
        pred_df = pd.DataFrame(af['catalog'])

    # Filter non-finite RA/Dec
    true_df = true_df[np.isfinite(true_df['ra']) & np.isfinite(true_df['dec'])]
    pred_df = pred_df[np.isfinite(pred_df['ra']) & np.isfinite(pred_df['dec'])]

    print(f"--- Analysis Results ---")
    print(f"Ground Truth Stars: {len(true_df)}")
    print(f"Detected Stars:     {len(pred_df)}")

    # Cross-match in Sky Space (2.0 arcsec radius)
    matched, dist_arcsec, idx = cross_match_sky(
        true_df['ra'].values, true_df['dec'].values,
        pred_df['ra'].values, pred_df['dec'].values,
        radius_arcsec=2.0
    )
    num_matched = np.sum(matched)
    
    print(f"Matched Stars:      {num_matched}")
    print(f"Completeness:       {num_matched/len(true_df)*100:.2f}%")
    print(f"Purity (Precision): {num_matched/len(pred_df)*100:.2f}%")

    if num_matched > 0:
        m_dist_arcsec = dist_arcsec[matched]
        m_true_mags = true_df['mag'].values[idx[matched]]
        m_pred_logflux = pred_df['mag'].values[matched]
        
        print(f"Positional Separation (mean): {np.mean(m_dist_arcsec):.3f} arcsec")
        
        inst_mag = -2.5 * m_pred_logflux
        zp = np.median(m_true_mags - inst_mag)
        residuals = m_true_mags - (inst_mag + zp)
        
        print(f"Estimated Zero-point: {zp:.3f}")
        print(f"Magnitude RMSE:       {np.std(residuals):.3f}")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Completeness vs Magnitude
        bins = np.linspace(true_df['mag'].min(), true_df['mag'].max(), 30)
        counts_true, _ = np.histogram(true_df['mag'], bins=bins)
        counts_matched, _ = np.histogram(m_true_mags, bins=bins)
        completeness = counts_matched / (counts_true + 1e-9)
        
        axes[0, 0].step(bins[:-1], completeness, where='post', color='blue')
        axes[0, 0].set_xlabel("True Magnitude")
        axes[0, 0].set_ylabel("Completeness")
        axes[0, 0].set_title("Completeness vs Magnitude")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Magnitude Residuals
        axes[0, 1].scatter(m_true_mags, residuals, alpha=0.3, s=5, color='green')
        axes[0, 1].axhline(0, color='r', linestyle='--')
        axes[0, 1].set_xlabel("True Magnitude")
        axes[0, 1].set_ylabel("True - Calibrated Pred")
        axes[0, 1].set_title("Photometric Residuals")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Detection Probability
        axes[1, 0].scatter(pred_df['mag'], pred_df['prob'], alpha=0.1, s=2, color='gray', label='All')
        axes[1, 0].scatter(pred_df['mag'][matched], pred_df['prob'][matched], alpha=0.3, s=5, color='blue', label='Matched')
        axes[1, 0].set_xlabel("Predicted Log-Flux")
        axes[1, 0].set_ylabel("Model Probability (p)")
        axes[1, 0].set_title("Detection Probability")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Visual Cutout
        cx, cy = 2044, 2044
        size = 128
        cutout = image_data[cy-size:cy+size, cx-size:cx+size]
        axes[1, 1].imshow(cutout, origin='lower', cmap='inferno', 
                          extent=[cx-size, cx+size, cy-size, cy+size],
                          vmin=np.percentile(cutout, 5), vmax=np.percentile(cutout, 99))
        
        mask_pred = (pred_df['x'] > cx-size) & (pred_df['x'] < cx+size) & \
                    (pred_df['y'] > cy-size) & (pred_df['y'] < cy+size)
        axes[1, 1].scatter(pred_df['x'][mask_pred], pred_df['y'][mask_pred], 
                           edgecolor='cyan', facecolor='none', s=40, label='Detected')
        
        axes[1, 1].set_title(f"Center Cutout (Detections Only)")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig("analysis_report.png")
        print(f"Analysis report saved to analysis_report.png")

if __name__ == "__main__":
    main()
