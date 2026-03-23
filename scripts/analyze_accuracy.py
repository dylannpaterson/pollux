import asdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os

def to_cartesian(ra, dec):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.column_stack((x, y, z))

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
            'ra': np.array(gt['ra']),
            'dec': np.array(gt['dec']),
            'ab_mag': np.array(gt['ab_mag'])
        })

    # Load Pipeline Results
    with asdf.open(catalog_path) as af:
        cat = af['catalog']
        pred_df = pd.DataFrame({
            'ra': np.array(cat['ra']),
            'dec': np.array(cat['dec']),
            'ab_mag': np.array(cat['ab_mag']),
            'prob': np.array(cat['prob'])
        })

    # Filter NaNs
    true_df = true_df.dropna().reset_index(drop=True)
    pred_df = pred_df.dropna().reset_index(drop=True)

    # Tight cross-match (1.5 pixels ~ 0.165")
    # This is consistent with the model's 2x2 cell search limit
    radius_arcsec = 0.165 
    radius_rad = np.deg2rad(radius_arcsec / 3600.0)
    chord_len = 2 * np.sin(radius_rad / 2.0)

    print(f"Strict 1-to-1 matching (within 1.5 pixels / {radius_arcsec:.3f}\")...")
    
    true_xyz = to_cartesian(true_df['ra'], true_df['dec'])
    pred_xyz = to_cartesian(pred_df['ra'], pred_df['dec'])
    
    tree = cKDTree(pred_xyz)
    dist, idx = tree.query(true_xyz, distance_upper_bound=chord_len)
    
    valid = dist < chord_len
    matches = pd.DataFrame({
        't_idx': np.arange(len(true_df))[valid],
        'p_idx': idx[valid],
        'dist': dist[valid]
    })
    
    # Enforce 1-to-1
    matches = matches.sort_values('dist').drop_duplicates('p_idx').drop_duplicates('t_idx')
    num_matched = len(matches)
    
    print(f"--- Pipeline Recovery Accuracy (Tight) ---")
    print(f"Matched: {num_matched} / {len(true_df)} ({num_matched/len(true_df)*100:.1f}%)")

    if num_matched > 0:
        m_true = true_df.iloc[matches['t_idx']].copy()
        m_pred = pred_df.iloc[matches['p_idx']].copy()
        
        # Astrometry
        dra = (m_pred['ra'].values - m_true['ra'].values) * 3600.0 * np.cos(np.deg2rad(m_true['dec'].values))
        ddec = (m_pred['dec'].values - m_true['dec'].values) * 3600.0
        sep = np.sqrt(dra**2 + ddec**2)
        print(f"Positional Error (arcsec): Mean={np.mean(sep):.4f}, Median={np.median(sep):.4f}")

        # Photometry
        mag_resid = m_true['ab_mag'].values - m_pred['ab_mag'].values
        print(f"AB Magnitude Residuals: Mean={np.mean(mag_resid):.4f}, RMSE={np.std(mag_resid):.4f}")

        # Plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Recovery Astrometry
        axes[0, 0].scatter(dra, ddec, alpha=0.3, s=2, color='blue')
        axes[0, 0].set_xlabel("ΔRA * cos(Dec) (arcsec)")
        axes[0, 0].set_ylabel("ΔDec (arcsec)")
        axes[0, 0].set_title("Astrometric Residuals (Tight Match)")
        axes[0, 0].set_aspect('equal')
        axes[0, 0].set_xlim(-0.2, 0.2); axes[0, 0].set_ylim(-0.2, 0.2)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Recovery Photometry
        axes[0, 1].scatter(m_true['ab_mag'], m_pred['ab_mag'], alpha=0.3, s=2, color='green')
        ax_min = min(m_true['ab_mag'].min(), m_pred['ab_mag'].min())
        ax_max = max(m_true['ab_mag'].max(), m_pred['ab_mag'].max())
        axes[0, 1].plot([ax_min, ax_max], [ax_min, ax_max], 'r--')
        axes[0, 1].set_xlabel("Truth AB Magnitude")
        axes[0, 1].set_ylabel("Pipeline Calibrated AB Magnitude")
        axes[0, 1].set_title("Photometric Correlation (Strict 1-to-1)")
        axes[0, 1].invert_xaxis(); axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Detection Completeness
        bins = np.linspace(true_df['ab_mag'].min(), true_df['ab_mag'].max(), 50)
        counts_total, _ = np.histogram(true_df['ab_mag'], bins=bins)
        counts_matched, _ = np.histogram(m_true['ab_mag'], bins=bins)
        comp = counts_matched / (counts_total + 1e-9)
        axes[1, 0].step(bins[:-1], comp, where='post', color='red', lw=2)
        axes[1, 0].set_xlabel("Truth AB Magnitude")
        axes[1, 0].set_ylabel("Completeness")
        axes[1, 0].set_title("Detection Completeness")
        axes[1, 0].invert_xaxis()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Recovery Residuals
        axes[1, 1].scatter(m_true['ab_mag'], mag_resid, alpha=0.3, s=2, color='purple')
        axes[1, 1].axhline(0, color='red', linestyle='--')
        axes[1, 1].set_xlabel("Truth AB Magnitude")
        axes[1, 1].set_ylabel("ΔMag (Truth - Pipeline)")
        axes[1, 1].set_title("Magnitude Residuals vs Magnitude")
        axes[1, 1].set_ylim(-3, 3)
        axes[1, 1].invert_xaxis()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("recovery_analysis.png")
        print("Analysis complete. Saved recovery_analysis.png")

if __name__ == "__main__":
    main()
