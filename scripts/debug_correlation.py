import asdf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os
import glob
import re

def extract_epoch(filename):
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else None

def main():
    tx, ty = 335.73, 157.41
    size = 15
    results_dir = "data/prototype/microlensing_test_stack/results"
    image_dir = "data/prototype/microlensing_test_stack"
    
    # 1. Corrected Diagnostic Plot for Epoch 0 and 50
    epochs_to_plot = [0, 50]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    for i, ep in enumerate(epochs_to_plot):
        img_path = os.path.join(image_dir, f"epoch_{ep:04d}.fits")
        cat_path = os.path.join(results_dir, f"epoch_{ep:04d}_catalog.asdf")
        
        with fits.open(img_path) as hdul:
            data = hdul[0].data
            magnif = hdul[0].header.get('MAGNIF', 1.0)
            zp = hdul[0].header.get('ZP', 26.5)

        with asdf.open(cat_path) as af:
            cat = af['catalog']
            df = pd.DataFrame({k: np.array(cat[k]) for k in cat.keys()})
        
        y0, y1 = int(round(ty - size//2)), int(round(ty + size//2))
        x0, x1 = int(round(tx - size//2)), int(round(tx + size//2))
        cutout = data[y0:y1+1, x0:x1+1]
        
        ax = axes[i]
        # Corrected extent: pixel center (x,y) is at coordinate (x,y)
        im = ax.imshow(cutout, origin='lower', extent=[x0-0.5, x1+0.5, y0-0.5, y1+0.5], 
                       cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='ADU')
        
        # Overlay catalog sources
        mask = (df['x'] >= x0-1) & (df['x'] <= x1+1) & (df['y'] >= y0-1) & (df['y'] <= y1+1)
        local = df[mask]
        if not local.empty:
            ax.scatter(local['x'], local['y'], s=100, facecolors='none', edgecolors='red', lw=1.5, label='Catalog')
            for _, row in local.iterrows():
                ax.text(row['x']+0.2, row['y']+0.2, f"{row['flux_raw']:.0f}", color='white', fontsize=8, fontweight='bold')

        ax.scatter([tx], [ty], marker='x', color='white', s=200, label='Target')
        ax.set_title(f"Epoch {ep} (Magnif: {magnif:.2f})")
        ax.legend()

    plt.tight_layout()
    plt.savefig("lightcurve_diagnostics_corrected.png")
    print("Saved lightcurve_diagnostics_corrected.png")

    # 2. Check for correlation with MAGNIF for all sources in the region
    print("\nChecking source correlations in 5x5 region around target...")
    catalog_files = sorted(glob.glob(os.path.join(results_dir, "epoch_*_catalog.asdf")))
    
    all_data = []
    for f in catalog_files:
        epoch = extract_epoch(f)
        img_path = os.path.join(image_dir, f"epoch_{epoch:04d}.fits")
        with fits.open(img_path) as hdul:
            magnif = hdul[0].header.get('MAGNIF', 1.0)
        
        with asdf.open(f) as af:
            cat = af['catalog']
            df = pd.DataFrame({k: np.array(cat[k]) for k in cat.keys()})
            mask = (df['x'] >= tx-2.5) & (df['x'] <= tx+2.5) & (df['y'] >= ty-2.5) & (df['y'] <= ty+2.5)
            near = df[mask].copy()
            near['epoch'] = epoch
            near['magnif'] = magnif
            all_data.append(near)
    
    if not all_data:
        print("No data found.")
        return
        
    combined = pd.concat(all_data)
    
    # We want to see which "star" (tracked by its approximate position) correlates with magnif
    # Group by x, y (rounded) to track individual stars across epochs
    combined['id'] = combined.apply(lambda r: f"{round(r['x']):.0f}_{round(r['y']):.0f}", axis=1)
    
    for star_id, group in combined.groupby('id'):
        if len(group) < 10: continue
        corr = np.corrcoef(group['magnif'], group['flux_raw'])[0, 1]
        print(f"Star at ~{star_id}: Corr={corr:.4f}, Count={len(group)}, Median Flux={group['flux_raw'].median():.1f}")
        
        # Test 21st mag hypothesis
        # m = ZP - 2.5 * log10(Flux / (Magnif * EXPTIME))
        exptime = 45.0
        baseline_flux_per_sec = (group['flux_raw'] / (group['magnif'] * exptime)).median()
        calc_mag = 26.5 - 2.5 * np.log10(baseline_flux_per_sec)
        print(f"  Estimated Baseline Mag (ZP=26.5, EXPTIME=45s): {calc_mag:.2f}")

if __name__ == "__main__":
    main()
