import asdf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os

def plot_epoch(epoch, target_x, target_y, size=15):
    img_path = f"data/prototype/microlensing_test_stack/epoch_{epoch:04d}.fits"
    cat_path = f"data/prototype/microlensing_test_stack/results/epoch_{epoch:04d}_catalog.asdf"
    
    if not os.path.exists(img_path) or not os.path.exists(cat_path):
        print(f"Missing files for epoch {epoch}")
        return None

    # Load image
    with fits.open(img_path) as hdul:
        data = hdul[0].data
        magnif = hdul[0].header.get('MAGNIF', 1.0)

    # Load catalog
    with asdf.open(cat_path) as af:
        cat = af['catalog']
        df = pd.DataFrame({k: np.array(cat[k]) for k in cat.keys()})
    
    # Filter for region and probability
    half = size / 2
    mask = (df['x'] >= target_x - half) & (df['x'] <= target_x + half) & \
           (df['y'] >= target_y - half) & (df['y'] <= target_y + half) & \
           (df['prob'] >= 0.5)
    local_stars = df[mask]

    # Cutout
    y0, y1 = int(round(target_y - half)), int(round(target_y + half))
    x0, x1 = int(round(target_x - half)), int(round(target_x + half))
    cutout = data[y0:y1, x0:x1]

    return {
        'cutout': cutout,
        'stars': local_stars,
        'extent': [x0, x1, y0, y1],
        'magnif': magnif
    }

def main():
    tx, ty = 256.0, 256.0
    epochs = [0, 50]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    for i, ep in enumerate(epochs):
        res = plot_epoch(ep, tx, ty)
        if res is None: continue
        
        ax = axes[i]
        im = ax.imshow(res['cutout'], origin='lower', extent=res['extent'], cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='ADU')
        
        stars = res['stars']
        if not stars.empty:
            # Size markers by flux (log scale for visibility)
            sizes = (np.log10(stars['flux_raw']) - 2) * 20 
            ax.scatter(stars['x'], stars['y'], s=sizes, facecolors='none', edgecolors='red', lw=1.5, label='Detected')
            
            for _, row in stars.iterrows():
                ax.text(row['x']+0.2, row['y']+0.2, f"{row['flux_raw']:.0f}", color='white', fontsize=8)

        ax.set_title(f"Epoch {ep} (Mag: {res['magnif']:.2f})")
        ax.axhline(ty, color='white', alpha=0.3, ls='--')
        ax.axvline(tx, color='white', alpha=0.3, ls='--')
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")

    plt.tight_layout()
    plt.savefig("lightcurve_diagnostics.png")
    print("Saved lightcurve_diagnostics.png")

if __name__ == "__main__":
    main()
