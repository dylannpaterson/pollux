import asdf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import os

def main():
    epoch = 0
    target_x, target_y = 256.0, 256.0
    size = 128 # 128x128 area
    
    img_path = f"data/prototype/microlensing_test_stack/epoch_{epoch:04d}.fits"
    cat_path = f"data/prototype/microlensing_test_stack/results/epoch_{epoch:04d}_catalog.asdf"
    
    if not os.path.exists(img_path) or not os.path.exists(cat_path):
        print("Missing files.")
        return

    # Load image
    with fits.open(img_path) as hdul:
        data = hdul[0].data
        
    # Load catalog
    with asdf.open(cat_path) as af:
        cat = af['catalog']
        df = pd.DataFrame({k: np.array(cat[k]) for k in cat.keys()})
    
    # Filter for region
    half = size / 2
    mask = (df['x'] >= target_x - half) & (df['x'] <= target_x + half) & \
           (df['y'] >= target_y - half) & (df['y'] <= target_y + half) & \
           (df['prob'] >= 0.1) # Lower threshold to see what it's thinking
    local_stars = df[mask]

    # Cutout
    y0, y1 = int(target_y - half), int(target_y + half)
    x0, x1 = int(target_x - half), int(target_x + half)
    cutout = data[y0:y1, x0:x1]

    plt.figure(figsize=(12, 12))
    # Use log stretch for image to see faint stars
    im = plt.imshow(np.log10(np.maximum(cutout, 1)), origin='lower', 
                    extent=[x0, x1, y0, y1], cmap='magma', interpolation='nearest')
    plt.colorbar(im, label='log10(ADU)')
    
    # Plot detections
    if not local_stars.empty:
        # High confidence in red
        high = local_stars[local_stars['prob'] >= 0.5]
        plt.scatter(high['x'], high['y'], s=40, facecolors='none', edgecolors='cyan', lw=1, label='p >= 0.5')
        
        # Low confidence in dashed yellow
        low = local_stars[local_stars['prob'] < 0.5]
        plt.scatter(low['x'], low['y'], s=20, facecolors='none', edgecolors='yellow', lw=0.5, alpha=0.5, label='p < 0.5')

    # Mark the target microlensing site
    plt.scatter([target_x], [target_y], marker='x', color='white', s=100, label='Target (256,256)')
    
    plt.legend()
    plt.title(f"Large Area Scan (Epoch {epoch}) - Center: ({target_x}, {target_y})")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    
    output = "large_area_scan.png"
    plt.savefig(output, dpi=200)
    print(f"Saved {output}")

if __name__ == "__main__":
    main()
