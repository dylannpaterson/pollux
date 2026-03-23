import argparse
import asdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches

def main():
    parser = argparse.ArgumentParser(description="Plot detection overlay for Roman images")
    parser.add_argument("image", help="Path to the input Roman ASDF image")
    parser.add_argument("catalog", help="Path to the output catalog ASDF")
    parser.add_argument("--output", default="detection_overlay.png", help="Path to save the output plot")
    parser.add_argument("--prob", type=float, default=0.9, help="Probability threshold for plotting")
    
    args = parser.parse_args()

    print(f"Loading image from {args.image}...")
    with asdf.open(args.image) as af:
        image = np.array(af['roman']['data'])

    # Fix the "black holes" (negative pixels)
    clean_image = np.copy(image)
    valid_mask = clean_image > 0
    if np.any(valid_mask):
        fill_val = np.percentile(clean_image[valid_mask], 99.9)
        clean_image[~valid_mask] = fill_val
    else:
        clean_image[~valid_mask] = 0

    print(f"Loading catalog from {args.catalog}...")
    with asdf.open(args.catalog) as af:
        cat = af['catalog']
        df = pd.DataFrame({
            'x': np.array(cat['x']),
            'y': np.array(cat['y']),
            'prob': np.array(cat['prob'])
        })

    # Filter for high-confidence sources
    df_filtered = df[df['prob'] > args.prob]
    print(f"Total detections (prob > {args.prob}): {len(df_filtered)}")

    # Visual Range
    vmin = max(0.1, np.percentile(clean_image, 5))
    vmax = np.percentile(clean_image, 99.9)
    print(f"Plotting with vmin={vmin:.2f}, vmax={vmax:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # --- Plot 1: Full Image ---
    print("Plotting full image...")
    im1 = ax1.imshow(clean_image, origin='lower', cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Counts')

    # Plot small subset for full view to avoid lag/huge file size
    if len(df_filtered) > 10000:
        sample_df = df_filtered.sample(10000)
    else:
        sample_df = df_filtered
    ax1.scatter(sample_df['x'], sample_df['y'], s=0.5, color='cyan', alpha=0.5, marker='.')
    ax1.set_title(f"Full SCA (Detections Overlaid)")

    # --- Plot 2: Chunk Cutout ---
    print("Plotting chunk cutout...")
    h, w = clean_image.shape
    cx, cy = w // 2, h // 2
    size = 128
    x0, x1 = max(0, cx - size), min(w, cx + size)
    y0, y1 = max(0, cy - size), min(h, cy + size)

    zoom_img = clean_image[y0:y1, x0:x1]
    ax2.imshow(zoom_img, origin='lower', cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax), extent=[x0, x1, y0, y1])
    
    # Roman PSF FWHM ~ 2 pixels. Circle with radius 1.5 pixels.
    psf_radius = 1.5 
    
    zoom_df = df_filtered[(df_filtered['x'] >= x0) & (df_filtered['x'] < x1) & 
                          (df_filtered['y'] >= y0) & (df_filtered['y'] < y1)]
    
    for _, star in zoom_df.iterrows():
        circ = patches.Circle((star['x'], star['y']), psf_radius, facecolor='none', 
                              edgecolor='cyan', linewidth=1.0, alpha=0.8)
        ax2.add_patch(circ)
    
    ax2.set_title(f"Center Cutout: 1.5px Radius Circles (Roman PSF Scale)")
    ax2.set_xlim(x0, x1)
    ax2.set_ylim(y0, y1)
    
    # Highlight zoom area on main plot
    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='cyan', facecolor='none')
    ax1.add_patch(rect)

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
