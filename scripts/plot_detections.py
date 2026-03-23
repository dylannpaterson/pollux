import argparse
import asdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from astropy.io import fits
import os

def load_image_and_header(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.asdf':
        with asdf.open(file_path) as af:
            return np.array(af['roman']['data']), None
    elif ext in ['.fits', '.fit', '.fz']:
        hdul = fits.open(file_path, memmap=True)
        if 'SCI' in hdul:
            return hdul['SCI'].data, hdul['SCI'].header
        for hdu in hdul:
            if hdu.data is not None:
                return hdu.data, hdu.header
    raise ValueError(f"Unsupported format: {ext}")

def main():
    parser = argparse.ArgumentParser(description="Plot detection overlay for Roman/FITS images")
    parser.add_argument("image", help="Path to the input image (ASDF/FITS)")
    parser.add_argument("catalog", help="Path to the output catalog ASDF")
    parser.add_argument("--output", default="detection_overlay.png", help="Path to save the output plot")
    parser.add_argument("--prob", type=float, default=0.9, help="Probability threshold for plotting")
    
    args = parser.parse_args()

    print(f"Loading image metadata from {args.image}...")
    image_data, header = load_image_and_header(args.image)
    h, w = image_data.shape
    print(f"Image shape: {w} x {h}")

    # For large images, load a downsampled version for the full plot
    # and only load the center cutout at full resolution
    max_dim = 2048
    if h > max_dim or w > max_dim:
        stride = max(h // max_dim, w // max_dim)
        print(f"Downsampling full field plot by factor of {stride}...")
        full_field_img = np.array(image_data[::stride, ::stride]).astype(np.float32)
    else:
        full_field_img = np.array(image_data).astype(np.float32)
        stride = 1

    print("Loading center cutout...")
    cx, cy = w // 2, h // 2
    z_size = 256
    x0, x1 = max(0, cx - z_size), min(w, cx + z_size)
    y0, y1 = max(0, cy - z_size), min(h, cy + z_size)
    zoom_img = np.array(image_data[y0:y1, x0:x1]).astype(np.float32)

    print("Loading catalog...")
    with asdf.open(args.catalog) as af:
        cat = af['catalog']
        df = pd.DataFrame({
            'x': np.array(cat['x']),
            'y': np.array(cat['y']),
            'prob': np.array(cat['prob'])
        })

    df_filtered = df[df['prob'] > args.prob]
    print(f"Total detections (prob > {args.prob}): {len(df_filtered)}")

    # Robust vmin/vmax from zoom
    vmin = max(0.1, np.percentile(zoom_img, 5))
    vmax = np.percentile(zoom_img, 99.5)
    print(f"Plotting with vmin={vmin:.2f}, vmax={vmax:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # --- Plot 1: Full Image ---
    print("Plotting full field...")
    # Adjust extent for downsampled image
    ax1.imshow(full_field_img, origin='lower', cmap='inferno', 
               norm=LogNorm(vmin=vmin, vmax=vmax),
               extent=[0, w, 0, h])
    
    if len(df_filtered) > 10000:
        sample_df = df_filtered.sample(10000)
    else:
        sample_df = df_filtered
    ax1.scatter(sample_df['x'], sample_df['y'], s=0.2, color='cyan', alpha=0.3, marker='.')
    ax1.set_title(f"Full Field (Subsampled Detections)")

    # --- Plot 2: Center Cutout ---
    print("Plotting cutout...")
    ax2.imshow(zoom_img, origin='lower', cmap='inferno', 
               norm=LogNorm(vmin=vmin, vmax=vmax), 
               extent=[x0, x1, y0, y1])
    
    psf_radius = 1.5 
    zoom_df = df_filtered[(df_filtered['x'] >= x0) & (df_filtered['x'] < x1) & 
                          (df_filtered['y'] >= y0) & (df_filtered['y'] < y1)]
    
    for _, star in zoom_df.iterrows():
        circ = patches.Circle((star['x'], star['y']), psf_radius, facecolor='none', 
                              edgecolor='cyan', linewidth=1.0, alpha=0.8)
        ax2.add_patch(circ)
    
    ax2.set_title(f"Center Cutout ({x1-x0}x{y1-y0} px)")
    ax2.set_xlim(x0, x1)
    ax2.set_ylim(y0, y1)
    
    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='cyan', facecolor='none')
    ax1.add_patch(rect)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
