import asdf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import re
import pandas as pd
import argparse
from astropy.io import fits
from astropy.wcs import WCS

def extract_epoch(filename):
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else None

def get_target_coords(image_dir):
    """Extract target RA/Dec from the first available epoch's header and WCS."""
    images = sorted(glob.glob(os.path.join(image_dir, "epoch_*.fits")))
    if not images:
        return None, None
    
    with fits.open(images[0]) as hdul:
        header = hdul[0].header
        w = WCS(header)
        tx, ty = header.get('OBJ_X'), header.get('OBJ_Y')
        if tx is None or ty is None:
            return None, None
        ra, dec = w.pixel_to_world_values(tx, ty)
        return float(ra), float(dec)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/prototype/microlensing_test_stack/results")
    parser.add_argument("--images", default="data/prototype/microlensing_test_stack")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--radius", type=float, default=0.5, help="Match radius in arcsec")
    args = parser.parse_args()

    target_ra, target_dec = get_target_coords(args.images)
    if target_ra is None:
        print("Error: Could not determine target RA/Dec from images.")
        return
    
    print(f"Tracking target at RA={target_ra:.6f}, Dec={target_dec:.6f} (radius={args.radius}\")")
    
    catalog_files = sorted(glob.glob(os.path.join(args.results, "epoch_*_catalog.asdf")))
    if not catalog_files: 
        print("No catalogs found.")
        return

    epochs, fluxes, magnifications = [], [], []

    for f in catalog_files:
        epoch = extract_epoch(f)
        if epoch is None: continue
        
        # 1. Load Magnification from Image
        img_path = os.path.join(args.images, f"epoch_{epoch:04d}.fits")
        if not os.path.exists(img_path): continue
        with fits.open(img_path) as hdul:
            magnif = hdul[0].header.get('MAGNIF', 1.0)

        # 2. Load Catalog and Match by RA/Dec
        with asdf.open(f) as af:
            cat = af['catalog']
            df = pd.DataFrame({'ra': np.array(cat['ra']), 'dec': np.array(cat['dec']), 'flux': np.array(cat['flux_raw'])})
        
        # Simple Euclidean distance in deg (fine for small areas)
        dist = np.sqrt(((df['ra'] - target_ra) * np.cos(np.deg2rad(target_dec)))**2 + (df['dec'] - target_dec)**2)
        idx = dist.argmin()
        
        if dist[idx] < (args.radius / 3600.0):
            epochs.append(epoch)
            fluxes.append(df.iloc[idx]['flux'])
            magnifications.append(magnif)

    if not epochs:
        print("No matches found across epochs.")
        return

    data = pd.DataFrame({
        'epoch': epochs, 
        'flux': fluxes, 
        'magnification': magnifications
    }).sort_values('epoch')

    # Plotting
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    
    # Model Flux
    lns1 = ax1.plot(data['epoch'], data['flux'], 'o-', color='C0', label='Model Flux', alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Model Flux (ADU)", color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.grid(True, alpha=0.3)

    # Magnification (Twin Axis)
    ax2 = ax1.twinx()
    lns2 = ax2.plot(data['epoch'], data['magnification'], 's--', color='C1', label='True Magnification', alpha=0.6)
    ax2.set_ylabel("Magnification Factor", color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    # Combined Legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')

    plt.title(f"Microlensing Lightcurve Analysis - {args.tag}")
    
    out_img = f"microlensing_lc_{args.tag}.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    print(f"Lightcurve saved to {out_img}")
    
    # Save CSV for backup
    data.to_csv(f"lightcurve_{args.tag}.csv", index=False)

if __name__ == "__main__":
    main()
