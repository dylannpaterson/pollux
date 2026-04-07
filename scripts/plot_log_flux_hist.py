import asdf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def main():
    catalog_path = "data/prototype/microlensing_test_stack/results/epoch_0000_catalog.asdf"
    
    if not os.path.exists(catalog_path):
        print(f"Error: Catalog {catalog_path} not found.")
        return

    print(f"Loading catalog from {catalog_path}...")
    with asdf.open(catalog_path) as af:
        flux = np.array(af['catalog']['flux_raw'])
        prob = np.array(af['catalog']['prob'])
    
    # Filter for high-confidence detections (p >= 0.5)
    mask = prob >= 0.5
    flux_filtered = flux[mask]
    
    print(f"Total detections: {len(flux)}")
    print(f"Detections with p >= 0.5: {len(flux_filtered)}")

    # Handle non-positive fluxes just in case
    flux_filtered = flux_filtered[flux_filtered > 0]
    log_flux = np.log10(flux_filtered)
    
    plt.figure(figsize=(10, 6))
    plt.hist(log_flux, bins=100, color='C0', alpha=0.7, edgecolor='white')
    
    plt.xlabel("log10(Flux) [counts]")
    plt.ylabel("Number of Stars")
    plt.title(f"Flux Distribution (p >= 0.5) - Epoch 0000\n{len(flux_filtered)} stars")
    plt.grid(True, alpha=0.3)
    
    output_plot = "epoch_0000_flux_hist.png"
    plt.savefig(output_plot, dpi=150)
    print(f"Histogram saved to {output_plot}")

if __name__ == "__main__":
    main()
