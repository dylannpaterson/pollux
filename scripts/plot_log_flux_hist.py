import asdf
import numpy as np
import matplotlib.pyplot as plt

def main():
    catalog_path = "mosaic_07_results.asdf"
    
    print(f"Loading catalog from {catalog_path}...")
    with asdf.open(catalog_path) as af:
        # Use flux_raw which is the physical flux output by the model
        flux = np.array(af['catalog']['flux_raw'])
        prob = np.array(af['catalog']['prob'])
    
    # Filter for high-confidence detections
    flux = flux[prob > 0.5]
    
    # Handle non-positive fluxes just in case
    flux = flux[flux > 0]
    log_flux = np.log10(flux)
    
    plt.figure(figsize=(10, 6))
    plt.hist(log_flux, bins=100, color='C0', alpha=0.7, edgecolor='white')
    
    plt.xlabel("log10(Flux) [counts]")
    plt.ylabel("Number of Stars")
    plt.title("Distribution of Detected Star Fluxes (log10)")
    plt.grid(True, alpha=0.3)
    
    output_plot = "mosaic_07_logflux_hist.png"
    plt.savefig(output_plot, dpi=150)
    print(f"Histogram saved to {output_plot}")

if __name__ == "__main__":
    main()
