import asdf
import numpy as np
from astropy.io import fits
import os

def export_asdf_to_fits(asdf_path, fits_path):
    print(f"Reading {asdf_path}...")
    with asdf.open(asdf_path) as af:
        # Extract image data
        image_data = np.array(af['roman']['data'])
        
        # Create Primary HDU
        hdu = fits.PrimaryHDU(image_data)
        
        # Add basic metadata manually since gWCS to FITS header is non-trivial
        meta = af['roman']['meta']
        hdu.header['FILTER'] = meta['instrument']['optical_element']
        hdu.header['DETECTOR'] = meta['instrument']['detector']
        hdu.header['TELESCOP'] = 'ROMAN'
        hdu.header['INSTRUME'] = 'WFI'
        
        # Write to file
        print(f"Writing to {fits_path}...")
        hdu.writeto(fits_path, overwrite=True)
        print("Success.")

if __name__ == "__main__":
    asdf_input = "data/prototype/pollux_prototype_l2.asdf"
    fits_output = "pollux_prototype.fits"
    export_asdf_to_fits(asdf_input, fits_output)
