import numpy as np
import pytest
import os
import asdf
from pollux.pipeline import PhotometryPipeline
from astropy.wcs import WCS
from astropy.io import fits

def test_extract_stars_center_crop():
    """
    Verify the Center-Crop Rule in _extract_stars_vectorized.
    Detections outside the [margin, margin+stride) range should be ignored.
    """
    # Create a pipeline without loading real weights
    pipeline = PhotometryPipeline.__new__(PhotometryPipeline)
    
    # [grid_h, grid_w, K, params]
    # K=3, params=86 (5 + 81 shape)
    grid_preds = np.zeros((64, 64, 3, 86))
    
    # 1. Star inside central area: grid cell (10, 10)
    grid_preds[10, 10, 0, 0] = 1.0 # prob
    grid_preds[10, 10, 0, 1] = 0.5 # dx
    grid_preds[10, 10, 0, 2] = 0.5 # dy
    grid_preds[10, 10, 0, 3] = 100.0 # flux_phys
    
    global_stars = []
    pipeline._extract_stars_vectorized(
        grid_preds, 
        x_offset=0, 
        y_offset=0, 
        threshold=0.5, 
        global_stars=global_stars, 
        tile_median=0,
        img_shape=(4088, 4088)
    )
    
    assert len(global_stars) == 1
    assert global_stars[0]['x'] == 40.5 

def test_extract_stars_global_mapping():
    """
    Verify global detector coordinate mapping with custom image shape.
    """
    pipeline = PhotometryPipeline.__new__(PhotometryPipeline)
    grid_preds = np.zeros((64, 64, 3, 86))
    
    # Star at grid (14, 24)
    grid_preds[14, 24, 0, 0] = 1.0
    grid_preds[14, 24, 0, 1] = 0.5 # dx
    grid_preds[14, 24, 0, 2] = 0.2 # dy
    grid_preds[14, 24, 0, 3] = 100.0
    
    x_offset, y_offset = 1000, 2000
    
    global_stars = []
    pipeline._extract_stars_vectorized(
        grid_preds, 
        x_offset, 
        y_offset, 
        threshold=0.5, 
        global_stars=global_stars, 
        tile_median=0,
        img_shape=(5000, 5000) # Custom size
    )
    
    assert len(global_stars) == 1
    assert global_stars[0]['x'] == pytest.approx(1096.5)
    assert global_stars[0]['y'] == pytest.approx(2056.2)

def test_build_catalog_wcs():
    """
    Verify _build_catalog uses WCS correctly.
    """
    pipeline = PhotometryPipeline.__new__(PhotometryPipeline)
    stars = [{'x': 100, 'y': 200, 'mag_raw': 2.0, 'flux_raw': 100, 'completeness': 0.9, 'prob': 1.0}]
    
    # Proper WCS initialization
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, -30.0]
    wcs.wcs.crpix = [0.0, 0.0]
    wcs.wcs.pc = [[-0.1, 0], [0, 0.1]]
    wcs.wcs.set() # Finalize
    
    expected_ra, expected_dec = wcs.wcs_pix2world(100, 200, 0)
    
    df = pipeline._build_catalog(stars, wcs)
    
    assert len(df) == 1
    assert df.iloc[0]['ra'] == pytest.approx(float(expected_ra))
    assert df.iloc[0]['dec'] == pytest.approx(float(expected_dec))

def test_load_fits_image(tmp_path):
    """
    Verify _load_image correctly parses a FITS file.
    """
    pipeline = PhotometryPipeline.__new__(PhotometryPipeline)
    
    # Create a dummy FITS file
    data = np.random.rand(100, 100).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header['FILTER'] = 'F146'
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL1'] = 150.0
    hdu.header['CRVAL2'] = -30.0
    
    fits_path = os.path.join(tmp_path, "test.fits")
    hdu.writeto(fits_path)
    
    img, wcs, filter_name = pipeline._load_image(fits_path)
    
    assert img.shape == (100, 100)
    assert filter_name == 'F146'
    assert isinstance(wcs, WCS)
    assert wcs.wcs.crval[0] == 150.0

def test_load_asdf_image(tmp_path):
    """
    Verify _load_image correctly parses an ASDF file.
    """
    pipeline = PhotometryPipeline.__new__(PhotometryPipeline)
    
    # Mock ASDF structure
    data = np.random.rand(50, 50).astype(np.float32)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    tree = {
        'roman': {
            'data': data,
            'meta': {
                'wcs': wcs,
                'instrument': {'optical_element': 'F158'}
            }
        }
    }
    
    asdf_path = os.path.join(tmp_path, "test.asdf")
    with asdf.AsdfFile(tree) as af:
        af.write_to(asdf_path)
        
    img, loaded_wcs, filter_name = pipeline._load_image(asdf_path)
    
    assert img.shape == (50, 50)
    assert filter_name == 'F158'
    assert isinstance(loaded_wcs, WCS)
