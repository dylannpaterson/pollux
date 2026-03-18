import asdf
import numpy as np
import pytest
import os

# Ensure we have the prototype file
PROTOTYPE_PATH = "data/prototype/pollux_prototype_l2.asdf"
CATALOG_PATH = "output_catalog.asdf"

@pytest.mark.skipif(not os.path.exists(PROTOTYPE_PATH), reason="Prototype ASDF not found")
def test_wcs_forward_transform():
    """
    Test that the WCS forward transformation (Pixel -> World) produces 
    RA/Dec values in the expected range for the prototype image.
    """
    with asdf.open(PROTOTYPE_PATH) as af:
        wcs = af['roman']['meta']['wcs']
        
        # Test corners and center of the 4088x4088 image
        x = np.array([0.0, 4087.0, 0.0, 4087.0, 2044.0])
        y = np.array([0.0, 0.0, 4087.0, 4087.0, 2044.0])
        
        ra, dec = wcs(x, y)
        
        # Expected ranges from ground truth (approximate)
        # RA: ~266.3 to 266.5
        # Dec: ~-29.1 to -28.9
        assert np.all(ra > 266.0) and np.all(ra < 267.0)
        assert np.all(dec > -30.0) and np.all(dec < -28.0)

@pytest.mark.skipif(not (os.path.exists(CATALOG_PATH) and os.path.exists(PROTOTYPE_PATH)), 
                    reason="Catalog or Prototype not found")
def test_output_catalog_wcs_bounds():
    """
    Verify that the detections in the generated catalog have RA/Dec 
    consistent with the ground truth bounds.
    """
    with asdf.open(CATALOG_PATH) as af:
        cat = af['catalog']
        # Convert to numpy while file is open
        ra = np.array(cat['ra'])
        dec = np.array(cat['dec'])
        
    # Bounds from ground truth
    RA_MIN, RA_MAX = 266.345, 266.455
    DEC_MIN, DEC_MAX = -29.055, -28.945
    
    # Check if RA/Dec are within these bounds (with a small buffer)
    assert np.all(ra >= RA_MIN - 0.01)
    assert np.all(ra <= RA_MAX + 0.01)
    assert np.all(dec >= DEC_MIN - 0.01)
    assert np.all(dec <= DEC_MAX + 0.01)

@pytest.mark.skipif(not os.path.exists(PROTOTYPE_PATH), reason="Prototype ASDF not found")
def test_pipeline_catalog_wcs_alignment():
    """
    Test that the pipeline correctly applies WCS to generated catalogs.
    We'll create a dummy list of stars and check if the RA/Dec results match direct WCS calls.
    """
    from pollux.pipeline import PhotometryPipeline
    
    with asdf.open(PROTOTYPE_PATH) as af:
        wcs = af['roman']['meta']['wcs']
    
    # Create dummy stars in detector coordinates
    dummy_stars = [
        {'x': 100.0, 'y': 100.0, 'mag': 15.0, 'completeness': 1.0, 'prob': 1.0},
        {'x': 2044.0, 'y': 2044.0, 'mag': 12.0, 'completeness': 1.0, 'prob': 1.0},
        {'x': 4000.0, 'y': 4000.0, 'mag': 18.0, 'completeness': 0.5, 'prob': 0.9},
    ]
    
    # Use a dummy model path
    pipeline = PhotometryPipeline(model_path="models/stage0/stage0_epoch_12.pth")
    catalog = pipeline._build_catalog(dummy_stars, wcs)
    
    # Manual WCS call
    expected_ra, expected_dec = wcs([s['x'] for s in dummy_stars], [s['y'] for s in dummy_stars])
    
    np.testing.assert_allclose(catalog['ra'].values, expected_ra)
    np.testing.assert_allclose(catalog['dec'].values, expected_dec)

def test_tiling_logic_boundaries():
    """
    Test that the tiling logic covers the full 4088x4088 image.
    """
    h, w = 4088, 4088
    stride = 224
    tile_size = 256
    
    ny = (h + stride - 1) // stride
    nx = (w + stride - 1) // stride
    
    last_y_start = (ny - 1) * stride
    last_x_start = (nx - 1) * stride
    
    assert ny == 19
    assert nx == 19
    assert last_y_start + tile_size >= h
    assert last_x_start + tile_size >= w
