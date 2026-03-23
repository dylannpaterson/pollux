import numpy as np
import pytest
from pollux.pipeline import PhotometryPipeline

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
    
    # cell_size = 4
    # margin = 16 pixels = 4 cells
    # stride = 224 pixels = 56 cells
    
    # 1. Star inside central area: grid cell (10, 10)
    # 10 is inside [4, 4+56=60)
    grid_preds[10, 10, 0, 0] = 1.0 # prob
    grid_preds[10, 10, 0, 1] = 0.5 # dx
    grid_preds[10, 10, 0, 2] = 0.5 # dy
    grid_preds[10, 10, 0, 3] = 100.0 # flux_phys
    
    # 2. Star outside (in margin): grid cell (2, 2)
    # 2 is outside [4, 60)
    grid_preds[2, 2, 0, 0] = 1.0
    grid_preds[2, 2, 0, 3] = 100.0
    
    global_stars = []
    # Using zero offsets
    pipeline._extract_stars_vectorized(grid_preds, x_offset=0, y_offset=0, threshold=0.5, global_stars=global_stars, tile_median=0)
    
    # Should only find 1 star
    assert len(global_stars) == 1
    # x = (cell_idx) * 4 + dx = 10 * 4 + 0.5 = 40.5
    assert global_stars[0]['x'] == 40.5 

def test_extract_stars_global_mapping():
    """
    Verify global detector coordinate mapping.
    """
    pipeline = PhotometryPipeline.__new__(PhotometryPipeline)
    grid_preds = np.zeros((64, 64, 3, 86))
    
    # Star at grid (10, 20) in the CROP
    # This means grid_preds[10+4, 20+4] = grid_preds[14, 24]
    grid_preds[14, 24, 0, 0] = 1.0
    grid_preds[14, 24, 0, 1] = 0.5 # dx
    grid_preds[14, 24, 0, 2] = 0.2 # dy
    grid_preds[14, 24, 0, 3] = 100.0
    
    # Tile starts at x_glob=1000, y_glob=2000 (EXCLUDING margin)
    # But x_offset passed to _extract_stars_vectorized is ix*stride - margin
    # If ix=5, stride=224, margin=16, then x_offset = 1120 - 16 = 1104
    x_offset, y_offset = 1000, 2000
    
    global_stars = []
    pipeline._extract_stars_vectorized(grid_preds, x_offset, y_offset, threshold=0.5, global_stars=global_stars, tile_median=0)
    
    # indices will be (10, 20) relative to crop
    # lx_tile = (20 + 4) * 4 + 0.5 = 96.5
    # ly_tile = (10 + 4) * 4 + 0.2 = 56.2
    # gx = 96.5 + 1000 = 1096.5
    # gy = 56.2 + 2000 = 2056.2
    
    assert len(global_stars) == 1
    assert global_stars[0]['x'] == pytest.approx(1096.5)
    assert global_stars[0]['y'] == pytest.approx(2056.2)
