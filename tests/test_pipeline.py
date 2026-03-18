import numpy as np
import pytest
from pollux.pipeline import PhotometryPipeline

def test_extract_stars_center_crop():
    """
    Verify the Center-Crop Rule in _extract_stars.
    Detections outside the [margin, margin+stride) range should be ignored.
    """
    # Dummy pipeline (no weights needed for this test)
    # Mocking the model initialization to avoid needing real weights
    import torch
    from unittest.mock import MagicMock
    
    # Create a pipeline without loading real weights
    PhotometryPipeline.__init__ = lambda self: None
    pipeline = PhotometryPipeline()
    
    # [grid_h, grid_w, K, params] = [128, 128, 3, 5+...]
    grid_preds = np.zeros((128, 128, 3, 10))
    
    # 1. Star inside central area (gx=10, gy=10)
    # Cell size is 2, so (10, 10) grid -> (20, 20) pixel in tile.
    # Margin is 16, so 20 is inside [16, 16+224=240).
    grid_preds[10, 10, 0, 0] = 1.0 # prob
    grid_preds[10, 10, 0, 3] = 2.0 # log_m
    
    # 2. Star outside (in margin) (gx=4, gy=4)
    # (4, 4) grid -> (8, 8) pixel in tile.
    # 8 is outside [16, 240).
    grid_preds[4, 4, 0, 0] = 1.0
    grid_preds[4, 4, 0, 3] = 2.0
    
    global_stars = []
    # Using dummy offsets
    pipeline._extract_stars(grid_preds, x_offset=0, y_offset=0, threshold=0.5, global_stars=global_stars)
    
    # Should only find 1 star
    assert len(global_stars) == 1
    assert global_stars[0]['x'] == 20.0 # 10 * 2 + 0 (dx=0)

def test_extract_stars_global_mapping():
    """
    Verify global detector coordinate mapping.
    """
    PhotometryPipeline.__init__ = lambda self: None
    pipeline = PhotometryPipeline()
    grid_preds = np.zeros((128, 128, 1, 10))
    
    # Star at grid (10, 20) -> Tile pixels (20, 40)
    grid_preds[20, 10, 0, 0] = 1.0
    grid_preds[20, 10, 0, 1] = 0.5 # dx
    grid_preds[20, 10, 0, 2] = 0.2 # dy
    
    # Tile starts at x_glob=1000, y_glob=2000 (excluding margin)
    # Tile-local pixel (0, 0) is at x_glob = 1000 - 16 = 984
    x_offset, y_offset = 1000 - 16, 2000 - 16
    
    global_stars = []
    pipeline._extract_stars(grid_preds, x_offset, y_offset, threshold=0.5, global_stars=global_stars)
    
    # x_glob = lx + x_offset = (10 * 2 + 0.5) + 984 = 20.5 + 984 = 1004.5
    # y_glob = ly + y_offset = (20 * 2 + 0.2) + 1984 = 40.2 + 1984 = 2024.2
    assert len(global_stars) == 1
    assert global_stars[0]['x'] == pytest.approx(1004.5)
    assert global_stars[0]['y'] == pytest.approx(2024.2)
