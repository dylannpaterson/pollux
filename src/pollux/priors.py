import sqlite3
import numpy as np
import pandas as pd
import os
from .base import WCSAdapter

def get_prior_catalog_from_db(db_path, wcs: WCSAdapter, img_shape, ra_col='ra_weighted', dec_col='dec_weighted', err_col='ra_rmse'):
    """
    Queries the database for targets within the RA/Dec footprint of the current image.
    Projects these targets into the CURRENT image's pixel coordinates.
    
    CRITICAL: Does NOT use 'x' or 'y' from the database as they are observation-specific.
    Uses RA/Dec to ensure correct alignment across different SCA pointings or orientations.
    """
    if not os.path.exists(db_path):
        print(f"⚠️ Warning: Prior database {db_path} not found.")
        return pd.DataFrame()
        
    conn = sqlite3.connect(db_path)
    try:
        # 1. Calculate RA/Dec footprint of the current image
        footprint = wcs.calc_footprint(img_shape)
        ra_min, ra_max = np.min(footprint[:, 0]), np.max(footprint[:, 0])
        dec_min, dec_max = np.min(footprint[:, 1]), np.max(footprint[:, 1])
        
        # Add a small buffer for safety (10 arcsec)
        pad = 10.0 / 3600.0
        ra_min -= pad; ra_max += pad
        dec_min -= pad; dec_max += pad
        
        # 2. Query targets. We use ra_weighted/dec_weighted as they represent the best known position.
        query = f"""
            SELECT uuid, {ra_col}, {dec_col}, {err_col} 
            FROM targets 
            WHERE {ra_col} >= ? AND {ra_col} <= ? 
              AND {dec_col} >= ? AND {dec_col} <= ?
        """
        df = pd.read_sql_query(query, conn, params=(ra_min, ra_max, dec_min, dec_max))
    finally:
        conn.close()
        
    if df.empty:
        return pd.DataFrame()
        
    # 3. Project RA/Dec to CURRENT pixel coordinates
    # This handles any rotation, shift, or distortion in the current WCS.
    px, py = wcs.world_to_pixel_values(df[ra_col].values, df[dec_col].values)
    df['x_current'], df['y_current'] = px, py
    
    # 4. Filter to stars actually within the image (plus a small margin for splatting)
    h, w = img_shape
    margin = 5
    mask = (px >= -margin) & (px < w + margin) & (py >= -margin) & (py < h + margin)
    
    return df[mask].reset_index(drop=True)

def render_prior_map(catalog, img_shape):
    """
    Renders a bilinear splat prior map from a projected catalog.
    Uses 'x_current' and 'y_current' coordinates.
    Returns a 2D numpy array [H, W].
    """
    h, w = img_shape
    prior = np.zeros((h, w), dtype=np.float32)
    
    if catalog is None or catalog.empty:
        return prior
        
    px, py = catalog['x_current'].values, catalog['y_current'].values
    
    # We use p=1.0 for all stars from the prior database.
    # If the database has a probability/objectness column, it could be used here.
    p = np.ones_like(px)
    
    # Filter for exact bounds before splatting to avoid index errors
    mask = (px >= 0) & (px < w - 1) & (py >= 0) & (py < h - 1)
    px, py, p = px[mask], py[mask], p[mask]
    
    if len(px) == 0:
        return prior
        
    # Bilinear Splatting
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    dx, dy = px - x0, py - y0
    
    w00 = (1 - dx) * (1 - dy) * p
    w10 = dx * (1 - dy) * p
    w01 = (1 - dx) * dy * p
    w11 = dx * dy * p
    
    def splat(ix, iy, weight):
        flat_idx = iy * w + ix
        # Use np.bincount for vectorized atomic addition
        prior.flat += np.bincount(flat_idx, weights=weight, minlength=prior.size)
        
    splat(x0, y0, w00)
    splat(x0 + 1, y0, w10)
    splat(x0, y0 + 1, w01)
    splat(x0 + 1, y0 + 1, w11)
    
    return prior
