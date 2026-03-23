import asdf
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from castor.models.dense_grid import DenseGridModel
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, GLOBAL_STRETCH_SCALE
from .astrometry import get_gaia_reference
from astropy.io import fits
from astropy.wcs import WCS
import os

class PhotometryPipeline:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Use constants from castor to ensure architecture matches
        self.model = DenseGridModel(
            K=MAX_CAPACITY_PER_CELL, 
            shape_size=SHAPE_SIZE, 
            cell_size=DEFAULT_CELL_SIZE
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def _load_image(self, file_path):
        """
        Loads image data, WCS, and metadata from ASDF or FITS.
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.asdf':
            with asdf.open(file_path) as af:
                image_data = np.array(af['roman']['data'])
                meta = af['roman']['meta']
                wcs = meta['wcs']
                filter_name = meta['instrument']['optical_element']
                return image_data, wcs, filter_name
        
        elif ext in ['.fits', '.fit', '.fz']:
            with fits.open(file_path) as hdul:
                # Standard Roman L2 FITS typically has image in 'SCI' or EXT 1
                # Compressed FITS (.fz) usually have data in the first extension
                if 'SCI' in hdul:
                    image_data = hdul['SCI'].data
                    header = hdul['SCI'].header
                else:
                    # Look for the first HDU with data
                    image_data = None
                    header = None
                    for hdu in hdul:
                        if hdu.data is not None:
                            image_data = hdu.data
                            header = hdu.header
                            break
                    
                    if image_data is None:
                        raise ValueError(f"No image data found in FITS file: {file_path}")
                
                wcs = WCS(header)
                # Attempt to find filter in common keywords
                filter_name = header.get('FILTER', header.get('OPT_ELEM', 'UNKNOWN'))
                return image_data, wcs, filter_name
        
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def process_image(self, input_path, threshold=0.5, batch_size=16, auto_calibrate=True):
        """
        Processes a Roman image (any size, FITS or ASDF) and performs automated Gaia calibration.
        """
        image_data, wcs, filter_name = self._load_image(input_path)
        
        h, w = image_data.shape
        tile_size, stride, margin = 256, 224, 16
        ny, nx = (h + stride - 1) // stride, (w + stride - 1) // stride
        pad_h, pad_w = (ny - 1) * stride + tile_size, (nx - 1) * stride + tile_size
        padded_image = np.pad(image_data, ((margin, pad_h - h - margin), (margin, pad_w - w - margin)), mode='reflect')
        
        global_stars = []
        tile_coords = [(ix, iy) for iy in range(ny) for ix in range(nx)]
        
        for i in tqdm(range(0, len(tile_coords), batch_size), desc="Batch Inference"):
            batch_coords = tile_coords[i:i + batch_size]
            batch_tiles = []
            batch_medians = []
            
            for ix, iy in batch_coords:
                tile = padded_image[iy*stride:iy*stride+tile_size, ix*stride:ix*stride+tile_size]
                median = np.median(tile)
                # Castor Preprocessing: arcsinh( (I - median) / scale )
                tile_stretched = np.arcsinh((tile - median) / GLOBAL_STRETCH_SCALE)
                batch_tiles.append(tile_stretched)
                batch_medians.append(median)
            
            batch_tensor = torch.from_numpy(np.stack(batch_tiles).astype(np.float32)).unsqueeze(1).to(self.device)
            with torch.no_grad():
                # Castor model returns physical flux (after sinh) in 'stars'
                batch_stars = self.model(batch_tensor)['stars'].cpu().numpy()
            
            for idx, (ix, iy) in enumerate(batch_coords):
                self._extract_stars_vectorized(
                    batch_stars[idx], 
                    ix*stride-margin, 
                    iy*stride-margin, 
                    threshold, 
                    global_stars,
                    batch_medians[idx],
                    img_shape=(h, w)
                )
        
        catalog = self._build_catalog(global_stars, wcs)
        
        if auto_calibrate and len(catalog) > 0:
            # Calibrate using the center of the image
            ra_c, dec_c = wcs.wcs_pix2world(w//2, h//2, 0)
            ref = get_gaia_reference(float(ra_c), float(dec_c), filter_name)
            if ref is not None:
                catalog = self.calibrate_catalog(catalog, ref['ra'], ref['dec'], ref['flux'])
                
        return catalog

    def _extract_stars_vectorized(self, grid_preds, x_offset, y_offset, threshold, global_stars, tile_median, img_shape):
        effective_threshold = max(threshold, 0.5)
        h_img, w_img = img_shape
        
        grid_h, grid_w, K, _ = grid_preds.shape
        cell_size = DEFAULT_CELL_SIZE
        
        # Grid indexing for the valid central region
        grid_margin = 16 // cell_size
        grid_stride = 224 // cell_size
        y_end, x_end = min(grid_h, grid_margin + grid_stride), min(grid_w, grid_margin + grid_stride)
        
        crop = grid_preds[grid_margin:y_end, grid_margin:x_end, :, :]
        mask = crop[..., 0] > effective_threshold
        if not np.any(mask): return

        indices = np.argwhere(mask)
        params = crop[mask]
        
        # Positions: (grid_index + margin) * cell_size + offset_within_cell
        ly_tile = (indices[:, 0] + grid_margin) * cell_size + params[:, 2]
        lx_tile = (indices[:, 1] + grid_margin) * cell_size + params[:, 1]
        
        gx, gy = lx_tile + x_offset, ly_tile + y_offset
        # Use dynamic image shape instead of hardcoded 4088
        valid = (gx >= 0) & (gx < w_img) & (gy >= 0) & (gy < h_img)
        
        # Castor outputs physical flux directly (not log10)
        flux_phys = params[:, 3]
        
        for i in range(len(gx)):
            if valid[i]:
                # mag_raw is log10(DN/s) for consistency with existing calibration logic
                # We need to handle 0 or negative flux safely before log10
                safe_flux = max(flux_phys[i], 1e-5)
                global_stars.append({
                    'x': gx[i], 'y': gy[i],
                    'mag_raw': np.log10(safe_flux), 
                    'flux_raw': flux_phys[i],
                    'completeness': params[i, 4],
                    'prob': params[i, 0]
                })

    def _build_catalog(self, stars, wcs):
        import pandas as pd
        df = pd.DataFrame(stars)
        if len(df) == 0: return df
        ra, dec = wcs.wcs_pix2world(df['x'].values, df['y'].values, 0)
        df['ra'], df['dec'] = ra, dec
        return df

    def calibrate_catalog(self, catalog, reference_ra, reference_dec, reference_flux_jy, radius_arcsec=1.0):
        """
        Calibrate model's log10(DN/s) to physical AB magnitudes using a reference catalog.
        """
        from scipy.spatial import cKDTree
        
        mask_cat = (np.isfinite(catalog['ra']) & np.isfinite(catalog['dec']) & 
                    (catalog['prob'] > 0.5))
        cat_valid = catalog[mask_cat].copy()
        
        if len(cat_valid) < 5: 
            return catalog

        mask_ref = np.isfinite(reference_ra) & np.isfinite(reference_dec)
        ref_ra, ref_dec, ref_f = reference_ra[mask_ref], reference_dec[mask_ref], reference_flux_jy[mask_ref]

        def to_xyz(ra, dec):
            r, d = np.deg2rad(ra), np.deg2rad(dec)
            return np.column_stack((np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)))

        tree = cKDTree(to_xyz(cat_valid['ra'].values, cat_valid['dec'].values))
        chord_len = 2 * np.sin(np.deg2rad(radius_arcsec/3600.0) / 2.0)
        dist, idx = tree.query(to_xyz(ref_ra, ref_dec), distance_upper_bound=chord_len)
        matched = dist < chord_len
        
        if np.sum(matched) < 5:
            print("Warning: Insufficient matches for Gaia calibration.")
            return catalog

        x_raw = cat_valid['mag_raw'].values[idx[matched]]
        y_true = -2.5 * np.log10(ref_f[matched] + 1e-15) + 8.90
        
        zp_offsets = y_true + 2.5 * x_raw
        slope = -2.5
        intercept = np.median(zp_offsets)
        
        print(f"Gaia Calibration (Gaia->ML): {np.sum(matched)} matches.")
        print(f"Fit: m_AB = {slope:.1f} * log10(f_dns) + {intercept:.4f} (Fixed Physical Slope)")
        
        catalog['ab_mag'] = slope * catalog['mag_raw'] + intercept
        catalog['log10_flux_jy'] = (catalog['ab_mag'] - 8.90) / -2.5
        catalog['flux_jy'] = 10**catalog['log10_flux_jy']
        
        return catalog

    def save_catalog(self, catalog, output_path):
        tree = {'catalog': {col: catalog[col].values for col in catalog.columns}}
        with asdf.AsdfFile(tree) as af:
            af.write_to(output_path)
