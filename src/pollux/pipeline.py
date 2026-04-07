import asdf
import numpy as np
import os
import yaml
import pandas as pd
from abc import ABC, abstractmethod
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
import onnxruntime as ort
from .astrometry import get_gaia_reference
from castor.constants import DEFAULT_CELL_SIZE, GLOBAL_STRETCH_SCALE

class PipelineContext:
    """Holds the state of the pipeline between steps."""
    def __init__(self):
        self.image_data = None
        self.wcs = None
        self.filter_name = None
        self.catalog = None
        self.metadata = {}

class PipelineStep(ABC):
    """Base class for all pipeline steps."""
    @abstractmethod
    def run(self, context: PipelineContext, config: dict):
        pass

class ImageLoaderStep(PipelineStep):
    """Loads image data and WCS from ASDF or FITS files."""
    def run(self, context: PipelineContext, config: dict):
        file_path = config.get('path')
        if not file_path:
            raise ValueError("ImageLoaderStep requires a 'path' in config.")
        
        print(f"Loading image from {file_path}...")
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.asdf':
            with asdf.open(file_path) as af:
                context.image_data = np.array(af['roman']['data'])
                context.metadata = af['roman']['meta']
                context.wcs = context.metadata['wcs']
                context.filter_name = context.metadata['instrument']['optical_element']
        
        elif ext in ['.fits', '.fit', '.fz']:
            with fits.open(file_path) as hdul:
                if 'SCI' in hdul:
                    context.image_data = hdul['SCI'].data
                    header = hdul['SCI'].header
                else:
                    context.image_data = None
                    header = None
                    for hdu in hdul:
                        if hdu.data is not None:
                            context.image_data = hdu.data
                            header = hdu.header
                            break
                    if context.image_data is None:
                        raise ValueError(f"No image data found in FITS file: {file_path}")
                
                context.wcs = WCS(header)
                context.filter_name = header.get('FILTER', header.get('OPT_ELEM', 'UNKNOWN'))
        else:
            raise ValueError(f"Unsupported file format: {ext}")

class PhotometryInferenceStep(PipelineStep):
    """Performs star detection and photometry using an ONNX model."""
    def run(self, context: PipelineContext, config: dict):
        model_path = config.get('model_path')
        threshold = config.get('threshold', 0.5)
        batch_size = config.get('batch_size', 16)
        
        if not model_path:
            raise ValueError("PhotometryInferenceStep requires 'model_path' in config.")
        
        print(f"Running inference with model {model_path}...")
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        image_data = context.image_data
        h, w = image_data.shape
        tile_size, stride, margin = 256, 224, 16
        ny, nx = (h + stride - 1) // stride, (w + stride - 1) // stride
        
        def get_padded_tile(ix, iy):
            """Surgically extracts a tile and pads only what is necessary."""
            y0, y1 = iy * stride - margin, iy * stride + tile_size - margin
            x0, x1 = ix * stride - margin, ix * stride + tile_size - margin
            
            # Bound the crop to the actual image
            cy0, cy1 = max(0, y0), min(h, y1)
            cx0, cx1 = max(0, x0), min(w, x1)
            
            tile_crop = image_data[cy0:cy1, cx0:cx1]
            
            # Calculate needed padding
            pad_top, pad_bottom = cy0 - y0, y1 - cy1
            pad_left, pad_right = cx0 - x0, x1 - cx1
            
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                return np.pad(tile_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
            return tile_crop

        global_stars = []
        tile_coords = [(ix, iy) for iy in range(ny) for ix in range(nx)]
        
        for i in tqdm(range(0, len(tile_coords), batch_size), desc="Batch Inference"):
            batch_coords = tile_coords[i:i + batch_size]
            batch_tiles = [get_padded_tile(ix, iy) for ix, iy in batch_coords]
            
            batch_input = np.stack(batch_tiles).astype(np.float32)[:, np.newaxis, :, :]
            # ONNX inference
            outputs = session.run(None, {input_name: batch_input})
            batch_stars = outputs[0] # [Batch, H, W, K, 7]

            # Collect results for the whole batch
            batch_results = []
            for idx, (ix, iy) in enumerate(batch_coords):
                res = self._extract_stars_vectorized(
                    batch_stars[idx], 
                    ix*stride-margin, 
                    iy*stride-margin, 
                    threshold, 
                    img_shape=(h, w)
                )
                if res is not None:
                    batch_results.append(res)

            if batch_results:
                # Efficiently combine results
                combined = {k: np.concatenate([r[k] for r in batch_results]) for k in batch_results[0].keys()}
                global_stars.append(combined)

        context.catalog = self._build_catalog(global_stars, context.wcs)

    def _build_catalog(self, global_stars_list, wcs):
        if not global_stars_list:
            return pd.DataFrame()
            
        # Combine all batch results into one dictionary of arrays
        full_results = {k: np.concatenate([batch[k] for batch in global_stars_list]) for k in global_stars_list[0].keys()}
        df = pd.DataFrame(full_results)
        
        ra, dec = wcs.wcs_pix2world(df['x'].values, df['y'].values, 0)
        df['ra'], df['dec'] = ra, dec
        return df

    def _extract_stars_vectorized(self, grid_preds, x_offset, y_offset, threshold, img_shape):
        effective_threshold = max(threshold, 0.5)
        h_img, w_img = img_shape
        
        grid_h, grid_w, K, _ = grid_preds.shape
        cell_size = DEFAULT_CELL_SIZE
        
        grid_margin = 16 // cell_size
        grid_stride = 224 // cell_size
        y_end, x_end = min(grid_h, grid_margin + grid_stride), min(grid_w, grid_margin + grid_stride)
        
        crop = grid_preds[grid_margin:y_end, grid_margin:x_end, :, :]
        mask = crop[..., 0] > effective_threshold
        if not np.any(mask): return None

        indices = np.argwhere(mask)
        params = crop[mask]
        
        ly_tile = (indices[:, 0] + grid_margin) * cell_size + params[:, 2]
        lx_tile = (indices[:, 1] + grid_margin) * cell_size + params[:, 1]
        
        gx, gy = lx_tile + x_offset, ly_tile + y_offset
        valid = (gx >= 0) & (gx < w_img) & (gy >= 0) & (gy < h_img)
        
        if not np.any(valid): return None

        # Collect valid detections as arrays
        v_params = params[valid]
        v_gx = gx[valid]
        v_gy = gy[valid]
        
        # mag_raw is log10(flux)
        # Handle zeros safely for the whole array
        flux_phys = v_params[:, 3]
        safe_flux = np.maximum(flux_phys, 1e-5)
        mag_raw_log10 = np.log10(safe_flux)
        
        return {
            'x': v_gx, 
            'y': v_gy,
            'mag_raw': mag_raw_log10, 
            'flux_raw': flux_phys,
            'prob': v_params[:, 0],
            'log_var_x': v_params[:, 4],
            'log_var_y': v_params[:, 5],
            'log_var_m': v_params[:, 6]
        }

class GaiaCalibrationStep(PipelineStep):
    """Calibrates the raw catalog using Gaia DR3 as reference."""
    def run(self, context: PipelineContext, config: dict):
        if context.catalog is None or len(context.catalog) == 0:
            print("Warning: No catalog to calibrate.")
            return

        radius_arcsec = config.get('radius_arcsec', 1.0)
        min_prob = config.get('min_prob', 0.5)
        h, w = context.image_data.shape
        ra_c, dec_c = context.wcs.wcs_pix2world(w//2, h//2, 0)
        
        print(f"Calibrating catalog using Gaia at center RA={ra_c:.5f}, Dec={dec_c:.5f}...")
        ref = get_gaia_reference(float(ra_c), float(dec_c), context.filter_name)
        
        if ref is not None:
            context.catalog = self.calibrate_catalog(
                context.catalog, 
                ref['ra'], ref['dec'], ref['flux'], 
                radius_arcsec=radius_arcsec,
                min_prob=min_prob
            )
        else:
            print("Warning: Gaia reference stars not found. Skipping calibration.")

    def calibrate_catalog(self, catalog, reference_ra, reference_dec, reference_flux_jy, radius_arcsec=1.0, min_prob=0.5):
        from scipy.spatial import cKDTree
        
        mask_cat = (np.isfinite(catalog['ra']) & np.isfinite(catalog['dec']) & 
                    (catalog['prob'] > min_prob))
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
        
        print(f"Gaia Calibration: {np.sum(matched)} matches found.")
        catalog['ab_mag'] = slope * catalog['mag_raw'] + intercept
        catalog['log10_flux_jy'] = (catalog['ab_mag'] - 8.90) / -2.5
        catalog['flux_jy'] = 10**catalog['log10_flux_jy']
        
        return catalog

class CatalogSaveStep(PipelineStep):
    """Saves the catalog to an ASDF file."""
    def run(self, context: PipelineContext, config: dict):
        output_path = config.get('path', 'output_catalog.asdf')
        if context.catalog is None:
            print("Warning: No catalog to save.")
            return

        print(f"Saving catalog to {output_path}...")
        tree = {'catalog': {col: context.catalog[col].values for col in context.catalog.columns}}
        with asdf.AsdfFile(tree) as af:
            af.write_to(output_path)

class Pipeline:
    """Orchestrates the execution of pipeline steps based on a YAML config."""
    STEP_MAPPING = {
        'load_image': ImageLoaderStep,
        'photometry': PhotometryInferenceStep,
        'calibrate': GaiaCalibrationStep,
        'save_catalog': CatalogSaveStep
    }

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.context = PipelineContext()

    def run(self):
        steps = self.config.get('pipeline', [])
        for step_config in steps:
            step_type = step_config.get('type')
            step_class = self.STEP_MAPPING.get(step_type)
            if not step_class:
                raise ValueError(f"Unknown step type: {step_type}")
            
            step = step_class()
            step.run(self.context, step_config)
        
        return self.context.catalog

# Backward compatibility
class PhotometryPipeline:
    def __init__(self, model_path):
        self.model_path = model_path

    def process_image(self, input_path, threshold=0.5, batch_size=16, auto_calibrate=True):
        # Create a temporary config to mimic old behavior
        config = {
            'pipeline': [
                {'type': 'load_image', 'path': input_path},
                {'type': 'photometry', 'model_path': self.model_path, 'threshold': threshold, 'batch_size': batch_size},
            ]
        }
        if auto_calibrate:
            config['pipeline'].append({'type': 'calibrate'})
        
        # Write temp config
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        
        try:
            pipeline = Pipeline(temp_config_path)
            catalog = pipeline.run()
            return catalog
        finally:
            os.remove(temp_config_path)

    def save_catalog(self, catalog, output_path):
        step = CatalogSaveStep()
        context = PipelineContext()
        context.catalog = catalog
        step.run(context, {'path': output_path})
