import asdf
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .model import DenseGridModel

class PhotometryPipeline:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = DenseGridModel(K=3, shape_size=9).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        # Check if it's a state_dict or a full checkpoint
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def process_image(self, asdf_path, threshold=0.5):
        with asdf.open(asdf_path) as af:
            image_data = np.array(af['roman']['data'])
            wcs = af['roman']['meta']['wcs']
            # Ground truth if available (for debugging)
            gt = af['ground_truth'] if 'ground_truth' in af else None

        h, w = image_data.shape
        tile_size = 256
        stride = 224
        margin = 16  # (256 - 224) / 2
        
        # Calculate number of tiles needed to cover the whole image
        ny = (h + stride - 1) // stride
        nx = (w + stride - 1) // stride
        
        # Calculate required padded size
        # We need to be able to extract a tile_size block starting at (ny-1)*stride
        pad_h = (ny - 1) * stride + tile_size
        pad_w = (nx - 1) * stride + tile_size
        
        # Initial padding for the 16-pixel margin
        # We also pad extra on the right/bottom to ensure full tiles
        padding = (
            (margin, pad_h - h - margin),
            (margin, pad_w - w - margin)
        )
        padded_image = np.pad(image_data, padding, mode='reflect')
        
        global_stars = []
        
        total_tiles = ny * nx
        pbar = tqdm(total=total_tiles, desc="Processing Tiles")
        
        for iy in range(ny):
            for ix in range(nx):
                y_start = iy * stride
                x_start = ix * stride
                
                # Extract tile from padded image
                tile = padded_image[y_start:y_start + tile_size, x_start:x_start + tile_size]
                
                # Run inference
                with torch.no_grad():
                    tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).to(self.device)
                    out = self.model(tile_tensor)
                    stars = out['stars'].squeeze(0).cpu().numpy() 
                    
                # Process detections
                # x_offset and y_offset should map tile-local (lx, ly) to global detector (gx, gy)
                # lx=margin maps to global=ix*stride
                # gx_glob = lx - margin + ix*stride
                self._extract_stars(stars, ix * stride - margin, iy * stride - margin, threshold, global_stars)
                pbar.update(1)
        
        pbar.close()
        
        # Convert to catalog
        catalog = self._build_catalog(global_stars, wcs)
        return catalog

    def _extract_stars(self, grid_preds, x_offset, y_offset, threshold, global_stars):
        """
        Extract stars from the grid, applying the Center-Crop Rule.
        """
        grid_h, grid_w, K, _ = grid_preds.shape
        cell_size = 2  # Input 256 -> Grid 128
        
        # Center-Crop Rule: Each tile is responsible for stars in its central stride x stride area.
        grid_margin = 8  # 16 / 2
        grid_stride = 112 # 224 / 2
        
        y_end = min(grid_h, grid_margin + grid_stride)
        x_end = min(grid_w, grid_margin + grid_stride)
        
        for gy in range(grid_margin, y_end):
            for gx in range(grid_margin, x_end):
                for k in range(K):
                    p, dx, dy, log_m, c = grid_preds[gy, gx, k, :5]
                    if p > threshold:
                        lx = (gx * cell_size) + dx
                        ly = (gy * cell_size) + dy
                        
                        gx_glob = lx + x_offset
                        gy_glob = ly + y_offset
                        
                        # Cap log_m to prevent overflows during 10**log_m conversion
                        # 10.0 is 10 billion DN/s, more than enough.
                        log_m_capped = np.clip(log_m, -5, 10.0)
                        
                        # Final check: is it within the 4088x4088 detector?
                        if 0 <= gx_glob < 4088 and 0 <= gy_glob < 4088:
                            global_stars.append({
                                'x': gx_glob,
                                'y': gy_glob,
                                'mag': log_m_capped, # Following design doc: mag is log-flux
                                'flux': 10**log_m_capped,
                                'completeness': c,
                                'prob': p
                            })

    def _build_catalog(self, stars, wcs):
        import pandas as pd
        df = pd.DataFrame(stars)
        if len(df) == 0:
            return df
            
        # WCS Transformation
        coords = wcs(df['x'].values, df['y'].values)
        df['ra'] = coords[0]
        df['dec'] = coords[1]
        
        return df

    def calibrate_catalog(self, catalog, reference_catalog_path=None):
        """
        Stage 5: Photometric Calibration.
        Aligns the predicted magnitudes with an external reference.
        """
        if reference_catalog_path is None:
            print("No reference catalog provided. Skipping calibration.")
            return catalog

        print(f"Calibrating against {reference_catalog_path}...")
        # TODO: Implement cross-match with Gaia/Pan-STARRS/etc.
        # For now, this is a placeholder for the Stage 5 logic.
        return catalog

    def save_catalog(self, catalog, output_path):
        # Save to ASDF as per design doc
        tree = {
            'catalog': {
                'x': catalog['x'].values,
                'y': catalog['y'].values,
                'ra': catalog['ra'].values,
                'dec': catalog['dec'].values,
                'mag': catalog['mag'].values,
                'flux': catalog['flux'].values,
                'completeness': catalog['completeness'].values,
                'prob': catalog['prob'].values
            }
        }
        with asdf.AsdfFile(tree) as af:
            af.write_to(output_path)
