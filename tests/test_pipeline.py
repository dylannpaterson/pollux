import numpy as np
import pytest
import os
import asdf
import yaml
import pandas as pd
from unittest.mock import MagicMock, patch
from pollux.pipeline import (
    PipelineContext, 
    ImageLoaderStep, 
    PhotometryInferenceStep, 
    GaiaCalibrationStep, 
    CatalogSaveStep,
    Pipeline
)
from astropy.wcs import WCS
from astropy.io import fits

def test_extract_stars_center_crop():
    """
    Verify the Center-Crop Rule in _extract_stars_vectorized.
    """
    step = PhotometryInferenceStep()
    
    # [grid_h, grid_w, K, params]
    # K=3, params=7
    grid_preds = np.zeros((64, 64, 3, 7))
    
    # Star inside central area: grid cell (10, 10)
    grid_preds[10, 10, 0, 0] = 1.0 # prob
    grid_preds[10, 10, 0, 1] = 0.5 # dx
    grid_preds[10, 10, 0, 2] = 0.5 # dy
    grid_preds[10, 10, 0, 3] = 100.0 # flux_phys
    
    global_stars = []
    step._extract_stars_vectorized(
        grid_preds, 
        x_offset=0, 
        y_offset=0, 
        threshold=0.5, 
        global_stars=global_stars, 
        img_shape=(4088, 4088)
    )
    
    assert len(global_stars) == 1
    assert global_stars[0]['x'] == 40.5 

def test_extract_stars_global_mapping():
    """
    Verify global detector coordinate mapping.
    """
    step = PhotometryInferenceStep()
    grid_preds = np.zeros((64, 64, 3, 7))
    
    # Star at grid (14, 24)
    grid_preds[14, 24, 0, 0] = 1.0
    grid_preds[14, 24, 0, 1] = 0.5 # dx
    grid_preds[14, 24, 0, 2] = 0.2 # dy
    grid_preds[14, 24, 0, 3] = 100.0
    
    x_offset, y_offset = 1000, 2000
    
    global_stars = []
    step._extract_stars_vectorized(
        grid_preds, 
        x_offset, 
        y_offset, 
        threshold=0.5, 
        global_stars=global_stars, 
        img_shape=(5000, 5000)
    )
    
    assert len(global_stars) == 1
    assert global_stars[0]['x'] == pytest.approx(1096.5)
    assert global_stars[0]['y'] == pytest.approx(2056.2)

def test_build_catalog_wcs():
    """
    Verify _build_catalog uses WCS correctly.
    """
    step = PhotometryInferenceStep()
    stars = [{'x': 100, 'y': 200, 'mag_raw': 2.0, 'flux_raw': 100, 'completeness': 0.9, 'prob': 1.0}]
    
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, -30.0]
    wcs.wcs.crpix = [0.0, 0.0]
    wcs.wcs.pc = [[-0.1, 0], [0, 0.1]]
    wcs.wcs.set()
    
    expected_ra, expected_dec = wcs.wcs_pix2world(100, 200, 0)
    
    df = step._build_catalog(stars, wcs)
    
    assert len(df) == 1
    assert df.iloc[0]['ra'] == pytest.approx(float(expected_ra))
    assert df.iloc[0]['dec'] == pytest.approx(float(expected_dec))

def test_load_fits_image(tmp_path):
    """
    Verify ImageLoaderStep with FITS.
    """
    step = ImageLoaderStep()
    context = PipelineContext()
    
    data = np.random.rand(100, 100).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header['FILTER'] = 'F146'
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL1'] = 150.0
    hdu.header['CRVAL2'] = -30.0
    
    fits_path = os.path.join(tmp_path, "test.fits")
    hdu.writeto(fits_path)
    
    step.run(context, {'path': fits_path})
    
    assert context.image_data.shape == (100, 100)
    assert context.filter_name == 'F146'
    assert isinstance(context.wcs, WCS)
    assert context.wcs.wcs.crval[0] == 150.0

def test_load_asdf_image(tmp_path):
    """
    Verify ImageLoaderStep with ASDF.
    """
    step = ImageLoaderStep()
    context = PipelineContext()
    
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
        
    step.run(context, {'path': asdf_path})
    
    assert context.image_data.shape == (50, 50)
    assert context.filter_name == 'F158'
    assert isinstance(context.wcs, WCS)

@patch('onnxruntime.InferenceSession')
def test_photometry_inference_step(mock_ort, tmp_path):
    """
    Verify PhotometryInferenceStep with mocked ONNX session.
    """
    step = PhotometryInferenceStep()
    context = PipelineContext()
    context.image_data = np.zeros((256, 256))
    context.wcs = WCS(naxis=2)
    
    # Mock session
    session_instance = mock_ort.return_value
    session_instance.get_inputs.return_value = [MagicMock(name='input')]
    # Mock output stars grid: (batch, h, w, K, params)
    # 1 tile, 64x64 grid cells, 3 stars per cell, 7 params
    session_instance.run.return_value = [np.zeros((1, 64, 64, 3, 7))]
    
    step.run(context, {'model_path': 'dummy.onnx', 'batch_size': 1})
    
    assert context.catalog is not None
    assert isinstance(context.catalog, pd.DataFrame)

def test_pipeline_execution(tmp_path):
    """
    Verify full Pipeline execution with mocks.
    """
    config_path = os.path.join(tmp_path, "config.yaml")
    config = {
        'pipeline': [
            {'type': 'load_image', 'path': 'dummy.asdf'}
        ]
    }
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    with patch.object(ImageLoaderStep, 'run') as mock_load:
        pipeline = Pipeline(config_path)
        pipeline.run()
        mock_load.assert_called_once()
