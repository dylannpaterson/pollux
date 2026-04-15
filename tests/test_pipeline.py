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

def test_extract_stars_batch_vectorized():
    """
    Verify the _extract_stars_batch_vectorized method.
    """
    step = PhotometryInferenceStep()
    
    # [batch, grid_h, grid_w, K, params]
    # batch=1, K=3, params=7
    batch_preds = np.zeros((1, 64, 64, 3, 7))
    
    # Star inside central area: grid cell (10, 10)
    # Note: center crop margin is 16//cell_size = 4
    batch_preds[0, 10, 10, 0, 0] = 1.0 # prob
    batch_preds[0, 10, 10, 0, 1] = 0.5 # dx
    batch_preds[0, 10, 10, 0, 2] = 0.5 # dy
    batch_preds[0, 10, 10, 0, 3] = 100.0 # flux_phys
    
    x_offsets = np.array([0])
    y_offsets = np.array([0])
    
    metadata = {'exptime': 100.0}
    
    res = step._extract_stars_batch_vectorized(
        batch_preds, 
        x_offsets,
        y_offsets,
        threshold=0.5, 
        img_shape=(4088, 4088),
        metadata=metadata
    )
    
    assert res is not None
    assert len(res['x']) == 1
    # Global pos = (grid_idx * cell_size) + offset + dx
    # Here grid_idx=10, cell_size=4 -> 40 + 0 + 0.5 = 40.5
    assert res['x'][0] == 40.5 
    # Instrumental Mag = -2.5 * log10(100 / 100) = 0.0
    assert res['mag_raw'][0] == 0.0

def test_build_catalog_wcs():
    """
    Verify _build_catalog uses WCS correctly.
    """
    step = PhotometryInferenceStep()
    # Mock global_stars_list which is a list of dicts of arrays
    stars_batch = {
        'x': np.array([100.0]), 
        'y': np.array([200.0]), 
        'mag_raw': np.array([2.0]), 
        'flux_raw': np.array([100.0]), 
        'prob': np.array([1.0]),
        'log_var_x': np.array([0.0]),
        'log_var_y': np.array([0.0]),
        'log_var_m': np.array([0.0])
    }
    
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, -30.0]
    wcs.wcs.crpix = [0.0, 0.0]
    wcs.wcs.pc = [[-0.1, 0], [0, 0.1]]
    wcs.wcs.set()
    
    expected_ra, expected_dec = wcs.pixel_to_world_values(100, 200)
    
    df = step._build_catalog([stars_batch], wcs)
    
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
    hdu.header['EXPTIME'] = 100.0
    
    fits_path = os.path.join(tmp_path, "test.fits")
    hdu.writeto(fits_path)
    
    step.run(context, {'path': fits_path})
    
    assert context.image_data.shape == (100, 100)
    assert context.filter_name == 'F146'
    assert context.metadata['exptime'] == 100.0
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
                'instrument': {'optical_element': 'F158', 'detector': 'WFI01'},
                'exposure': {'exposure_time': 100.0, 'start_time': 61344.0, 'ma_table_number': 1002, 'nresultants': 5},
                'photometry': {'conversion_microjanskys': 0.09}
            }
        }
    }
    
    asdf_path = os.path.join(tmp_path, "test.asdf")
    with asdf.AsdfFile(tree) as af:
        af.write_to(asdf_path)
        
    step.run(context, {'path': asdf_path})
    
    assert context.image_data.shape == (50, 50)
    assert context.filter_name == 'F158'
    assert context.metadata['exptime'] == 100.0
    assert context.metadata['phot_conv'] == 0.09
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
    context.metadata = {'exptime': 100.0}
    
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
