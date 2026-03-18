# Architecture Design: Pollux (Automated Roman Photometry Pipeline)

## 1. Objective
To develop a high-throughput, production-ready photometry pipeline capable of processing full-scale ($4088 \times 4088$) Roman Level 2 images. **Pollux** serves as the deployment arm of the **Castor** ecosystem, utilizing its trained models to perform automated detection, photometry, and completeness mapping across the entire Wide Field Instrument (WFI) focal plane.

## 2. Input & Output Specifications

### Input (The Full Image)
*   **Format:** Roman Level 2 ASDF.
*   **Dimensions:** $4088 \times 4088$ (Single SCA).
*   **Preprocessing:** Background estimation (using Castor's background head) and noise-floor normalization.

### Output (The Master Catalog)
*   **Format:** ASDF.
*   **Primary Columns:**
    1.  **x, y:** Global Detector Coordinates (Sub-pixel).
    2.  **ra, dec:** World Coordinates (WCS-transformed).
    3.  **mag:** Measured Log-Flux / Magnitude.
    4.  **completeness:** Predicted recoverability score ($0.0 \to 1.0$).
    5.  **psf_shape:** (Optional) The 81-parameter PSF profile for residual analysis.

## 3. Pipeline Architecture

### Stage 1: Sliding Window Tiling
The $4088 \times 4088$ image is decomposed into $256 \times 256$ chunks.
*   **Overlap:** 32 pixels on all sides (16-pixel safety margin).
*   **Stride:** 224 pixels.
*   **Padding:** The outer boundaries of the SCA are mirror-padded by 16 pixels to ensure full context for edge sources.

### Stage 2: Parallel Batch Inference
Tiles are batched and passed through the **Castor** ResNet-34 model.
*   **Inference Mode:** Full evaluation (No gradients, half-precision FP16 recommended).
*   **Background Stitching:** The predicted local background values ($b$) are stitched into a global background map for the entire SCA.

### Stage 3: Duplicate Resolution (Effective Area Strategy)
To eliminate edge artifacts and redundant detections in the overlap zones:
*   **The Center-Crop Rule:** Each tile is only responsible for detections whose $(x, y)$ falls within its **central $224 \times 224$ area**.
*   Detections in the 16-pixel "outer margin" are discarded, as they will be "centered" in a neighboring tile.
*   This ensures zero duplicate merging logic is required during stitching.

### Stage 4: Global Catalog Assembly
*   **Coordinate Re-mapping:** Tile-local coordinates are offset to global detector coordinates: 
    $x_{global} = x_{tile} + (\text{tile\_col} \times \text{stride})$.
*   **WCS Transformation:** Global $x, y$ are converted to RA/Dec using the SCA's WCS information.

### Stage 5: Photometric Calibration & Alignment
To ensure the predicted magnitudes ($m$) are physically calibrated:
*   **Cross-matching:** Automated cross-match between the generated catalog and an external reference (e.g., Gaia, Pan-STARRS, or high-precision Roman "standard" fields).
*   **Zero-point Refinement:** Calculate the global photometric zero-point ($Zp$) by minimizing the residual between model-predicted magnitudes and reference magnitudes.
*   **Color-term Corrections:** (Optional) If multiple filters are available, apply color-dependent corrections to align the catalog with standard photometric systems.
*   **Calibration Application:** Apply the calculated offsets to the master catalog, providing a finalized, calibrated product.

## 4. Diagnostic & Validation Tools

### The Residual Map
Pollux can generate a full-sized residual image:
$\text{Residual} = \text{Original Image} - \sum (\text{Predicted PSF Shapes})$.
This is the ultimate test of the model's fidelity—a "clean" residual indicates successful photometry of all sources.

### The Depth Map
By plotting the predicted completeness scores ($c$) across the SCA, Pollux produces a high-resolution map of the survey's effective depth, highlighting regions of extreme crowding or noise.

## 5. Performance Targets
*   **Throughput:** < 10 seconds per SCA on an NVIDIA T4/L4 GPU.
*   **Scaling:** Designed to scale to the full 18-SCA Roman focal plane via parallel process workers.
