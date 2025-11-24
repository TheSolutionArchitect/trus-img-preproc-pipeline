# Transrectal Ultrasound (TRUS) Image Preprocessing Pipeline

**Purpose**
High–dose–rate (HDR) prostate brachytherapy requires accurate intraoperative localization of needle tips in transrectal ultrasound (TRUS) images, which are often degraded by speckle noise and low contrast. We aim to design a configurable, reproducible preprocessing pipeline tailored to TRUS that improves image quality and enhances downstream needle-spot detection using a Swin UNETR-based model.

**Methods**
We developed a standalone preprocessing module that reads parameters exclusively from an external YAML configuration file (`config.yml`), avoiding hardcoded settings. The pipeline supports selectable noise-reduction and contrast-enhancement operators—including Gaussian filtering, anisotropic diffusion (edge-preserving), histogram equalization, and contrast-limited adaptive histogram equalization (CLAHE)—followed by intensity normalization. It automatically generates side-by-side visual composites (pre/post) for quality assurance and organizes outputs by patient ID and timestamp to facilitate auditability and batch processing. We trained a Swin UNETR model for needle-spot detection/segmentation on TRUS images with expert ground-truth masks, and performed ablations comparing: (i) no preprocessing, (ii) normalization-only baselines, and (iii) combinations of anisotropic diffusion and CLAHE. We assessed image quality with contrast-to-noise ratio (CNR) and edge-strength metrics, and evaluated detection performance using standard segmentation and detection measures (Dice score, precision, recall, and localization error).

**Results**
The proposed pipeline consistently improved TRUS image quality, increasing CNR and enhancing edge sharpness around fine, high-intensity needle structures while suppressing speckle. Among variants, anisotropic diffusion followed by CLAHE yielded the best trade-off between noise suppression and detail preservation. When used for training and inference, this preprocessing improved Swin-UNETR’s needle-spot detection—raising overlap-based metrics and reducing false positives, particularly near heterogeneous boundaries—relative to normalization-only and no-preprocessing baselines. The composite visual outputs enabled rapid human verification of parameter choices and supported reproducible experimentation.

**Conclusion**
A configuration-driven TRUS preprocessing workflow centered on anisotropic diffusion and CLAHE can materially enhance image quality and improve Swin UNETR-based needle-spot detection for HDR prostate brachytherapy. The modular design, YAML-controlled parameters, and automated visual QA make the approach practical for clinical research pipelines and scalable to larger datasets and multi-institutional studies.

### How to run
Use Python 3.11 and other libs as mentioned in main.py. Please update the input and output directories in config.yml. Make sure to add patient-wise subdirectories in the input base directories. 
```
python main.py --config config.yml
```
### Sequence of steps in the configured pipeline

1. Load → grayscale
2. Normalization
3. Noise reduction
4. Contrast enhancement
5. Final min–max normalization
6. QA visualization (composites, histograms, zoom)
7. Optional threshold preview
8. Optional ablation variants (for analysis, not the main output)

Here is a table focused on the image‑processing stages:

| Seq | Stage (conceptual)            | Method (from config/code)                                                                 | Significance for TRUS HDR brachy preprocessing |
|-----|-------------------------------|-------------------------------------------------------------------------------------------|-----------------------------------------------|
| 0   | Image loading & grayscale     | `cv2.imread` + `cv2.cvtColor(..., COLOR_BGR2GRAY)` when `general.to_grayscale: true`     | Forces all TRUS images to a single gray channel with consistent intensity format, regardless of input RGB/PNG encoding. This avoids channel‑wise variability and is appropriate because B‑mode TRUS is intrinsically grayscale. |
| 1   | Initial intensity normalization | `pipeline.normalization.method: "minmax"` with `percentiles: [1.0, 99.0]` (via `normalize_image`) | Robustly scales intensities to [0,1] using 1st–99th percentiles, reducing the effect of outliers (e.g. very bright needle tips or artifacts). This makes different patients/slices comparable and stabilizes subsequent denoising and contrast steps. |
| 2   | Noise reduction (speckle suppression) | `pipeline.noise_reduction.method: "anisotropic"` → Perona–Malik anisotropic diffusion with `niter=15, k=0.05, gamma=0.2, option=1` | TRUS is dominated by speckle noise. Anisotropic diffusion reduces speckle in relatively homogeneous regions (e.g., within the prostate gland) while preserving edges (prostate capsule, needle boundaries). This is crucial for later segmentation and for improving CNR without blurring clinically important structures. |
| 3   | Contrast enhancement          | `pipeline.contrast_enhancement.method: "clahe"` with `clip_limit=0.01`, `nbins=256` (via `apply_contrast_enhancement`) | CLAHE (adaptive histogram equalization) boosts local contrast, especially around the prostate, urethra, and needles, while limiting contrast amplification in very bright regions via clipping. For HDR brachy, this helps visually and algorithmically separate prostate vs surrounding tissues and highlight needles. |
| 4   | Final normalization to [0,1]  | `pipeline.final_minmax: true` → `normalize_image(..., method="minmax", percentiles=[0,100])` | Ensures that after denoising and contrast enhancement, the final image is again normalized into [0,1]. This standardizes dynamic range for: (a) saving outputs, (b) threshold preview, and (c) downstream ML models (e.g., segmentation networks) and metrics. |
| 5   | QA composites & histograms    | `qa.generate_composites: true`, `add_hist: true`, `zoom_from_mask: true`, `cmap: "gray"` (via `make_composite`) | Produces multi‑panel visuals: raw vs processed, difference maps, zoomed crops near ROI (mask), and histograms. This lets us qualitatively verify that speckle is reduced, CNR improved, and edges preserved around the prostate and needles for each patient. |
| 6   | Threshold preview (segmentation QA) | `qa.threshold_preview.enabled: true`, `mode: "otsu"`/`"adaptive_mean"`/`"fixed"` (via `threshold_preview`) with morphological cleaning | Provides a quick binary segmentation preview on the processed image using Otsu or adaptive mean thresholding. Helpful for assessing whether the current preprocessing makes prostate or needle segmentation feasible and to tune thresholding strategies. |
| 7   | Ablation variants generation  | `ablation.enabled: true`, `ablation_variants: ["raw","norm_only","adiff","adiff+clahe","gauss+clahe"]` (via `generate_variants`) | Systematically generates alternative preprocessing variants per image (e.g., with/without anisotropic diffusion or CLAHE, or substituting Gaussian for anisotropic). Then compare their CNR and edge strength to understand which steps truly help TRUS HDR tasks. These are for analysis, not the “official” pipeline output. |

### Output sample: After Anisotropic Diffusion

<img width="613" height="409" alt="image" src="https://github.com/user-attachments/assets/ff8f32d9-38be-4c1c-8adc-50112546a6ec" />
