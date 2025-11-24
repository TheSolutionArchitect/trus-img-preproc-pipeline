import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_opening, remove_small_objects, disk
from skimage.segmentation import find_boundaries


# -------------------------------
# YAML / I/O utilities
# -------------------------------
def load_yaml(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def timestamp_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def to_float01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = img * 0.0
    return img


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = to_float01(img)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def load_image(path: Path, force_gray: bool = True) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if force_gray and im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


# -------------------------------
# Dataset scanning
# -------------------------------
def list_patient_dirs(input_root: Path, allowed_exts: List[str]) -> List[Path]:
    """Return all subdirectories under input_root that contain at least one image."""
    patient_dirs = []
    for child in sorted(input_root.iterdir()):
        if child.is_dir():
            has_img = any((f.suffix.lower() in allowed_exts) for f in child.glob("*"))
            if has_img:
                patient_dirs.append(child)
    return patient_dirs


def find_images_in_dir(pdir: Path, allowed_exts: List[str]) -> List[Path]:
    return [p for p in sorted(pdir.glob("*")) if p.suffix.lower() in allowed_exts]


def parse_slice_number(fname: str) -> Optional[int]:
    m = re.search(r"slice[_\-](\d+)", fname)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def select_highest_slice(img_paths: List[Path]) -> Optional[Path]:
    candidates = []
    for p in img_paths:
        sn = parse_slice_number(p.name)
        if sn is not None:
            candidates.append((sn, p))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    return img_paths[-1] if img_paths else None


# -------------------------------
# Mask mapping
# -------------------------------
def map_mask_path(image_path: Path, masks_cfg: Dict[str, Any]) -> Optional[Path]:
    if not masks_cfg.get("enabled", False):
        return None

    masks_dir = Path(masks_cfg.get("masks_dir", image_path.parent))
    method = masks_cfg.get("mapping", "prefix")  # "prefix" | "suffix" | "regex"

    if method == "prefix":
        prefix = masks_cfg.get("mask_prefix", "gt_")
        mname = prefix + image_path.name
        cand = masks_dir / mname
        return cand if cand.exists() else None

    elif method == "suffix":
        suffix = masks_cfg.get("mask_suffix", "")
        mname = f"{image_path.stem}{suffix}{image_path.suffix}"
        cand = masks_dir / mname
        return cand if cand.exists() else None

    elif method == "regex":
        pattern = masks_cfg.get("pattern", r"^trus_(.*)$")
        replace = masks_cfg.get("replace", r"gt_\1")
        mname = re.sub(pattern, replace, image_path.name)
        cand = masks_dir / mname
        return cand if cand.exists() else None

    return None


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    if not mask_path or not mask_path.exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype(bool)


# -------------------------------
# Normalization
# -------------------------------
def normalize_image(im: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    method = cfg.get("method", "minmax")
    im = im.astype(np.float32)

    if method == "minmax":
        lpc, upc = cfg.get("percentiles", [0.0, 100.0])
        lo, hi = np.percentile(im, [lpc, upc])
        if hi > lo:
            im = np.clip(im, lo, hi)
        im = to_float01(im)
        return im

    elif method == "zscore":
        mu = float(im.mean())
        sd = float(im.std() + 1e-8)
        im = (im - mu) / sd
        im = np.clip(im, -3, 3)
        return to_float01(im)

    else:
        return to_float01(im)


# -------------------------------
# Denoising
# -------------------------------
def anisotropic_diffusion(im01: np.ndarray, niter: int, k: float, gamma: float, option: int) -> np.ndarray:
    I = im01.astype(np.float32).copy()
    for _ in range(int(niter)):
        nablaN = np.zeros_like(I)
        nablaS = np.zeros_like(I)
        nablaE = np.zeros_like(I)
        nablaW = np.zeros_like(I)
        nablaN[1:, :] = I[1:, :] - I[:-1, :]
        nablaS[:-1, :] = I[:-1, :] - I[1:, :]
        nablaE[:, :-1] = I[:, :-1] - I[:, 1:]
        nablaW[:, 1:] = I[:, 1:] - I[:, :-1]
        if option == 1:
            cN = np.exp(-(nablaN / k) ** 2)
            cS = np.exp(-(nablaS / k) ** 2)
            cE = np.exp(-(nablaE / k) ** 2)
            cW = np.exp(-(nablaW / k) ** 2)
        else:
            cN = 1.0 / (1.0 + (nablaN / k) ** 2)
            cS = 1.0 / (1.0 + (nablaS / k) ** 2)
            cE = 1.0 / (1.0 + (nablaE / k) ** 2)
            cW = 1.0 / (1.0 + (nablaW / k) ** 2)
        I = I + gamma * (cN * nablaN + cS * nablaS + cE * nablaE + cW * nablaW)
        I = np.clip(I, 0, 1)
    return I


def apply_noise_reduction(im01: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    method = cfg.get("method", "none").lower()
    if method == "none":
        return im01
    if method == "gaussian":
        sigma = float(cfg.get("sigma", 1.0))
        return gaussian(im01, sigma=sigma, preserve_range=True)
    if method == "anisotropic":
        niter = int(cfg.get("niter", 15))
        k = float(cfg.get("k", 0.05))
        gamma = float(cfg.get("gamma", 0.2))
        option = int(cfg.get("option", 1))
        return anisotropic_diffusion(im01, niter=niter, k=k, gamma=gamma, option=option)
    return im01


# -------------------------------
# Contrast enhancement
# -------------------------------
def apply_contrast_enhancement(im01: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    method = cfg.get("method", "none").lower()
    if method == "none":
        return im01
    if method == "hist_eq":
        return equalize_hist(im01)
    if method == "clahe":
        clip = float(cfg.get("clip_limit", 0.01))
        nbins = int(cfg.get("nbins", 256))
        return equalize_adapthist(im01, clip_limit=clip, nbins=nbins)
    return im01


# -------------------------------
# Metrics
# -------------------------------
def cnr_metric(im01: np.ndarray, roi: np.ndarray, bg: Optional[np.ndarray] = None) -> Dict[str, float]:
    roi = roi.astype(bool)
    if roi.sum() == 0:
        return {"CNR": float("nan"), "mean_roi": float("nan"), "mean_bg": float("nan"), "std_roi": float("nan"), "std_bg": float("nan")}
    if bg is None:
        bg = ~roi
    bg = bg.astype(bool)
    if bg.sum() == 0:
        return {"CNR": float("nan"), "mean_roi": float("nan"), "mean_bg": float("nan"), "std_roi": float("nan"), "std_bg": float("nan")}
    m1, s1 = im01[roi].mean(), im01[roi].std() + 1e-8
    m0, s0 = im01[bg].mean(), im01[bg].std() + 1e-8
    cnr = (m1 - m0) / np.sqrt(s1**2 + s0**2)
    return {"CNR": float(cnr), "mean_roi": float(m1), "mean_bg": float(m0), "std_roi": float(s1), "std_bg": float(s0)}


def edge_strength(im01: np.ndarray, roi: np.ndarray) -> float:
    b = find_boundaries(roi, mode="outer")
    if b.sum() == 0:
        return float("nan")
    gx = np.zeros_like(im01); gy = np.zeros_like(im01)
    gx[:, 1:-1] = (im01[:, 2:] - im01[:, :-2]) / 2.0
    gy[1:-1, :] = (im01[2:, :] - im01[:-2, :]) / 2.0
    gradmag = np.hypot(gx, gy)
    return float(gradmag[b].mean())


# -------------------------------
# Threshold preview 
# -------------------------------
def threshold_preview(im01: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    mode = cfg.get("mode", "otsu").lower()
    if mode == "otsu":
        t = threshold_otsu(im01)
        bw = im01 > t
    elif mode == "adaptive_mean":
        win_sigma = float(cfg.get("sigma", 3.0))
        local_mean = gaussian(im01, sigma=win_sigma)
        bw = im01 > (local_mean + float(cfg.get("offset", 0.0)))
    else:
        t = float(cfg.get("fixed_t", 0.5))
        bw = im01 > t
    bw = binary_opening(bw, disk(1))
    bw = remove_small_objects(bw, min_size=int(cfg.get("min_size", 10)))
    return bw.astype(bool)


# -------------------------------
# Visualization: composite panels
# -------------------------------
def make_composite(
    raw01: np.ndarray,
    proc01: np.ndarray,
    mask: Optional[np.ndarray],
    out_path: Path,
    title: str,
    add_hist: bool = True,
    zoom_from_mask: bool = True,
    cmap: str = "gray",
    hist_bins: int = 64
):
    ensure_dir(out_path.parent)
    plt.figure(figsize=(12, 8))
    rows, cols = (2, 3) if add_hist else (2, 2)

    # Raw
    plt.subplot(rows, cols, 1)
    plt.imshow(raw01, cmap=cmap)
    if mask is not None:
        b = find_boundaries(mask, mode="outer")
        plt.contour(b, colors="y", linewidths=0.5)
    plt.title("Raw (normalized)")
    plt.axis("off")

    # Processed
    plt.subplot(rows, cols, 2)
    plt.imshow(proc01, cmap=cmap)
    if mask is not None:
        b = find_boundaries(mask, mode="outer")
        plt.contour(b, colors="c", linewidths=0.5)
    plt.title("Processed")
    plt.axis("off")

    # Absolute difference
    plt.subplot(rows, cols, 3)
    diff = np.abs(proc01 - raw01)
    plt.imshow(diff, cmap="magma")
    plt.title("|Processed - Raw|")
    plt.axis("off")

    # Zoom crop around mask bbox or center
    plt.subplot(rows, cols, 4)
    h, w = raw01.shape
    if zoom_from_mask and mask is not None and mask.sum() > 0:
        ys, xs = np.where(mask)
        y0, y1 = max(ys.min() - 10, 0), min(ys.max() + 10, h)
        x0, x1 = max(xs.min() - 10, 0), min(xs.max() + 10, w)
    else:
        ch, cw = min(128, h), min(128, w)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        y1, x1 = y0 + ch, x0 + cw
    raw_zoom = raw01[y0:y1, x0:x1]
    proc_zoom = proc01[y0:y1, x0:x1]
    zoom = np.concatenate([raw_zoom, proc_zoom], axis=1)
    plt.imshow(zoom, cmap=cmap)
    plt.title("Zoom (Raw | Processed)")
    plt.axis("off")

    # Histograms
    if add_hist:
        plt.subplot(rows, cols, 5)
        plt.hist(raw01.ravel(), bins=hist_bins, alpha=0.5, label="Raw")
        plt.hist(proc01.ravel(), bins=hist_bins, alpha=0.5, label="Processed")
        plt.legend()
        plt.title("Intensity histograms")

        plt.subplot(rows, cols, 6)
        diff_zoom = np.abs(proc_zoom - raw_zoom)
        plt.imshow(diff_zoom, cmap="inferno")
        plt.title("|Processed - Raw| (Zoom)")
        plt.axis("off")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------------
# Variant generation
# -------------------------------
def generate_variants(raw: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Return dict of variants in [0,1] for display/comparison.
    Variants are specified in cfg['ablation_variants'].
    """
    base_norm_cfg = cfg["pipeline"]["normalization"]
    noise_cfg = cfg["pipeline"]["noise_reduction"]
    contrast_cfg = cfg["pipeline"]["contrast_enhancement"]
    variants_cfg = cfg.get("ablation_variants", ["raw", "norm_only", "adiff", "adiff+clahe", "gauss+clahe"])

    out = {}

    # raw
    if "raw" in variants_cfg:
        out["raw"] = normalize_image(raw, {"method": "minmax", "percentiles": [0.0, 100.0]})

    # norm_only
    if "norm_only" in variants_cfg:
        out["norm_only"] = normalize_image(raw, base_norm_cfg)

    # adiff
    if "adiff" in variants_cfg:
        im = normalize_image(raw, base_norm_cfg)
        im = apply_noise_reduction(im, {"method": "anisotropic", **cfg["pipeline"]["noise_reduction"]})
        out["adiff"] = normalize_image(im, {"method": "minmax", "percentiles": [0.0, 100.0]})

    # adiff+clahe
    if "adiff+clahe" in variants_cfg:
        im = normalize_image(raw, base_norm_cfg)
        im = apply_noise_reduction(im, {"method": "anisotropic", **cfg["pipeline"]["noise_reduction"]})
        im = apply_contrast_enhancement(im, {"method": "clahe", **cfg["pipeline"]["contrast_enhancement"]})
        out["adiff+clahe"] = normalize_image(im, {"method": "minmax", "percentiles": [0.0, 100.0]})

    # gauss+clahe
    if "gauss+clahe" in variants_cfg:
        im = normalize_image(raw, base_norm_cfg)
        im = apply_noise_reduction(im, {"method": "gaussian", **cfg["pipeline"]["noise_reduction"]})
        im = apply_contrast_enhancement(im, {"method": "clahe", **cfg["pipeline"]["contrast_enhancement"]})
        out["gauss+clahe"] = normalize_image(im, {"method": "minmax", "percentiles": [0.0, 100.0]})

    return out


# -------------------------------
# Main processing 
# -------------------------------
def process_single_image(
    img_path: Path,
    cfg: Dict[str, Any],
    patient_out_dir: Path,
    timestamp: str,
    patient_metrics: List[Dict[str, Any]],
    visualize_this_image: bool
):
    qa_cfg = cfg.get("qa", {})
    hist_bins = int(qa_cfg.get("hist_bins", 64))
    pipeline = cfg["pipeline"]

    # load image and optional mask
    raw = load_image(img_path, force_gray=bool(cfg["general"].get("to_grayscale", True)))
    mask_path = map_mask_path(img_path, cfg.get("masks", {}))
    mask = load_mask(mask_path) if mask_path else None

    # normalization
    raw_norm = normalize_image(raw, pipeline["normalization"])

    # noise reduction + contrast enhancement
    denoised = apply_noise_reduction(raw_norm, pipeline["noise_reduction"])
    enhanced = apply_contrast_enhancement(denoised, pipeline["contrast_enhancement"])

    # final minmax to [0,1]
    if pipeline.get("final_minmax", True):
        processed01 = normalize_image(enhanced, {"method": "minmax", "percentiles": [0.0, 100.0]})
    else:
        processed01 = enhanced

    # directories
    processed_dir = patient_out_dir / "processed"
    composite_dir = patient_out_dir / "composites"
    ablation_dir = patient_out_dir / "ablation_images"
    metrics_dir = patient_out_dir / "metrics"
    for d in [processed_dir, composite_dir, ablation_dir, metrics_dir]:
        ensure_dir(d)

    # save processed image (preserve filename)
    out_img_path = processed_dir / img_path.name
    cv2.imwrite(str(out_img_path), to_uint8(processed01))

    # composite for main pipeline
    if visualize_this_image and qa_cfg.get("generate_composites", True):
        cpath = composite_dir / f"{img_path.stem}_composite.png"
        make_composite(
            raw01=raw_norm,
            proc01=processed01,
            mask=mask,
            out_path=cpath,
            title=f"{img_path.name} | pipeline",
            add_hist=qa_cfg.get("add_hist", True),
            zoom_from_mask=qa_cfg.get("zoom_from_mask", True),
            cmap=qa_cfg.get("cmap", "gray"),
            hist_bins=hist_bins
        )

        # threshold preview
        tp_cfg = qa_cfg.get("threshold_preview", {})
        if tp_cfg.get("enabled", False):
            bw = threshold_preview(processed01, tp_cfg)
            tp_path = composite_dir / f"{img_path.stem}_thr_preview.png"
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(processed01, cmap="gray"); plt.title("Processed"); plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(bw, cmap="gray"); plt.title(f"Threshold: {tp_cfg.get('mode','otsu')}"); plt.axis("off")
            plt.tight_layout()
            plt.savefig(tp_path, dpi=200); plt.close()

    # metrics for main pipeline
    row = {
        "image": img_path.name,
        "patient_id": patient_out_dir.parent.name,
        "timestamp": timestamp,
        "variant": "pipeline",
        "out_image": out_img_path.name,
    }
    if mask is not None:
        m = cnr_metric(processed01, mask)
        row.update(m)
        row["edge_strength"] = edge_strength(processed01, mask)
    else:
        row.update({"CNR": float("nan"), "mean_roi": float("nan"), "mean_bg": float("nan"),
                    "std_roi": float("nan"), "std_bg": float("nan"), "edge_strength": float("nan")})
    patient_metrics.append(row)

    # ablations: images, composites, and metrics per variant
    if cfg.get("ablation", {}).get("enabled", True) and visualize_this_image:
        variants = generate_variants(raw, cfg)
        for name, im01 in variants.items():
            # save variant image
            ab_img_path = ablation_dir / f"{img_path.stem}_{name}{img_path.suffix}"
            cv2.imwrite(str(ab_img_path), to_uint8(im01))

            # composite
            v_comp = composite_dir / f"{img_path.stem}_ablation_{name}.png"
            make_composite(
                raw01=normalize_image(raw, {"method": "minmax", "percentiles": [0.0, 100.0]}),
                proc01=im01,
                mask=mask,
                out_path=v_comp,
                title=f"{img_path.name} | {name}",
                add_hist=True,
                zoom_from_mask=True,
                cmap=qa_cfg.get("cmap", "gray"),
                hist_bins=hist_bins
            )

            # metrics for variant
            vrow = {
                "image": img_path.name,
                "patient_id": patient_out_dir.parent.name,
                "timestamp": timestamp,
                "variant": name,
                "out_image": ab_img_path.name
            }
            if mask is not None:
                vm = cnr_metric(im01, mask)
                vrow.update(vm)
                vrow["edge_strength"] = edge_strength(im01, mask)
            else:
                vrow.update({"CNR": float("nan"), "mean_roi": float("nan"), "mean_bg": float("nan"),
                             "std_roi": float("nan"), "std_bg": float("nan"), "edge_strength": float("nan")})
            patient_metrics.append(vrow)


def write_metrics_csv(metrics: List[Dict[str, Any]], csv_path: Path):
    if not metrics:
        return
    ensure_dir(csv_path.parent)
    keys = sorted(metrics[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in metrics:
            w.writerow(r)


def snapshot_config(cfg: Dict[str, Any], out_dir: Path):
    ensure_dir(out_dir)
    with open(out_dir / "config_snapshot.yml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="TRUS preprocessing pipeline (YAML-driven)")
    ap.add_argument("--config", required=True, help="Path to config-imgprepros.yml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    cfg = load_yaml(cfg_path)
    general = cfg["general"]

    input_root = Path(general["input_dir"])
    if not input_root.exists():
        print(f"[ERROR] Input directory not found: {input_root}")
        sys.exit(1)
    preprocess_root = Path(general["output_root"])
    ts = general.get("timestamp", "auto")
    if ts == "auto" or ts is None:
        ts = timestamp_str()
    out_root = preprocess_root / ts
    ensure_dir(out_root)
    
    out_root = Path(general["output_root"])
    ensure_dir(out_root)

    ts = general.get("timestamp", "auto")
    if ts == "auto" or ts is None:
        ts = timestamp_str()

    allowed_exts = [e.lower() for e in general.get("allowed_img_exts", [".png"])]

    # Aggregate / batch output
    batch_dir = out_root / f"batch_{ts}"
    ensure_dir(batch_dir)
    snapshot_config(cfg, batch_dir)

    all_metrics: List[Dict[str, Any]] = []

    # Enumerate patients
    patient_dirs = list_patient_dirs(input_root, allowed_exts)
    if not patient_dirs:
        print(f"[WARN] No patient directories with images found in {input_root}")
        sys.exit(0)

    print(f"[INFO] Processing {len(patient_dirs)} patient directories. Outputs -> {out_root}")

    for pdir in patient_dirs:
        pid = pdir.name
        print(f"[INFO] Patient: {pid}")

        imgs = find_images_in_dir(pdir, allowed_exts)
        if not imgs:
            continue
        highest = select_highest_slice(imgs)
        visualize_all = bool(cfg.get("qa", {}).get("visualize_all_images", False))

        patient_out_dir = out_root / pid / ts
        ensure_dir(patient_out_dir)

        patient_metrics: List[Dict[str, Any]] = []

        for ipath in imgs:
            visualize_this_image = visualize_all or (ipath == highest)
            process_single_image(
                img_path=ipath,
                cfg=cfg,
                patient_out_dir=patient_out_dir,
                timestamp=ts,
                patient_metrics=patient_metrics,
                visualize_this_image=visualize_this_image
            )

        # write per-patient metrics
        write_metrics_csv(patient_metrics, patient_out_dir / "metrics" / "metrics_all.csv")
        all_metrics.extend(patient_metrics)

    # batch summary and global metrics
    with open(batch_dir / "summary.json", "w") as f:
        json.dump({"num_patients": len(patient_dirs), "timestamp": ts}, f, indent=2)

    write_metrics_csv(all_metrics, batch_dir / "metrics_all.csv")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
