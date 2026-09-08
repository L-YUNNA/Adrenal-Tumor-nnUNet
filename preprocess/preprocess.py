"""
================================================================================
Adrenal CT preprocessing pipeline (for the nnU-Net adrenal gland / tumor
segmentation model)
================================================================================

WHAT THIS SCRIPT DOES
--------------------------------------------------------------------------------
It converts raw abdominal CT volumes into exactly the same image representation
that was used to train the released nnU-Net model. Running it is what makes an
external dataset comparable to the training data.

The pipeline consists of three operations:

  1. Voxel normalization : resample to isotropic 1 x 1 x 1 mm spacing (linear
                           interpolation) and apply a soft-tissue CT window
                           (WW = 400, WL = 40 -> clipped to [-160, 240] HU).
  2. In-plane resize     : slice-wise center crop / pad to `target_size` x
                           `target_size` pixels (default 192).
  3. Depth crop          : keep `target_depth` axial slices (default 128)
                           centered on the adrenal gland.
                           The center slice is found automatically by a
                           pre-trained 2D ResNet-50 classifier that predicts,
                           for every axial slice, whether an adrenal gland is
                           present ("VOI extraction").

The final volume is therefore (128, 192, 192) in numpy / SimpleITK axis order,
i.e. (depth/z, height/y, width/x), at 1 mm isotropic spacing.

--------------------------------------------------------------------------------
DIRECTORY LAYOUT (this is the only thing an external user has to set up)
--------------------------------------------------------------------------------
Set `base_dir` (or the ADRENAL_BASE_DIR environment variable) to any folder you
like, then place your CT volumes like this:

    <BASE_DIR>/
      data/
        <INSTITUTION>/                 e.g. "AMC", "MyHospital"
          <DISEASE>/                   e.g. "Adenoma", "ACC", "NF"
            precontrast/
              <CaseID>.nii.gz          one file per case
            postcontrast/
              <CaseID>.nii.gz

    Both `<CaseID>.nii(.gz)` and `<CaseID>/<CaseID>.nii(.gz)` are accepted.
    <CaseID> can be any string; it is carried through the whole pipeline and
    is used as the nnU-Net case identifier.

Everything the script writes goes under:

    <BASE_DIR>/
      preprocessed/
        <INSTITUTION>/<DISEASE>/
          <VERSION>/                   FINAL preprocessed volumes  <- use these
          <VERSION>_scaled/            output of run_voxnorm_only() (optional)
          _intermediate/               intermediate + log files (safe to delete)
            resize_<VERSION>/            after step 1+2, before depth crop
            voi_npy_<VERSION>/           2D slices fed to the ResNet classifier
            voi_npy_<VERSION>_list.csv
            infer_result_<VERSION>.csv   per-slice adrenal probability
            voi_info_<VERSION>.csv       per-case adrenal center slice index
            *_history.csv                per-stage processing log
        nnunet_input/<INSTITUTION>_<DISEASE>_<VERSION>/
                                       final volumes renamed to <CaseID>_0000.nii.gz
                                       (this is the folder you pass to
                                        `nnUNetv2_predict -i`)

No path in this file points into the original project any more. If you never
touch `base_dir`, everything is written next to this repository.

NOTE ON IMAGE GEOMETRY
--------------------------------------------------------------------------------
Output volumes carry 1 mm isotropic spacing with a zero origin and an identity
direction; the original scan's origin/direction are not preserved. Predictions
inherit the geometry of whatever you feed nnU-Net, so this is self-consistent,
but a prediction cannot be overlaid on the *original* CT without re-registration.

--------------------------------------------------------------------------------
CONTRAST PHASE ("version")
--------------------------------------------------------------------------------
    "precontrast"  : use the pre-contrast scan only
    "postcontrast" : use the post-contrast scan only
    "total"        : one scan per case - post-contrast if available, else pre

--------------------------------------------------------------------------------
TYPICAL USE (see notebooks/adrenal_pipeline.ipynb for the annotated version)
--------------------------------------------------------------------------------
    from preprocess import *

    cfg = build_pre_config(institution="MyHospital",
                           disease="Adenoma",   # any cohort label: NF, ACC, CS, ...
                           version="precontrast",
                           base_dir="<YOUR_PROJECT_ROOT>")

    run_stage1(cfg)             # voxel norm + in-plane resize  -> _intermediate
    make_voi_npy(cfg)           # 3D -> per-slice 2D .npy
    run_inference(cfg)          # ResNet-50: is there an adrenal gland on this slice?
    search_voi_index(cfg)       # longest positive run -> center slice per case
    run_stage2_depthcrop(cfg)   # depth crop around that center -> FINAL volumes
    export_for_nnunet(cfg)      # rename to <CaseID>_0000.nii.gz for nnU-Net
    qc_show(cfg)                # visual check of the mid slice

    # or, all of the above in one call:
    run_all(cfg)

    # If you already know the adrenal center slice for each case, you can skip
    # the classifier entirely and supply your own CSV
    # (columns: fname, voi_center_idx):
    #   run_stage2_depthcrop(cfg, voi_csv="<YOUR_PATH>/voi_info.csv")

Command line equivalent:

    python preprocess.py --base-dir <YOUR_PROJECT_ROOT> \
                         --institution MyHospital --disease Adenoma \
                         --version precontrast --all
================================================================================
"""

import os
import re
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import itk
import SimpleITK as sitk
from torchvision import transforms


# ==============================================================================
# DEFAULT PATHS
#   Nothing here is tied to a specific machine. Priority:
#     explicit argument  >  ADRENAL_BASE_DIR env var  >  repository root
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]

# Weights of the 2D ResNet-50 slice classifier used to locate the adrenal gland.
# It ships with this repository; override with `checkpoint_path=` if you moved it.
DEFAULT_CHECKPOINT = REPO_ROOT / "preprocess" / "VOI_extraction" / "Best.pth"


def default_base_dir():
    """Root that holds ./data and ./preprocessed. Override with ADRENAL_BASE_DIR."""
    return os.environ.get("ADRENAL_BASE_DIR", str(REPO_ROOT))


# ==============================================================================
# CONFIG
# ==============================================================================
# Default preprocessing settings. These reproduce the distribution the released
# nnU-Net segmentation model was trained on, and they are applied to EVERY cohort.
#
# `disease` below is nothing but a folder name - use any cohort label you like
# ("Adenoma", "NF", "ACC", "CS", ...) and it will be passed through unchanged.
# To use different settings for a particular cohort, pass them explicitly:
#
#     cfg = build_pre_config(..., disease="ACC", target_size=320, do_depth_crop=False)
#
DEFAULT_SETTINGS = {"target_size": 192, "do_depth_crop": True, "target_depth": 128}

# CHOOSING target_size / target_depth
# -----------------------------------
# The 192 x 192 x 128 field of view was chosen for adrenal adenomas, which are
# small. Large lesions - adrenocortical carcinoma in particular - can easily
# exceed it, and anything outside the crop is simply thrown away before the model
# ever sees it. Always check qc_show() output: if the lesion touches the border of
# the cropped volume, enlarge the field of view, e.g.
#
#     build_pre_config(..., target_size=320, target_depth=192)
#
# or skip cropping altogether with run_voxnorm_only(), which keeps the full CT.
# This costs inference time but no accuracy: nnU-Net uses sliding-window
# inference and accepts any input size (see README section 4).


def build_pre_config(
    institution="MyInstitution",
    disease="Adenoma",               # free-form cohort label = folder name under data/<INSTITUTION>/
    version="total",                 # "precontrast" | "postcontrast" | "total"
    base_dir=None,                   # root holding ./data and ./preprocessed
    data_root=None,                  # input root; defaults to <base_dir>/data
    preprocessed_root=None,          # output root; defaults to <base_dir>/preprocessed
    ww=400, wl=40,                   # CT window width / level (soft tissue)
    target_size=None,                # in-plane size; None -> disease default
    target_depth=None,               # number of axial slices kept
    do_depth_crop=None,
    # --- adrenal (VOI) localization ---
    checkpoint_path=None,            # ResNet-50 weights; None -> DEFAULT_CHECKPOINT
    infer_batch_size=128,
    infer_gpu="0",                   # value for CUDA_VISIBLE_DEVICES; None -> leave as is
    nnunet_input_root=None,          # where export_for_nnunet() writes
    verbose=True,                    # print the effective settings once
):
    """Build the dict that every other function in this file consumes.

    All paths are derived from `base_dir`, so an external user only has to point
    this at their own workspace.

    `institution` and `disease` are only folder names; any label works.
    """
    assert version in ("precontrast", "postcontrast", "total"), \
        f"version must be precontrast/postcontrast/total, got {version!r}"

    d = DEFAULT_SETTINGS
    base_dir = base_dir or default_base_dir()
    data_root = data_root or os.path.join(base_dir, "data")
    preprocessed_root = preprocessed_root or os.path.join(base_dir, "preprocessed")
    case_root = os.path.join(preprocessed_root, institution, disease)
    inter = os.path.join(case_root, "_intermediate")
    nnunet_input_root = nnunet_input_root or os.path.join(preprocessed_root, "nnunet_input")

    cfg = {
        "institution": institution,
        "disease": disease,
        "version": version,
        "WW": ww, "WL": wl,
        "target_size": target_size if target_size is not None else d["target_size"],
        "target_depth": target_depth if target_depth is not None else d["target_depth"],
        "do_depth_crop": do_depth_crop if do_depth_crop is not None else d["do_depth_crop"],

        # roots
        "base_dir": base_dir,
        "data_root": data_root,
        "preprocessed_root": preprocessed_root,
        "case_root": case_root,
        "inter_dir": inter,

        # input phase folders
        "pre_dir":  os.path.join(data_root, institution, disease, "precontrast"),
        "post_dir": os.path.join(data_root, institution, disease, "postcontrast"),

        # final output
        "out_dir": os.path.join(case_root, version),

        # intermediate products (voxel_norm + resize result, before depth crop)
        "stage_resize_dir": os.path.join(inter, f"resize_{version}"),
        # 2D slices used as ResNet input
        "voi_npy_dir": os.path.join(inter, f"voi_npy_{version}"),
        "voi_npy_list_csv": os.path.join(inter, f"voi_npy_{version}_list.csv"),
        # classifier / search outputs
        "infer_result_csv": os.path.join(inter, f"infer_result_{version}.csv"),
        "voi_info_csv": os.path.join(inter, f"voi_info_{version}.csv"),

        # nnU-Net ready folder (files renamed to <CaseID>_0000.nii.gz)
        "nnunet_input_dir": os.path.join(
            nnunet_input_root, f"{institution}_{disease}_{version}"),

        # inference settings
        "checkpoint_path": str(checkpoint_path or DEFAULT_CHECKPOINT),
        "infer_batch_size": infer_batch_size,
        "infer_gpu": infer_gpu,
    }

    if verbose:
        print(f"[config] {institution}/{disease}/{version} | "
              f"target_size={cfg['target_size']}, target_depth={cfg['target_depth']}, "
              f"depth_crop={cfg['do_depth_crop']}")
    return cfg


def print_config(cfg):
    """Print the resolved paths so you can sanity-check them before running."""
    keys = ["base_dir", "pre_dir", "post_dir", "out_dir", "stage_resize_dir",
            "voi_info_csv", "nnunet_input_dir", "checkpoint_path",
            "target_size", "target_depth", "do_depth_crop", "WW", "WL"]
    width = max(len(k) for k in keys)
    print("-" * 78)
    for k in keys:
        print(f"  {k:<{width}} : {cfg[k]}")
    print("-" * 78)
    for k, label in [("pre_dir", "precontrast input"), ("post_dir", "postcontrast input"),
                     ("checkpoint_path", "ResNet checkpoint")]:
        if not os.path.exists(cfg[k]):
            print(f"  [WARN] {label} does not exist: {cfg[k]}")


# ==============================================================================
# COMMON IO
# ==============================================================================
def load_nii(path):
    """Read a NIfTI file. Returns (sitk image, numpy array in (z, y, x) order)."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    return img, arr


def _ensure(d):
    os.makedirs(d, exist_ok=True)
    return d


def fname_from_path(path):
    """File path -> case identifier (strips folders and the .nii/.nii.gz suffix)."""
    base = os.path.basename(path)
    base = re.sub(r"\.nii(\.gz)?$", "", base)
    return base


_SLICE_RE = re.compile(r"^(?P<case>.+)_slice_index(?P<idx>\d+)\.npy$")


def _parse_slice_name(npy_name):
    """'<CaseID>_slice_index007.npy' -> ('<CaseID>', 7).

    Case IDs of any length/format are supported (the original code assumed a
    fixed 8-character ID, which breaks on other naming conventions).
    """
    m = _SLICE_RE.match(os.path.basename(npy_name))
    if not m:
        raise ValueError(f"unexpected slice file name: {npy_name}")
    return m.group("case"), int(m.group("idx"))


def list_phase_cases(phase_dir):
    """Scan one phase folder and return {CaseID: file path}.

    Accepted layouts:
      A) <phase>/<CaseID>.nii(.gz)
      B) <phase>/<CaseID>/<CaseID>.nii(.gz)
    """
    out = {}
    if not os.path.isdir(phase_dir):
        return out
    for p in glob(os.path.join(phase_dir, "*.nii")) + glob(os.path.join(phase_dir, "*.nii.gz")):
        out[fname_from_path(p)] = p
    for sub in sorted(os.listdir(phase_dir)):
        subdir = os.path.join(phase_dir, sub)
        if os.path.isdir(subdir):
            cands = glob(os.path.join(subdir, "*.nii")) + glob(os.path.join(subdir, "*.nii.gz"))
            if cands:
                out.setdefault(sub, sorted(cands)[0])
    return out


def resolve_inputs(cfg):
    """Pick exactly one input volume per case according to cfg['version'].

    Returns a list of dicts: {Fname, src_path, src_phase}.
    """
    pre = list_phase_cases(cfg["pre_dir"])
    post = list_phase_cases(cfg["post_dir"])
    version = cfg["version"]

    rows = []
    if version == "precontrast":
        for f, p in pre.items():
            rows.append({"Fname": f, "src_path": p, "src_phase": "precontrast"})
    elif version == "postcontrast":
        for f, p in post.items():
            rows.append({"Fname": f, "src_path": p, "src_phase": "postcontrast"})
    else:  # total: prefer post-contrast, fall back to pre-contrast
        for f in sorted(set(pre) | set(post)):
            if f in post:
                rows.append({"Fname": f, "src_path": post[f], "src_phase": "postcontrast"})
            else:
                rows.append({"Fname": f, "src_path": pre[f], "src_phase": "precontrast"})
    rows.sort(key=lambda r: r["Fname"])

    if not rows:
        print(f"[WARN] no input volumes found. Checked:\n"
              f"       {cfg['pre_dir']}\n       {cfg['post_dir']}")
    return rows


# ==============================================================================
# 1) Voxel normalization: 1 mm isotropic resampling + CT windowing
# ==============================================================================
def window_image(image, window_center, window_width):
    """Clip HU values to the [WL - WW/2, WL + WW/2] window."""
    img_min = window_center - (window_width // 2)
    img_max = window_center + (window_width // 2)
    out = image.copy()
    out[out < img_min] = img_min
    out[out > img_max] = img_max
    return out


def voxel_resizing(input_file_name, Interpol=None):
    """Resample a volume to 1 x 1 x 1 mm isotropic spacing.

    Interpol: 'Near' (nearest neighbour, for masks) | 'Linear' (for CT) | None
    """
    PixelType = itk.SS  # signed short, sufficient for CT HU values
    input_image = itk.imread(input_file_name, PixelType)
    input_size = itk.size(input_image)
    input_spacing = itk.spacing(input_image)
    input_origin = itk.origin(input_image)
    input_direction = input_image.GetDirection()
    Dimension = input_image.GetImageDimension()

    # target spacing is 1 mm in every direction
    new_spacing = np.array(input_spacing) / np.array([1, 1, 1])
    output_size = [int(input_size[d] * new_spacing[d]) for d in range(Dimension)]
    output_spacing = [input_spacing[d] / new_spacing[d] for d in range(Dimension)]
    output_origin = [input_origin[d] + 0.5 * (output_spacing[d] - input_spacing[d])
                     for d in range(Dimension)]

    scale_transform = itk.ScaleTransform[itk.D, Dimension].New()
    if Interpol == "Near":
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(input_image)
    elif Interpol == "Linear":
        interpolator = itk.LinearInterpolateImageFunction.New(input_image)
    else:
        interpolator = None

    resampled = itk.resample_image_filter(
        input_image,
        transform=scale_transform,
        interpolator=interpolator,
        size=output_size,
        output_spacing=output_spacing,
        output_origin=output_origin,
        output_direction=input_direction,
    )
    return resampled


def voxnorm_to_array(src_path, ttype, ww, wl):
    """Isotropic resampling (+ windowing for CT) -> numpy array in (z, y, x).

    The `[::-1]` flip along z reproduces the axis convention used when the
    training data was built; do not remove it.
    """
    if ttype == "mask":
        voxel_resized = voxel_resizing(src_path, Interpol="Near")
        arr = np.asarray(voxel_resized)
        arr = np.ascontiguousarray(arr[::-1, :, :])
        return arr
    else:  # CT
        voxel_resized = voxel_resizing(src_path, Interpol="Linear")
        arr = np.asarray(voxel_resized)
        arr = window_image(arr, wl, ww)               # (array, window_center=WL, window_width=WW)
        arr = np.ascontiguousarray(arr[::-1, :, :])
        return arr


# ==============================================================================
# 2) In-plane resize: slice-wise center crop, pad first if the slice is too small
# ==============================================================================
def resize_volume(nii_array, target_size):
    """Center crop (or pad then crop) every axial slice to target_size x target_size.

    Padding uses the volume minimum (i.e. the lower window bound) so that padded
    background is indistinguishable from air outside the body.
    """
    min_val = float(nii_array.min())
    fill_val = int(min_val) if float(min_val).is_integer() else min_val
    D, H, W = nii_array.shape
    need_pad = target_size > max(H, W)

    if need_pad:
        transform = transforms.Compose([
            transforms.Pad(padding=target_size, fill=fill_val),
            transforms.CenterCrop(target_size),
        ])
    else:
        transform = transforms.CenterCrop(target_size)

    out = []
    for i in range(D):
        sl = Image.fromarray(nii_array[i, :, :])
        sl = transform(sl)
        out.append(np.array(sl))
    stacked = np.stack(out, axis=0)
    # PIL widens int16 to int32; cast back so the saved NIfTI keeps the input
    # dtype. Voxel values are unchanged.
    if np.can_cast(stacked.dtype, nii_array.dtype, casting="unsafe"):
        stacked = stacked.astype(nii_array.dtype, copy=False)
    return stacked


# ==============================================================================
# 3) Depth crop: keep `target_depth` slices centered on the adrenal gland
# ==============================================================================
def depth_crop_volume(nii_array, voi_center_idx, target_depth, target_size,
                      pad_value=0):
    """Crop `target_depth` axial slices around `voi_center_idx`.

    The window is shifted back inside the volume if it would run past the last
    slice; if the volume is shorter than target_depth it is zero padded
    symmetrically (pad_value=0 reproduces the training data exactly).
    """
    ad_middle = int(voi_center_idx)
    start = max(ad_middle - (target_depth // 2), 0)
    end = start + target_depth
    if end > nii_array.shape[0]:
        end = nii_array.shape[0]
        start = max(end - target_depth, 0)

    img = nii_array[start:end, :, :]
    if img.shape[0] < target_depth:
        pad = target_depth - img.shape[0]
        pb, pa = pad // 2, pad - pad // 2
        img = np.pad(img, ((pb, pa), (0, 0), (0, 0)),
                     mode="constant", constant_values=pad_value)

    assert img.shape == (target_depth, target_size, target_size), \
        f"depth crop shape mismatch: {img.shape} != {(target_depth, target_size, target_size)}"
    return img


# ==============================================================================
# Stage 1 : voxel normalization + in-plane resize (done in memory, one pass)
# ==============================================================================
def run_stage1(cfg, ttype="CT", save_intermediate=None, limit=None):
    """Resample to 1 mm iso, apply the CT window, center crop to target_size.

    - depth crop enabled  -> writes to cfg['stage_resize_dir'] (intermediate)
    - depth crop disabled -> writes straight to cfg['out_dir'] (final)
    Set save_intermediate=True/False to force either destination.

    Returns a DataFrame with one row per case.
    """
    inputs = resolve_inputs(cfg)
    if limit:
        inputs = inputs[:limit]

    will_depth_crop = cfg["do_depth_crop"]
    if save_intermediate is None:
        dst_dir = cfg["stage_resize_dir"] if will_depth_crop else cfg["out_dir"]
    else:
        dst_dir = cfg["stage_resize_dir"] if save_intermediate else cfg["out_dir"]
    _ensure(dst_dir)
    _ensure(cfg["inter_dir"])

    print(f"[Stage1] {cfg['institution']}/{cfg['disease']}/{cfg['version']} "
          f"| cases={len(inputs)} | target_size={cfg['target_size']} "
          f"| depth_crop={will_depth_crop} -> {dst_dir}")

    rec = []
    for r in tqdm(inputs):
        fname, src = r["Fname"], r["src_path"]
        try:
            arr = voxnorm_to_array(src, ttype, cfg["WW"], cfg["WL"])
            resized = resize_volume(arr, cfg["target_size"])
            out_path = os.path.join(dst_dir, f"{fname}.nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(resized), out_path)
            rec.append({**r, "stage1_out": out_path, "shape": resized.shape,
                        "is_final": (dst_dir == cfg["out_dir"]), "status": "OK", "error": None})
        except Exception as e:
            rec.append({**r, "stage1_out": None, "shape": None,
                        "is_final": False, "status": "FAILED", "error": str(e)})
            print(f"  [FAIL] {fname}: {e}")

    df = pd.DataFrame(rec)
    df.to_csv(os.path.join(cfg["inter_dir"], f"stage1_{cfg['version']}_history.csv"),
              index=False)
    n_ok = int((df["status"] == "OK").sum()) if len(df) else 0
    print(f"  done OK={n_ok}/{len(df)}")
    return df


# ==============================================================================
# Stage 1-A : voxel normalization only (no resize, no depth crop)
#   Output: <preprocessed>/<INSTITUTION>/<DISEASE>/<VERSION>_scaled/<CaseID>.nii.gz
# ==============================================================================
def get_scaled_dir(cfg):
    """Path of the '<version>_scaled' folder, next to the final output folder."""
    return os.path.join(cfg["case_root"], f"{cfg['version']}_scaled")


def run_voxnorm_only(cfg, ttype="CT", out_dir=None, limit=None,
                     set_spacing=True, overwrite=True):
    """Only isotropic resampling + CT windowing; the field of view is untouched.

    Useful when you want to run nnU-Net on the full CT instead of the cropped
    128 x 192 x 192 volume (the network accepts any input size - see README).
    Output shape therefore differs from case to case.
    """
    inputs = resolve_inputs(cfg)
    if limit:
        inputs = inputs[:limit]

    dst_dir = _ensure(out_dir or get_scaled_dir(cfg))
    _ensure(cfg["inter_dir"])
    print(f"[VoxNorm] {cfg['institution']}/{cfg['disease']}/{cfg['version']} "
          f"| cases={len(inputs)} | WW/WL={cfg['WW']}/{cfg['WL']} "
          f"| resize=None -> {dst_dir}")

    rec = []
    for r in tqdm(inputs):
        fname, src = r["Fname"], r["src_path"]
        out_path = os.path.join(dst_dir, f"{fname}.nii.gz")
        try:
            if (not overwrite) and os.path.exists(out_path):
                rec.append({**r, "scaled_out": out_path, "shape": None,
                            "status": "SKIP_EXIST", "error": None})
                continue

            arr = voxnorm_to_array(src, ttype, cfg["WW"], cfg["WL"])  # (z, y, x)
            img = sitk.GetImageFromArray(arr)
            if set_spacing:
                img.SetSpacing((1.0, 1.0, 1.0))   # voxel norm output is 1 mm isotropic
            sitk.WriteImage(img, out_path)

            rec.append({**r, "scaled_out": out_path, "shape": arr.shape,
                        "status": "OK", "error": None})
        except Exception as e:
            rec.append({**r, "scaled_out": None, "shape": None,
                        "status": "FAILED", "error": str(e)})
            print(f"  [FAIL] {fname}: {e}")

    df = pd.DataFrame(rec)
    df.to_csv(os.path.join(cfg["inter_dir"], f"voxnorm_{cfg['version']}_history.csv"),
              index=False)
    n_ok = int((df["status"] == "OK").sum()) if len(df) else 0
    print(f"  done OK={n_ok}/{len(df)}")
    return df


# ==============================================================================
# VOI localization : 2D slices -> ResNet inference -> adrenal center slice index
# ==============================================================================
def make_voi_npy(cfg):
    """Split the Stage-1 volumes into single-slice .npy files for the classifier.

    Next steps: run_inference(cfg) -> search_voi_index(cfg)
    """
    src_dir = cfg["stage_resize_dir"]
    out_dir = _ensure(cfg["voi_npy_dir"])
    files = sorted(glob(os.path.join(src_dir, "*.nii.gz")))
    print(f"[VOI npy] {len(files)} volumes -> {out_dir}")

    fnames = []
    for f in tqdm(files):
        fname = fname_from_path(f)
        _img, arr = load_nii(f)
        for z in range(arr.shape[0]):
            nf = f"{fname}_slice_index{z:04d}.npy"
            np.save(os.path.join(out_dir, nf), arr[z])
            fnames.append(nf)

    df = pd.DataFrame({"fname": sorted(fnames)})
    df.to_csv(cfg["voi_npy_list_csv"], index=False)
    print(f"  slice list -> {cfg['voi_npy_list_csv']}")
    print("  next: run_inference(cfg) -> search_voi_index(cfg)")
    return out_dir, cfg["voi_npy_list_csv"]


class _SliceNpyDataset:
    """Loads the 2D .npy slices as single-channel tensors (inference only, no labels)."""

    def __init__(self, npy_dir, image_names, transform=None):
        self.npy_dir = npy_dir
        self.image_names = list(image_names)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, i):
        arr = np.load(os.path.join(self.npy_dir, self.image_names[i])).astype(np.float32)
        if self.transform is not None:
            x = self.transform(arr)            # ToTensor -> (1, H, W)
        else:
            import torch
            x = torch.from_numpy(arr)[None]    # (1, H, W)
        return x


def _build_resnet(num_classes, checkpoint_path, device):
    """ResNet-50 with a 1-channel stem and a 2-class head (adrenal present / absent)."""
    import torch
    from torch import nn
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"ResNet checkpoint not found: {checkpoint_path}\n"
            f"Pass checkpoint_path='<YOUR_PATH>/Best.pth' to build_pre_config().")
    try:
        import timm
        model = timm.models.resnet50()
    except Exception:
        from torchvision.models import resnet50
        model = resnet50(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def run_inference(cfg, num_classes=2, num_workers=4):
    """Classify every axial slice as adrenal / non-adrenal -> infer_result_csv."""
    import torch
    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    if cfg.get("infer_gpu") is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["infer_gpu"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    names = pd.read_csv(cfg["voi_npy_list_csv"])["fname"].astype(str).tolist()
    # sort by (case id, slice index) so that slice order is guaranteed regardless
    # of how the case identifier is formatted
    names = sorted(names, key=_parse_slice_name)
    transform = T.Compose([T.ToTensor()])  # (H, W) -> (1, H, W), float

    ds = _SliceNpyDataset(cfg["voi_npy_dir"], names, transform=transform)
    loader = DataLoader(ds, batch_size=cfg["infer_batch_size"],
                        num_workers=num_workers, shuffle=False)

    model = _build_resnet(num_classes, cfg["checkpoint_path"], device)
    print(f"[infer] {len(names)} slices | ckpt={cfg['checkpoint_path']} | device={device}")

    probs = []
    with torch.no_grad():
        for xb in tqdm(loader):
            xb = xb.to(device)
            logit = model(xb)
            p = torch.softmax(logit, dim=1).cpu().numpy()
            probs.append(p)
    prob = np.concatenate(probs, axis=0)
    pred = np.argmax(prob, axis=1)

    parsed = [_parse_slice_name(n) for n in names]
    df = pd.DataFrame()
    df["Fname"] = names
    df["ID"] = [c for c, _ in parsed]          # case identifier, any length
    df["slice_index"] = [z for _, z in parsed]
    df["Pred"] = pred
    df["Neg_prob"] = prob[:, 0]
    df["Pos_prob"] = prob[:, 1]
    df.to_csv(cfg["infer_result_csv"], index=False)
    print(f"  inference result -> {cfg['infer_result_csv']}")
    return df


def _cal_voi_idx(label_array, num_features):
    """Return (size, center index) of the longest run of positive slices."""
    max_size, max_center = 0, -1
    for i in range(1, num_features + 1):
        idx = np.where(label_array == i)[0]
        size = len(idx)
        if size > max_size:
            max_size = size
            max_center = idx[size // 2]
    return max_size, max_center


def search_voi_index(cfg):
    """Per case: longest consecutive run of adrenal-positive slices -> center index."""
    from scipy.ndimage import label as _label

    res = pd.read_csv(cfg["infer_result_csv"], dtype={"ID": str})
    if "slice_index" not in res.columns:  # backward compatibility with older CSVs
        res["slice_index"] = [_parse_slice_name(n)[1] for n in res["Fname"]]
    res = res.sort_values(["ID", "slice_index"]).reset_index(drop=True)
    case_list = sorted(res["ID"].unique().tolist())

    sizes, centers = [], []
    for case in case_list:
        pred = res.loc[res["ID"] == case, "Pred"].tolist()
        lab, n = _label(pred)
        size, center = _cal_voi_idx(lab, n)
        sizes.append(size)
        centers.append(center)

    voi = pd.DataFrame({"fname": case_list, "voi_center_idx": centers, "voi_size": sizes})
    voi.to_csv(cfg["voi_info_csv"], index=False)
    print(f"[search] cases={len(case_list)} -> {cfg['voi_info_csv']}")
    bad = voi[voi["voi_center_idx"] < 0]
    if len(bad):
        print(f"  [WARN] no adrenal gland detected in {len(bad)} case(s): {bad['fname'].tolist()}\n"
              f"         inspect these manually and, if needed, fix voi_center_idx in "
              f"{cfg['voi_info_csv']} before running run_stage2_depthcrop().")
    return voi


# ==============================================================================
# Stage 2 : depth crop (needs the VOI center index from the step above)
# ==============================================================================
def run_stage2_depthcrop(cfg, voi_csv=None, fname_col="fname", center_col="voi_center_idx"):
    """Stage-1 volumes + voi_info.csv -> depth cropped FINAL volumes in cfg['out_dir'].

    Supply your own CSV via `voi_csv=` if you localized the adrenal gland by
    other means; it needs the columns given by fname_col / center_col.
    """
    if not cfg["do_depth_crop"]:
        print(f"[Stage2] depth crop disabled for '{cfg['disease']}' "
              f"(the Stage-1 output is already final).")
        return None

    voi_csv = voi_csv or cfg["voi_info_csv"]
    voi = pd.read_csv(voi_csv, dtype={fname_col: str})
    voi_map = dict(zip(voi[fname_col].astype(str), voi[center_col]))

    src_dir = cfg["stage_resize_dir"]
    out_dir = _ensure(cfg["out_dir"])
    _ensure(cfg["inter_dir"])
    files = sorted(glob(os.path.join(src_dir, "*.nii.gz")))
    print(f"[Stage2] depth crop {len(files)} volumes -> {out_dir} "
          f"(depth={cfg['target_depth']}) | voi={voi_csv}")

    rec = []
    for f in tqdm(files):
        fname = fname_from_path(f)
        try:
            if fname not in voi_map or int(voi_map[fname]) < 0:
                rec.append({"Fname": fname, "out": None, "status": "NO_VOI_INFO", "error": None})
                print(f"  [skip] {fname}: no usable VOI info")
                continue
            _img, arr = load_nii(f)
            cropped = depth_crop_volume(arr, voi_map[fname],
                                        cfg["target_depth"], cfg["target_size"])
            out_path = os.path.join(out_dir, f"{fname}.nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(cropped), out_path)
            rec.append({"Fname": fname, "out": out_path, "shape": cropped.shape,
                        "voi_center": int(voi_map[fname]), "status": "OK", "error": None})
        except Exception as e:
            rec.append({"Fname": fname, "out": None, "status": "FAILED", "error": str(e)})
            print(f"  [FAIL] {fname}: {e}")

    df = pd.DataFrame(rec)
    df.to_csv(os.path.join(cfg["inter_dir"], f"stage2_{cfg['version']}_history.csv"),
              index=False)
    n_ok = int((df["status"] == "OK").sum()) if len(df) else 0
    print(f"  done OK={n_ok}/{len(df)}")
    return df


# ==============================================================================
# Export : rename the final volumes for nnU-Net inference
# ==============================================================================
def export_for_nnunet(cfg, out_dir=None, channel="0000", copy=True):
    """Copy the final volumes to <CaseID>_0000.nii.gz in a flat folder.

    nnU-Net v2 requires the 4-digit channel suffix on every input file. Pass the
    resulting folder to `nnUNetv2_predict -i`. The predicted segmentations come
    back named <CaseID>.nii.gz (without the suffix).
    """
    import shutil

    src_dir = cfg["out_dir"]
    dst_dir = _ensure(out_dir or cfg["nnunet_input_dir"])
    files = sorted(glob(os.path.join(src_dir, "*.nii.gz")))
    if not files:
        print(f"[export] nothing to export - {src_dir} is empty.")
        return dst_dir

    for f in tqdm(files):
        fname = fname_from_path(f)
        dst = os.path.join(dst_dir, f"{fname}_{channel}.nii.gz")
        if copy:
            shutil.copy2(f, dst)
        else:
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(f), dst)
    print(f"[export] {len(files)} volumes -> {dst_dir}  (pass this to nnUNetv2_predict -i)")
    return dst_dir


# ==============================================================================
# Convenience : whole pipeline in one call
# ==============================================================================
def run_all(cfg, limit=None, export=True):
    """Stage 1 -> VOI localization -> Stage 2 -> nnU-Net export."""
    print_config(cfg)
    run_stage1(cfg, limit=limit)
    if cfg["do_depth_crop"]:
        make_voi_npy(cfg)
        run_inference(cfg)
        search_voi_index(cfg)
        run_stage2_depthcrop(cfg)
    if export:
        export_for_nnunet(cfg)
    return cfg["out_dir"]


# ==============================================================================
# QC
# ==============================================================================
def qc_show(cfg, which="final", show_range=(None, None)):
    """Display the mid axial slice of the processed volumes for a quick visual check."""
    import matplotlib.pyplot as plt
    d = cfg["out_dir"] if which == "final" else cfg["stage_resize_dir"]
    files = sorted(glob(os.path.join(d, "*.nii.gz")))[show_range[0]:show_range[1]]
    if not files:
        print(f"[qc] no volumes found in {d}")
        return
    for f in files:
        _img, arr = load_nii(f)
        z = arr.shape[0] // 2
        plt.figure(figsize=(3, 3))
        plt.imshow(arr[z], cmap="gray")
        plt.title(f"{fname_from_path(f)}\n{arr.shape} z={z}", fontsize=8)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# ==============================================================================
# CLI
# ==============================================================================
def _build_argparser():
    p = argparse.ArgumentParser(
        description="Adrenal CT preprocessing for the nnU-Net segmentation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-dir", default=None,
                   help="root holding ./data and ./preprocessed "
                        "(default: ADRENAL_BASE_DIR env var, else the repository root)")
    p.add_argument("--data-root", default=None, help="override <base-dir>/data")
    p.add_argument("--preprocessed-root", default=None, help="override <base-dir>/preprocessed")
    p.add_argument("--institution", default="MyInstitution")
    p.add_argument("--disease", default="Adenoma")
    p.add_argument("--version", default="total",
                   choices=["precontrast", "postcontrast", "total"])
    p.add_argument("--target-size", type=int, default=None)
    p.add_argument("--target-depth", type=int, default=None)
    p.add_argument("--no-depth-crop", action="store_true")
    p.add_argument("--checkpoint", default=None, help="path to the VOI ResNet Best.pth")
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--limit", type=int, default=None, help="process only the first N cases")
    p.add_argument("--all", action="store_true", help="run the full pipeline")
    p.add_argument("--voxnorm-only", action="store_true",
                   help="only 1 mm resampling + windowing (no resize / depth crop)")
    return p


def main():
    args = _build_argparser().parse_args()
    cfg = build_pre_config(
        institution=args.institution, disease=args.disease, version=args.version,
        base_dir=args.base_dir, data_root=args.data_root,
        preprocessed_root=args.preprocessed_root,
        target_size=args.target_size, target_depth=args.target_depth,
        do_depth_crop=False if args.no_depth_crop else None,
        checkpoint_path=args.checkpoint, infer_gpu=args.gpu,
    )
    if args.voxnorm_only:
        print_config(cfg)
        run_voxnorm_only(cfg, limit=args.limit)
    elif args.all:
        run_all(cfg, limit=args.limit)
    else:
        print_config(cfg)
        print("\nNothing to do. Add --all (full pipeline) or --voxnorm-only.")


if __name__ == "__main__":
    main()
