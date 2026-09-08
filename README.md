# Adrenal Gland and Tumor Segmentation (nnU-Net v2)

Trained weights, CT preprocessing code, and inference instructions for automatic
segmentation of the adrenal gland and adrenal tumors on abdominal CT.

| | |
|---|---|
| Task | `Dataset001_AdrenalTumor` |
| Architecture | nnU-Net v2, `3d_fullres`, `nnUNetTrainer__nnUNetPlans` |
| Input | single-channel CT, 1 x 1 x 1 mm isotropic, soft-tissue window (WW 400 / WL 40) |
| Training volume size | **(128, 192, 192)** = (depth/z, height/y, width/x) |
| Network patch size | (96, 160, 160), sliding-window inference |
| Labels | `0` background, `1` adrenal gland, `2` tumor |
| Training cases | 373, 5-fold cross-validation |

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── tools/
│   └── make_requirements.py    regenerate requirements.txt from your environment
├── notebooks/
│   └── adrenal_pipeline.ipynb      annotated end-to-end walkthrough  <- start here
├── preprocess/
│   ├── preprocess.py               CT preprocessing pipeline
│   └── VOI_extraction/
│       ├── Best.pth                ResNet-50 slice classifier (adrenal localization)
│       └── log.txt                 its training log
└── nnUNet-master/                  nnU-Net v2 (source + trained weights)
    ├── nnunetv2/
    ├── nnUNet_results/Dataset001_AdrenalTumor/
    │   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
    │       ├── plans.json, dataset.json
    │       └── fold_0 ... fold_4/checkpoint_best.pth
    └── documentation/
```

---

## 1. Installation

```bash
git clone <THIS_REPOSITORY_URL>
cd <REPO_ROOT>

conda create -n adrenal python=3.9 -y
conda activate adrenal

# PyTorch matching your CUDA version - see https://pytorch.org/get-started/locally/
pip install torch torchvision

pip install -e ./nnUNet-master
pip install -r requirements.txt
```

Set the nnU-Net environment variables. Only `nnUNet_results` matters for inference:

```bash
export nnUNet_raw="<REPO_ROOT>/nnUNet-master/nnUNet_raw"
export nnUNet_preprocessed="<REPO_ROOT>/nnUNet-master/nnUNet_preprocessed"
export nnUNet_results="<REPO_ROOT>/nnUNet-master/nnUNet_results"
```

Reference: [nnU-Net environment variables](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md)

### Regenerating `requirements.txt`

The shipped `requirements.txt` is unpinned. To pin it to the versions that actually work
on your machine, run this from the repository root inside your environment:

```bash
python tools/make_requirements.py -o requirements.txt
```

It scans the repo (including notebook code cells) for imports, drops standard-library and
local modules, and writes `package==installed_version` for the rest.

---

## 2. Prepare your data

Pick any folder as `<YOUR_PROJECT_ROOT>` and arrange your CT volumes like this.
`<CaseID>` can be any string; it is carried through to the predicted segmentation filename.

```
<YOUR_PROJECT_ROOT>/
└── data/
    └── <INSTITUTION>/
        └── <DISEASE>/
            ├── precontrast/<CaseID>.nii.gz
            └── postcontrast/<CaseID>.nii.gz
```

`<CaseID>/<CaseID>.nii.gz` (one subfolder per case) is also accepted.

---

## 3. Preprocessing

```bash
python preprocess/preprocess.py \
    --base-dir <YOUR_PROJECT_ROOT> \
    --institution <YOUR_INSTITUTION> \
    --disease Adenoma \
    --version precontrast \
    --all
```

or, from Python / the notebook:

```python
from preprocess import build_pre_config, run_all

cfg = build_pre_config(institution="<YOUR_INSTITUTION>",
                       disease="<YOUR_COHORT>",     # any label: Adenoma, NF, ACC, CS, ...
                       version="precontrast",
                       base_dir="<YOUR_PROJECT_ROOT>")
run_all(cfg)
```

`version` selects the contrast phase: `precontrast`, `postcontrast`, or `total`
(post-contrast when available, otherwise pre-contrast). `institution` and `disease` are
only folder names - any label works, and every cohort is processed with the same settings
(`target_size=192`, `target_depth=128`, the ones the released model was trained with).
Pass `target_size=` / `target_depth=` / `do_depth_crop=` explicitly to change them.

### What it does

| # | Step | Operation |
|---|---|---|
| 1 | Voxel normalization | resample to 1 x 1 x 1 mm isotropic (linear interpolation), clip HU to WW 400 / WL 40 |
| 2 | In-plane resize | slice-wise center crop to 192 x 192 (pad with the volume minimum if smaller) |
| 3 | Depth crop | keep 128 axial slices centered on the adrenal gland |

The crop center in step 3 comes from a pre-trained 2D ResNet-50
(`preprocess/VOI_extraction/Best.pth`) that labels each axial slice as adrenal
present/absent; the middle of the longest positive run becomes the center.

Intensity normalization is **not** applied here - nnU-Net applies its own `CTNormalization`
internally at inference time.

### What it writes

Everything lands under `<YOUR_PROJECT_ROOT>/preprocessed/`:

```
preprocessed/
├── <INSTITUTION>/<DISEASE>/
│   ├── <VERSION>/                      FINAL volumes, (128, 192, 192)
│   ├── <VERSION>_scaled/               run_voxnorm_only() output (optional)
│   └── _intermediate/                  safe to delete after a successful run
│       ├── resize_<VERSION>/           after steps 1-2
│       ├── voi_npy_<VERSION>/          2D slices fed to the ResNet
│       ├── infer_result_<VERSION>.csv  per-slice adrenal probability
│       ├── voi_info_<VERSION>.csv      per-case adrenal center slice index
│       └── *_history.csv               per-stage processing log
└── nnunet_input/<INSTITUTION>_<DISEASE>_<VERSION>/
                                        <CaseID>_0000.nii.gz  -> nnUNetv2_predict -i
```

Call `print_config(cfg)` to see every resolved path before processing.

### Options

* **Supply your own VOI center.** Skip the classifier and pass a CSV with columns
  `fname`, `voi_center_idx`:
  `run_stage2_depthcrop(cfg, voi_csv="<YOUR_PATH>/voi_info.csv")`
* **Skip resize and depth crop.** `run_voxnorm_only(cfg)` applies only the 1 mm resampling
  and the CT window, keeping the original field of view. The model handles this too.
* **Field of view — important for large tumors.** The 192 x 192 x 128 crop was chosen for
  adrenal adenomas, which are small. Large lesions, adrenocortical carcinoma (ACC) in
  particular, can exceed it, and anything outside the crop is discarded before the model
  sees it. Check the `qc_show()` images: if a lesion touches the border, enlarge the field
  of view with `build_pre_config(..., target_size=320, target_depth=192)`, or keep the full
  CT with `run_voxnorm_only(cfg)`. This costs inference time but no accuracy — nnU-Net
  accepts any input size (see section 4).
* **Image geometry.** Output volumes carry 1 mm isotropic spacing, a zero origin and an
  identity direction; the original scan's origin/direction are not preserved. Predictions
  inherit the geometry of the input you give nnU-Net, so this is self-consistent, but
  overlaying a prediction on the *original* CT requires re-registration.
* **Cases with no detection.** `voi_center_idx = -1` means the classifier found no adrenal
  slice; those cases are skipped by the depth crop. Review them and set the index manually
  in `voi_info_<version>.csv`.

---

## 4. Inference

Input files must be named `<CaseID>_0000.nii.gz` (`_0000` is the nnU-Net channel suffix).
`export_for_nnunet(cfg)`, which `run_all` calls for you, produces exactly that.

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> nnUNetv2_predict \
    -i <YOUR_PROJECT_ROOT>/preprocessed/nnunet_input/<INSTITUTION>_<DISEASE>_<VERSION> \
    -o <YOUR_OUTPUT_PATH>/predictions \
    -d 001 \
    -f 0 \
    -c 3d_fullres \
    -chk checkpoint_best.pth
```

* `-f 0` uses fold 0 only; `-f 0 1 2 3 4` ensembles all five folds (slower, usually more robust).
* `--save_probabilities` additionally writes the softmax `.npz` per case - only needed for
  ensembling across runs.
* `-device cpu` runs without a GPU. `-npp 1 -nps 1` lowers RAM usage.

### On input size

The model was trained on **(128, 192, 192)** volumes, in numpy/SimpleITK axis order
`(z, y, x)` = (depth, height, width): 128 axial slices of 192 x 192 pixels. The same volume
read through SimpleITK's image API (`GetSize()`) reports the reversed `(x, y, z)` =
(192, 192, 128).

**Your inputs do not have to match this size.** nnU-Net performs sliding-window inference
with a (96, 160, 160) patch and stitches the result, so any shape works - volumes smaller
than the patch are padded automatically, larger ones simply take more windows. Full
uncropped 1 mm CTs from `run_voxnorm_only()` are valid input.

What does matter is the **voxel spacing**: the plans file specifies 1 x 1 x 1 mm and nnU-Net
resamples anything else to it before inference. The preprocessing above already produces
that spacing, so no additional interpolation is introduced.

---

## 5. Output

Segmentations are written as `<CaseID>.nii.gz` with the same geometry as the input.

| Value | Meaning |
|---|---|
| 0 | background |
| 1 | adrenal gland |
| 2 | tumor |

Labels are region-based: label `1` is the gland *including* the tumor and label `2` is the
tumor inside it. The whole gland is therefore `prediction > 0`, and the tumor alone is
`prediction == 2`.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `nnUNet_results is not defined` | export the environment variables before importing `nnunetv2` |
| nnU-Net finds 0 cases | input files are missing the `_0000` suffix - run `export_for_nnunet(cfg)` |
| `RuntimeError` about the dataset id | `-d 001` must match `Dataset001_AdrenalTumor` inside `nnUNet_results` |
| `voi_center_idx = -1` | no adrenal slice detected; check orientation and QC images, then set it manually |
| `depth crop shape mismatch` | Stage-1 output is not `target_size` x `target_size`; re-run `run_stage1` with the matching size |
| Predictions look shifted or mirrored | run the section-3 preprocessing first; the model expects that orientation |
| CUDA out of memory | `-device cpu`, `-npp 1 -nps 1`, or predict cropped volumes instead of full CTs |

---

## Citation and license

nnU-Net is developed by the Division of Medical Image Computing, DKFZ, and is distributed
under the Apache 2.0 license.

> Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021).
> nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
> *Nature Methods*, 18(2), 203-211.

<!-- Add your own citation, license, and contact information here. -->
